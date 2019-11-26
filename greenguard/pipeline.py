# -*- coding: utf-8 -*-

import logging
import os
from collections import defaultdict

import cloudpickle
import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold

from greenguard.metrics import METRICS

LOGGER = logging.getLogger(__name__)


PIPELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pipelines'))


def get_pipelines():
    pipelines = dict()
    for filename in os.listdir(PIPELINES_DIR):
        if filename.endswith('.json'):
            name = os.path.basename(filename)[:-len('.json')]
            path = os.path.join(PIPELINES_DIR, filename)
            pipelines[name] = path

    return pipelines


class GreenGuardPipeline(object):

    template = None
    fitted = False
    cv_score = None

    _cv_class = None
    _metric = None
    _cost = False
    _tuner = None
    _pipeline = None
    _splits = None

    def _get_cv(self, stratify, cv_splits, shuffle, random_state):
        if stratify:
            cv_class = StratifiedKFold
        else:
            cv_class = KFold

        return cv_class(n_splits=cv_splits, shuffle=shuffle, random_state=random_state)

    def _count_static_steps(self):
        tunable_hyperparams = self._pipeline.get_tunable_hyperparameters()
        for index, block_name in enumerate(self._pipeline.blocks.keys()):
            if tunable_hyperparams[block_name]:
                return index + 1

    def __init__(self, template, metric, cost=False, hyperparameters=None, stratify=True,
                 cv_splits=5, shuffle=True, random_state=0, preprocessing=0):

        self._cv = self._get_cv(stratify, cv_splits, shuffle, random_state)

        if isinstance(metric, str):
            metric, cost = METRICS[metric]

        self._metric = metric
        self._cost = cost

        self.template = template
        self._pipeline = MLPipeline(template)
        self._static = self._count_static_steps()
        self._preprocessing = preprocessing

        if self._preprocessing >= self._static:
            raise ValueError('Preprocessing cannot be bigger than static')

        if hyperparameters:
            self._pipeline.set_hyperparameters(hyperparameters)

    def get_hyperparameters(self):
        return self._pipeline.get_hyperparameters()

    def set_hyperparameters(self, hyperparameters):
        self._pipeline.set_hyperparameters(hyperparameters)
        self.fitted = False

    @staticmethod
    def _clone_pipeline(pipeline):
        return MLPipeline.from_dict(pipeline.to_dict())

    def _is_better(self, score):
        if self._cost:
            return score < self.cv_score

        return score > self.cv_score

    def _generate_splits(self, X, y, readings):
        if self._preprocessing:
            pipeline = MLPipeline(self.template)
            LOGGER.debug('Running %s preprocessing steps', self._preprocessing)
            context = pipeline.fit(X=X, y=y, readings=readings, output_=self._preprocessing - 1)
            del context['X']
            del context['y']
        else:
            context = {
                'readings': readings
            }

        splits = list()
        for fold, (train_index, test_index) in enumerate(self._cv.split(X, y)):
            LOGGER.debug('Running static steps for fold %s', fold)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline = MLPipeline(self.template)
            fit = pipeline.fit(X_train, y_train, output_=self._static - 1,
                               start_=self._preprocessing, **context)
            predict = pipeline.predict(X_test, output_=self._static - 1,
                                       start_=self._preprocessing, **context)

            splits.append((fold, pipeline, fit, predict, y_test))

        return splits

    def cross_validate(self, X=None, y=None, readings=None, params=None):
        if self._splits is None:
            LOGGER.info('Running static steps before cross validation')
            self._splits = self._generate_splits(X, y, readings)

        scores = []
        for fold, pipeline, fit, predict, y_test in self._splits:
            LOGGER.debug('Scoring fold %s', fold)

            if params:
                pipeline.set_hyperparameters(params)
            else:
                pipeline.set_hyperparameters(self._pipeline.get_hyperparameters())

            pipeline.fit(start_=self._static, **fit)
            predictions = pipeline.predict(start_=self._static, **predict)

            score = self._metric(y_test, predictions)

            LOGGER.debug('Fold fold %s score: %s', fold, score)
            scores.append(score)

        return np.mean(scores)

    def _to_dicts(self, hyperparameters):
        params_tree = defaultdict(dict)
        for (block, hyperparameter), value in hyperparameters.items():
            if isinstance(value, np.integer):
                value = int(value)

            elif isinstance(value, np.floating):
                value = float(value)

            elif isinstance(value, np.ndarray):
                value = value.tolist()

            elif value == 'None':
                value = None

            params_tree[block][hyperparameter] = value

        return params_tree

    def _to_tuples(self, params_tree, tunable_keys):
        param_tuples = defaultdict(dict)
        for block_name, params in params_tree.items():
            for param, value in params.items():
                key = (block_name, param)
                if key in tunable_keys:
                    param_tuples[key] = 'None' if value is None else value

        return param_tuples

    def _get_tunables(self):
        tunables = []
        tunable_keys = []
        for block_name, params in self._pipeline.get_tunable_hyperparameters().items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = 'string' if param_type == 'str' else param_type

                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                tunable_keys.append(key)

        return tunables, tunable_keys

    def _get_tuner(self):
        tunables, tunable_keys = self._get_tunables()
        tuner = GP(tunables)

        # Inform the tuner about the score that the default hyperparmeters obtained
        param_tuples = self._to_tuples(self._pipeline.get_hyperparameters(), tunable_keys)
        tuner.add(param_tuples, self.cv_score)

        return tuner

    def tune(self, X=None, y=None, readings=None, iterations=10):
        if not self._tuner:
            LOGGER.info('Scoring the default pipeline')
            self.cv_score = self.cross_validate(X, y, readings)

            LOGGER.info('Default Pipeline score: %s', self.cv_score)

            self._tuner = self._get_tuner()

        for i in range(iterations):
            LOGGER.info('Scoring pipeline %s', i + 1)

            params = self._tuner.propose(1)
            param_dicts = self._to_dicts(params)

            try:
                score = self.cross_validate(params=param_dicts)

                LOGGER.info('Pipeline %s score: %s', i + 1, score)

                if self._is_better(score):
                    self.cv_score = score
                    self.set_hyperparameters(param_dicts)

                self._tuner.add(params, score)

            except Exception:
                failed = '\n'.join('{}: {}'.format(k, v) for k, v in params.items())
                LOGGER.exception("Caught an exception scoring pipeline %s with params:\n%s",
                                 i + 1, failed)

    def fit(self, X, y, readings):
        self._pipeline.fit(X, y, readings=readings)
        self.fitted = True

    def predict(self, X, readings):
        if not self.fitted:
            raise NotFittedError()

        return self._pipeline.predict(X, readings=readings)

    def save(self, path):
        with open(path, 'wb') as pickle_file:
            cloudpickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as pickle_file:
            return cloudpickle.load(pickle_file)
