# -*- coding: utf-8 -*-

import json
import logging
import os
from collections import defaultdict

import cloudpickle
import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

LOGGER = logging.getLogger(__name__)


PIPELINES_DIR = os.path.join(os.path.dirname(__file__), 'pipelines')

METRICS = {
    'accuracy': (accuracy_score, False),
    'r2': (r2_score, False),
    'mse': (mean_squared_error, True),
    'mae': (mean_absolute_error, True)
}


class GreenGuardPipeline(object):

    template = None
    fitted = False
    score = None

    _cv_class = None
    _metric = None
    _cost = False
    _tuner = None
    _pipeline = None

    def _get_cv(self, stratify, cv_splits, shuffle, random_state):
        if stratify:
            cv_class = StratifiedKFold
        else:
            cv_class = KFold

        return cv_class(n_splits=cv_splits, shuffle=shuffle, random_state=random_state)

    def _load_template(self, template):
        if not os.path.isfile(template):
            template = os.path.join(PIPELINES_DIR, template + '.json')

        with open(template, 'r') as template_file:
            return json.load(template_file)

    def _load_mlpipeline(self, template):
        if not isinstance(template, dict):
            template = self._load_template(template)

        self.template = template

        return MLPipeline.from_dict(template)

    def __init__(self, template, metric, cost=False, hyperparameters=None,
                 stratify=True, cv_splits=5, shuffle=True, random_state=0):

        self._cv = self._get_cv(stratify, cv_splits, shuffle, random_state)

        if isinstance(metric, str):
            metric, cost = METRICS[metric]

        self._metric = metric
        self._cost = cost

        self._pipeline = self._load_mlpipeline(template)

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
            return score < self.score

        return score > self.score

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

    def _score_pipeline(self, pipeline, X, y, tables):
        scores = []

        for fold, (train_index, test_index) in enumerate(self._cv.split(X, y)):
            LOGGER.debug('Scoring fold %s', fold)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline = self._clone_pipeline(pipeline)
            pipeline.fit(X_train, y_train, **tables)

            predictions = pipeline.predict(X_test, **tables)
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

    def _get_tuner(self):
        tunables, tunable_keys = self._get_tunables()
        tuner = GP(tunables)

        # Inform the tuner about the score that the default hyperparmeters obtained
        param_tuples = self._to_tuples(self._pipeline.get_hyperparameters(), tunable_keys)
        tuner.add(param_tuples, self.score)

        return tuner

    def tune(self, X, y, tables, iterations=10):
        tables.setdefault('entityset', None)
        if not self._tuner:
            LOGGER.info('Scoring the default pipeline')
            self.score = self._score_pipeline(self._pipeline, X, y, tables)
            LOGGER.info('Default Pipeline score: %s', self.score)

            self._tuner = self._get_tuner()

        for i in range(iterations):
            LOGGER.info('Scoring pipeline %s', i + 1)

            params = self._tuner.propose(1)
            param_dicts = self._to_dicts(params)

            candidate = self._clone_pipeline(self._pipeline)
            candidate.set_hyperparameters(param_dicts)

            try:
                score = self._score_pipeline(candidate, X, y, tables)

                LOGGER.info('Pipeline %s score: %s', i + 1, score)

                if self._is_better(score):
                    self.score = score
                    self.set_hyperparameters(param_dicts)

                self._tuner.add(params, score)

            except Exception:
                failed = '\n'.join('{}: {}'.format(k, v) for k, v in params.items())
                LOGGER.exception("Caught an exception scoring pipeline %s with params:\n%s",
                                 i + 1, failed)

    def fit(self, X, y, tables):
        tables.setdefault('entityset', None)
        self._pipeline.fit(X, y, **tables)
        self.fitted = True

    def predict(self, X, tables):
        if not self.fitted:
            raise NotFittedError()

        tables.setdefault('entityset', None)
        return self._pipeline.predict(X, **tables)

    def save(self, path):
        with open(path, 'wb') as pickle_file:
            cloudpickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as pickle_file:
            return cloudpickle.load(pickle_file)
