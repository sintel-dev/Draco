# -*- coding: utf-8 -*-

import logging
import os
from collections import defaultdict
from copy import deepcopy

import cloudpickle
import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from mlblocks.discovery import load_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold

from greenguard.metrics import METRICS

LOGGER = logging.getLogger(__name__)


PIPELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pipelines'))


def get_pipelines(pattern='', path=False, unstacked=False):
    """Get the list of available pipelines.

    Optionally filter the names using a patter or obtain
    the paths to the pipelines alongside their name.

    Args:
        pattern (str):
            Pattern to search for in the pipeline names
        path (bool):
            Whether to return a dictionary containing the pipeline
            paths instead of only a list with the names.
        unstacked (bool):
            Whether to load the pipelines that expect the readings
            to be already unstacked by signal_id. Defaults to ``False``.

    Return:
        list or dict:
            List of available and matching pipeline names.
            If `path=True`, return a dict containing the pipeline
            names as keys and their absolute paths as values.
    """
    pipelines = dict()
    pipelines_dir = PIPELINES_DIR
    if unstacked:
        pipelines_dir = os.path.join(pipelines_dir, 'unstacked')

    for filename in os.listdir(pipelines_dir):
        if filename.endswith('.json') and pattern in filename:
            name = os.path.basename(filename)[:-len('.json')]
            pipeline_path = os.path.join(PIPELINES_DIR, filename)
            pipelines[name] = pipeline_path

    if not path:
        pipelines = list(pipelines)

    return pipelines


class GreenGuardPipeline(object):
    """Main Machine Learning component in the GreenGuard project.

    The ``GreenGuardPipeline`` represents the abstraction of a Machine
    Learning pipeline architecture specialized on the GreenGuard data
    format.

    In order to use it, an MLBlocks pipeline template needs to be given,
    alongside information about how to evaluate its performance using
    cross validation.

    Attributes:
        template (MLPipeline):
            MLPipeline instance used as the template for tuning.
        template_name:
            Name of the template being used.
        fitted (bool):
            Whether this GreenGuardPipeline has already been fitted or not.
        steps (list):
            List of primitives that compose this template.
        preprocessing (list):
            List of preprocessing steps. These steps have no learning stage
            and are executed only once on the complete training dataset, before
            partitioning it for cross validation.
        static (list):
            List of static steps. These are all the steps in the pipeline that
            come after the preprocessing ones but have no hyperparameters.
            These are executed on each cross validation split only once, when
            the data is partitioned, and their output is cached to be reused
            later on at every tuning iteration.
        tunable (list):
            List of steps that have hyperparameters and will be tuned during
            the tuning loop.

    Args:
        template (str or MLPipeline):
            Template to use. If a ``str`` is given, load the corresponding
            ``MLPipeline``.
        metric (str or function):
            Metric to use. If an ``str`` is give it must be one of the metrics
            defined in the ``greenguard.metrics.METRICS`` dictionary.
        cost (bool):
            Whether the metric is a cost function (the lower the better) or not.
            Defaults to ``False``.
        init_params (dict):
            Initial parameters to pass to the underlying MLPipeline if something
            other than the defaults need to be used.
            Defaults to ``None``.
        stratify (bool):
            Whether to stratify the data when partitioning for cross validation.
            Defaults to ``True``.
        cv_splits (int):
            Number of cross validation folds to use. Defaults to ``5``.
        shuffle (bool):
            Whether to shuffle the data when partitioning for cross validation.
            Defaults to ``True``.
        random_state (int or RandomState):
            random state to use for the cross validation partitioning.
            Defaults to ``0``.
        preprocessing (int):
            Number of steps to execute during the preprocessing stage.
            The number of preprocessing steps cannot be higher than the
            number of static steps in the given template.
            Defaults to ``0``.
    """

    template = None
    template_name = None
    fitted = False
    cv_score = None

    _cv_class = None
    _metric = None
    _cost = False
    _tuner = None
    _pipeline = None
    _splits = None
    _static = None

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
                return index

        return 0

    def _build_pipeline(self):
        self._pipeline = MLPipeline(self.template)
        if self._hyperparameters:
            self._pipeline.set_hyperparameters(self._hyperparameters)

        self.fitted = False

    @staticmethod
    def _update_params(old, new):
        for name, params in new.items():
            if '#' not in name:
                name = name + '#1'

            block_params = old.setdefault(name, dict())
            for param, value in params.items():
                block_params[param] = value

    def set_init_params(self, init_params):
        """Set new init params for the template and pipeline.

        Args:
            init_params (dict):
                New init_params to use.
        """
        template_params = self.template['init_params']
        self._update_params(template_params, init_params)
        self._build_pipeline()

    def __init__(self, template, metric, cost=False, init_params=None, stratify=True,
                 cv_splits=5, shuffle=True, random_state=0, preprocessing=0):

        self._cv = self._get_cv(stratify, cv_splits, shuffle, random_state)

        if isinstance(metric, str):
            metric, cost = METRICS[metric]

        self._metric = metric
        self._cost = cost

        if isinstance(template, str):
            self.template_name = template
            self.template = load_pipeline(template)
        else:
            self.template = template

        # Make sure to have block number in all init_params names
        template_params = self.template.setdefault('init_params', dict())
        for name, params in list(template_params.items()):
            if '#' not in name:
                template_params[name + '#1'] = template_params.pop(name)

        self._hyperparameters = dict()
        if init_params:
            self.set_init_params(init_params)
        else:
            self._build_pipeline()

        self._static = self._count_static_steps()
        self._preprocessing = preprocessing

        self.steps = self._pipeline.primitives.copy()
        self.preprocessing = self.steps[:self._preprocessing]
        self.static = self.steps[self._preprocessing:self._static]
        self.tunable = self.steps[self._static:]

        if self._preprocessing and (self._preprocessing > self._static):
            raise ValueError('Preprocessing cannot be bigger than static')

    def __repr__(self):
        return (
            "GreenGuardPipeline({})\n"
            "  preprocessing:\n{}\n"
            "  static:\n{}\n"
            "  tunable:\n{}\n"
        ).format(
            self.template_name,
            '\n'.join('    {}'.format(step) for step in self.preprocessing),
            '\n'.join('    {}'.format(step) for step in self.static),
            '\n'.join('    {}'.format(step) for step in self.tunable),
        )

    def get_hyperparameters(self):
        """Get the current hyperparameters.

        Returns:
            dict:
                Current hyperparameters.
        """
        return deepcopy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameters for this pipeline instance.

        The template ``init_params`` remain unmodified.

        Args:
            hyperparameters (dict):
                New hyperparameters to use.
        """
        self._update_params(self._hyperparameters, hyperparameters)
        self._build_pipeline()

    @staticmethod
    def _clone_pipeline(pipeline):
        return MLPipeline.from_dict(pipeline.to_dict())

    def _is_better(self, score):
        if self._cost:
            return score < self.cv_score

        return score > self.cv_score

    def _generate_splits(self, X, y, readings, turbines=None):
        if self._preprocessing:
            pipeline = MLPipeline(self.template)
            LOGGER.debug('Running %s preprocessing steps', self._preprocessing)
            context = pipeline.fit(X=X, y=y, readings=readings,
                                   turbines=turbines, output_=self._preprocessing - 1)
            del context['X']
            del context['y']
        else:
            context = {
                'readings': readings,
                'turbines': turbines,
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

    def cross_validate(self, X=None, y=None, readings=None, turbines=None, params=None):
        """Compute cross validation score using the given data.

        If the splits have not been previously computed, compute them now.
        During this computation, the data is partitioned using the indicated
        cross validation parameters and later on processed using the
        pipeline static steps.

        The results of the fit and produce executions are cached and reused
        in subsequent calls to this method.

        Args:
            X (pandas.DataFrame):
                ``target_times`` data, without the ``target`` column.
                Only needed if the splits have not been previously computed.
            y (pandas.Series or numpy.ndarray):
                ``target`` vector corresponding to the passed ``target_times``.
                Only needed if the splits have not been previously computed.
            readings (pandas.DataFrame):
                ``readings`` table. Only needed if the splits have not been
                previously computed.
            turbines (pandas.DataFrame):
                ``turbines`` table. Only needed if the splits have not been
                previously computed.
            params (dict):
                hyperparameter values to use.

        Returns:
            float:
                Computed cross validation score. This score is the average
                of the scores obtained accross all the cross validation folds.
        """

        if self._splits is None:
            LOGGER.info('Running static steps before cross validation')
            self._splits = self._generate_splits(X, y, readings, turbines)

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

        cv_score = np.mean(scores)
        if self.cv_score is None:
            self.cv_score = cv_score

        return cv_score

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

    def tune(self, target_times=None, readings=None, turbines=None, iterations=10):
        """Tune this pipeline for the indicated number of iterations.

        Args:
            target_times (pandas.DataFrame):
                ``target_times`` table, containing the ``turbine_id``, ``cutoff_time``
                and ``target`` columns.
                Only needed if the splits have not been previously computed.
            readings (pandas.DataFrame):
                ``readings`` table. Only needed if the splits have not been
                previously computed.
            turbines (pandas.DataFrame):
                ``turbines`` table. Only needed if the splits have not been
                previously computed.
            iterations (int):
                Number of iterations to perform.
        """
        if not self._tuner:
            LOGGER.info('Scoring the default pipeline')
            X = target_times[['turbine_id', 'cutoff_time']]
            y = target_times['target']
            self.cv_score = self.cross_validate(X, y, readings, turbines)

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

    def fit(self, target_times, readings, turbines=None):
        """Fit this pipeline to the given data.

        Args:
            target_times (pandas.DataFrame):
                ``target_times`` table, containing the ``turbine_id``, ``cutoff_time``
                and ``target`` columns.
            readings (pandas.DataFrame):
                ``readings`` table.
            turbines (pandas.DataFrame):
                ``turbines`` table.
        """
        X = target_times[['turbine_id', 'cutoff_time']]
        y = target_times['target']
        self._pipeline.fit(X, y, readings=readings, turbines=turbines)
        self.fitted = True

    def predict(self, target_times, readings, turbines=None):
        """Make predictions using this pipeline.

        Args:
            target_times (pandas.DataFrame):
                ``target_times`` table, containing the ``turbine_id``, ``cutoff_time``
                and ``target`` columns.
            readings (pandas.DataFrame):
                ``readings`` table.
            turbines (pandas.DataFrame):
                ``turbines`` table.

        Returns:
            numpy.ndarray:
                Vector of predictions.
        """
        if not self.fitted:
            raise NotFittedError()

        X = target_times[['turbine_id', 'cutoff_time']]
        return self._pipeline.predict(X, readings=readings, turbines=turbines)

    def save(self, path):
        """Serialize and save this pipeline using cloudpickle.

        Args:
            path (str):
                Path to the file where the pipeline will be saved.
        """
        with open(path, 'wb') as pickle_file:
            cloudpickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        """Load a previously saved pipeline from a file.

        Args:
            path (str):
                Path to the file where the pipeline is saved.

        Returns:
            GreenGuardPipeline:
                Loaded GreenGuardPipeline instance.
        """
        with open(path, 'rb') as pickle_file:
            return cloudpickle.load(pickle_file)
