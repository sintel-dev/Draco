# -*- coding: utf-8 -*-

import gc
import json
import logging
import os
import pickle
import tempfile
from copy import deepcopy
from hashlib import md5

import cloudpickle
import keras
import numpy as np
from btb import BTBSession
from btb.tuning import Tunable
from mlblocks import MLPipeline
from mlblocks.discovery import load_pipeline
from mlprimitives.adapters.keras import Sequential
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold

from greenguard.metrics import METRICS

LOGGER = logging.getLogger(__name__)


PIPELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pipelines'))


# Patch Keras to save on disk without a model trained
def __getstate__(self):
    state = self.__dict__.copy()
    if 'model' in state:
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(state.pop('model'), fd.name, overwrite=True)
            state['model_str'] = fd.read()

    return state


def __setstate__(self, state):
    if 'model_str' in state:
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state.pop('model_str'))
            fd.flush()

            state['model'] = keras.models.load_model(fd.name)

    self.__dict__ = state


Sequential.__getstate__ = __getstate__
Sequential.__setstate__ = __setstate__


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
        templates (str, MLPipeline or list):
            Template to use. If a ``str`` is given, load the corresponding
            ``MLPipeline``. Also can be a list combining both.
        metric (str or function):
            Metric to use. If an ``str`` is give it must be one of the metrics
            defined in the ``greenguard.metrics.METRICS`` dictionary.
        cost (bool):
            Whether the metric is a cost function (the lower the better) or not.
            Defaults to ``False``.
        init_params (dict or list):
            There are three possible values for init_params:

                * Init params ``dict``: It will be used for all templates.
                * ``dict`` with the name of the template as a key and dictionary with its
                  init params.
                * ``list``: each value will be assigned to the corresponding position of
                  self.templates.

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
        preprocessing (int, dict or list):
            There are three possible values for preprocessing:

                * ``int``: the value will be used for all templates.
                * ``dict`` with the template name as a key and a number as a value, will
                  be used for that template.
                * ``list``: each value will be assigned to the corresponding position of
                  self.templates.

            Defaults to ``0``.
        cache_path (str):
            If given, cache the generated cross validation splits in this folder.
            Defatuls to ``None``.
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
    _static = None
    _init_params = None
    _preprocessing = None

    def _get_cv(self, stratify, cv_splits, shuffle, random_state):
        if stratify:
            cv_class = StratifiedKFold
        else:
            cv_class = KFold

        return cv_class(n_splits=cv_splits, shuffle=shuffle, random_state=random_state)

    def _set_hyperparameters(self, new_hyperparameters):
        self._hyperparameters = deepcopy(new_hyperparameters)

    def _set_template(self, template_name):
        self.template_name = template_name
        self.template = self._template_dicts[self.template_name]

    @staticmethod
    def _update_params(old, new):
        for name, params in new.items():
            if '#' not in name:
                name = name + '#1'

            block_params = old.setdefault(name, dict())
            for param, value in params.items():
                block_params[param] = value

    def _count_static_steps(self, pipeline):
        tunable_hyperparams = pipeline.get_tunable_hyperparameters()
        for index, block_name in enumerate(pipeline.blocks.keys()):
            if tunable_hyperparams[block_name]:
                return index

        return 0

    def _get_templates(self, templates):
        template_dicts = dict()
        template_names = list()
        for template in templates:
            if isinstance(template, str):
                template_name = template
                template = load_pipeline(template_name)
            else:
                template_name = md5(json.dumps(template)).digest()
            template_dicts[template_name] = template
            template_names.append(template_name)

        return template_names, template_dicts

    def _generate_init_params(self, init_params):
        if not init_params:
            self._init_params = {}
        elif isinstance(init_params, list):
            self._init_params = dict(zip(self._template_names, init_params))
        elif any(name in init_params for name in self._template_names):
            self._init_params = init_params

    def _generate_preprocessing(self, preprocessing):
        if isinstance(preprocessing, int):
            self._preprocessing = {name: preprocessing for name in self._template_names}
        else:
            if isinstance(preprocessing, list):
                preprocessing = dict(zip(self._template_names, preprocessing))

            self._preprocessing = {
                name: preprocessing.get(name, 0)
                for name in self._template_names
            }

    def _build_pipeline(self):
        self._pipeline = MLPipeline(self.template)

        if self._hyperparameters:
            self._pipeline.set_hyperparameters(self._hyperparameters)

        self.fitted = False

    def __init__(self, templates, metric='accuracy', cost=False, init_params=None, stratify=True,
                 cv_splits=5, shuffle=True, random_state=0, preprocessing=0, cache_path=None):

        if isinstance(metric, str):
            metric, cost = METRICS[metric]

        self._metric = metric
        self._cost = cost
        self._cv = self._get_cv(stratify, cv_splits, shuffle, random_state)
        self.cv_score = np.inf if cost else -np.inf

        if not isinstance(templates, list):
            templates = [templates]

        self.templates = templates
        self._template_names, self._template_dicts = self._get_templates(templates)
        self._default_init_params = {}
        self._generate_init_params(init_params)

        for name, template in self._template_dicts.items():
            init_params = self._init_params.get(name, self._default_init_params)
            template_params = template.setdefault('init_params', {})
            self._update_params(template_params, init_params)

        self._generate_preprocessing(preprocessing)
        self._set_template(self._template_names[0])
        self._hyperparameters = dict()
        self._build_pipeline()
        self._cache_path = cache_path
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)

    def get_hyperparameters(self):
        """Get the current hyperparameters.

        Returns:
            dict:
                Current hyperparameters.
        """
        return deepcopy(self._hyperparameters)

    def _is_better(self, score):
        if self._cost:
            return score < self.cv_score

        return score > self.cv_score

    def _generate_splits(self, template_name, target_times, readings, turbines=None):
        template = self._template_dicts.get(template_name)
        pipeline = MLPipeline(template)
        preprocessing = self._preprocessing.get(template_name)
        static = self._count_static_steps(pipeline)
        X = target_times[['turbine_id', 'cutoff_time']]
        y = target_times['target']

        if preprocessing:
            if preprocessing > static:
                raise ValueError('Preprocessing cannot be bigger than static')

            LOGGER.debug('Running %s preprocessing steps', preprocessing)
            context = pipeline.fit(X=X, y=y, readings=readings,
                                   turbines=turbines, output_=preprocessing - 1)
            del context['X']
            del context['y']
            gc.collect()

        else:
            context = {
                'readings': readings,
                'turbines': turbines,
            }

        splits = list()
        for fold, (train_index, test_index) in enumerate(self._cv.split(X, y)):
            LOGGER.debug('Running static steps for fold %s', fold)
            gc.collect()
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline = MLPipeline(template)
            fit = pipeline.fit(X_train, y_train, output_=static - 1,
                               start_=preprocessing, **context)
            predict = pipeline.predict(X_test, output_=static - 1,
                                       start_=preprocessing, **context)

            split = (fold, pipeline, fit, predict, y_test, static)

            if self._cache_path:
                split_name = '{}_{}.pkl'.format(template_name, fold)
                split_path = os.path.join(self._cache_path, split_name)

                with open(split_path, 'wb') as split_file:
                    pickle.dump(split, split_file)

                split = split_path

            splits.append(split)

        gc.collect()
        return splits

    def _cross_validate(self, template_splits, hyperparams):
        scores = []
        for split in template_splits:
            gc.collect()
            if self._cache_path:
                with open(split, 'rb') as split_file:
                    split = pickle.load(split_file)

            fold, pipeline, fit, predict, y_test, static = split

            LOGGER.debug('Scoring fold %s', fold)
            pipeline.set_hyperparameters(hyperparams)
            pipeline.fit(start_=static, **fit)
            predictions = pipeline.predict(start_=static, **predict)

            score = self._metric(y_test, predictions)
            LOGGER.debug('Fold fold %s score: %s', fold, score)
            scores.append(score)

        return np.mean(scores)

    def _make_btb_scorer(self, target_times, readings, turbines):
        splits = {}

        def scorer(template_name, config):
            template_splits = splits.get(template_name)
            if template_splits is None:
                template_splits = self._generate_splits(
                    template_name, target_times, readings, turbines)

                splits[template_name] = template_splits

            cv_score = self._cross_validate(template_splits, config)
            if self._is_better(cv_score):
                _config = '\n'.join('      {}: {}'.format(n, v) for n, v in config.items())
                LOGGER.info(('New configuration found:\n'
                             '  Template: %s \n'
                             '    Hyperparameters: \n'
                             '%s'), template_name, _config)

                self.cv_score = cv_score
                self._set_template(template_name)
                self._set_hyperparameters(config)
                self._build_pipeline()

            return cv_score

        return scorer

    def cross_validate(self, target_times, readings, turbines,
                       template_name=None, hyperparams=None):
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
        if not template_name:
            template_name = self.template_name
            if hyperparams is None:
                hyperparams = self.get_hyperparameters()

        elif hyperparams is None:
            hyperparams = {}

        template_splits = self._generate_splits(template_name, target_times, readings, turbines)
        return self._cross_validate(template_splits, hyperparams)

    @classmethod
    def _get_tunables(cls, template_dicts):
        tunables = {}
        for name, template in template_dicts.items():
            pipeline = MLPipeline(template)
            pipeline_tunables = pipeline.get_tunable_hyperparameters(flat=True)
            tunables[name] = Tunable.from_dict(pipeline_tunables)

        return tunables

    def tune(self, target_times, readings, turbines=None):
        """Create a tuning session object that tunes and selects the templates.

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
        """
        scoring_function = self._make_btb_scorer(target_times, readings, turbines)
        tunables = self._get_tunables(self._template_dicts)
        return BTBSession(tunables, scoring_function, maximize=not self._cost)

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
