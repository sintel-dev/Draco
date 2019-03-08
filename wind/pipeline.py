# -*- coding: utf-8 -*-

import json
import logging
import os
from collections import defaultdict

import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

LOGGER = logging.getLogger(__name__)


class WindPipeline(object):

    template = None
    fitted = False
    score = None

    _cv_class = None
    _score = None
    _cost = None
    _tuner = None
    _pipeline = None

    def _get_cv(self, cv_splits, random_state):
        return self._cv_class(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def _load_mlpipeline(self, template):
        if not isinstance(template, dict):
            template_name = template
            if os.path.isfile(template_name):
                with open(template_name, 'r') as template_file:
                    template = json.load(template_file)

            # elif self._db:
            #     template = self._db.load_template(template_name)

            if not template:
                raise ValueError('Unknown template {}'.format(template_name))

            self.template = template

        return MLPipeline.from_dict(template)

    def __init__(self, template=None, hyperparameters=None,
                 scorer=None, cost=False, cv=None, cv_splits=5, random_state=0):

        self._cv = cv or self._get_cv(cv_splits, random_state)

        if scorer:
            self._score = scorer

        self._cost = cost

        # self._db = db

        self._pipeline = self._load_mlpipeline(template or self.template)

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
            score = self._score(y_test, predictions)

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
            self._tuner = self._get_tuner()

        # dataset = data['dataset_name']
        # table = data['target_entity']
        # column = data['target_column']

        for i in range(iterations):
            LOGGER.info('Scoring pipeline %s', i + 1)
            params = '\n'.join('{}: {}'.format(k, v) for k, v in proposal.items())
            LOGGER.info("Scoring pipeline %s: %s\n%s", i + 1, pipeline.id, params)

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

            # if self._db:
            #     self._db.insert_pipeline(candidate, score, dataset, table, column)

    def fit(self, X, y, tables):
        tables.setdefault('entityset', None)
        self._pipeline.fit(X, y, **tables)
        self.fitted = True

    def predict(self, X, tables):
        if not self.fitted:
            raise NotFittedError()

        tables.setdefault('entityset', None)
        return self._pipeline.predict(X, **tables)


class WindClassifier(WindPipeline):
    _cv_class = StratifiedKFold
    template = {
        'primitives': [
            'pandas.DataFrame.resample',
            'pandas.DataFrame.unstack',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.add_relationship',
            'featuretools.dfs',
            'mlprimitives.custom.feature_extraction.CategoricalEncoder',
            'sklearn.impute.SimpleImputer',
            'sklearn.preprocessing.StandardScaler',
            'xgboost.XGBClassifier',
        ],
        'init_params': {
            'pandas.DataFrame.resample#1': {
                'rule': '1D',
                'time_index': 'timestamp',
                'groupby': ['turbine_id', 'signal_id'],
                'aggregation': 'mean'
            },
            'pandas.DataFrame.unstack#1': {
                'level': 'signal_id',
                'reset_index': True
            },
            'featuretools.EntitySet.entity_from_dataframe#1': {
                'entity_id': 'readings',
                'index': 'index',
                'make_index': True,
                'time_index': 'timestamp'
            },
            'featuretools.EntitySet.entity_from_dataframe#2': {
                'entity_id': 'turbines',
                'index': 'turbine_id',
            },
            'featuretools.EntitySet.entity_from_dataframe#3': {
                'entity_id': 'signals',
                'index': 'signal_id',
            },
            'featuretools.EntitySet.add_relationship#1': {
                'parent': 'turbines',
                'parent_column': 'turbine_id',
                'child': 'readings',
                'child_column': 'turbine_id',
            },
            'featuretools.EntitySet.add_relationship#2': {
                'parent': 'signals',
                'parent_column': 'signal_id',
                'child': 'readings',
                'child_column': 'signal_id',
            },
            'featuretools.dfs#1': {
                'target_entity': 'turbines',
                'index': 'turbine_id',
                'time_index': 'timestamp',
                'encode': False
            },
        },
        'input_names': {
            'pandas.DataFrame.resample#1': {
                'X': 'readings'
            },
            'pandas.DataFrame.unstack#1': {
                'X': 'readings'
            },
            'featuretools.EntitySet.entity_from_dataframe#1': {
                'dataframe': 'readings',
            },
            'featuretools.EntitySet.entity_from_dataframe#2': {
                'dataframe': 'turbines',
            },
            'featuretools.EntitySet.entity_from_dataframe#3': {
                'dataframe': 'signals',
            },
        },
        'output_names': {
            'pandas.DataFrame.resample#1': {
                'X': 'readings'
            },
            'pandas.DataFrame.unstack#1': {
                'X': 'readings'
            },
        }
    }

    def _score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class WindRegressor(WindPipeline):
    _cv_class = KFold
    template = {
        'primitives': [
            'pandas.DataFrame.resample',
            'pandas.DataFrame.unstack',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.entity_from_dataframe',
            'featuretools.EntitySet.add_relationship',
            'featuretools.dfs',
            'mlprimitives.custom.feature_extraction.CategoricalEncoder',
            'sklearn.impute.SimpleImputer',
            'sklearn.preprocessing.StandardScaler',
            'xgboost.XGBRegressor',
        ],
        'init_params': {
            'pandas.DataFrame.resample#1': {
                'rule': '1D',
                'time_index': 'timestamp',
                'groupby': ['turbine_id', 'signal_id'],
                'aggregation': 'mean'
            },
            'pandas.DataFrame.unstack#1': {
                'level': 'signal_id',
                'reset_index': True
            },
            'featuretools.EntitySet.entity_from_dataframe#1': {
                'entity_id': 'readings',
                'index': 'index',
                'make_index': True,
                'time_index': 'timestamp'
            },
            'featuretools.EntitySet.entity_from_dataframe#2': {
                'entity_id': 'turbines',
                'index': 'turbine_id',
            },
            'featuretools.EntitySet.entity_from_dataframe#3': {
                'entity_id': 'signals',
                'index': 'signal_id',
            },
            'featuretools.EntitySet.add_relationship#1': {
                'parent': 'turbines',
                'parent_column': 'turbine_id',
                'child': 'readings',
                'child_column': 'turbine_id',
            },
            'featuretools.EntitySet.add_relationship#2': {
                'parent': 'signals',
                'parent_column': 'signal_id',
                'child': 'readings',
                'child_column': 'signal_id',
            },
            'featuretools.dfs#1': {
                'target_entity': 'turbines',
                'index': 'turbine_id',
                'time_index': 'timestamp',
                'encode': False
            },
        },
        'input_names': {
            'pandas.DataFrame.resample#1': {
                'X': 'readings'
            },
            'pandas.DataFrame.unstack#1': {
                'X': 'readings'
            },
            'featuretools.EntitySet.entity_from_dataframe#1': {
                'dataframe': 'readings',
            },
            'featuretools.EntitySet.entity_from_dataframe#2': {
                'dataframe': 'turbines',
            },
            'featuretools.EntitySet.entity_from_dataframe#3': {
                'dataframe': 'signals',
            },
        },
        'output_names': {
            'pandas.DataFrame.resample#1': {
                'X': 'readings'
            },
            'pandas.DataFrame.unstack#1': {
                'X': 'readings'
            },
        }
    }

    def _score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
