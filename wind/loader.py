# -*- coding: utf-8 -*-

import os

import pandas as pd


class WindLoader(object):

    def __init__(self, dataset_path, target, target_column,
                 readings='readings', turbines='turbines', signals='signals'):

        self._dataset_path = dataset_path
        self._target = target
        self._target_column = target_column
        self._readings = readings
        self._turbines = turbines
        self._signals = signals

    def read_csv(self, table, timestamp=False):
        if timestamp:
            timestamp = ['timestamp']

        path = os.path.join(self._dataset_path, table + '.csv')
        return pd.read_csv(path, parse_dates=timestamp, infer_datetime_format=True)

    def load(self, target=True):
        tables = {
            'readings': self.read_csv(self._readings, True),
            'signals': self.read_csv(self._signals),
            'turbines': self.read_csv(self._turbines),
        }

        X = self.read_csv(self._target, True)
        if target:
            y = X.pop(self._target_column)
            return X, y, tables

        else:
            return X, tables
