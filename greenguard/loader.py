# -*- coding: utf-8 -*-

import logging
import os
import shutil

import pandas as pd

LOGGER = logging.getLogger(__name__)


class GreenGuardLoader(object):

    def __init__(self, dataset_path, target='targets', target_column='target',
                 readings='readings', turbines='turbines', signals='signals', gzip=False):

        self._dataset_path = dataset_path
        self._target = target
        self._target_column = target_column
        self._readings = readings
        self._turbines = turbines
        self._signals = signals
        self._gzip = gzip

    def read_csv(self, table, timestamp=False):
        if timestamp:
            timestamp = ['timestamp']

        path = os.path.join(self._dataset_path, table + '.csv')
        if self._gzip:
            path += '.gz'

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


def load_demo():
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')
    if os.path.exists(demo_path):
        loader = GreenGuardLoader(demo_path, gzip=True)
        return loader.load()

    else:
        os.mkdir(demo_path)
        try:
            loader = GreenGuardLoader('https://d3-ai-greenguard.s3.amazonaws.com/', gzip=True)
            X, tables = loader.load(target=False)
            X.to_csv(os.path.join(demo_path, 'targets.csv.gz'), index=False)
            for name, table in tables.items():
                table.to_csv(os.path.join(demo_path, name + '.csv.gz'), index=False)

            y = X.pop('target')
            return X, y, tables
        except Exception:
            shutil.rmtree(demo_path)
            raise
