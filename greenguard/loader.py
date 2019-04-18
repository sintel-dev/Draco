# -*- coding: utf-8 -*-

import logging
import os
import shutil

import pandas as pd

LOGGER = logging.getLogger(__name__)


class GreenGuardLoader(object):
    """GreenGuardLoader class.

    The GreenGuardLoader class provides a simple interface to load a relational
    dataset in the format expected by the GreenGuard Pipelines.

    Args:
        dataset_path (str): Path to the root folder of the dataset.
        target (str): Name of the target table.
        target_column (str): Name of the target column within the target table.
        readings (str): Name of the readings table.
        turbines (str): Name of the turbines table.
        signals (str): Name of the signals table.
        gzip (bool): Whether the CSV files will be in GZipped. If `True`, the filenames
            are expected to have the `.csv.gz` extension.
    """

    def __init__(self, dataset_path, target='targets', target_column='target',
                 readings='readings', turbines='turbines', signals='signals', gzip=False):

        self._dataset_path = dataset_path
        self._target = target
        self._target_column = target_column
        self._readings = readings
        self._turbines = turbines
        self._signals = signals
        self._gzip = gzip

    def _read_csv(self, table, timestamp=False):
        if timestamp:
            timestamp = ['timestamp']

        path = os.path.join(self._dataset_path, table + '.csv')
        if self._gzip:
            path += '.gz'

        return pd.read_csv(path, parse_dates=timestamp, infer_datetime_format=True)

    def load(self, target=True):
        """Load the dataset.

        Args:
            target (bool): If True, return the target column as a separated vector.
                Otherwise, the target column is expected to be already missing from
                the target table.

        Returns:
            (tuple):
                * ``X (pandas.DataFrame)``: A pandas.DataFrame with the contents of the
                  target table.
                * ``y (pandas.Series, optional)``: A pandas.Series with the contents of
                  the target column.
                * ``tables (dict)``: A dictionary containing the readings, turbines and
                  signals tables as pandas.DataFrames.
        """
        tables = {
            'readings': self._read_csv(self._readings, True),
            'signals': self._read_csv(self._signals),
            'turbines': self._read_csv(self._turbines),
        }

        X = self._read_csv(self._target, True)
        if target:
            y = X.pop(self._target_column)
            return X, y, tables

        else:
            return X, tables


def load_demo():
    """Load the demo included in the GreenGuard project.

    The first time that this function is executed, the data will be downloaded
    and cached inside the `greenguard/demo` folder.
    Subsequent calls will load the cached data instead of downloading it again.
    """
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
