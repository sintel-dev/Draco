# -*- coding: utf-8 -*-

import logging
import os

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)
BUCKET = 'd3-ai-green-guard'
S3_URL = 'https://{}.s3.amazonaws.com/{}.csv'


def download(name):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com)
    and then cached inside the `data` folder, within the `orion` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `orion/data` folder without contacting S3.

    Args:
        name (str): Name of the CSV to load.

    Returns:
        If no test_size is given, a single pandas.DataFrame is returned containing all
        the data. If test_size is given, a tuple containing one pandas.DataFrame for
        the train split and another one for the test split is returned.
    """

    filename = os.path.join(DATA_PATH, name + '.csv')
    if os.path.exists(filename):
        data = pd.read_csv(filename)

    else:
        url = S3_URL.format(BUCKET, name)

        LOGGER.debug('Downloading CSV %s from %s', name, url)
        os.makedirs(DATA_PATH, exist_ok=True)
        data = pd.read_csv(url)
        data.to_csv(filename, index=False)

    return data


class GreenGuardLoader(object):

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

        try:
            path = os.path.join(self._dataset_path, table + '.csv')
            return pd.read_csv(path, parse_dates=timestamp, infer_datetime_format=True)
        except FileNotFoundError:
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
    loader = GreenGuardLoader('https://d3-ai-green-guard.s3.amazonaws.com/', 'labels', 'label')

    return loader.load()
