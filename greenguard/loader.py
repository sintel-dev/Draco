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
        dataset_path (str):
            Path to the root folder of the dataset.
        target_times (str):
            Name of the target_times file.
        target_column (str):
            Name of the target column within the target_times file.
        readings (str):
            Name of the readings file.
        turbines (str):
            Name of the turbines file.
        signals (str):
            Name of the signals file.
        gzip (bool):
            Whether the CSV files will be in GZipped. If `True`, the filenames
            are expected to have the `.csv.gz` extension.
    """

    def __init__(self, dataset_path, target_times='target_times', target_column='target',
                 readings='readings', turbines=None, signals=None, gzip=False):

        self._dataset_path = dataset_path
        self._target_times = target_times
        self._target_column = target_column
        self._readings = readings
        self._turbines = turbines
        self._signals = signals
        self._gzip = gzip

    def _read_csv(self, table, timestamp=None):
        if timestamp:
            parse_dates = [timestamp]
        else:
            parse_dates = False

        if '.csv' not in table:
            table += '.csv'
            if self._gzip:
                table += '.gz'

        path = os.path.join(self._dataset_path, table)

        return pd.read_csv(path, parse_dates=parse_dates, infer_datetime_format=True)

    def load(self, return_target=True):
        """Load the dataset.

        Args:
            return_target (bool):
                If True, return the target column as a separated vector.
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
            'readings': self._read_csv(self._readings, 'timestamp'),
        }

        if self._signals:
            tables['signals'] = self._read_csv(self._signals)

        if self._turbines:
            tables['turbines'] = self._read_csv(self._turbines)

        X = self._read_csv(self._target_times, 'cutoff_time')
        if return_target:
            y = X.pop(self._target_column)
            return X, y, tables

        else:
            return X, tables


class GreenGuardRawLoader(object):
    """GreenGuardRawLoader class.

    The GreenGuardRawLoader class provides a simple interface to load a
    time series data provided turbine in raw format and return it in the
    format expected by the GreenGuard Pipelines.

    This raw format has the following characteristics:

        * All the data from all the turbines is inside a single folder.
        * Inside the data folder, a folder exists for each turbine.
          This folders are named exactly like each turbine id, and inside it one or more
          CSV files can be found. The names of these files is not relevant.
        * Each CSV file will have the the following columns:

            * timestamp: timestemp of the reading.
            * signal: name or id of the signal.
            * value: value of the reading.

    And the output is the following 3 elements:

        * `X`: target times table containing:
            * `turbine_id`: Unique identifier of the turbine which this target corresponds to.
            * `cutoff_timestamp`: The timestamp at which the target value is deemed to be known.
              This timestamp is used to filter data such that only data prior to this is used
              for featurize.
        * `y`: 1d vector of value sthat we want to predict. This can either be a numerical value
              or a categorical label. This column can also be skipped when preparing data that
              will be used only to make predictions and not to fit any pipeline.
        * readings:
            * `reading_id`: Unique identifier of this reading.
            * `turbine_id`: Unique identifier of the turbine which this reading comes from.
            * `signal_id`: Unique identifier of the signal which this reading comes from.
            * `timestamp`: Time where the reading took place, as an ISO formatted datetime.
            * `value`: Numeric value of this reading.

    Args:
        target_times (pd.DataFrame or str):
            target_times DataFrame or path to the target_times CSV file.
        readings_path (str):
            Path to the folder containing all the readings data.
    """

    def __init__(self, target_times, readings_path):
        self._target_times = target_times
        self._readings_path = readings_path

    def _load_readings_file(self, turbine_file, signals):
        LOGGER.info('Loading file %s', turbine_file)
        data = pd.read_csv(turbine_file)
        data.columns = data.columns.str.lower()
        data.rename(columns={'signal': 'signal_id'}, inplace=True)

        if 'unnamed: 0' in data.columns:
            # Someone forgot to drop the index before
            # storing the DataFrame as a CSV
            del data['unnamed: 0']

        if signals:
            LOGGER.info('Filtering by signal')
            data = data[data.signal_id.isin(signals)]

        LOGGER.info('Parsing timestamps')
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%m/%d/%y %H:%M:%S')

        return data

    def _filter_by_window_size(self, filenames, data, window_size):
        window_size = pd.to_timedelta(window_size)
        min_timestamp = (data.cutoff_time - window_size).dt.strftime('%Y-%m-.csv')
        max_timestamp = data.cutoff_time.dt.strftime('%Y-%m-.csv')

        for filename in filenames:
            if ((min_timestamp <= filename) & (filename <= max_timestamp)).any():
                yield filename

    def _load_turbine_readings(self, turbine_id, data, signals, window_size):
        turbine_path = os.path.join(self._readings_path, turbine_id)
        filenames = sorted(os.listdir(turbine_path))

        if window_size:
            filenames = self._filter_by_window_size(filenames, data, window_size)

        readings = list()
        for readings_file in filenames:
            readings_file_path = os.path.join(turbine_path, readings_file)
            readings.append(self._load_readings_file(readings_file_path, signals))

        return pd.concat(readings)

    def _load_readings(self, X, signals, window_size):
        turbine_ids = X.turbine_id.unique()

        readings = list()
        for turbine_id in sorted(turbine_ids):
            data = X[X['turbine_id'] == turbine_id]
            LOGGER.info('Loading turbine %s readings', turbine_id)
            turbine_readings = self._load_turbine_readings(turbine_id, data, signals, window_size)
            turbine_readings['turbine_id'] = turbine_id
            readings.append(turbine_readings)

        return pd.concat(readings)

    def load(self, return_target=True, signals=None, window_size=None):
        """Load the dataset.

        Args:
            return_target (bool):
                If ``True``, return the target column as a separated vector.
                Otherwise, the target column is expected to be already missing from
                the target table.
            signals (list):
                List of signals to load from the readings files. If not given, load
                all the signals available.
            window_size (str):
                Rule indicating how long back before the cutoff times we have to go
                when loading the data.

        Returns:
            (tuple):
                * ``X (pandas.DataFrame)``: A pandas.DataFrame with the contents of the
                  target_times table.
                * ``y (pandas.Series, optional)``: A pandas.Series with the contents of
                  the target column.
                * ``readings (pandas.DataFrame)``: A pandas.DataFrame containing the readings.
        """
        if isinstance(self._target_times, pd.DataFrame):
            X = self._target_times.copy()
        else:
            X = pd.read_csv(self._target_times)
            X['cutoff_time'] = pd.to_datetime(X['cutoff_time'])

        readings = self._load_readings(X, signals, window_size)
        LOGGER.info('Loaded %s turbine readings', len(readings))

        if return_target:
            y = X.pop('target')
            return X, y, readings

        else:
            return X, readings


def load_demo():
    """Load the demo included in the GreenGuard project.

    The first time that this function is executed, the data will be downloaded
    and cached inside the `greenguard/demo` folder.
    Subsequent calls will load the cached data instead of downloading it again.
    """
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')
    if os.path.exists(demo_path):
        loader = GreenGuardLoader(demo_path, gzip=True)
        X, y, tables = loader.load()
        return X, y, tables['readings']

    else:
        os.mkdir(demo_path)
        try:
            loader = GreenGuardLoader('https://d3-ai-greenguard.s3.amazonaws.com/', gzip=True)
            X, tables = loader.load(target=False)
            X.to_csv(os.path.join(demo_path, 'targets.csv.gz'), index=False)
            for name, table in tables.items():
                table.to_csv(os.path.join(demo_path, name + '.csv.gz'), index=False)

            y = X.pop('target')
            return X, y, tables['readings']
        except Exception:
            shutil.rmtree(demo_path)
            raise
