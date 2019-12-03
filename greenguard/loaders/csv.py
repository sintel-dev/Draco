import logging
import os

import dask
import pandas as pd

LOGGER = logging.getLogger(__name__)


class CSVLoader:

    def __init__(self, readings_path='.', rule=None, aggregation='mean', unstack=True):
        self._readings_path = readings_path
        self._rule = rule
        self._aggregation = aggregation
        self._unstack = unstack

    @dask.delayed
    def __filter_by_signal(self, readings, signals):
        if signals is not None:
            LOGGER.debug('Filtering by signal')
            readings = readings[readings.signal_id.isin(signals)]

        LOGGER.debug('Selected %s readings by signal', len(readings))

        return readings.copy()

    @dask.delayed
    def __filter_by_timestamp(self, readings, timestamps):
        LOGGER.debug('Parsing timestamps')
        readings_ts = pd.to_datetime(readings['timestamp'], format='%m/%d/%y %H:%M:%S')
        readings['timestamp'] = readings_ts

        LOGGER.debug('Filtering by timestamp')

        related = [False] * len(readings)
        for row in timestamps.itertuples():
            lower = row.start <= readings_ts
            upper = readings_ts <= row.stop
            related |= lower & upper

        readings = readings[related]

        LOGGER.debug('Selected %s readings by timestamp', len(readings))

        return readings.copy()

    @dask.delayed
    def __load_readings_file(self, turbine_file, timestamps, signals):
        LOGGER.debug('Loading file %s', turbine_file)
        data = pd.read_csv(turbine_file, low_memory=False)
        data.columns = data.columns.str.lower()
        data = data.rename(columns={'signal': 'signal_id'})

        if 'unnamed: 0' in data.columns:
            # Someone forgot to drop the index before
            # storing the DataFrame as a CSV
            del data['unnamed: 0']

        LOGGER.debug('Loaded %s readings from file %s', len(data), turbine_file)

        return data

    @dask.delayed
    def __consolidate(self, readings, turbine_id):
        readings = pd.concat(readings)
        try:
            readings['value'] = readings['value'].astype(float)
        except ValueError:
            signals = readings[readings['value'].str.isnumeric()].signal_id.unique()
            raise ValueError('Signals contain non-numerical values: {}'.format(signals))

        readings['turbine_id'] = turbine_id

        LOGGER.info('Loaded %s readings from turbine %s', len(readings), turbine_id)

        return readings

    @staticmethod
    def _get_filenames(turbine_path, timestamps):
        min_csv = timestamps.start.dt.strftime('%Y-%m-.csv')
        max_csv = timestamps.stop.dt.strftime('%Y-%m-.csv')

        for filename in sorted(os.listdir(turbine_path)):
            if ((min_csv <= filename) & (filename <= max_csv)).any():
                yield os.path.join(turbine_path, filename)

    @staticmethod
    def _join_names(names):
        """Join the names of a multi-level index with an underscore."""

        levels = (str(name) for name in names if name != '')
        return '_'.join(levels)

    @dask.delayed
    def __resample(self, readings):
        LOGGER.info('Resampling: %s - %s', self._rule, self._aggregation)
        grouped = readings.groupby(['turbine_id', 'signal_id'])
        dfr = grouped.resample(rule=self._rule, on='timestamp')
        agg = dfr.agg(self._aggregation)
        if self._unstack:
            agg = agg.unstack(level='signal_id').reset_index()
            agg.columns = agg.columns.map(self._join_names)
            return agg
        else:
            return agg.reset_index()

    def _load_turbine(self, turbine_id, timestamps, signals=None):
        if 'turbine_id' in timestamps:
            timestamps = timestamps[timestamps.turbine_id == turbine_id]

        turbine_path = os.path.join(self._readings_path, turbine_id)
        filenames = self._get_filenames(turbine_path, timestamps)

        readings = list()
        for filename in filenames:
            file_readings = self.__load_readings_file(filename, timestamps, signals)
            file_readings = self.__filter_by_signal(file_readings, signals)
            file_readings = self.__filter_by_timestamp(file_readings, timestamps)
            readings.append(file_readings)

        if readings:
            readings = self.__consolidate(readings, turbine_id)

            if self._rule:
                readings = self.__resample(readings)

        return readings

    @staticmethod
    def _get_timestamps(target_times, window_size):
        cutoff_times = target_times.cutoff_time
        min_times = cutoff_times - window_size

        return pd.DataFrame({
            'turbine_id': target_times.turbine_id,
            'start': min_times,
            'stop': cutoff_times,
        })

    def load(self, target_times, window_size, signals=None, debug=False):
        if isinstance(target_times, str):
            target_times = pd.read_csv(target_times)
            target_times['cutoff_time'] = pd.to_datetime(target_times['cutoff_time'])

        if isinstance(signals, pd.DataFrame):
            signals = signals.signal_id

        window_size = pd.to_timedelta(window_size)
        timestamps = self._get_timestamps(target_times, window_size)

        readings = list()
        for turbine_id in timestamps.turbine_id.unique():
            readings.append(self._load_turbine(turbine_id, timestamps, signals))

        dask_scheduler = 'single-threaded' if debug else None
        computed = dask.compute(*readings, scheduler=dask_scheduler)
        readings = pd.concat(c for c in computed if len(c))

        LOGGER.info('Loaded %s turbine readings', len(readings))

        return readings
