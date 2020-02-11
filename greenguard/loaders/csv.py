import logging
import os

import dask
import pandas as pd

from greenguard.targets import select_valid_targets

LOGGER = logging.getLogger(__name__)


class CSVLoader:
    """Load the required readings from CSV files.

    The CSVLoader class is responsible for analyzing the target_times table
    and then load the required readings from CSV files.

    Also, optionally, it can perform a resampling aggregation while loading
    the data, reducing the amount of memory requirements.

    The CSVLoader class uses Dask to parallelize all the IO and resampling
    computation and reduce loading times.

    Args:
        readings_path (str):
            Path to the readings folder, where a folder exist for each turbine.
        rule (str):
            Resampling rule, as expected by ``DataFrame.resmple``. The rule is a
            string representation of a TimeDelta, which includes a number and a
            unit. For example: ``3d``, ``1w``, ``6h``.
            If ``None``, resampling is disabled.
        aggregation (str):
            Name of the aggregation to perform during the resampling.
        unstack (bool):
            Whether to unstack the resampled data, generating one column per signal.
            Only used when resampling. Defaults to ``False``.
    """

    DEFAULT_DATETIME_FMT = '%m/%d/%y %M:%H:%S'
    DEFAULT_FILENAME_FMT = '%Y-%m-.csv'

    def __init__(self, readings_path='.', rule=None, aggregation='mean', unstack=False,
                 datetime_fmt=DEFAULT_DATETIME_FMT, filename_fmt=DEFAULT_FILENAME_FMT):
        self._readings_path = readings_path
        self._rule = rule
        self._aggregation = aggregation
        self._unstack = unstack
        self._datetime_fmt = datetime_fmt
        self._filename_fmt = filename_fmt

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
        readings_ts = pd.to_datetime(readings['timestamp'], format=self._datetime_fmt)
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
        readings = pd.concat(readings, ignore_index=True)
        try:
            readings['value'] = readings['value'].astype(float)
        except ValueError:
            signals = readings[readings['value'].str.isnumeric()].signal_id.unique()
            raise ValueError('Signals contain non-numerical values: {}'.format(signals))

        readings.insert(0, 'turbine_id', turbine_id)

        LOGGER.info('Loaded %s readings from turbine %s', len(readings), turbine_id)

        return readings

    def _get_filenames(self, turbine_path, timestamps):
        min_csv = timestamps.start.dt.strftime(self._filename_fmt)
        max_csv = timestamps.stop.dt.strftime(self._filename_fmt)

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

    def load(self, target_times, window_size, signals=None, debug=False, select_valid=True):
        """Load the readings needed for the given target_times and window_size.

        Optionally filter the signals that are loaded and discard the rest.

        Args:
            target_times (str or pandas.DataFrame):
                target_times ``DataFrame`` or path to the corresponding CSV file.
                The table must have three volumns, ``turbine_id``, ``target`` and
                ``cutoff_time``.
            window_size (str):
                Amount of data to load before each cutoff time, specified as a string
                representation of a TimeDelta, which includes a number and a
                unit. For example: ``3d``, ``1w``, ``6h``.
            signals (list or pandas.DataFrame):
                List of signal names or table that has a ``signal_id`` column to
                use as the signal names list.
            debug (bool):
                Force single thread execution for easy debugging. Defaults to ``False``.

        Returns:
            pandas.DataFrame:
                Table of readings for the target times, including the columns ``turbine_id``,
                ``signal_id``, ``timestamp`` and ``value``.
        """
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

        found_readings = [c for c in computed if len(c)]
        if not found_readings:
            msg = 'No readings found for the given target times in {}'.format(self._readings_path)
            raise ValueError(msg)

        readings = pd.concat(found_readings, ignore_index=True, sort=False)

        LOGGER.info('Loaded %s turbine readings', len(readings))

        if select_valid:
            target_times = select_valid_targets(target_times, readings, window_size)
            return target_times, readings

        return readings
