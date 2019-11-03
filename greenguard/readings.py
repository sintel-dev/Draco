"""readings module.

This module contains functions to work directly with turbine readins in raw format.

This raw format has the following characteristics:

    * All the data from all the turbines is inside a single folder.
    * Inside the data folder, a folder exists for each turbine.
      This folders are named exactly like each turbine id, and inside it one or more
      CSV files can be found. The names of these files is not relevant.
    * Each CSV file will have the the following columns:

        * timestamp: timestemp of the reading.
        * signal: name or id of the signal.
        * value: value of the reading.
"""

import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def make_targets(target_times, window_size, target, new_targets=None):
    target_times = target_times.sort_values('cutoff_time', ascending=True)
    cutoff_times = target_times.cutoff_time
    window_size = pd.to_timedelta(window_size)
    original_size = len(target_times)
    current_size = original_size
    new_targets = new_targets or current_size

    for index in range(len(cutoff_times) - 1):
        timestamp = cutoff_times.iloc[index]
        next_time = cutoff_times.iloc[index + 1]

        if timestamp + (window_size * 2) >= next_time:
            continue

        span_start = timestamp + window_size
        span_end = next_time - window_size
        span_length = (span_end - span_start).total_seconds()

        delay = pd.to_timedelta(np.random.randint(span_length), unit='s')
        cutoff_time = span_start + delay

        target_times = target_times.append(pd.Series({
            'turbine_id': target_times.iloc[index].turbine_id,
            'cutoff_time': cutoff_time,
            'target': target
        }), ignore_index=True)

        current_size = len(target_times)
        if current_size == original_size + new_targets:
            return target_times.sort_values('cutoff_time', ascending=True)

    if current_size == original_size:
        warnings.warn('There is no space left between to add more targets.')
        return target_times

    new_targets = new_targets - (current_size - original_size)
    return make_targets(target_times, window_size, target, new_targets)


def _filter_by_filename(target_times, filenames):
    max_csv = target_times.end.dt.strftime('%Y-%m-.csv')
    min_csv = target_times.start.dt.strftime('%Y-%m-.csv')

    for filename in filenames:
        if ((min_csv <= filename) & (filename <= max_csv)).any():
            yield filename


def _load_readings_file(turbine_file):
    LOGGER.info('Loading file %s', turbine_file)
    data = pd.read_csv(turbine_file)
    data.columns = data.columns.str.lower()
    data.rename(columns={'signal': 'signal_id'}, inplace=True)

    if 'unnamed: 0' in data.columns:
        # Someone forgot to drop the index before
        # storing the DataFrame as a CSV
        del data['unnamed: 0']

    LOGGER.info('Loaded %s readings from file %s', len(data), turbine_file)

    return data


def _filter_by_signal(data, signals):
    if signals is not None:
        LOGGER.info('Filtering by signal')
        data = data[data.signal_id.isin(signals.signal_id)]

    LOGGER.info('Selected %s readings by signal', len(data))

    return data


def _filter_by_timestamp(data, target_times):
    LOGGER.info('Parsing timestamps')
    timestamps = pd.to_datetime(data['timestamp'], format='%m/%d/%y %H:%M:%S')
    data['timestamp'] = timestamps

    LOGGER.info('Filtering by timestamp')

    related = [False] * len(timestamps)
    for row in target_times.itertuples():
        related |= (row.start <= timestamps) & (timestamps <= row.end)

    data = data[related]

    LOGGER.info('Selected %s readings by timestamp', len(data))

    return data


def _load_turbine_readings(readings_path, target_times, signals):
    turbine_id = target_times.turbine_id.iloc[0]
    turbine_path = os.path.join(readings_path, turbine_id)
    filenames = sorted(os.listdir(turbine_path))
    filenames = _filter_by_filename(target_times, filenames)

    readings = list()
    for readings_file in filenames:
        readings_file_path = os.path.join(turbine_path, readings_file)
        data = _load_readings_file(readings_file_path)
        data = _filter_by_signal(data, signals)
        data = _filter_by_timestamp(data, target_times)

        readings.append(data)

    if readings:
        readings = pd.concat(readings)
    else:
        readings = pd.DataFrame(columns=['timestamp', 'signal_id', 'value', 'turbine_id'])

    LOGGER.info('Loaded %s readings from turbine %s', len(readings), turbine_id)

    return readings


def _get_times(target_times, window_size):
    cutoff_times = target_times.cutoff_time
    if window_size:
        window_size = pd.to_timedelta(window_size)
        min_times = cutoff_times - window_size
    else:
        min_times = [datetime.min] * len(cutoff_times)

    return pd.DataFrame({
        'turbine_id': target_times.turbine_id,
        'start': min_times,
        'end': cutoff_times,
    })


def _load_readings(readings_path, target_times, signals, window_size):
    turbine_ids = target_times.turbine_id.unique()

    target_times = _get_times(target_times, window_size)

    readings = list()
    for turbine_id in sorted(turbine_ids):
        turbine_target_times = target_times[target_times['turbine_id'] == turbine_id]
        LOGGER.info('Loading turbine %s readings', turbine_id)
        turbine_readings = _load_turbine_readings(readings_path, turbine_target_times, signals)
        turbine_readings['turbine_id'] = turbine_id
        readings.append(turbine_readings)

    return pd.concat(readings)


def extract_readings(readings_path, target_times, signals=None, window_size=None,
                     add_targets=False, new_target_value=None):
    """Extract raw readings data for the given target_times.

    The ``target_times`` table is examined to decide from which turbines found
    in the ``reading_pathp`` which data to load.

    And the output is a ``pandas.DataFrame`` containing:

        * `turbine_id`: Unique identifier of the turbine which this reading comes from.
        * `signal_id`: Unique identifier of the signal which this reading comes from.
        * `timestamp`: Time where the reading took place, as an ISO formatted datetime.
        * `value`: Numeric value of this reading.

    Args:
        readings_path (str):
            Path to the folder containing all the readings data.
        target_times (pd.DataFrame or str):
            target_times DataFrame or path to the target_times CSV file.
        signals (list):
            List of signals to load from the readings files. If not given, load
            all the signals available.
        window_size (str):
            Rule indicating how long back before the cutoff times we have to go
            when loading the data.
        add_targets (bool):
            Whether to add new target times with random cutoff times.
        new_target_value (str):
            Target value to use when adding target times.

    Returns:
        pandas.DataFrame
    """
    if isinstance(target_times, pd.DataFrame):
        target_times = target_times.copy()
    else:
        target_times = pd.read_csv(target_times)

    target_times['cutoff_time'] = pd.to_datetime(target_times['cutoff_time'])

    if add_targets:
        if not new_target_value:
            raise ValueError('Cannot add targets without a new target value')
        target_times = make_targets(target_times, window_size, new_target_value)

    without_duplicates = target_times.drop_duplicates(subset=['cutoff_time', 'turbine_id'])
    if len(target_times) != len(without_duplicates):
        raise ValueError("Duplicate rows found in target_times")

    if isinstance(signals, list):
        signals = pd.DataFrame({'signal_id': signals})
    elif isinstance(signals, str):
        signals = pd.read_csv(signals)

    readings = _load_readings(readings_path, target_times, signals, window_size)
    LOGGER.info('Loaded %s turbine readings', len(readings))

    return readings
