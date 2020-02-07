"""Targets module.

This module contains functions to work with target_times.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import trange

LOGGER = logging.getLogger(__name__)


def make_targets(target_times, window_size, target, new_targets=None):
    target_times = target_times.sort_values('cutoff_time', ascending=True)
    cutoff_times = target_times.cutoff_time
    window_size = pd.to_timedelta(window_size)
    original_size = len(target_times)
    current_size = original_size
    new_targets = new_targets or current_size

    for index in trange(len(cutoff_times) - 1):
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


def _to_timedelta(specification):
    if isinstance(specification, int):
        specification = '{}s'.format(specification)

    return pd.to_timedelta(specification)


def make_target_times(failure_dates, step, start=None, end=None, forecast_window=0,
                      prediction_window=0, before=0, after=0, offset=0, max_true=None,
                      max_false=None, shuffle=True):

    step = _to_timedelta(step)
    start = start or failure_dates.timestamp.min()
    start = start or failure_dates.min()

    forecast_window = _to_timedelta(forecast_window)
    prediction_window = _to_timedelta(prediction_window)
    before = _to_timedelta(before)
    after = _to_timedelta(after)
    offset = _to_timedelta(offset)

    target_times = pd.DataFrame()
    turbines = failure_dates.turbine_id.unique()
    failures = failure_dates.set_index(['turbine_id', 'date'])

    for turbine in turbines:
        turbine_failures = failures.loc[turbine]

        min_failure_date = turbine_failures.index.min() - before
        last_failure_date = turbine_failures.index.max() + after
        turbine_targets = list()
        while min_failure_date < last_failure_date:
            max_failure_date = min_failure_date + prediction_window
            day_failures = turbine_failures.loc[min_failure_date:max_failure_date]

            min_failure_date = min_failure_date + offset

            turbine_targets.append({
                'turbine_id': turbine,
                'target': int(bool(len(day_failures))),
                'cutoff_time': min_failure_date - forecast_window
            })

        turbine_targets = pd.DataFrame(turbine_targets)
        failed = turbine_targets[turbine_targets.target == 1]
        target_times = target_times.append(failed)

        non_failed = turbine_targets[turbine_targets.target == 0]
        non_failed = non_failed.sample(min(max_false, len(non_failed)))

        target_times = target_times.append(non_failed)

    if shuffle:
        target_times = target_times.sample(len(target_times))

    return target_times


def _valid_targets(timestamps):
    def apply_function(row):
        cutoff = row.cutoff_time
        try:
            times = timestamps.loc[row.turbine_id]
        except KeyError:
            return False

        return times['min'] < cutoff < times['max']

    return apply_function


def select_valid_targets(target_times, readings, window_size):
    """Filter out target_times without enough data for this window_size.

    The table_times table is scanned and checked against the readings table
    considering the window_size. All the target times entries that do not
    have enough data are dropped.

    Args:
        target_times (pandas.DataFrame):
            Target times table, with at least turbined_id and cutoff_time fields.
        readings (pandas.DataFrame):
            Readings table, with at least turbine_id, signal_id, and timestamp ields.
        window_size (str or pandas.TimeDelta):
            TimeDelta specification that indicates the lenght of the training window.

    Returns:
        pandas.DataFrame:
            New target_times table without the invalid targets.
    """

    timestamps = readings.groupby('turbine_id').timestamp.agg(['min', 'max'])
    timestamps['min'] += pd.to_timedelta(window_size)

    valid = target_times.apply(_valid_targets(timestamps), axis=1)
    valid_targets = target_times[valid].copy()

    LOGGER.info('Dropped %s invalid targets', len(target_times) - len(valid_targets))

    return valid_targets
