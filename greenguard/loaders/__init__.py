import logging

LOGGER = logging.getLogger(__name__)


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
    timestamps = readings.groupby('turbine_id').timestamp.agg(['min', 'max'])
    timestamps['min'] += window_size

    valid = target_times.apply(_valid_targets(timestamps), axis=1)
    valid_targets = target_times[valid].copy()

    LOGGER.info('Dropped %s invalid targets', len(target_times) - len(valid_targets))

    return valid_targets
