# -*- coding: utf-8 -*-

import logging
import os

import pandas as pd

LOGGER = logging.getLogger(__name__)

S3_URL = 'https://d3-ai-greenguard.s3.amazonaws.com/'
DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')

_FILES = {
    'DEFAULT': [
        ('target_times', 'cutoff_time'),
        ('readings', 'timestamp')
    ],
    'RUL': [
        ('rul_train_target_times', 'cutoff_time'),
        ('rul_test_target_times', 'cutoff_time'),
        ('rul_readings', 'timestamp')
    ]
}

def _load_or_download(filename, dates):
    filename += '.csv.gz'
    file_path = os.path.join(DEMO_PATH, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path, compression='gzip', parse_dates=[dates])

    os.makedirs(DEMO_PATH, exist_ok=True)
    url = S3_URL + filename

    LOGGER.info('Downloading %s from %s', filename, url)
    data = pd.read_csv(url, compression='gzip', parse_dates=[dates])
    data.to_csv(file_path, index=False, compression='gzip')

    return data


def load_demo(name='default', load_readings=True):
    """Load the demo included in the Draco project.

    The first time that this function is executed, the data will be downloaded
    and cached inside the `draco/demo` folder.
    Subsequent calls will load the cached data instead of downloading it again.
    
    Args:
        rul (str):
            Name of the dataset to load. If "RUL", load NASA's CMAPSS dataset
            https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan.
            If "default" then load default demo.
        load_readings (bool):
            Whether to load the ``readings`` table or not.

    Returns:
        tuple[pandas.DataFrame]:
            target_times and readings tables
    """
    files = _FILES[name.upper()]

    if not load_readings:
        files = files[:-1]

    output = list()
    for filename, dates in files:
        output.append(_load_or_download(filename, dates))

    return tuple(output)


def generate_raw_readings(output_path='demo'):
    """Generate raw readings based on the demo data.

    Args:
        path (str):
            Path where the readings will be generated.
    """
    target_times, readings = load_demo()

    for turbine_id in target_times.turbine_id.unique():
        turbine_path = os.path.join(output_path, turbine_id)
        os.makedirs(turbine_path, exist_ok=True)
        data = readings[readings.turbine_id == turbine_id]
        for month in range(1, 13):
            month_data = data[data.timestamp.dt.month == month].copy()
            month_data['timestamp'] = month_data['timestamp'].dt.strftime('%m/%d/%y %H:%M:%S')
            month_path = os.path.join(turbine_path, '2013-{:02d}.csv'.format(month))
            LOGGER.info('Generating file %s', month_path)
            month_data[['signal_id', 'timestamp', 'value']].to_csv(month_path, index=False)

    return target_times
