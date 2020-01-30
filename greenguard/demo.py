# -*- coding: utf-8 -*-

import logging
import os

import pandas as pd

LOGGER = logging.getLogger(__name__)

S3_URL = 'https://d3-ai-greenguard.s3.amazonaws.com/'
DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')


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


def load_demo():
    """Load the demo included in the GreenGuard project.
    The first time that this function is executed, the data will be downloaded
    and cached inside the `greenguard/demo` folder.
    Subsequent calls will load the cached data instead of downloading it again.
    """
    target_times = _load_or_download('target_times', 'cutoff_time')
    readings = _load_or_download('readings', 'timestamp')

    return target_times, readings
