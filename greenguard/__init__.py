# -*- coding: utf-8 -*-

"""Top-level package for GreenGuard."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.1-dev'

import os

from greenguard.demo import load_demo
from greenguard.pipeline import GreenGuardPipeline, get_pipelines

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PIPELINES = os.path.join(_BASE_PATH, 'pipelines')


__all__ = (
    'GreenGuardPipeline',
    'get_pipelines',
    'load_demo'
)
