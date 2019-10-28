# -*- coding: utf-8 -*-

"""Top-level package for GreenGuard."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.1-dev'

from greenguard.loader import load_demo
from greenguard.pipeline import GreenGuardPipeline, get_pipelines

__all__ = (
    'GreenGuardPipeline',
    'get_pipelines'
)
