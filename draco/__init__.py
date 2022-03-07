# -*- coding: utf-8 -*-

"""Top-level package for Draco."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.1.dev0'

import os

from draco.pipeline import DracoPipeline, get_pipelines

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives')
MLBLOCKS_PIPELINES = tuple(
    dirname
    for dirname, _, _ in os.walk(os.path.join(_BASE_PATH, 'pipelines'))
)

__all__ = (
    'DracoPipeline',
    'get_pipelines',
)
