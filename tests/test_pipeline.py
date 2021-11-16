#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `draco.pipeline` module."""
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from draco.pipeline import DracoPipeline


class TestDracoPipeline(TestCase):

    def _get_data(self):
        target_times = pd.DataFrame({
            'turbine_id': ['T001'],
            'cutoff_time': [pd.Timestamp('2010-01-01')],
            'target': [1]
        })
        readings = pd.DataFrame({
            'turbine_id': ['T001'],
            'timestamp': [pd.Timestamp('2010-01-01')],
            'signal_id': ['S1'],
            'value': [0.1]
        })
        return target_times, readings

    @patch('draco.pipeline.MLPipeline')
    @patch('draco.pipeline.load_pipeline')
    def test_fit(self, load_pipeline_mock, mlpipeline_mock):
        load_pipeline_mock.return_value = dict()

        # Run
        instance = DracoPipeline('a_pipeline', 'accuracy')
        target_times, readings = self._get_data()
        instance.fit(target_times, readings)

        # Asserts
        assert instance.fitted

    @patch('draco.pipeline.MLPipeline')
    @patch('draco.pipeline.load_pipeline')
    def test_predict(self, load_pipeline_mock, mlpipeline_mock):
        load_pipeline_mock.return_value = dict()

        # Run
        instance = DracoPipeline('a_pipeline', 'accuracy')
        instance.fitted = True
        target_times, readings = self._get_data()
        instance.predict(target_times, readings)
