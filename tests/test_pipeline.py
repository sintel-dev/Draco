#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `wind.pipeline` module."""
from unittest import TestCase
from unittest.mock import Mock, patch

from wind.pipeline import WindPipeline


class TestWindPipeline(TestCase):
    """Tests for `TimeSeriesClassifier`."""

    @patch('wind.pipeline.MLPipeline.from_dict')
    def test_fit(self, from_dict_mock):
        """fit prepare the pipeline to make predictions based on the given data."""

        # Setup
        pipeline_mock = Mock()
        from_dict_mock.return_value = pipeline_mock

        # Run
        instance = WindPipeline(dict(), 'accuracy')
        instance.fit('an_X', 'a_y', {'some': 'tables'})

        # Asserts
        from_dict_mock.assert_called_once_with(dict())
        assert instance._pipeline == pipeline_mock

        pipeline_mock.fit.assert_called_once_with('an_X', 'a_y', entityset=None, some='tables')

        assert instance.fitted

    @patch('wind.pipeline.MLPipeline.from_dict')
    def test_predict(self, from_dict_mock):
        """predict produces results using the pipeline."""
        # Setup
        pipeline_mock = Mock()
        from_dict_mock.return_value = pipeline_mock

        # Run
        instance = WindPipeline(dict(), 'accuracy')
        instance.fitted = True
        instance.predict('an_X', {'some': 'tables'})

        # Asserts
        pipeline_mock.predict.assert_called_once_with('an_X', entityset=None, some='tables')
