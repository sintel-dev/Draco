#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `greenguard.pipeline` module."""
from unittest import TestCase
from unittest.mock import Mock, patch

from greenguard.pipeline import GreenGuardPipeline


class TestGreenGuardPipeline(TestCase):
    """Tests for `TimeSeriesClassifier`."""

    @patch('greenguard.pipeline.MLPipeline')
    def test_fit(self, pipeline_class_mock):
        """fit prepare the pipeline to make predictions based on the given data."""
        # Run
        instance = GreenGuardPipeline('pipeline_name', 'accuracy')
        instance.fit('an_X', 'a_y', 'readings')

        # Asserts
        pipeline_mock = pipeline_class_mock.return_value
        pipeline_class_mock.assert_called_once_with('pipeline_name')
        assert instance._pipeline == pipeline_mock

        pipeline_mock.fit.assert_called_once_with('an_X', 'a_y', readings='readings')

        assert instance.fitted

    @patch('greenguard.pipeline.MLPipeline')
    def test_predict(self, pipeline_mock):
        """predict produces results using the pipeline."""
        # Run
        instance = GreenGuardPipeline('pipeline_name', 'accuracy')
        instance.fitted = True
        instance.predict('an_X', 'readings')

        # Asserts
        pipeline_mock.return_value.predict.assert_called_once_with('an_X', readings='readings')
