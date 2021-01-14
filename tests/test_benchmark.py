"""Tests for `greenguard.benchmark` module."""
import numpy as np

from greenguard.benchmark import evaluate_templates
from greenguard.demo import load_demo


def test_predict():
    # setup
    templates = [
        'probability.unstack_lstm_timeseries_classifier'
    ]

    window_size_rule = [
        ('1d', '1h')
    ]

    target_times, readings = load_demo()
    target_times = target_times.head(40)
    readings = readings.head(100)

    # run
    scores_df = evaluate_templates(
        target_times=target_times,
        readings=readings,
        templates=templates,
        window_size_rule=window_size_rule,
        tuning_iterations=1,
        cv_splits=2
    )

    # assert
    expected_columns = [
        'problem_name',
        'window_size',
        'resample_rule',
        'template',
        'default_test',
        'default_cv',
        'tuned_cv',
        'tuned_test',
        'tuning_metric',
        'tuning_metric_kwargs',
        'fit_predict_time',
        'default_cv_time',
        'average_cv_time',
        'total_time',
        'status',
        'accuracy_threshold/0.5',
        'f1_threshold/0.5',
        'fpr_threshold/0.5',
        'tpr_threshold/0.5',
    ]

    expected_dtypes = [
        np.dtype('O'),
        np.dtype('O'),
        np.dtype('O'),
        np.dtype('O'),
        np.dtype('float64'),
        np.dtype('float64'),
        np.dtype('float64'),
        np.dtype('float64'),
        np.dtype('O'),
        np.dtype('O'),
        np.dtype('<m8[ns]'),
        np.dtype('<m8[ns]'),
        np.dtype('<m8[ns]'),
        np.dtype('<m8[ns]'),
        np.dtype('O'),
        np.dtype('float64'),
        np.dtype('float64'),
        np.dtype('float64'),
        np.dtype('float64')
    ]

    assert (scores_df.columns.to_list() == expected_columns)
    assert (scores_df.tuned_test.notnull)
    assert (scores_df.dtypes.to_list() == expected_dtypes)
