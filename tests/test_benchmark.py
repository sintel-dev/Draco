"""Tests for `greenguard.benchmark` module."""
from sklearn.metrics import f1_score

from greenguard.benchmark import evaluate_templates
from greenguard.demo import load_demo


def test_predict():
    # setup
    templates = [
        'unstack_lstm_timeseries_classifier'
    ]

    window_size_rule = [
        ('1d', '1h')
    ]

    target_times, readings = load_demo()
    target_times = target_times.head(10)
    readings = readings.head(100)

    # run
    scores_df = evaluate_templates(
        target_times=target_times,
        readings=readings,
        templates=templates,
        window_size_rule=window_size_rule,
        metric=f1_score,
        tuning_iterations=1,
        cv_splits=2
    )

    # assert
    expected_columns = [
        'template',
        'window_size',
        'resample_rule',
        'default_test',
        'default_cv',
        'tuned_cv',
        'tuned_test',
        'status'
    ]

    expected_dtypes = [
        'object',
        'object',
        'object',
        'float64',
        'float64',
        'float64',
        'float64',
        'object'
    ]

    assert (scores_df.columns.to_list() == expected_columns)
    assert (scores_df.tuned_test.notnull)
    assert (scores_df.dtypes.to_list() == expected_dtypes)
