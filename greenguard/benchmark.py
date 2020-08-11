import logging
from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split

from greenguard.demo import load_demo
from greenguard.metrics import METRICS
from greenguard.pipeline import GreenGuardPipeline, generate_init_params, generate_preprocessing

LOGGER = logging.getLogger(__name__)


def _build_init_params(template, window_size, rule, template_params):
    if 'dfs' in template:
        window_size_rule_params = {
            'pandas.DataFrame.resample#1': {
                'rule': rule,
            },
            'featuretools.dfs.json#1': {
                'training_window': window_size,
            }
        }
    elif 'lstm' in template:
        window_size_rule_params = {
            'pandas.DataFrame.resample#1': {
                'rule': rule,
            },
            'mlprimitives.custom.timeseries_preprocessing.cutoff_window_sequences#1': {
                'window_size': window_size,
            }
        }

    for primitive, params in window_size_rule_params.items():
        primitive_params = template_params.get(primitive, {})
        primitive_params.update(params)

    return template_params


def evaluate_template(template, target_times, readings, metric='f1', tuning_iterations=50,
                      preprocessing=0, init_params=None, cost=False, test_size=0.25,
                      cv_splits=3, random_state=0, cache_path=None):
    """Returns the scores for a given template.

    Args:
        template (str):
            Given template to evaluate.
        target_times (DataFrame):
            Contains the specefication problem that we are solving, which has three columns:

                * turbine_id: Unique identifier of the turbine which this label corresponds to.
                * cutoff_time: Time associated with this target.
                * target: The value that we want to predict. This can either be a numerical value
                          or a categorical label. This column can also be skipped when preparing
                          data that will be used only to make predictions and not to fit any
                          pipeline.

        readings (DataFrame):
            Contains the signal data from different sensors, with the following columns:

                * turbine_id: Unique identifier of the turbine which this reading comes from.
                * signal_id: Unique identifier of the signal which this reading comes from.
                * timestamp (datetime): Time where the reading took place, as a datetime.
                * value (float): Numeric value of this reading.

        metric (function or str):
            Metric to use. If an ``str`` is give it must be one of the metrics
            defined in the ``greenguard.metrics.METRICS`` dictionary.
        tuning_iterations (int):
            Number of iterations to be used.
        preprocessing (int, list or dict):
            Number of preprocessing steps to be used.
        init_params (list):
            Initialization parameters for the pipeline.
        cost (bool):
            Wheter the metric is a cost function (the lower the better) or not.
        test_size (float):
            Percentage of the data set to be used for the test.
        cv_splits (int):
            Amount of splits to create.
        random_state (int):
            Random number of train_test split.
        cache_path (str):
            If given, cache the generated cross validation splits in this folder.
            Defatuls to ``None``.

    Returns:
        scores (dict):
            Stores the four types of scores that are being evaluate.
    """
    scores = dict()

    train, test = train_test_split(target_times, test_size=test_size, random_state=random_state)

    if isinstance(metric, str):
        metric, cost = METRICS[metric]

    pipeline = GreenGuardPipeline(
        template,
        metric,
        cost=cost,
        cv_splits=cv_splits,
        init_params=init_params,
        preprocessing=preprocessing,
        cache_path=cache_path
    )

    # Computing the default test score
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)

    scores['default_test'] = metric(test['target'], predictions)

    # Computing the default cross validation score
    session = pipeline.tune(train, readings)
    session.run(1)

    scores['default_cv'] = pipeline.cv_score

    # Computing the cross validation score with tuned hyperparameters
    session.run(tuning_iterations)

    scores['tuned_cv'] = pipeline.cv_score

    # Computing the test score with tuned hyperparameters
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)

    scores['tuned_test'] = metric(test['target'], predictions)

    return scores


def evaluate_templates(templates, window_size_rule, metric='f1',
                       tuning_iterations=50, init_params=None, target_times=None,
                       readings=None, preprocessing=0, cost=False, test_size=0.25,
                       cv_splits=3, random_state=0, cache_path=None, output_path=None):
    """Execute the benchmark process and optionally store the result as a ``CSV``.

    Args:
        templates (list):
            List of templates to try.
        window_size_rule (list):
            List of tupples (int, str or Timedelta object).
        metric (function or str):
            Metric to use. If an ``str`` is give it must be one of the metrics
            defined in the ``greenguard.metrics.METRICS`` dictionary.
        tuning_iterations (int):
            Number of iterations to be used.
        init_params (dict):
            Initialization parameters for the pipelines.
        target_times (DataFrame):
            Contains the specefication problem that we are solving, which has three columns:

                * turbine_id: Unique identifier of the turbine which this label corresponds to.
                * cutoff_time: Time associated with this target.
                * target: The value that we want to predict. This can either be a numerical value
                          or a categorical label. This column can also be skipped when preparing
                          data that will be used only to make predictions and not to fit any
                          pipeline.

        readings (DataFrame):
            Contains the signal data from different sensors, with the following columns:

                * turbine_id: Unique identifier of the turbine which this reading comes from.
                * signal_id: Unique identifier of the signal which this reading comes from.
                * timestamp (datetime): Time where the reading took place, as a datetime.
                * value (float): Numeric value of this reading.

        preprocessing (int, list or dict):
            Number of preprocessing steps to be used.
        cost (bool):
            Wheter the metric is a cost function (the lower the better) or not.
        test_size (float):
            Percentage of the data set to be used for the test.
        cv_splits (int):
            Amount of splits to create.
        random_state (int):
            Random number of train_test split.
        output_path (str):
            Path where to save the benchmark report.
        cache_path (str):
            If given, cache the generated cross validation splits in this folder.
            Defatuls to ``None``.

    Returns:
        pandas.DataFrame or None:
            If ``output_path`` is ``None`` it will return a ``pandas.DataFrame`` object,
            else it will dump the results in the specified ``output_path``.

    Example:
        >>> from sklearn.metrics import f1_score
        >>> templates = [
        ...    'normalize_dfs_xgb_classifier',
        ...    'unstack_lstm_timeseries_classifier'
        ... ]
        >>> window_size_rule = [
        ...     ('30d','12h'),
        ...     ('7d','4h')
        ... ]
        >>> preprocessing = [0, 1]
        >>> scores_df = evaluate_templates(
        ...                 templates=templates,
        ...                 window_size_rule=window_size_rule,
        ...                 metric=f1_score,
        ...                 tuning_iterations=5,
        ...                 preprocessing=preprocessing,
        ...                 cost=False,
        ...                 test_size=0.25,
        ...                 cv_splits=3,
        ...                 random_state=0
        ...             )
        >>> scores_df
                                 template window_size resample_rule  default_test  default_cv  tuned_cv  tuned_test status
    0  unstack_lstm_timeseries_classifier         30d           12h      0.720000    0.593634  0.627883    0.775510     OK
    1  unstack_lstm_timeseries_classifier          7d            4h      0.723404    0.597440  0.610766    0.745098     OK
    2        normalize_dfs_xgb_classifier         30d           12h      0.581818    0.619698  0.637123    0.596491     OK
    3        normalize_dfs_xgb_classifier          7d            4h      0.581818    0.619698  0.650367    0.603774     OK

    """  # noqa

    if readings is None and target_times is None:
        target_times, readings = load_demo()

    init_params = generate_init_params(templates, init_params)
    preprocessing = generate_preprocessing(templates, preprocessing)

    scores_list = []
    for template, window_rule in product(templates, window_size_rule):
        window_size, rule = window_rule

        scores = dict()
        scores['template'] = template
        scores['window_size'] = window_size
        scores['resample_rule'] = rule

        try:
            template_params = init_params[template]
            template_params = _build_init_params(template, window_size, rule, template_params)
            template_preprocessing = preprocessing[template]

            result = evaluate_template(
                template=template,
                target_times=target_times,
                readings=readings,
                metric=metric,
                tuning_iterations=tuning_iterations,
                preprocessing=template_preprocessing,
                init_params=template_params,
                cost=cost,
                test_size=test_size,
                cv_splits=cv_splits,
                random_state=random_state,
                cache_path=cache_path
            )

            scores.update(result)
            scores['status'] = 'OK'

        except Exception:
            scores['status'] = 'ERRORED'
            LOGGER.exception('Could not score template %s ', template)

        scores_list.append(scores)

    results = pd.DataFrame.from_records(scores_list)
    results = results.reindex(['template', 'window_size', 'resample_rule', 'default_test',
                               'default_cv', 'tuned_cv', 'tuned_test', 'status'], axis=1)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)
    else:
        return results
