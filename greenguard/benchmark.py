import argparse
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from itertools import product

import pandas as pd
import tabulate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from greenguard import get_pipelines
from greenguard.demo import load_demo
from greenguard.loaders import CSVLoader
from greenguard.metrics import (METRICS, accuracy_score, f1_score,
                                fpr_score, tpr_score, threshold_score)
from greenguard.pipeline import GreenGuardPipeline, generate_init_params, generate_preprocessing
from greenguard.results import load_results, write_results

LOGGER = logging.getLogger(__name__)

DEFAULT_TUNING_METRIC_KWARGS = {'threshold': 0.5}
LEADERBOARD_COLUMNS = [
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
]


def _scorer(metric, metric_args):
    if isinstance(metric, str):
        metric, cost = METRICS[metric]

    def f(expected, observed):
        try:
            return metric(expected, observed, **metric_args)
        except TypeError:
            if 'threshold' not in metric_args:
                raise

            kwargs = metric_args.copy()
            threshold = kwargs.pop('threshold')
            observed = observed >= threshold
            return metric(expected, observed, **kwargs)

    return f


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
        primitive_params = template_params.setdefault(primitive, {})
        primitive_params.update(params)

    return template_params


def evaluate_template(
    template,
    target_times,
    readings,
    tuning_iterations=50,
    init_params=None,
    preprocessing=0,
    metrics=None,
    threshold=None,
    tpr=None,
    tuning_metric='roc_auc_score',
    tuning_metric_kwargs=DEFAULT_TUNING_METRIC_KWARGS,
    cost=False,
    cv_splits=3,
    test_size=0.25,
    random_state=0,
    cache_path=None,
    scores={}
):
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
    start_time = datetime.utcnow()
    scores['tuning_metric'] = str(tuning_metric)
    scores['tuning_metric_kwargs'] = tuning_metric_kwargs
    tuning_metric = _scorer(tuning_metric, tuning_metric_kwargs)

    train, test = train_test_split(target_times, test_size=test_size, random_state=random_state)

    pipeline = GreenGuardPipeline(
        template,
        metric=tuning_metric,
        cost=cost,
        cv_splits=cv_splits,
        init_params=init_params,
        preprocessing=preprocessing,
        cache_path=cache_path
    )

    # Computing the default test score
    fit_predict_time = datetime.utcnow()
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)
    fit_predict_time = datetime.utcnow() - fit_predict_time

    scores['default_test'] = tuning_metric(test['target'], predictions)

    # Computing the default cross validation score
    default_cv_time = datetime.utcnow()
    session = pipeline.tune(train, readings)
    session.run(1)
    default_cv_time = datetime.utcnow() - default_cv_time

    scores['default_cv'] = pipeline.cv_score

    # Computing the cross validation score with tuned hyperparameters
    average_cv_time = datetime.utcnow()
    session.run(tuning_iterations)
    average_cv_time = (datetime.utcnow() - average_cv_time) / tuning_iterations

    scores['tuned_cv'] = pipeline.cv_score

    # Computing the test score with tuned hyperparameters
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)
    ground_truth = test['target']

    # compute different metrics
    if tpr:
        tpr = tpr if isinstance(tpr, list) else [tpr]
        for value in tpr:
            threshold = threshold_score(ground_truth, predictions, tpr)
            scores[f'fpr_tpr/{value}'] = fpr_score(ground_truth, predictions, tpr=tpr)
            predictions_classes = predictions >= threshold
            scores[f'accuracy_tpr/{value}'] = accuracy_score(ground_truth, predictions_classes)
            scores[f'f1_tpr/{value}'] = f1_score(ground_truth, predictions_classes)
            scores[f'threshold_tpr/{value}'] = threshold_score(ground_truth, predictions, value)

            if f'accuracy_tpr/{value}' not in LEADERBOARD_COLUMNS:
                LEADERBOARD_COLUMNS.extend([
                    f'accuracy_tpr/{value}',
                    f'f1_tpr/{value}',
                    f'fpr_tpr/{value}',
                    f'threshold_tpr/{value}',
                ])

    else:
        threshold = 0.5 if threshold is None else threshold
        threshold = threshold if isinstance(threshold, list) else [threshold]

        for value in threshold:
            scores[f'fpr_threshold/{value}'] = fpr_score(
                ground_truth, predictions, threshold=value)

            predictions_classes = predictions >= threshold
            scores[f'accuracy_threshold/{value}'] = accuracy_score(
                ground_truth, predictions_classes)

            scores[f'f1_threshold/{value}'] = f1_score(ground_truth, predictions_classes)
            scores[f'tpr_threshold/{value}'] = tpr_score(ground_truth, predictions, value)

            if f'accuracy_threshold/{value}' not in LEADERBOARD_COLUMNS:
                LEADERBOARD_COLUMNS.extend([
                    f'accuracy_threshold/{value}',
                    f'f1_threshold/{value}',
                    f'fpr_threshold/{value}',
                    f'tpr_threshold/{value}',
                ])

    scores['tuned_test'] = tuning_metric(test['target'], predictions)
    scores['fit_predict_time'] = fit_predict_time
    scores['default_cv_time'] = default_cv_time
    scores['average_cv_time'] = average_cv_time
    scores['total_time'] = datetime.utcnow() - start_time

    return scores


def evaluate_templates(
    templates,
    window_size_rule,
    tuning_iterations=50,
    init_params=None,
    preprocessing=0,
    metrics=None,
    threshold=None,
    tpr=None,
    tuning_metric='roc_auc_score',
    tuning_metric_kwargs=DEFAULT_TUNING_METRIC_KWARGS,
    target_times=None,
    readings=None,
    cost=False,
    test_size=0.25,
    cv_splits=3,
    random_state=0,
    cache_path=None,
    cache_results=None,
    problem_name=None,
    output_path=None,
    progress_bar=None,
    multiprocess=False
):
    """Execute the benchmark process and optionally store the result as a ``CSV``.

    Args:
        templates (list):
            List of templates to try.
        window_size_rule (list):
            List of tuples (int, str or Timedelta object).
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


        try:
            LOGGER.info('Evaluating template %s on problem %s (%s, %s)',
                        template, problem_name, window_size, rule)

            template_params = init_params[template]
            template_params = _build_init_params(template, window_size, rule, template_params)
            template_preprocessing = preprocessing[template]
            if multiprocess:
                manager = mp.Manager()
                scores = manager.dict()
                process = mp.Process(
                    target=evaluate_template,
                    args=(
                        template,
                        target_times,
                        readings,
                        tuning_iterations,
                        init_params,
                        preprocessing,
                        metrics,
                        threshold,
                        tpr,
                        tuning_metric,
                        tuning_metric_kwargs,
                        cost,
                        cv_splits,
                        test_size,
                        random_state,
                        cache_path,
                        scores
                    )
                )

                process.start()
                process.join()
                if 'tuned_test' not in scores:
                    scores['status'] = 'ERRORED'

                scores = dict(scores)  # parse the managed dict to dict for pandas.

            else:
                scores = dict()
                scores['problem_name'] = problem_name
                scores['template'] = template
                scores['window_size'] = window_size
                scores['resample_rule'] = rule
                result = evaluate_template(
                    template=template,
                    target_times=target_times,
                    readings=readings,
                    metrics=metrics,
                    tuning_metric=tuning_metric,
                    tuning_metric_kwargs=tuning_metric_kwargs,
                    threshold=threshold,
                    tpr=tpr,
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

        if cache_results:
            os.makedirs(cache_results, exist_ok=True)
            template_name = template
            if os.path.isfile(template_name):
                template_name = os.path.basename(template_name).replace('.json', '')

            file_name = '{}_{}_{}_{}.csv'.format(problem_name, template_name, window_size, rule)
            df = pd.DataFrame([scores]).reindex(LEADERBOARD_COLUMNS, axis=1)
            df.to_csv(os.path.join(cache_results, file_name), index=False)

        scores_list.append(scores)

        if progress_bar:
            progress_bar.update(1)

    results = pd.DataFrame.from_records(scores_list)
    results = results.reindex(LEADERBOARD_COLUMNS, axis=1)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)
    else:
        return results


def _generate_target_times_readings(target_times, readings_path, window_size, rule, signals):
    """
    Returns:
        pandas.DataFrame:
            Table of readings for the target times, including the columns ``turbine_id``,
            ``signal_id``, ``timestamp`` and ``value``.
    """
    csv_loader = CSVLoader(
        readings_path,
        rule=rule,
    )

    return csv_loader.load(target_times, window_size=window_size, signals=signals)


def make_problems(target_times_paths, readings_path, window_size_resample_rule,
                  output_path=None, signals=None):
    """Make problems with the target times and readings for each window size and resample rule.

    Create problems in the accepted format by ``run_benchmark`` as pickle files containing:

        * ``target_times``: ``pandas.DataFrame`` containing the target times.
        * ``readings``: ``pandas.DataFrame`` containing the readings for the target times.
        * ``window_size``: window size value used.
        * ``resample_rule``: resample rule value used.

    Or return a ``dict`` containing as keys the names of the problems generated and tuples with
    the previously specified fields of target times, readings, window size and resample rule.

    Args:
        target_times_paths (list):
            List of paths to CSVs that contain target times.
        readings_path (str):
            Path to the folder where readings in raw CSV format can be found.
        window_size_resample_rule (list):
            List of tuples (int, str or Timedelta object).
        output_path (str):
            Path to save the generated problems.
        signals (str):
            List of signal names or csv file that has a `signal_id` column to use as the signal
            names list.
    """
    if isinstance(target_times_paths, str):
        target_times_paths = [target_times_paths]
    if isinstance(target_times_paths, list):
        target_times_paths = {
            os.path.basename(path).replace('.csv', ''): path
            for path in target_times_paths
        }

    if output_path:
        generated_problems = list()
    else:
        generated_problems = {}

    if isinstance(signals, str) and os.path.exists(signals):
        signals = pd.read_csv(signals).signal_id

    for problem_name, target_time_path in tqdm(target_times_paths.items()):
        for window_size, rule in window_size_resample_rule:
            target_times = pd.read_csv(target_time_path, parse_dates=['cutoff_time'])
            new_target_times, readings = _generate_target_times_readings(
                target_times,
                readings_path,
                window_size,
                rule,
                signals=signals,
            )

            pickle_name = '{}_{}_{}'.format(problem_name, window_size, rule)

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                output_pickle_path = os.path.join(output_path, pickle_name + '.pkl')
                with open(output_pickle_path, 'wb') as pickle_file:
                    pickle.dump((new_target_times, readings, window_size, rule), pickle_file)

                generated_problems.append(output_pickle_path)

            else:
                generated_problems[pickle_name] = (new_target_times, readings, window_size, rule)

    return generated_problems


def run_benchmark(templates, problems, window_size_resample_rule=None,
                  tuning_iterations=50, signals=None, preprocessing=0, init_params=None,
                  metrics=None, threshold=None, tpr=None, tuning_metric='roc_auc_score',
                  tuning_metric_kwargs=DEFAULT_TUNING_METRIC_KWARGS, cost=False, cv_splits=5,
                  test_size=0.33, random_state=0, cache_path=None, cache_results=None,
                  output_path=None, multiprocess=False):
    """Execute the benchmark function and optionally store the result as a ``CSV``.

    This function provides a user-friendly interface to interact with the ``evaluate_templates``
    function. It allows the user to specify an ``output_path`` where the results can be
    stored. If this path is not provided, a ``pandas.DataFrame`` will be returned.

    This function evaluates each template against each problem for each window size and resample
    rule possible, and will tune each teamplate for the given amount of tuning iterations.

    The problems can be a pickle file that contains the following values:

        * ``target_times``: ``pandas.DataFrame`` containing the target times.
        * ``readings``: ``pandas.DataFrame`` containing the readings for the target times.
        * ``window_size``: window size value used.
        * ``resample_rule``: resample rule value used.

    Or it can be dictionary containing the problem's name and as values either a path to a pickle
    file or a tuple containing the previously specified fields.

    Args:
        templates (str or list):
            Name of the json pipelines that will be evaluated against the problems.
        problems (str, list or dict):
            There are three possible values for problems:

                * ``str``: Path to a given problem stored as a pickle file (pkl).
                * ``list``: List of paths to given problems stored as a pickle files (pkl).
                * ``dict``: A dict containing as keys the name of the problem and as value the
                            path to a pickle file or a tuple with target times and readings data
                            frames and the window size and resample rule used to generate this
                            problem.

            The pickle files has to contain a tuple with target times and readings data frames and
            the window size and resample rule used to generate that problem. We recommend using
            the function ``make_problems`` to generate those files.

        window_size_resample_rule (list):
            List of tuples (int, str or Timedelta object).
        tuning_iterations (int):
            Amount of tuning iterations to perfrom over each template.
        signals (str or list):
            Path to a csv file containing ``signal_id`` column that we would like to use or a
            ``list`` of signals that we would like to use. If ``None`` use all the signals from
            the readings.
        preprocessing (int, dict or list):
            There are three possible values for preprocessing:

                * ``int``: the value will be used for all templates.
                * ``dict`` with the template name as a key and a number as a value, will
                  be used for that template.
                * ``list``: each value will be assigned to the corresponding position of
                  self.templates.

            Defaults to ``0``.
        init_params (dict or list):
            There are three possible values for init_params:

                * Init params ``dict``: It will be used for all templates.
                * ``dict`` with the name of the template as a key and dictionary with its
                  init params.
                * ``list``: each value will be assigned to the corresponding position of
                  self.templates.

            Defaults to ``None``.
        metric (function or str):
            Metric to use. If an ``str`` is give it must be one of the metrics
            defined in the ``greenguard.metrics.METRICS`` dictionary.
        cost (bool):
            Whether the metric is a cost function (the lower the better) or not.
            Defaults to ``False``.
        cv_splits (int):
            Number of cross validation folds to use. Defaults to ``5``.
        test_size (float):
            Amount of data that will be saved for test, represented in percentage between 0 and 1.
        random_state (int or RandomState):
            random state to use for the cross validation partitioning. Defaults to ``0``.
        cache_path (str):
            If given, cache the generated cross validation splits in this folder.
            Defatuls to ``None``.
        cache_results (str):
            If provided, store the progress of each pipeline and each problem while runing.
        output_path (str):
            If provided, store the results to the given filename. Defaults to ``None``.
    """
    templates = templates if isinstance(templates, (list, tuple)) else [templates]
    results = list()
    if isinstance(problems, str):
        problems = [problems]
    if isinstance(problems, list):
        problems = {
            os.path.basename(problem).replace('.pkl', ''): problem
            for problem in problems
        }

    if signals is not None:
        if isinstance(signals, str) and os.path.exists(signals):
            signals = pd.read_csv(signals).signal_id

    total_runs = len(templates) * len(problems) * len(window_size_resample_rule or [1])
    pbar = tqdm(total=total_runs)

    for problem_name, problem in problems.items():
        # remove window_size resample_rule nomenclature from the problem's name
        problem_name = re.sub(r'\_\d+[DdHhMmSs]', r'', problem_name)

        if isinstance(problem, str):
            with open(problem, 'rb') as pickle_file:
                target_times, readings, orig_window_size, orig_rule = pickle.load(pickle_file)
        else:
            target_times, readings, orig_window_size, orig_rule = problem

        if signals is not None:
            readings = readings[readings.signal_id.isin(signals)]

        wsrr = window_size_resample_rule or [(orig_window_size, orig_rule)]

        orig_window_size = pd.to_timedelta(orig_window_size)
        orig_rule = pd.to_timedelta(orig_rule)

        for window_size, resample_rule in wsrr:

            # window_size can be only smaller than pickle window size
            # resample rule can be only bigger than picke rule
            if (orig_window_size >= pd.to_timedelta(window_size)
                    and orig_rule <= pd.to_timedelta(resample_rule)): # noqa W503

                df = evaluate_templates(
                    templates,
                    [(window_size, resample_rule)],
                    metrics=metrics,
                    tuning_iterations=tuning_iterations,
                    threshold=threshold,
                    tpr=tpr,
                    init_params=init_params,
                    target_times=target_times,
                    readings=readings,
                    preprocessing=preprocessing,
                    cost=cost,
                    test_size=test_size,
                    cv_splits=cv_splits,
                    random_state=random_state,
                    cache_path=cache_path,
                    cache_results=cache_results,
                    problem_name=problem_name,
                    output_path=None,
                    progress_bar=pbar,
                    multiprocess=multiprocess,
                )

                results.append(df)

                if cache_results:
                    file_name = '{}_{}_{}.csv'.format(problem_name, window_size, resample_rule)
                    df.to_csv(os.path.join(cache_results, file_name), index=False)

            else:
                pbar.update(1)

                msg = 'Invalid window size or resample rule {}.'.format(
                    (window_size, orig_window_size, resample_rule, orig_rule))

                LOGGER.warn(msg)

    pbar.close()

    results = pd.concat(results, ignore_index=True)
    if output_path:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)

    else:
        return results


def _setup_logging(args):
    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(filename=args.logfile, level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


def _run(args):
    _setup_logging(args)
    if args.templates is None:
        args.templates = get_pipelines()

    window_size_resample_rule = None
    if args.window_size_resample_rule:
        pattern = re.compile(r'\d+[DdHhMmSs]')
        window_size_resample_rule = [
            tuple(pattern.findall(item))
            for item in args.window_size_resample_rule
        ]

    if args.tuning_metric_kwargs:
        args.tuning_metric_kwargs = json.loads(args.tuning_metric_kwargs)

    else:
        args.tuning_metric_kwargs = DEFAULT_TUNING_METRIC_KWARGS

    # run
    results = run_benchmark(
        templates=args.templates,
        problems=args.problems,
        window_size_resample_rule=window_size_resample_rule,
        cv_splits=args.cv_splits,
        metrics=args.metrics,
        threshold=args.threshold,
        tpr=args.tpr,
        tuning_metric=args.tuning_metric,
        tuning_metric_kwargs=args.tuning_metric_kwargs,
        test_size=args.test_size,
        random_state=args.random_state,
        cache_path=args.cache_path,
        cache_results=args.cache_results,
        tuning_iterations=args.iterations,
        output_path=args.output_path,
        signals=args.signals,
        multiprocess=args.multiprocess
    )

    if not args.output_path:
        print(tabulate.tabulate(
            results,
            tablefmt='github',
            headers=results.columns
        ))


def summarize_results(input_paths, output_path):
    """Load multiple benchmark results CSV files and compile a summary.

    The result is an Excel file with one tab for each results CSV file
    and an additional Number of Wins tab with a summary.

    Args:
        inputs_paths (list[str]):
            List of paths to CSV files where the benchmarks results are stored.
            These files must have one column per Tuner and one row per Challenge.
        output_path (str):
            Path, including the filename, where the Excel file will be created.
    """
    results = load_results(input_paths)
    write_results(results, output_path)


def _summarize_results(args):
    summarize_results(args.input, args.output)


def _make_problems(args):
    window_size_resample_rule = list(product(args.window_size, args.resample_rule))
    make_problems(
        args.target_times_paths,
        args.readings_path,
        window_size_resample_rule,
        output_path=args.output_path,
        signals=args.signals
    )


def _get_parser():
    parser = argparse.ArgumentParser(description='GreenGuard Benchmark Command Line Interface.')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # Run action
    run = action.add_parser('run', help='Run the GreenGuard Benchmark')
    run.set_defaults(action=_run)
    run.set_defaults(user=None)

    run.add_argument('-v', '--verbose', action='count', default=0,
                     help='Be verbose. Use -vv for increased verbosity.')
    run.add_argument('-l', '--logfile',
                     help='Log file.')
    run.add_argument('-t', '--templates', nargs='+',
                     help='Perform benchmarking over the given list of templates.')
    run.add_argument('-p', '--problems', nargs='+', required=False,
                     help='Perform benchmarking over a list of pkl problems.')
    run.add_argument('-w', '--window-size-resample-rule', nargs='+', required=False,
                     help='List of window sizes values to benchmark.')
    run.add_argument('-o', '--output_path', type=str,
                     help='Output path where to store the results.')
    run.add_argument('-s', '--cv-splits', type=int, default=5,
                     help='Amount of cross validation splits to use.')
    run.add_argument('-m', '--metrics', nargs='+',
                     help='Names of metric functions to be used for the benchmarking.')
    run.add_argument('-T', '--threshold', nargs='+', type=float,
                     help='Threhshold values for the metrics.')
    run.add_argument('-P', '--tpr', nargs='+', type=float,
                     help='TPR vales for the metrics, if provided threshold will be ignored.')
    run.add_argument('-n', '--random-state', type=int, default=0,
                     help='Random state for the cv splits.')
    run.add_argument('-e', '--test-size', type=float, default=0.33,
                     help='Percentage of the data set to be used for the test.')
    run.add_argument('-c', '--cache-path', type=str,
                     help='Path to cache the generated cross validation splits in.')
    run.add_argument('-R', '--cache-results', type=str,
                     help='Path to store the csv files for each problem and template.')
    run.add_argument('-i', '--iterations', type=int, default=100,
                     help='Number of iterations to perform per challenge with each candidate.')
    run.add_argument('-S', '--signals', type=str,
                     help='Path to csv file that has signal_id column to use as the signal')
    run.add_argument('-k', '--tuning-metric', type=str, default='roc_auc_score',
                     help='Tuning metric to be used.')
    run.add_argument('-K', '--tuning-metric-kwargs', type=str,
                     help='Tuning metric args to be used with the metric.')
    run.add_argument('-u', '--multiprocess', action='store_true',
                     help='Wether or not to spawn a separate process and avoid crashing.')


    # Summarize action
    summary = action.add_parser('summarize-results',
                                help='Summarize the GreenGuard Benchmark results')
    summary.set_defaults(action=_summarize_results)
    summary.add_argument('input', nargs='+', help='Input path with results.')
    summary.add_argument('output', help='Output file.')

    # Make problems action
    problems = action.add_parser('make-problems', help='Create GreenGuard problems')
    problems.set_defaults(action=_make_problems)
    problems.add_argument('target-times-paths', nargs='+', help='List of target times paths.')
    problems.add_argument('readings-path', type=str, help='Path to the readings folder.')
    problems.add_argument('-w', '--window-size', nargs='+', required=False,
                          help='List of window sizes values to benchmark.')
    problems.add_argument('-r', '--resample-rule', nargs='+', required=False,
                          help='List of resample rule to benchmark.')
    problems.add_argument('-o', '--output', type=str,
                          help='Output path where to save the generated problems.')
    problems.add_argument('-s', '--signals', type=str,
                          help='Path to csv file that has signal_id column to use as the signal')

    return parser


def main():
    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
