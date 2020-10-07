import argparse
import logging
import os
import pickle
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
from greenguard.metrics import METRICS
from greenguard.pipeline import GreenGuardPipeline, generate_init_params, generate_preprocessing
from greenguard.results import summarize_results
from greenguard.utils import as_list

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
    start_time = datetime.utcnow()

    scores = dict()
    scores['metric'] = metric

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
    fit_predict_time = datetime.utcnow()
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)
    fit_predict_time = datetime.utcnow() - fit_predict_time

    scores['default_test'] = metric(test['target'], predictions)

    # Computing the default cross validation score
    cv_time = datetime.utcnow()
    session = pipeline.tune(train, readings)
    session.run(1)
    cv_time = datetime.utcnow() - cv_time

    scores['default_cv'] = pipeline.cv_score

    # Computing the cross validation score with tuned hyperparameters
    session.run(tuning_iterations)

    scores['tuned_cv'] = pipeline.cv_score

    # Computing the test score with tuned hyperparameters
    pipeline.fit(train, readings)
    predictions = pipeline.predict(test, readings)

    scores['tuned_test'] = metric(test['target'], predictions)
    scores['fit_predict_time'] = fit_predict_time
    scores['cv_time'] = cv_time
    scores['total_time'] = datetime.utcnow() - start_time

    return scores


def evaluate_templates(templates, window_size_rule, metric='f1', tuning_iterations=50,
                       init_params=None, target_times=None, readings=None, preprocessing=0,
                       cost=False, test_size=0.25, cv_splits=3, random_state=0, cache_path=None,
                       cache_results=None, problem_name=None, output_path=None):
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
        scores['problem_name'] = problem_name
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

        if cache_results:
            os.makedirs(cache_results, exist_ok=True)
            template_name = template
            if os.path.isfile(template_name):
                template_name = os.path.basename(template_name).replace('.json', '')

            file_name = '{}_{}_{}_{}.csv'.format(problem_name, template_name, window_size, rule)
            pd.DataFrame([scores]).to_csv(os.path.join(cache_results, file_name), index=False)

        scores_list.append(scores)

    results = pd.DataFrame.from_records(scores_list)
    columns = [
        'problem_name',
        'window_size',
        'resample_rule',
        'template',
        'default_test',
        'default_cv',
        'tuned_cv',
        'tuned_test',
        'fit_predict_time',
        'cv_time',
        'total_time',
        'status',
    ]

    results = results.reindex(columns, axis=1)

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
    """
    Args:
        target_times_paths (list):
            List of paths to CSVs that contain target times.
        readings_path (str):
            Path to the folder where readings in raw CSV format can be found.
        window_size_resample_rule (list):
            List of tupples (int, str or Timedelta object).
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

    for problem_name, target_time_path in tqdm(target_times_paths.values()):
        for window_size, rule in window_size_resample_rule:
            target_times = pd.read_csv(target_time_path, parse_dates=['cutoff_time'])
            new_target_times, readings = _generate_target_times_readings(
                target_times,
                readings_path,
                window_size,
                rule,
                signals=signals,
            )

            problem_name = '{}_{}_{}.pkl'.format(problem_name, window_size, rule)

            if output_path:
                output_pickle_path = os.path.join(output_path, problem_name)
                with open(output_pickle_path, 'wb') as pickle_file:
                    pickle.dump((new_target_times, readings, window_size, rule), pickle_file)

                generated_problems.append(output_pickle_path)

            else:
                generated_problems[problem_name] = (new_target_times, readings, window_size, rule)

    return generated_problems


def run_benchmark(templates, problem_paths=None, window_size_resample_rule=None,
                  tuning_iterations=100, preprocessing=0, init_params=None, cost=False,
                  cv_splits=5, metric='f1', test_size=0.33, random_state=0, cache_path=None,
                  output_path=None, cache_results=None):
    """
    Args:
        templates (list):
        problem_paths (list):
        window_size_resample_rule (list):
            List of tupples (int, str or Timedelta object).
        tuning_iterations (int):
        preprocessing :
        init_params :
        cost :
        test_size :
        cv_splits (int):
        cache_path (str):
        output_path (str):
    """
    templates = as_list(templates)
    results = list()
    problem_paths = as_list(problem_paths)

    for problem_path in tqdm(problem_paths):
        with open(problem_path, 'rb') as pickle_file:
            target_times, readings, pickle_window_size, pickle_rule = pickle.load(pickle_file)

        problem_name = '_'.join(os.path.basename(problem_path).split('_')[:-2])

        if window_size_resample_rule is None:
            window_size_resample_rule = [(pickle_window_size, pickle_rule)]

        for window_size, resample_rule in window_size_resample_rule:

            # window_size can be only smaller than pickle window size
            # resample rule can be only bigger than picke rule
            if (pd.to_timedelta(pickle_window_size) >= pd.to_timedelta(window_size)
                    and pd.to_timedelta(pickle_rule) <= pd.to_timedelta(resample_rule)): # noqa W503

                df = evaluate_templates(
                    templates,
                    [(window_size, resample_rule)],
                    metric=metric,
                    tuning_iterations=tuning_iterations,
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
                    output_path=None
                )

                results.append(df)

                if cache_results:
                    file_name = '{}_{}_{}.csv'.format(problem_name, window_size, resample_rule)
                    df.to_csv(os.path.join(cache_results, file_name), index=False)

            else:
                msg = 'Invalid window size or resample rule {}.'.format(
                    (window_size, pickle_window_size, resample_rule, pickle_rule))

                LOGGER.info(msg)

    results = pd.concat(results, ignore_index=True)

    if output_path:
        results.to_csv(output_path, index=False)

    else:
        return results


def _run(args):
    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("ax").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    if args.templates is None:
        args.templates = get_pipelines()

    window_size_resample_rule = None
    if args.window_size:
        window_size_resample_rule = list(product(args.window_size, args.resample_rule))

    # run
    results = run_benchmark(
        templates=args.templates,
        problem_paths=args.problem_paths,
        window_size_resample_rule=window_size_resample_rule,
        cv_splits=args.cv_splits,
        metric=args.metric,
        test_size=args.test_size,
        random_state=args.random_state,
        cache_path=args.cache_path,
        cache_results=args.cache_results,
        tuning_iterations=args.iterations,
        output_path=args.output_path,
    )

    if not args.output_path:
        print(tabulate.tabulate(
            results,
            tablefmt='github',
            headers=results.columns
        ))


def _summary(args):
    summarize_results(args.input, args.output)


def _create(args):
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
    run.add_argument('-t', '--templates', nargs='+',
                     help='Perform benchmarking over the given list of templates.')
    run.add_argument('-p', '--problem-paths', nargs='+', required=False,
                     help='Perform benchmarking over a list of pkl problems.')
    run.add_argument('-w', '--window-size', nargs='+', required=False,
                     help='List of window sizes values to benchmark.')
    run.add_argument('-r', '--resample-rule', nargs='+', required=False,
                     help='List of resample rule to benchmark.')
    run.add_argument('-o', '--output_path', type=str,
                     help='Output path where to store the results.')
    run.add_argument('-s', '--cv-splits', type=int, default=5,
                     help='Amount of cross validation splits to use.')
    run.add_argument('-m', '--metric', type=str, default='f1',
                     help='Name of metric function to be used during benchmarking.')
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

    # Summarize action
    summary = action.add_parser('summary', help='Summarize the GreenGuard Benchmark results')
    summary.set_defaults(action=_summary)
    summary.add_argument('input', nargs='+', help='Input path with results.')
    summary.add_argument('output', help='Output file.')

    # Create action
    create = action.add_parser('create', help='Create GreenGuard problems')
    create.set_defaults(action=_create)
    create.add_argument('target-times-paths', nargs='+', help='List of target times paths.')
    create.add_argument('readings-path', type=str, help='Path to the readings folder.')
    create.add_argument('-w', '--window-size', nargs='+', required=False,
                        help='List of window sizes values to benchmark.')
    create.add_argument('-r', '--resample-rule', nargs='+', required=False,
                        help='List of resample rule to benchmark.')
    create.add_argument('-o', '--output', type=str,
                        help='Output path where to save the generated problems.')
    create.add_argument('-s', '--signals', type=str,
                        help='Path to csv file that has signal_id column to use as the signal')

    return parser


def main():
    warnings.filterwarnings("ignore")

    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
