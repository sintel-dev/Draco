import os
from random import random

import pandas as pd


def load_results(files):
    problems_results = dict()
    for filename in files:
        problem = os.path.basename(filename).replace('.csv', '')
        problems_results[problem] = pd.read_csv(filename).round(6)

    return problems_results


def get_wins_by_problems(results):
    df = results.groupby('problem_name')['template', 'window_size', 'resample_rule', 'fpr_threshold=0.5']
    df = df.apply(max)
    df = df.rename(columns={'fpr_threshold=0.5': 'score'})

    return df


def get_exclusive_wins(scores, column, pivot_columns=['window_size', 'resample_rule']):
    summary = {}
    for problem in scores.problem_name.unique():
        df = scores[scores['problem_name'] == problem]
        df['wr'] = df.apply(
            lambda row: '{}_{}_{}'.format(row[pivot_columns[0]], row[pivot_columns[1]], random()), axis=1)
        df = df.pivot(index='wr', columns=column, values='fpr_threshold=0.5')

        is_winner = df.T.rank(method='min', ascending=False) == 1
        num_winners = is_winner.sum()
        is_exclusive = num_winners == 1
        is_exclusive_winner = is_winner & is_exclusive
        summary[problem] = is_exclusive_winner.sum(axis=1)

    summary_df = pd.DataFrame(summary)
    summary_df.index.name = 'template'
    columns = summary_df.columns.sort_values(ascending=False)
    return summary_df[columns]


def add_sheet(dfs, name, writer, cell_fmt, index_fmt, header_fmt):
    startrow = 0
    widths = [0]
    if not isinstance(dfs, dict):
        dfs = {None: dfs}

    for df_name, df in dfs.items():
        df = df.reset_index()
        startrow += bool(df_name)
        df.to_excel(writer, sheet_name=name, startrow=startrow + 1, index=False, header=False)

        worksheet = writer.sheets[name]

        if df_name:
            worksheet.write(startrow - 1, 0, df_name, index_fmt)
            widths[0] = max(widths[0], len(df_name))

        for idx, column in enumerate(df.columns):
            worksheet.write(startrow, idx, column, header_fmt)
            width = max(len(column), *df[column].astype(str).str.len()) + 1
            if len(widths) > idx:
                widths[idx] = max(widths[idx], width)
            else:
                widths.append(width)

        startrow += len(df) + 2

    for idx, width in enumerate(widths):
        fmt = cell_fmt if idx else index_fmt
        worksheet.set_column(idx, idx, width + 1, fmt)


def write_results(results, output):
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    cell_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10"
    })
    index_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
    })
    header_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
        "bottom": 1
    })

    if isinstance(results, dict):
        results = pd.concat(list(results.values()), ignore_index=True)

    window = get_exclusive_wins(results, 'window_size', ['window_size', 'fpr_threshold=0.5'])

    resample_pivots = ['resample_rule', ['problem_name', 'fpr_threshold=0.5']]
    resample = get_exclusive_wins(results, 'resample_rule', resample_pivots)

    summary = {
        'Best pipeline by Problem': get_wins_by_problems(results),
        'Rankings - Number of wins': get_exclusive_wins(results, 'template'),
        'Resample Rule': resample,
        'Window Size': window
    }
    add_sheet(summary, 'Summary', writer, cell_fmt, index_fmt, header_fmt)

    for problem in results['problem_name'].unique():
        add_sheet(
            results[results['problem_name'] == problem],
            problem,
            writer,
            cell_fmt,
            index_fmt,
            header_fmt
        )

    writer.save()
