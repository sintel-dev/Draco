import os

import pandas as pd


def read_csv(dataset_path, table, timestamp=False):
    if timestamp:
        timestamp = ['timestamp']

    path = os.path.join(dataset_path, table + '.csv')
    return pd.read_csv(path, parse_dates=timestamp, infer_datetime_format=True)


def load_data(dataset_path, target_table, target_column, readings_table='readings',
              turbines_table='turbines', signals_table='signals'):

    tables = {
        'readings': read_csv(dataset_path, readings_table, True),
        'signals': read_csv(dataset_path, signals_table),
        'turbines': read_csv(dataset_path, turbines_table),
    }

    X = read_csv(dataset_path, target_table, True)
    y = X.pop(target_column)

    return X, y, tables
