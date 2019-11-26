# -*- coding: utf-8 -*-

from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score)


def f1_macro(exp, obs):
    return f1_score(exp, obs, average='macro')


METRICS = {
    'accuracy': (accuracy_score, False),
    'f1': (f1_score, False),
    'f1_macro': (f1_macro, False),
    'r2': (r2_score, False),
    'mse': (mean_squared_error, True),
    'mae': (mean_absolute_error, True)
}
