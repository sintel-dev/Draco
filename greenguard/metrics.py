# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_curve, r2_score)

LOGGER = logging.getLogger(__name__)


def f1_macro(exp, obs):
    return f1_score(exp, obs, average='macro')


METRICS = {
    'accuracy': (accuracy_score, False),
    'f1': (f1_score, False),
    'f1_macro': (f1_macro, False),
    'r2': (r2_score, False),
    'mse': (mean_squared_error, True),
    'mae': (mean_absolute_error, True),
    'fpr_score'; (fpr_score, False)
}


def fpr_score(ground_true, probabilities, tpr=1):
    """
    Args:
        ground_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions.
        probabilities (numpy.ndarray):
            ``numpy.ndarray`` with the generated predictions in probability.
        tpr (float):
            ``float`` value representing the percentage of True Positive Rate
            to be satisfied.

    Returns:
        ``float``:
            The percentage on base 1 of the false positive rate regarding the specified ``tpr``.
    """
    roc_fpr, roc_tpr, roc_threshold = roc_curve(ground_true, probabilities, pos_label=1)
    try:
        index = np.where(roc_tpr >= tpr)[0][0]
    except:
        LOGGER.warn('Could not find a threshold that satisfies the requested True Positive Rate')
        index = -1

    return 1 - roc_fpr[index]
