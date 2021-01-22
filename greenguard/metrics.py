# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_curve, roc_auc_score, r2_score)

LOGGER = logging.getLogger(__name__)


def f1_macro(exp, obs):
    return f1_score(exp, obs, average='macro')


def threshold_score(ground_truth, probabilities, tpr):
    roc_fpr, roc_tpr, roc_threshold = roc_curve(ground_truth, probabilities, pos_label=1)
    try:
        index = np.where(roc_tpr >= tpr)[0][0]
    except:
        LOGGER.warn('Could not find a threshold that satisfies the requested True Positive Rate')
        index = -1

    return roc_threshold[index]


def tpr_score(ground_truth, probabilities, threshold):
    roc_fpr, roc_tpr, roc_threshold = roc_curve(ground_truth, probabilities, pos_label=1)
    try:
        index = np.where(roc_threshold >= threshold)[0][0]
    except:
        LOGGER.warn('Could not find a tpr that satisfies the requested threshold')
        index = -1

    return roc_tpr[index]


def fpr_score(ground_truth, probabilities, tpr=None, threshold=None):
    """Compute the False Positive Rate associated with the given True Positive Rate.

    This metric computes the False Positive Rate that needs to be assumed in order
    to achieve the desired True Positive Rate.
    The metric is computed by finding the minimum necessary threshold to ensure
    that the TPR is satisfied and then computing the associated FPR. The final output
    is 1 minus the found FPR to produce a maximization score between 0 and 1.

    Args:
        ground_truth (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions.
        probabilities (numpy.ndarray):
            ``numpy.ndarray`` with the generated predictions in probability.
        tpr (float):
            ``float`` value representing the percentage of True Positive Rate
            to be satisfied.

    Returns:
        float:
            Value between 0 and 1, where bigger is better.
    """
    roc_fpr, roc_tpr, roc_threshold = roc_curve(ground_truth, probabilities, pos_label=1)
    try:
        if tpr:
            index = np.where(roc_tpr >= tpr)[0][0]
        elif threshold:
            index = np.where(roc_threshold >= threshold)[0][0]

    except:
        LOGGER.warn('Could not find a threshold that satisfies the requested True Positive Rate')
        index = -1

    return 1 - roc_fpr[index]


METRICS = {
    'accuracy': (accuracy_score, False),
    'f1': (f1_score, False),
    'f1_macro': (f1_macro, False),
    'r2': (r2_score, False),
    'mse': (mean_squared_error, True),
    'mae': (mean_absolute_error, True),
    'fpr': (fpr_score, False),
    'roc_auc_score': (roc_auc_score, False)
}
