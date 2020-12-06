import numpy as np

from greenguard.metrics import fpr_score


def test_fpr_score_perfect_scenario():
    truth = [0, 0, 0, 1, 1, 1]
    false_probs = [0.2, 0.4, 0.6]
    true_probs = [0.8, 0.7, 0.9]
    probs = np.concatenate([false_probs, true_probs])
    score = fpr_score(truth, probs, tpr=1)
    assert score == 1


def test_fpr_score_predict_over_half():
    truth = [0, 0, 0, 0, 1, 1, 1, 1]
    false_probs = [0.1, 0.2, 0.4, 0.6]
    true_probs = [0.5, 0.7, 0.8, 0.9]
    probs = np.concatenate([false_probs, true_probs])
    score = fpr_score(truth, probs, tpr=1)
    assert score == 0.75


def test_fpr_score_predict_half():
    truth = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    false_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    true_probs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    probs = np.concatenate([false_probs, true_probs])
    score = fpr_score(truth, probs, tpr=1)
    assert score == 0.5


def test_fpr_score_predict_one_third():
    truth = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    false_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    true_probs = [0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    probs = np.concatenate([false_probs, true_probs])
    score = fpr_score(truth, probs, tpr=1)
    assert round(score, 4) == 0.3333
