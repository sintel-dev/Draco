from greenguard.metrics import fpr_score

def test_perfect_predict():
    truth = [0] * 50 + [1] * 50
    false_probs = np.random.random(0, 0.1, size=50)
    true_probs = np.random.random(0.9, 1, size=50)
    probs = np.concatenate([false_probs, true_probs])

    score = fpr_score(truth, probs, tpr=1)
    assert score == 1
￼
￼
def test_one_third_predict():
    truth = [0] * 50 + [1] * 50
    false_probs = np.random.random(0, 0.6, size=50)
    true_probs = np.random.random(0.4, 1, size=50)
    probs = np.concatenate([false_probs, true_probs])

    score = fpr_score(truth, probs, tpr=1)
    assert score == 1 / 3


def test_one_sixt_predict():
    truth = [0] * 50 + [1] * 50
    false_probs = np.random.random(0, 0.6, size=50)
    true_probs = np.random.random(0.4, 1, size=50)
    probs = np.concatenate([false_probs, true_probs])

    score = fpr_score(truth, probs, tpr=(5 / 6))
    assert score == 1 / 6
    ￼
￼
def test_two_thirds_predict():
    truth = [0] * 50 + [1] * 50
    false_probs = np.random.random(0, 0.6, size=50)
    true_probs = np.random.random(0.4, 1, size=50)
    probs = np.concatenate([false_probs, true_probs])

    score = fpr_score(truth, probs, tpr=(2 / 3))
    assert score == 1
