"""Functions for ranking features."""
import numpy as np

from third_party import ccm


# -----------------------------------------------------------------------------
# ---- Feature rankers --------------------------------------------------------
# -----------------------------------------------------------------------------
# pylint: disable=invalid-name
def pearson_rank(dataset, num_features=None):
    """Ranks features using the Pearson correlation criteria.

    Arguments:
        dataset (torch.utils.data.Dataset): The dataset to rank features on.
        num_features (int, optional): If not none, the top num_features
            features will be selected; otherwise, rankings for all features
            will be returned.

    Returns:
        rankings for the top num_features features.
    """
    assert dataset, "Cannot select features from an empty dataset"

    x = [x.tolist() for x, y in dataset]
    y = [float(y) for x, y in dataset]
    m = len(dataset)
    n = len(x[0])

    x_hat = [sum([x[k][i] for k in range(m)]) / m for i in range(n)]
    y_hat = sum([y[k] for k in range(m)]) / m

    def R(i):
        numerator = sum([(x[k][i] - x_hat[i]) * (y[k] - y_hat) for k in range(m)
                        ])
        a = sum([pow(x[k][i] - x_hat[i], 2) for k in range(m)])
        b = sum([pow(y[k] - y_hat, 2) for k in range(m)])
        denominator = pow(a * b, 1 / 2)
        return numerator / denominator

    scores = [R(i) for i in range(n)]
    features = [i + 1 for i in range(n)]
    features.sort(key=lambda feature: scores[feature - 1], reverse=True)
    num_features = num_features if num_features else m
    return features[:num_features]


def ccm_rank(dataset, num_features=None):
    """Ranks features using conditional covariance minimization.

    See https://github.com/Jianbo-Lab/CCM.

    Arguments:
        dataset (torch.utils.data.Dataset): The dataset to rank features on.
        num_features (int, optional): If not none, the top num_features
            features will be selected; otherwise, rankings for all features
            will be returned.

    Returns:
        rankings for the top num_features features.
    """
    assert dataset, "Cannot select features from an empty dataset"

    num_features = num_features if num_features else len(dataset[0][0])
    # An N x K array where N is the number of samples and K is the number of features
    design_matric = np.array(
        [list(observation) for observation, label in dataset])
    # An length-N 1-dimensional array where N is the number of samples
    label_matrix = np.array(
        [1 if label == 1 else -1 for observation, label in dataset])
    epsilon = 0.001
    type_Y = 'binary'
    rankings = ccm.ccm(design_matric,
                       label_matrix,
                       num_features,
                       type_Y,
                       epsilon,
                       verbose=False)
    selected_features = np.argsort(rankings)[:num_features]
    return selected_features


# -----------------------------------------------------------------------------
# ---- Utility functions ------------------------------------------------------
# -----------------------------------------------------------------------------
def wrap_ranker(ranker, num_features):
    """Returns a ranker that takes one argument and selects num_features.

    This is usefull for the evaluations.evaluate_stability function.

    Arguments:
        ranker (func): A two-argument function that accepts a dataset and an
            integer representing the number of features to select
        num_features (int): A positive integer representing the number of
            features to select.
    """
    assert num_features > 0

    def wrapped_ranker(dataset):
        return ranker(dataset, num_features)

    return wrapped_ranker
