"""Functions for ranking features."""
import numpy as np

from third_party import ccm


# -----------------------------------------------------------------------------
# ---- Feature rankers --------------------------------------------------------
# -----------------------------------------------------------------------------
# pylint: disable=invalid-name
def pearson_rank(dataset):
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
        if denominator == 0:
            print([x[k][i] for k in range(m)])
            print(x_hat[i])
        return numerator / denominator

    scores = [R(i) for i in range(n)]
    ordered_features = [i for i in range(n)]
    ordered_features.sort(key=lambda feature: scores[feature], reverse=True)

    rankings = [ordered_features.index(feature) + 1 for feature in range(n)]
    return rankings


def ccm_rank(dataset):
    assert dataset, "Cannot select features from an empty dataset"

    # An N x K array where N is the number of samples and K is the number of features
    design_matrix = np.array(
        [list(observation) for observation, label in dataset])
    # An length-N 1-dimensional array where N is the number of samples
    label_matrix = np.array(
        [1 if label == 1 else -1 for observation, label in dataset])
    epsilon = 0.001
    num_features = len(dataset[0][0])
    type_Y = 'binary'
    rankings = ccm.ccm(design_matrix,
                       label_matrix,
                       num_features,
                       type_Y,
                       epsilon,
                       verbose=False)
    return list(rankings)


# -----------------------------------------------------------------------------
# ---- Feature selectors ------------------------------------------------------
# -----------------------------------------------------------------------------
def pearson_select(dataset, num_features):
    rankings = pearson_rank(dataset)
    top_features = np.argsort(rankings)[:num_features]
    feature_subset = [(1 if i in top_features else 0) for i in range(len(rankings))]
    return feature_subset


def ccm_select(dataset, num_features):
    assert dataset, "Cannot select features from an empty dataset"

    num_features = num_features if num_features else len(dataset[0][0])
    # An N x K array where N is the number of samples and K is the number of features
    design_matrix = np.array(
        [list(observation) for observation, label in dataset])
    # An length-N 1-dimensional array where N is the number of samples
    label_matrix = np.array(
        [1 if label == 1 else -1 for observation, label in dataset])
    epsilon = 0.001
    type_Y = 'binary'
    rankings = ccm.ccm(design_matrix,
                       label_matrix,
                       num_features,
                       type_Y,
                       epsilon,
                       verbose=False)
    top_features = np.argsort(rankings)[:num_features]
    feature_subset = [1 if i in top_features else 0 for i in range(len(rankings))]
    return feature_subset
