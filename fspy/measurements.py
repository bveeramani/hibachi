"""Functions for evaluating the effectiveness of feature selection algorithms."""
from itertools import combinations

import torch

from fspy.training import train, test
from fspy.models import LogisticRegressionModel
from fspy import utils


# -----------------------------------------------------------------------------
# ---- Stability measures -----------------------------------------------------
# -----------------------------------------------------------------------------
# pylint: disable=invalid-name
def pearson_coefficient(w, w_prime):
    """Computes Pearson's correlation coefficient between two weighting-scorings

    A weighting-scoring is an element of R^m, where m is the total number of
    features.

    Arguments:
        w (list): A weighting-scoring
        w_prime (list): A weighting-scoring

    Returns:
        Pearson's correlation coefficient between w and w_prime.
    """
    w_hat = sum(w) / len(w)
    w_prime_hat = sum(w_prime) / len(w_prime)
    numerator = sum([
        (w[i] - w_hat) * (w_prime[i] - w_prime_hat) for i in range(len(w))
    ])
    a = sum([pow(w[i] - w_hat, 2) for i in range(len(w))])
    b = sum([pow(w_prime[i] - w_prime_hat, 2) for i in range(len(w_prime))])
    denominator = pow(a * b, 1 / 2)
    return numerator / denominator


# pylint: disable=invalid-name
def spearman_coefficient(r, r_prime):
    """Computes Spearman's rank correlation coefficient between two rankings.

    See https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient.

    Let m represent the total number of features. Then, a ranking r is a list
    (r_0, r_1, ..., r_m-1) where 1 <= r_i <= m for all i, r_i is the rank of
    feature i in ranking r, and each element of {1,..., m} appears exactly once
    in r.

    Arguments:
        r (list): a ranking
        r_prime (list): a ranking

    Returns:
        Spearman's rank correlation coefficient between the inputs.
    """
    assert len(r) == len(r_prime), "r and r_prime must be same length"

    m = len(r)
    if m == 1:
        return 1 if r == r_prime else 0

    f = lambda x, y: pow((x - y), 2) / (m * (pow(m, 2) - 1))
    return 1 - 6 * sum([f(r[i], r_prime[i]) for i in range(m)])


# pylint: disable=invalid-name
def tanimoto_distance(s, s_prime):
    """Computes the Tanimoto distance between two sets.

    See https://en.wikipedia.org/wiki/Jaccard_index.

    A feature subset is a subset of {0,..., m} where m is the total number of
    features.

    Arguments:
        s (list): a feature subset
        s_prime (list): a feature subset

    Returns:
        The Jaccard distance between the inputs.
    """
    a = len(s) + len(s_prime)
    b = len(s.intersection(s_prime))
    return 1 - (a - 2 * b) / (a - b)


# -----------------------------------------------------------------------------
# ---- Evaluation methods -----------------------------------------------------
# -----------------------------------------------------------------------------
def measure_accuracy(selected_features, dataset, num_folds=10):
    """Returns the accuracy from training a logistic regression model on the
    selected features.

    This function averages accuracies as measured by k-fold cross-validation.

    A feature subset is a subset of {0,..., m} where m is the total number of
    features.

    Arguments:
        selected_featurse: A feature subset.
        dataset (list): A list of (x, y) pairs.
        num_folds (int, optional): The number of folds to cross-validate on.

    Returns:
        The average prediction accuracy over the folds.
    """
    assert dataset, "dataset cannot be empty"

    num_features = len(dataset[0][0])
    features = {i for i in range(num_features)}
    unselected_features = features - selected_features

    dataset = utils.eliminate(dataset, unselected_features)
    folds = utils.fold(dataset, num_folds)

    accuracies = []
    for training_fold, testing_fold in folds:
        model = LogisticRegressionModel(len(selected_features))
        model = train(model, training_fold, quiet=True)
        accuracy = test(model, testing_fold, quiet=True)

        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / len(accuracies)

    return average_accuracy


def measure_stability(selection_algorithm,
                      dataset,
                      stability_measure=tanimoto_distance,
                      num_folds=10):
    """Returns the stability of a feature selection algorithm with respect to a
    stability measure as measured by k-fold cross validation.

    The stability is approximated by subsampling the dataset via k-fold cross-
    validation and generating features on the subsamples. Each pair of features
    is compared using the stability measure. The average stability measure
    across all pairs of features is returned.

    Arguments:
        selection_algorithm (func): A one-argument function that accepts a
            dataset as an argument and returns a feature selection.
        dataset (list): A list of (x, y) pairs.
        stability_measure (func, optional): A real-valued function that accepts
            two feature selections as arguments.
        num_folds (int, optional): The number of folds to measure stability on.

    Returns:
        The average stability across all pairs of features generated.
    """
    folds = utils.fold(dataset, num_folds)

    feature_selections = []
    for training_fold, testing_fold in folds:
        selected_features = selection_algorithm(training_fold)
        feature_selections.append(selected_features)

    stabilities = []
    for f, f_prime in combinations(feature_selections, 2):
        stability = stability_measure(f, f_prime)
        stabilities.append(stability)
    average_stability = sum(stabilities) / len(stabilities)

    return average_stability
