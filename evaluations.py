"""Functions for evaluating the effectiveness of feature selection algorithms."""
from itertools import combinations

import torch
from torch.utils.data import random_split, ConcatDataset

from train import train, test
from models import LogisticRegressionModel


# -----------------------------------------------------------------------------
# ---- Stability measures -----------------------------------------------------
# -----------------------------------------------------------------------------
# pylint: disable=invalid-name
def spearman_coefficient(r, r_prime):
    """Computes Spearman's rank correlation coefficient between two rankings.

    See https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient.

    A ranking is a list r = (r_1, r_2, ..., r_m) where 1 <= r_i <= m for all i
    and each element of {1, ..., m} appears exactly once in r.

    Arguments:
        r (list): a ranking
        r_prime (list): a ranking

    Returns:
        Spearman's rank correlation coefficient between the inputs.
    """
    assert len(r) == len(r_prime), "r and r_prime must be same length"

    m = len(r)
    f = lambda x, y: pow((x - y), 2) / (m * (pow(m, 2) - 1))
    return 1 - 6 * sum([f(r[i], r_prime[i]) for i in range(m)])


# pylint: disable=invalid-name
def tanimoto_distance(s, s_prime):
    """Computes the Tanimoto distance between two sets.

    See https://en.wikipedia.org/wiki/Jaccard_index.

    Arguments:
        s (list): a list of integers
        s_prime (list): a list of integers

    Returns:
        The Jaccard distance between the inputs.
    """
    s, s_prime = set(s), set(s_prime)
    a = len(s) + len(s_prime)
    b = len(s.intersection(s_prime))
    return 1 - (a - 2 * b) / (a - b)


# -----------------------------------------------------------------------------
# ---- Evaluation methods -----------------------------------------------------
# -----------------------------------------------------------------------------
def evaluate_accuracy(selected_features, dataset, num_folds=10):
    """Returns the accuracy from training a logistic regression model on the
    selected features.

    This function averages accuracies from k-fold cross-validation.

    Arguments:
        selected_features (list): A one-indexed list of selected features.
        dataset (torch.utils.data.Dataset): The training dataset.
        num_folds (int, optional): The number of folds to cross-validate on.

    Returns:
        The average prediction accuracy over the folds.
    """
    dataset = remove_unselected_features(dataset, selected_features)
    folds = create_folds(dataset, num_folds)

    accuracies = []
    for training_fold, testing_fold in folds:
        model = LogisticRegressionModel(len(selected_features))

        model = train(model, training_fold)
        accuracy = test(model, testing_fold)

        accuracies.append(accuracy)
    average_accuracy = sum(accuracies) / len(accuracies)

    return average_accuracy


def evaluate_stability(selection_algorithm,
                       dataset,
                       stability_measure=tanimoto_distance,
                       num_folds=10):
    """Returns the stability of a feature selection algorithm as measured by a
    stability measure.

    The stability is approximated by subsampling the dataset via k-fold cross-
    validation and generating features on the subsamples. Each pair of features
    is compared using some stability measure. The average stability measure
    across all pairs of features is returned.

    Arguments:
        selection_algorithm (func): A function that accepts a dataset as an
            arguments and returns a list of one-indexed features.
        dataset (torch.utils.data.Dataset): The dataset to evaluate stability on.
        stability_measure (func, optional): A real-valued function that accepts
            two lists of integers as arguments.
        num_folds (int, optional): The number of folds to measure stability on.

    Returns:
        The average stability across all pairs of features generated.
    """
    folds = create_folds(dataset, num_folds)

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


# -----------------------------------------------------------------------------
# ---- Utility functions ------------------------------------------------------
# -----------------------------------------------------------------------------
def remove_unselected_features(dataset, selected_features):
    """Returns a copy of the dataset without the selected features.

    Arguments:
        dataset (torch.utils.data.Dataset): The dataset to remove features from.
        selected_features (list): A one-indexed list of feature positions.

    Returns:
        A copy of the dataset with the selected features removed.
    """
    assert dataset, "dataset cannot be empty"
    num_features = len(dataset[0])

    def transform(observation, label):
        observation = [
            observation[i]
            for i in range(num_features)
            if i + 1 in selected_features
        ]
        observation = torch.tensor(observation)
        return observation, label

    dataset = [transform(observation, label) for observation, label in dataset]
    return dataset


def create_folds(dataset, num_subsamples):
    """Creates a list of training-fold test-fold pairs.

    Use this function for k-fold cross validation.

    Arguments:
        dataset (list): An iterable of x, y pairs.
        num_subsamples (int): The nubmer of folds to create.

    Returns:
        A list of pairs. The first element of the i-th pair is a dataset
        containing all of the data except for the data in the i-th fold. The
        second element is a dataset containing all of the data in the i-th fold.
    """
    subsample_size = int(len(dataset) / num_subsamples)
    subsample_sizes = tuple(
        subsample_size if i < num_subsamples - 1 else len(dataset) -
        subsample_size * (num_subsamples - 1) for i in range(num_subsamples))
    subsamples = random_split(dataset, subsample_sizes)

    folds = []
    for i in range(num_subsamples):
        testing_fold = subsamples[i]
        training_fold = ConcatDataset(
            [subsamples[j] for j in range(num_subsamples) if i is not j])
        folds.append(tuple([training_fold, testing_fold]))
    return folds
