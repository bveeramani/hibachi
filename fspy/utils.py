from torch.utils.data import Dataset, ConcatDataset, random_split

from fspy.datasets import ListDataset
import torch

def eliminate(dataset, features):
    """Returns a copy of the dataset with the specified features removed.

    Arguments:
        dataset (list): A list of x, y pairs.
        features (set): A set containing the zero-indexed positions of features
            to remove.

    Returns:
        A copy of the dataset with the specified features removed.
    """
    assert dataset, "dataset cannot be empty"
    num_features = len(dataset[0][0])

    def transform(observation):
        observation = torch.tensor([
            observation[i].item() for i in range(num_features) if i not in features
        ])
        return observation

    observations = [transform(observation) for observation, label in dataset]
    labels = [label for observation, label in dataset]
    return ListDataset(observations, labels)


def fold(dataset, num_folds):
    """Creates a list of training-fold test-fold pairs.

    Use this function for k-fold cross validation.

    Arguments:
        dataset (list): A list of x, y pairs.
        num_folds (int): The nubmer of folds to create.

    Returns:
        A list of pairs. The first element of the i-th pair is a dataset
        containing all of the data except for the data in the i-th fold. The
        second element is a dataset containing all of the data in the i-th fold.
    """
    assert num_folds > 1, "num_folds must be greater than one"

    subsample_size = int(len(dataset) / num_folds)
    subsample_sizes = tuple(
        subsample_size if i < num_folds - 1 else len(dataset) - subsample_size *
        (num_folds - 1) for i in range(num_folds))
    subsamples = random_split(dataset, subsample_sizes)

    folds = []
    for i in range(num_folds):
        testing_fold = subsamples[i]
        training_fold = ConcatDataset(
            [subsamples[j] for j in range(num_folds) if i is not j])
        folds.append(tuple([training_fold, testing_fold]))
    return folds
