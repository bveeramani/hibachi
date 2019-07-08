"""Classes for preprocessing and loading datasets."""
import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ---- Synthetic datasets -----------------------------------------------------
# -----------------------------------------------------------------------------

class OrangeSkin(Dataset):
    """Synthetic binary classification dataset.

    Given Y = -1, (X_1, ..., X_10) ~ N(0, I_10). Given Y = 1, (X_5, ..., X_10) ~
    N(0, I_5) and (X_1, ... X_4) ~ N(0, I_4) conditioned on 9 <= (X_1)^2 + ... +
    (X_4)^2 <= 16.

    Arguments:
        size: The number of samples to generate.
    """

    def __init__(self, size):
        self.data = []
        for _ in range(size // 2):
            observation = torch.randn(10)
            label = torch.tensor(-1)

            sample = observation, label
            self.data.append(sample)

        for _ in range(size - size // 2):
            observation = torch.randn(4)

            while not 9 <= torch.sum(torch.pow(observation, 2)) <= 16:
                observation = torch.randn(4)

            observation = torch.cat((observation, torch.randn(6)))
            label = torch.tensor(1)

            sample = observation, label
            self.data.append(sample)

        random.shuffle(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ANR(Dataset):
    """Synthetic additive nonlinear regression dataset.

    Data is generated using the following distribution:

        Y = -2sin(2X_1) + max(X_2, 0) + X_3 + exp(-X_4) + epsilon

    where (X_1, ..., X_10) ~ N(0, I_10) and epsilon ~ N(0, 1).

    Arguments:
        size: The number of samples to generate.
    """
    def __init__(self, size, std=1):
        self.data = []

        for _ in range(size):
            X = torch.randn(10)
            epsilon = torch.squeeze(torch.normal(torch.tensor(0.), torch.tensor(float(std))))
            Y = -2 * torch.sin(2 * X[0]) + max(X[1], 0) + X[2] + torch.exp(-X[3]) + epsilon

            sample = X, Y
            self.data.append(sample)

        random.shuffle(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# -----------------------------------------------------------------------------
# ---- Real-world datasets ----------------------------------------------------
# -----------------------------------------------------------------------------

class Lactamase(Dataset):
    """Real-world binary classification dataset containing non-categorical data
    taken from a Cytochrome P450 gene library.

    Arguments:
        filename (string): Path to a CSV file containing the data.
        transform (callable, optional): A two-argument function that takes
            a list of 51 numbers and a binary label, and returns a transformed
            version of each.
    """

    NUM_FEATURES = 51

    def __init__(self, filename, transform=None):
        with open(filename) as file:
            self.data = file.readlines()[1:]
        self.transform = transform

        random.shuffle(self.data)


    def __getitem__(self, index):
        # First line contains headings so we add 1 to index
        line = self.data[index]

        observation = line.split(",")[2:]
        observation = [float(number) for number in observation]

        label = line.split(",")[1]
        label = int(float(label))

        if self.transform:
            observation, label = self.transform(observation, label)

        return observation, label

    def __len__(self):
        return len(self.data)


class P450(Dataset):
    """Real-world binary classification dataset containing non-categorical data
    taken from a Cytochrome P450 gene library.

    Arguments:
        filename (string): Path to a CSV file containing the data.
        transform (callable, optional): A two-argument function that takes
            a list of 51 numbers and a binary label, and returns a transformed
            version of each.
    """

    NUM_FEATURES = 51

    def __init__(self, filename, transform=None):
        with open(filename) as file:
            self.lines = file.readlines()
        self.transform = transform

    def __getitem__(self, index):
        # First line contains headings so we add 1 to index
        line = self.lines[index + 1]

        observation = line.split(",")[2:]
        observation = [float(number) for number in observation]

        label = line.split(",")[1]
        label = int(float(label))

        if self.transform:
            observation, label = self.transform(observation, label)

        return observation, label

    def __len__(self):
        # First line contains headings so we substract 1 to length
        return len(self.lines) - 1


class P450C(Dataset):
    """Real-world binary classification dataset containing categorical data
    taken from a Cytochrome P450 gene library.

    The data set contains eight categories, with each category belonging to one
    of three classes. Each category is encoded using a one-hot encoding scheme.
    As a result, the data set contains twenty-four features.

    Arguments:
        filename (string): Path to a CSV file containing the data.
        transform (callable, optional): A function that takes in a list of 24
            bits and a binary label, and returns a transformed version of each.
    """

    NUM_FEATURES = 24

    def __init__(self, filename, transform=None):
        with open(filename) as file:
            self.lines = file.readlines()
        self.transform = transform

    def __getitem__(self, index):
        line = self.lines[index]

        observation = line.split(",")[0]
        # We're substracting by 1 so that values lay in {0, 1, 2} instead of {1, 2, 3}
        observation = [
            int(categorical_code) - 1 for categorical_code in observation
        ]
        observation = torch.tensor(observation)
        observation = [
            F.one_hot(categorical_code, num_classes=3)
            for categorical_code in observation
        ]
        observation = torch.cat(observation)
        observation = observation.tolist()

        label = line.split(",")[1]
        label = int(label)

        if self.transform:
            observation, label = self.transform(observation, label)

        return observation, label

    def __len__(self):
        return len(self.lines)


# -----------------------------------------------------------------------------
# ---- Utility datasets -------------------------------------------------------
# -----------------------------------------------------------------------------

class ListDataset(Dataset):
    """Dataset wrapping two lists."""

    def __init__(self, x, y):
        assert len(x) == len(y), "input lists must be same length"

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class PredictionDataset(Dataset):
    """Dataset wrapping another dataset.

    Use this wrapper to ensure that non-tensor datasets function properly
    when used with torch model predictions.
    """

    def __init__(self, dataset):
        self.x = torch.tensor([x for x, y in dataset])
        self.y = torch.tensor([y for x, y in dataset])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
