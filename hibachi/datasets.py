"""Classes for preprocessing and loading datasets."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class HibachiDataset(Dataset):

    def select(self, indices):
        raise NotImplementedError

    @property
    def dimensionality(self):
        raise NotImplementedError


class TensorDataset(HibachiDataset):

    def __init__(self, X, y, transform=None):
        if not X.dim() == 2:
            raise ValueError("Invalid number of X dimensions: {}.".format(
                X.dim()))
        if not y.dim() == 1:
            raise ValueError("Invalid number of y dimensions: {}.".format(
                y.dim()))
        if not len(X) == len(y):
            raise ValueError("Length mismatch: {} != {}.".format(
                len(X), len(y)))
        self.X, self.y = torch.clone(X), torch.clone(y)
        self.transform = transform

    def __getitem__(self, index):
        observation, response = self.X[index], self.y[index]

        if self.transform:
            observation, response = self.transform(observation, response)

        return observation, response

    def __len__(self):
        return len(self.X)

    def select(self, indices):
        X = self.X[:, list(indices)]
        return TensorDataset(X, self.y)

    @property
    def dimensionality(self):
        if self.x is None:
            raise RuntimeError(
                "Dimensionality is undefined for empty datasets.")
        return len(self.x[0])


class NumpyDataset(HibachiDataset):

    def __init__(self, x, y, transform=None):
        if not X.ndim == 2:
            raise ValueError("Invalid number of X dimensions: {}.".format(
                x.ndim))
        if not y.ndim == 1:
            raise ValueError("Invalid number of y dimensions: {}.".format(
                y.ndim))
        if not len(X) == len(y):
            raise ValueError("Length mismatch: {} != {}.".format(
                len(X), len(y)))
        self.X = torch.tensor(np.copy(X), dtype=torch.float)
        self.y = torch.tensor(np.copy(y), dtype=torch.float)
        self.transform = transform

    def __getitem__(self, index):
        observation, response = self.x[index], self.y[index]
        if self.transform:
            observation, response = self.transform(observation, response)
        return observation, response

    def __len__(self):
        return len(self.X)

    def select(self, indices):
        X = self.X[:, list(indices)]
        return NumpyDataset(X, self.y)

    @property
    def dimensionality(self):
        if self.X is None:
            raise RuntimeError(
                "Dimensionality is undefined for empty datasets.")
        return len(self.X[0])


class OrangeSkin(HibachiDataset):
    """Synthetic binary classification dataset.

    Given Y = -1, (X_1, ..., X_10) ~ N(0, I_10). Given Y = 1, (X_5, ..., X_10) ~
    N(0, I_5) and (X_1, ... X_4) ~ N(0, I_4) conditioned on 9 <= (X_1)^2 + ... +
    (X_4)^2 <= 16.

    Arguments:
        size: The number of samples to generate.
    """

    def __init__(self, n, transform=None):
        torch.manual_seed(0)

        observations, labels = [], []
        for _ in range(n // 2):
            observation = torch.randn(10)
            label = torch.tensor(-1, dtype=torch.long)

            observations.append(observation)
            labels.append(label)

        for _ in range(n - n // 2):
            observation = torch.randn(4)

            while not 9 <= torch.sum(torch.pow(observation, 2)) <= 16:
                observation = torch.randn(4)

            observation = torch.cat((observation, torch.randn(6)))
            label = torch.tensor(1, dtype=torch.long)

            observations.append(observation)
            labels.append(label)

        self.X = torch.stack(observations)
        self.y = torch.stack(labels)
        self.transform = transform


    def __getitem__(self, index):
        observation, label = self.X[index], self.y[index]
        if self.transform:
            observation, label = self.transform(observation, label)
        return observation, label

    def __len__(self):
        return len(self.X)

    @property
    def dimensionality(self):
        return 10

    def select(self, indices):
        X = self.X[:, list(indices)]
        return TensorDataset(X, self.y)


class ANR(Dataset):
    """Synthetic additive nonlinear regression dataset.

    Data is generated using the following distribution:

        Y = -2sin(2X_1) + max(X_2, 0) + X_3 + exp(-X_4) + epsilon

    where (X_1, ..., X_10) ~ N(0, I_10) and epsilon ~ N(0, 1).

    Arguments:
        size: The number of samples to generate.
    """

    def __init__(self, n, std=1, transform=None):
        observations, responses = [], []

        for _ in range(n):
            observation = torch.randn(10)
            epsilon = torch.squeeze(
                torch.normal(torch.tensor(0.), torch.tensor(float(std))))
            response = -2 * torch.sin(2 * X[0]) + max(
                X[1], 0) + X[2] + torch.exp(-X[3]) + epsilon

            observations.append(observation)
            responses.append(response)

        self.X = torch.stack(observation)
        self.y = torch.stack(responses)
        self.transform = transform

    def __getitem__(self, index):
        observation, response = self.X[index], self.y[index]
        if self.transform:
            observation, response = self.transform(observation, response)
        return observation, response

    def __len__(self):
        return len(self.X)

    @property
    def dimensionality(self):
        return 10

    def select(self, indices):
        X = self.X[:, list(indices)]
        return TensorDataset(X, self.y)


class FunctionDataset(Dataset):

    def __init__(self, function, domain, transform=None):
        self.x = [torch.tensor(x, dtype=torch.float) for x in domain]
        self.y = [torch.tensor(function(x), dtype=torch.float) for x in domain]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

    def __len__(self):
        return len(self.x)
