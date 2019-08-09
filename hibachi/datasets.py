"""Classes for preprocessing and loading datasets."""
import random

import torch
from torch.utils.data import Dataset


class OrangeSkin(Dataset):
    """Synthetic binary classification dataset.

    Given Y = -1, (X_1, ..., X_10) ~ N(0, I_10). Given Y = 1, (X_5, ..., X_10) ~
    N(0, I_5) and (X_1, ... X_4) ~ N(0, I_4) conditioned on 9 <= (X_1)^2 + ... +
    (X_4)^2 <= 16.

    Arguments:
        size: The number of samples to generate.
    """

    def __init__(self, size):
        torch.manual_seed(0)

        self.data = []
        for _ in range(size // 2):
            observation = torch.randn(10)
            label = torch.tensor(-1, dtype=torch.float)

            sample = observation, label
            self.data.append(sample)

        for _ in range(size - size // 2):
            observation = torch.randn(4)

            while not 9 <= torch.sum(torch.pow(observation, 2)) <= 16:
                observation = torch.randn(4)

            observation = torch.cat((observation, torch.randn(6)))
            label = torch.tensor(1, dtype=torch.float)

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
            epsilon = torch.squeeze(
                torch.normal(torch.tensor(0.), torch.tensor(float(std))))
            Y = -2 * torch.sin(2 * X[0]) + max(
                X[1], 0) + X[2] + torch.exp(-X[3]) + epsilon

            sample = X, Y
            self.data.append(sample)

        random.shuffle(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)