"""Classes for preprocessing and loading datasets."""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class OteyP450(Dataset):
    """Otey P450 protein dataset.

    The Otey P450 dataset is encoded in a two-column CSV file. The first column
    contains an eight-digit number where each digit is either a 1, 2, or 3.
    The second column containers a binary label.

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
