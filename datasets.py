"""Implements classes for preprocessing and loading datasets."""
import torch
from torch.utils.data import Dataset


class OteyP450(Dataset):
    """Otey P450 protein dataset.

    The Otey P450 dataset is encoded in a two-column CSV file. The first column
    contains an eight-digit number where each digit is either a 1, 2, or 3.
    The second column containers a label, either 0 or 1.

    Arguments:
        filename (string): Path to a CSV file containing the data.
        transform (callable, optional): A function that takes in a list of eight
            predictors and a scalar label and returns a transformed version of
            each.
    """

    def __init__(self, filename, transform=None):
        with open(filename) as file:
            self.lines = file.readlines()
        self.transform = transform

    def __getitem__(self, index):
        line = self.lines[index]

        features = line.split(",")[0]
        features = [int(feature) for feature in features]
        features = torch.tensor(features)
        features = features.type(torch.FloatTensor)

        label = line.split(",")[1]
        label = torch.tensor(int(label))
        label = label.type(torch.FloatTensor)

        return self.transform(features, label)

    def __len__(self):
        return len(self.lines)
