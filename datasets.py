"""Implements classes for preprocessing and loading datasets."""
import torch
from torch.utils.data import Dataset


class OteyP450(Dataset):
    """Abstraction for the Otey P450 Dataset."""

    def __init__(self, filename):
        with open(filename) as file:
            self.lines = file.readlines()

    def __getitem__(self, index):
        line = self.lines[index]

        features = line.split(",")[0]
        features = [int(feature) for feature in features]
        features = torch.tensor(features)
        features = features.type(torch.FloatTensor)

        label = line.split(",")[1]
        label = torch.tensor(int(label))
        label = label.type(torch.FloatTensor)

        return features, label

    def __len__(self):
        return len(self.lines)
