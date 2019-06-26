"""Implements classes for preprocessing and loading datasets."""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class OteyP450(Dataset):
    """Otey P450 protein dataset.

    The Otey P450 dataset is encoded in a two-column CSV file. The first column
    contains an eight-digit number where each digit is either a 1, 2, or 3.
    The second column containers a binary label.

    Features and labels are converted to torch.FloatTensor after transformations
    have beem applied (if any).

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

        obvservation = line.split(",")[0]
        # We're substracting by 1 so that values lay in {0, 1, 2} instead of {1, 2, 3}
        obvservation = [
            int(categorical_code) - 1 for categorical_code in obvservation
        ]
        obvservation = torch.tensor(obvservation)
        obvservation = [
            F.one_hot(categorical_code, num_classes=3)
            for categorical_code in obvservation
        ]
        obvservation = torch.cat(obvservation)

        label = line.split(",")[1]
        label = torch.tensor(int(label))

        if self.transform:
            obvservation, label = self.transform(obvservation, label)

        obvservation = obvservation.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        return obvservation, label

    def __len__(self):
        return len(self.lines)
