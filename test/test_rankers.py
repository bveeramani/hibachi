"""Tests for selectors.py"""
import unittest

import torch
from torch.utils.data import Dataset

from rankers import pearson_rank


class RankersTest(unittest.TestCase):

    def test_pearson_select(self):
        dataset = DummyDataset()
        self.assertEqual(pearson_rank(dataset), [2, 1])


class DummyDataset(Dataset):

    def __init__(self):
        self.x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = torch.tensor([0, 1, 0, 1])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return 3
