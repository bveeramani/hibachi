"""Tests for selectors.py"""
import unittest

import torch
from torch.utils.data import Dataset

from fspy import algorithms


class PearsonTest(unittest.TestCase):

    def test_pearson_rank(self):
        dataset = DummyDataset()
        self.assertEqual(algorithms.pearson_rank(dataset), [2, 1])

    def test_pearson_select(self):
        dataset = DummyDataset()
        self.assertEqual(algorithms.pearson_select(dataset, 1), set([1]))


class CCMTest(unittest.TestCase):

    def test_ccm_rank(self):
        dataset = DummyDataset()
        self.assertEqual(algorithms.ccm_rank(dataset), [2, 1])

    def test_ccm_select(self):
        dataset = DummyDataset()
        self.assertEqual(algorithms.ccm_select(dataset, 1), set([1]))


class DummyDataset(Dataset):

    def __init__(self):
        self.x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [0, 1, 0, 1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return 4
