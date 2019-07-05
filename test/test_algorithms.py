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


class RegularizerTest(unittest.TestCase):

    def test_lasso_penalty(self):
        model = StubModel([1, 1])
        self.assertEqual(algorithms.lasso_penalty(model), 2)

        model = StubModel([1, -1])
        self.assertEqual(algorithms.lasso_penalty(model), 2)

        model = StubModel([1, 0])
        self.assertEqual(algorithms.lasso_penalty(model, 2), 2)


class DummyDataset(Dataset):

    def __init__(self):
        self.x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [0, 1, 0, 1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return 4


class StubModel(torch.nn.Module):

    def __init__(self, weights):
        super(StubModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))

    def forward(self, x):  # pylint: disable=arguments-differ
        return x
