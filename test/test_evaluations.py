"""Tests for evaluations.py"""
import unittest

import torch
from torch.utils.data import Dataset, TensorDataset

from evaluations import spearman_coefficient, tanimoto_distance, remove_unselected_features, create_folds


class SimilarityMeasureTest(unittest.TestCase):

    def test_spearman_coefficient(self):
        r = [1, 2, 3]
        r_prime = [3, 1, 2]
        self.assertAlmostEqual(spearman_coefficient(r, r_prime), -0.5)

    def test_tanimoto_distance(self):
        s = [1, 2]
        s_prime = [3, 1, 2]
        self.assertAlmostEqual(tanimoto_distance(s, s_prime), 2 / 3)


class UtilityFunctionTest(unittest.TestCase):

    #TODO: Rewrite me
    def test_remove_unselected_features(self):
        dataset = DummyDataset()
        dataset = remove_unselected_features(dataset, [1, 0])

        x = torch.tensor([[0], [0], [1], [1]])
        y = torch.tensor([0, 1, 0, 1])
        reference_dataset = TensorDataset(x, y)

        actual = [(a, b) for a, b in dataset]
        expected = [(a, b) for a, b in reference_dataset]

        self.assertEqual(actual, expected)

    def test_create_folds(self):
        dataset = DummyDataset()
        folds = create_folds(dataset, 3)

        self.assertEqual(len(folds), 3)

        # Three folds of size 1 x 1 x 2
        for i in range(2):
            self.assertEqual(len(folds[i][0]), 3)
            self.assertEqual(len(folds[i][1]), 1)

        self.assertEqual(len(folds[2][0]), 2)
        self.assertEqual(len(folds[2][1]), 2)


class DummyDataset(Dataset):

    def __init__(self):
        # f(x, y) = y over a 2 x 2 region
        self.x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = torch.tensor([0, 1, 0, 1])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return 4
