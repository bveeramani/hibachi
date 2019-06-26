"""Tests for evaluations.py"""
import unittest

import torch
from torch.utils.data import Dataset

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

    def test_remove_unselected_features(self):
        dataset = DummyDataset(2)
        dataset = remove_unselected_features(dataset, [1])
        expected = [(torch.tensor([0]), torch.tensor(0)),
                    (torch.tensor([0]), torch.tensor(1)),
                    (torch.tensor([1]), torch.tensor(1)),
                    (torch.tensor([1]), torch.tensor(2))]
        self.assertEqual(dataset, expected)

    def test_create_folds(self):
        dataset = DummyDataset(3)
        folds = create_folds(dataset, 4)
        self.assertEqual(len(folds), 4)
        for i in range(3):
            for j in range(3):
                self.assertEqual(len(folds[i][0]), 7)
                self.assertEqual(len(folds[i][1]), 2)
        self.assertEqual(len(folds[3][0]), 6)
        self.assertEqual(len(folds[3][1]), 3)


class DummyDataset(Dataset):

    def __init__(self, size):
        self.domain = [(x, y) for x in range(size) for y in range(size)]
        f = lambda x, y: x + y
        self.image = [f(x, y) for x, y in self.domain]

    def __getitem__(self, index):
        observation = torch.tensor(self.domain[index])
        response = torch.tensor(self.image[index])
        return observation, response

    def __len__(self):
        return len(self.domain)
