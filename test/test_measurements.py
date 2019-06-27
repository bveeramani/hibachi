"""Tests for evaluations.py"""
import unittest

import torch
from torch.utils.data import Dataset, TensorDataset

from fspy import measurements


class SimilarityMeasureTest(unittest.TestCase):

    def test_pearson_coefficient(self):
        w = [1, 0, 2]
        w_prime = [3, 1, 2]
        self.assertAlmostEqual(measurements.pearson_coefficient(w, w_prime),
                               1 / 2)

    def test_spearman_coefficient(self):
        r = [1, 2, 3]
        r_prime = [3, 1, 2]
        self.assertAlmostEqual(measurements.spearman_coefficient(r, r_prime),
                               -1 / 2)

    def test_tanimoto_distance(self):
        s = set([1, 2])
        s_prime = set([3, 1, 2])
        self.assertAlmostEqual(measurements.tanimoto_distance(s, s_prime),
                               2 / 3)
