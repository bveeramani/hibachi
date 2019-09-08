# Copyright 2019 Balaji Veeramani. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for hibachi.filters"""
import unittest

import torch
from torch.utils.data import Dataset

from hibachi import filters, datasets


class FiltersTest(unittest.TestCase):

    def test_VarianceThreshold(self):
        class StubDataset(Dataset):

            def __init__(self):
                x0 = torch.tensor([0, 0], dtype=torch.float)
                x1 = torch.tensor([0, 1], dtype=torch.float)
                self.X = torch.stack([x0, x1])

            def __getitem__(self, index):
                return self.X[index], None

            def __len__(self):
                return len(self.X)
        criterian = criteria.Variance
        filter = filters.VarianceThreshold()
        dataset = StubDataset()

    def test_correlation(self):
        dataset = StubDataset(n=50, d=2)

        actual = rankers.correlation(dataset)
        expected = torch.tensor([1, 2])

        self.assertTrue(isinstance(actual, torch.LongTensor))
        self.assertTrue(torch.equal(actual, expected))

    def test_ccm(self):
        dataset = StubDataset(n=50, d=2)

        actual = rankers.ccm(dataset, num_iterations=10)
        expected = torch.tensor([1, 2])

        self.assertTrue(torch.equal(actual, expected))

    def test_ccm_correctly_ranks_anr(self):
        dataset = datasets.ANR(100)

        ranks = rankers.ccm(dataset, m=4, epsilon=0.1, num_iterations=1000)

        actual = torch.median(ranks[0:4])
        expected = int((4 + 1) / 2)

        self.assertEqual(actual, expected)

    def test_ccm_correctly_ranks_orange_skin(self):
        dataset = datasets.OrangeSkin(100)

        ranks = rankers.ccm(dataset, m=4, epsilon=0.001, num_iterations=100)

        actual = torch.median(ranks[0:4])
        expected = int((4 + 1) / 2)

        self.assertEqual(actual, expected)


class StubDataset(Dataset):

    def __init__(self, n, d):
        self.samples = []

        for _ in range(n):
            x = torch.rand(d, dtype=torch.float)
            y = x[0]
            self.samples.append((x, y))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
