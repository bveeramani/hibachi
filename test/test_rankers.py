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
"""Tests for hibachi.rankers"""
import unittest

import torch
from torch.utils.data import Dataset

from hibachi import rankers, datasets


class RankersTest(unittest.TestCase):

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

    def test_ccm_returns_long(self):
        dataset = StubDataset(n=50, d=4)

        ranks = rankers.ccm(dataset, m=2, num_iterations=1)

        self.assertTrue(isinstance(ranks, torch.LongTensor))

    def test_ccm_correctly_ranks_anr(self):
        dataset = datasets.ANR(100)

        ranks = rankers.ccm(dataset, m=4, epsilon=0.1, num_iterations=1000)

        actual = torch.sum(ranks[0:4])
        expected = torch.tensor(10) # 1 + 2 + 3 + 4

        self.assertEqual(actual, expected)

    def test_ccm_correctly_ranks_orange_skin(self):
        dataset = datasets.OrangeSkin(100)

        ranks = rankers.ccm(dataset, m=4, epsilon=0.001, num_iterations=100)

        actual = torch.sum(ranks[0:4])
        expected = torch.tensor(10) # 1 + 2 + 3 + 4

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
