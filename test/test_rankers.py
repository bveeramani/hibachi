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

from hibachi import rankers


class RankersTest(unittest.TestCase):

    def test_correlation(self):
        dataset = StubDataset(n=100, d=2)

        actual = rankers.correlation(dataset)
        expected = torch.tensor([1, 2])

        self.assertTrue(isinstance(actual, torch.LongTensor))
        self.assertTrue(torch.equal(actual, expected))

    def test_ccm(self):
        dataset = StubDataset(n=100, d=2)
        self.assertEqual(rankers.ccm(dataset), torch.tensor([1, 2]))


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
