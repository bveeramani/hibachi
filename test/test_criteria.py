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
"""Tests for hibachi.criteria"""
import unittest

import torch
from torch.utils.data import Dataset

from hibachi import criteria


class CriteriaTest(unittest.TestCase):

    def test_correlation(self):
        dataset = StubDataset()

        actual = criteria.correlation(dataset)
        expected = torch.tensor([1, 0.78072], dtype=torch.float)

        self.assertTrue(isinstance(actual, torch.FloatTensor))
        self.assertTrue(torch.norm(actual - expected) < 0.00001)


class StubDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index):
        if index == 0:
            return torch.tensor([0, 0], dtype=torch.float), torch.tensor(
                0, dtype=torch.float)
        elif index == 1:
            return torch.tensor([1, 3], dtype=torch.float), torch.tensor(
                1, dtype=torch.float)
        elif index == 2:
            return torch.tensor([2, 1], dtype=torch.float), torch.tensor(
                2, dtype=torch.float)
        elif index == 3:
            return torch.tensor([3, 6], dtype=torch.float), torch.tensor(
                3, dtype=torch.float)
        else:
            raise IndexError("dataset index out of range")

    def __len__(self):
        return 3
