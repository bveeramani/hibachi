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
"""Tests for hibachi.datasets"""
import unittest

import numpy as np
import torch

from hibachi import datasets


class NumpyDatasetTests(unittest.TestCase):

    def test_NumpyDataset_select(self):
        x = np.arange(1, 10).reshape(3, 3)
        y = np.arange(1, 4)
        dataset = datasets.NumpyDataset(x, y)

        selected_dataset = dataset.select({0})

        x0_actual, y0_actual = selected_dataset[0]
        x0_expected = torch.tensor([1], dtype=torch.float)
        y0_expected = torch.tensor(1, dtype=torch.float)

        self.assertTrue(x0_actual.equal(x0_expected))
        self.assertTrue(y0_actual.equal(y0_expected))

    def test_NumpyDataset_dimensionality(self):
        x = np.arange(1, 9).reshape(2, 4)
        y = np.arange(1, 3)
        dataset = datasets.NumpyDataset(x, y)

        self.assertEqual(dataset.dimensionality, 4)

    def test_NumpyDataset_getitem(self):
        x = np.arange(1, 10).reshape(3, 3)
        y = np.arange(1, 4)
        dataset = datasets.NumpyDataset(x, y)

        x0_actual, y0_actual = dataset[0]
        x0_expected = torch.tensor([1, 2, 3], dtype=torch.float)
        y0_expected = torch.tensor(1, dtype=torch.float)

        self.assertTrue(x0_actual.equal(x0_expected))
        self.assertTrue(y0_actual.equal(y0_expected))

    def test_NumpyDataset_len(self):
        x = np.arange(1, 10).reshape(3, 3)
        y = np.arange(1, 4)
        dataset = datasets.NumpyDataset(x, y)

        self.assertEqual(len(dataset), 3)
