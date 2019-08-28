# Copyright 2019 Balaji Veeramani. All Rights Reserved.n
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
"""Tests for hibachi.measures"""
import unittest

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from hibachi import measures, datasets


class MeasuresTest(unittest.TestCase):

    def test_NegativeMSE(self):
        # f_hat(x) = 0
        model = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)

        # f(x) = x
        dataset = datasets.FunctionDataset(function=lambda x: x, domain={0, 1})
        measure = measures.NegativeMSE()
        score = measure(model, dataset)

        # -1 / 2 = -((0 - 0)^2  + (1 - 0)^2) / 2
        self.assertEqual(score, -1 / 2)

    def test_ClassificationAccuracy(self):

        class StubModel(torch.nn.Module):

            def forward(self, inputs):
                return torch.tensor([1, 0], dtype=torch.float)

        # f(x) = 0, g(x) = 0
        model = StubModel()

        def transform(x, y):
            return x.type(torch.FloatTensor), y.type(torch.LongTensor)

        # g(x) = x % 2
        dataset = datasets.FunctionDataset(function=lambda x: x % 3,
                                           domain={0, 1, 2, 3},
                                           transform=transform)
        measure = measures.ClassificationAccuracy()
        score = measure(model, dataset)

        # 1 / 2 = (1 + 0 + 1 + 0) / 4
        self.assertEqual(score, 1 / 2)
