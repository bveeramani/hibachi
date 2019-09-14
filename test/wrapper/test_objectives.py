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
"""Tests for hibachi.wrapper.objectives"""
import unittest

import torch

from hibachi import objectives
from hibachi.algorithms import RegressionTrainingAlgorithm


class StubModel(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()
        # We need this to prevent "ValueError: optimizer got an empty parameter list"
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, inputs):
        return torch.zeros(len(inputs), 1, requires_grad=True)


class NegativeMSETest(unittest.TestCase):

    def test_NegativeMSE(self):
        X = torch.tensor([[0], [1]], dtype=torch.float)
        y = torch.tensor([0, 1], dtype=torch.float)
        objective = objectives.NegativeMSE(StubModel, X, y)
        score = objective(torch.tensor([1]))

        # -1 / 2 = -((0 - 0)^2  + (1 - 0)^2) / 2
        self.assertEqual(score, -1 / 2)
