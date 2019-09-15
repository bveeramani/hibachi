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
# pylint: disable=missing-docstring
import unittest

import torch

from hibachi.wrapper import objectives


class MeanSquaredErrorTest(unittest.TestCase):

    def test_evaluate(self):
        X = torch.zeros(1, 1)
        y = torch.ones(1)
        objective = objectives.MeanSquaredError(torch.nn.Identity, X, y)

        features = torch.ones(1)
        actual = objective(features)
        expected = torch.ones(1)

        self.assertEqual(actual, expected)


class ClassificationErrorTest(unittest.TestCase):

    def test_evaluate(self):
        X = torch.arange(1, 10).reshape(3, 3)
        y = torch.arange(0, 3)
        objective = objectives.ClassificationError(torch.nn.Identity, X, y)

        features = torch.ones(2)
        actual = objective(features)
        expected = torch.tensor(2 / 3)

        self.assertEqual(actual, expected)
