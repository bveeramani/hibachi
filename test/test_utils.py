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
"""Tests for hibachi.utils"""
import unittest

import torch

from hibachi import utils


class UtilsTest(unittest.TestCase):

    def test_project(self):
        v = torch.ones(2, dtype=torch.float)

        actual = utils.project(v, 1)
        expected = torch.tensor([0.5, 0.5], dtype=torch.float)

        self.assertTrue(isinstance(actual, torch.FloatTensor))
        self.assertTrue(torch.equal(actual, expected))

    def test_center(self):
        X = torch.arange(1, 5).reshape(2, 2)

        actual = utils.center(X)
        expected = torch.zeros(2, 2, dtype=torch.float)

        self.assertTrue(isinstance(actual, torch.FloatTensor))
        self.assertTrue(torch.equal(actual, expected))
