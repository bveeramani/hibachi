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
import unittest

import torch

from hibachi.interpretation import explainers


class StubModel(torch.nn.Module):

    def __init__(self, function):
        super(StubModel, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class SaliencyTest(unittest.TestCase):

    def test_explain(self):
        explain = explainers.Saliency()
        model = StubModel(lambda x: x**2)
        input = torch.tensor([2.])

        actual = explain(model, input)
        expected = torch.tensor([4.])

        self.assertTrue(actual.equal(expected))
        self.assertTrue(actual.dtype == torch.float)

    def test_repr(self):
        explain = explainers.Saliency()
        self.assertEqual(repr(explain), "Saliency()")

        explain = explainers.Saliency(target=0)
        self.assertEqual(repr(explain), "Saliency(target=0)")


class GradientStarInputTest(unittest.TestCase):

    def test_explain(self):
        explain = explainers.GradientStarInput()
        model = StubModel(lambda x: x**2)
        input = torch.tensor([2.])

        actual = explain(model, input)
        expected = torch.tensor([8.])

        self.assertTrue(actual.equal(expected))
        self.assertTrue(actual.dtype == torch.float)

    def test_repr(self):
        explain = explainers.GradientStarInput()
        self.assertEqual(repr(explain), "Saliency()")

        explain = explainers.GradientStarInput(target=0)
        self.assertEqual(repr(explain), "Saliency(target=0)")
