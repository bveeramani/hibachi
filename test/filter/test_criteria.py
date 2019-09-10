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
# pylint: disable=missing-docstring
import unittest

import torch

from hibachi.filter import criteria
from hibachi import datasets


class PercentMissingValuesTest(unittest.TestCase):

    def test_call(self):
        criterion = criteria.PercentMissingValues()
        X = torch.ones(2, 1)  # pylint: disable=invalid-name

        actual = criterion(X, None)
        expected = torch.zeros(1, dtype=torch.float)

        self.assertTrue(actual.equal(expected))

    def test_call2(self):
        criterion = criteria.PercentMissingValues()
        X = torch.tensor([[1], [float("nan")]])  # pylint: disable=invalid-name

        actual = criterion(X, None)
        expected = torch.tensor([0.5], dtype=torch.float)

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        criterion = criteria.PercentMissingValues()

        actual = repr(criterion)
        expected = "PercentMissingValues()"

        self.assertEqual(actual, expected)


class VarianceTest(unittest.TestCase):

    def test_call(self):
        criterion = criteria.Variance()
        X = torch.arange(1, 5).reshape(2, 2)  # pylint: disable=invalid-name

        actual = criterion(X, None)
        expected = torch.tensor([2, 2], dtype=torch.float)

        self.assertTrue(actual.equal(expected))

    def test_call2(self):
        criterion = criteria.Variance(unbiased=False)
        X = torch.arange(1, 5).reshape(2, 2)  # pylint: disable=invalid-name

        actual = criterion(X, None)
        expected = torch.tensor([1, 1], dtype=torch.float)

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        criterion = criteria.Variance()

        actual = repr(criterion)
        expected = "Variance()"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        criterion = criteria.Variance(unbiased=False)

        actual = repr(criterion)
        expected = "Variance(unbiased=False)"

        self.assertEqual(actual, expected)


class CorrelationTest(unittest.TestCase):

    def test_call(self):
        criterion = criteria.Correlation()
        X = torch.tensor([[0, 0], [1, 3], [2, 1], [3, 6]])  # pylint: disable=invalid-name
        y = torch.tensor([0, 1, 2, 3])  # pylint: disable=invalid-name

        actual = criterion(X, y)
        expected = torch.tensor([1, 0.78072], dtype=torch.float)

        self.assertTrue(torch.norm(actual - expected) < 0.00001)

    def test_call2(self):
        criterion = criteria.Correlation(square=True)
        X = torch.tensor([[0, 0], [1, 3], [2, 1], [3, 6]])  # pylint: disable=invalid-name
        y = torch.tensor([0, 1, 2, 3])  # pylint: disable=invalid-name

        actual = criterion(X, y)
        expected = torch.tensor([1, 0.6095237184], dtype=torch.float)

        self.assertTrue(torch.norm(actual - expected) < 0.00001)

    def test_repr(self):
        criterion = criteria.Correlation()

        actual = repr(criterion)
        expected = "Correlation()"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        criterion = criteria.Correlation(square=True)

        actual = repr(criterion)
        expected = "Correlation(square=True)"

        self.assertEqual(actual, expected)


class CCMTest(unittest.TestCase):

    def test_call(self):
        dataset = datasets.OrangeSkin(n=100)
        X = torch.stack([x for x, y in dataset])  # pylint: disable=invalid-name
        y = torch.stack([y for x, y in dataset])  # pylint: disable=invalid-name
        criterion = criteria.CCM(m=4)

        actual = criterion(X, y)

        self.assertTrue(all(actual[:4] > 0.8))
        self.assertTrue(all(actual[4:] < 0.2))

    def test_repr(self):
        criterion = criteria.CCM(m=1)

        actual = repr(criterion)
        expected = "CCM(m=1)"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        criterion = criteria.CCM(m=1, epsilon=1, iterations=1, lr=1)

        actual = repr(criterion)
        expected = "CCM(m=1, epsilon=1, iterations=1, lr=1)"

        self.assertEqual(actual, expected)


class CollinearityTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.Collinearity()

        actual = repr(criterion)
        expected = "Collinearity()"

        self.assertEqual(actual, expected)


class MutualInformationTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.MutualInformation()

        actual = repr(criterion)
        expected = "MutualInformation()"

        self.assertEqual(actual, expected)


class ChiSquareTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.ChiSquare()

        actual = repr(criterion)
        expected = "ChiSquare()"

        self.assertEqual(actual, expected)


class ANOVATest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.ANOVA()

        actual = repr(criterion)
        expected = "ANOVA()"

        self.assertEqual(actual, expected)
