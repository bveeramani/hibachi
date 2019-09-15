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


class PercentMissingValuesTest(unittest.TestCase):

    def test_call(self):
        X = torch.tensor([[1], [float("nan")]])  # pylint: disable=invalid-name

        criterion = criteria.PercentMissingValues()
        actual = criterion(X, None)
        expected = torch.tensor([0.5], dtype=torch.float)
        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        criterion = criteria.PercentMissingValues()
        self.assertEqual(repr(criterion), "PercentMissingValues()")


class VarianceTest(unittest.TestCase):

    def test_call(self):
        X = torch.arange(1, 5).reshape(2, 2)  # pylint: disable=invalid-name

        criterion = criteria.Variance()
        actual = criterion(X, None)
        expected = torch.tensor([2, 2], dtype=torch.float)
        self.assertTrue(actual.equal(expected))

        criterion = criteria.Variance(unbiased=False)
        actual = criterion(X, None)
        expected = torch.tensor([1, 1], dtype=torch.float)
        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        criterion = criteria.Variance()
        self.assertEqual(repr(criterion), "Variance()")

        criterion = criteria.Variance(unbiased=False)
        self.assertEqual(repr(criterion), "Variance(unbiased=False)")


class CorrelationTest(unittest.TestCase):

    def test_call(self):
        X = torch.tensor([[0, 0], [1, 3], [2, 1], [3, 6]])  # pylint: disable=invalid-name
        y = torch.tensor([0, 1, 2, 3])  # pylint: disable=invalid-name

        criterion = criteria.Correlation()
        actual = criterion(X, y)
        expected = torch.tensor([1, 0.78072], dtype=torch.float)
        self.assertTrue(torch.allclose(actual, expected))

        criterion = criteria.Correlation(square=True)
        actual = criterion(X, y)
        expected = torch.tensor([1, 0.6095237184], dtype=torch.float)
        self.assertTrue(torch.allclose(actual, expected))

    def test_repr(self):
        criterion = criteria.Correlation()
        self.assertEqual(repr(criterion), "Correlation()")

        criterion = criteria.Correlation(square=True)
        self.assertEqual(repr(criterion), "Correlation(square=True)")


class CCMTest(unittest.TestCase):

    def test_call(self):
        X = torch.randn(100, 10)  # pylint: disable=invalid-name
        y = torch.sum(X[:, [0, 1, 2, 3]], dim=1)  # pylint: disable=invalid-name

        criterion = criteria.CCM(m=4)
        actual = criterion(X, y)
        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                dtype=torch.float)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-05))

    def test_repr(self):
        criterion = criteria.CCM(m=1)
        self.assertEqual(repr(criterion), "CCM(m=1)")

        criterion = criteria.CCM(m=1, epsilon=1, iterations=1, lr=1)
        self.assertEqual(repr(criterion),
                         "CCM(m=1, epsilon=1, iterations=1, lr=1)")


class CollinearityTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.Collinearity()
        self.assertEqual(repr(criterion), "Collinearity()")


class MutualInformationTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.MutualInformation()
        self.assertEqual(repr(criterion), "MutualInformation()")


class ChiSquareTest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.ChiSquare()
        self.assertEqual(repr(criterion), "ChiSquare()")


class ANOVATest(unittest.TestCase):

    def test_call(self):
        raise NotImplementedError

    def test_repr(self):
        criterion = criteria.ANOVA()
        self.assertEqual(repr(criterion), "ANOVA()")
