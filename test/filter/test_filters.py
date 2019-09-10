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
"""Tests for hibachi.filter.filters"""
# pylint: disable=missing-docstring
import unittest

import torch

from hibachi import algorithms
from hibachi.filter import filters, criteria


class StubCriterion(criteria.Criterion):

    def __call__(self, X, y):
        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))
        return torch.arange(0, X.shape[1])

    def __repr__(self):
        return "StubCriterion()"


class StubFilter(algorithms.Selector):

    def __init__(self, indices):
        self.indices = indices

    def __select__(self, X, y):
        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))
        return X[:, self.indices]

    def __repr__(self):
        return "StubFilter(indices={0})".format(self.indices)


class FilterTest(unittest.TestCase):

    def test_call(self):
        select = filters.Filter(lambda score: score > 0,
                                criterion=StubCriterion())
        X = torch.arange(1, 5).reshape(2, 2)  # pylint: disable=invalid-name

        actual = select(X, None)
        expected = torch.tensor([[2], [4]])

        self.assertTrue(actual.equal(expected))

    def test_repr(self):

        class StubFunction:

            def __call__(self, score):
                return True

            def __repr__(self):
                return "StubFunction()"

        selector = filters.Filter(function=StubFunction(),
                                  criterion=StubCriterion())

        actual = repr(selector)
        expected = "Filter(function=StubFunction(), criterion=StubCriterion())"

        self.assertEqual(actual, expected)


class SelectTest(unittest.TestCase):

    def test_call(self):
        select = filters.Select(k=1, criterion=StubCriterion())
        X = torch.arange(1, 5).reshape(2, 2)  # pylint: disable=invalid-name

        actual = select(X, None)
        expected = torch.tensor([[2], [4]])

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.Select(k=1, criterion=StubCriterion())

        actual = repr(select)
        expected = "Select(k=1, criterion=StubCriterion())"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        select = filters.Select(k=1, criterion=StubCriterion(), minimize=True)

        actual = repr(select)
        expected = "Select(k=1, criterion=StubCriterion(), minimize=True)"

        self.assertEqual(actual, expected)


class ComposeTest(unittest.TestCase):

    def test_call(self):
        select = filters.Compose(
            [StubFilter(indices=[1, 2]),
             StubFilter(indices=[0])])
        X = torch.arange(1, 10).reshape(3, 3)  # pylint: disable=invalid-name

        actual = select(X, None)
        expected = torch.tensor([[2], [5], [8]])

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.Compose(
            [StubFilter(indices=[1, 2]),
             StubFilter(indices=[0])])

        actual = repr(select)
        expected = "Compose([\n    StubFilter(indices=[1, 2]),\n    StubFilter(indices=[0])\n])"

        self.assertEqual(actual, expected)


class ThresholdTest(unittest.TestCase):

    def test_call(self):
        select = filters.Threshold(cutoff=1, criterion=StubCriterion())
        X = torch.arange(1, 10).reshape(3, 3)  # pylint: disable=invalid-name

        actual = select(X, None)
        expected = X[:, [1, 2]]

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.Threshold(cutoff=1, criterion=StubCriterion())

        actual = repr(select)
        expected = "Threshold(cutoff=1, criterion=StubCriterion())"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        select = filters.Threshold(cutoff=1,
                                   criterion=StubCriterion(),
                                   minimize=True)

        actual = repr(select)
        expected = "Threshold(cutoff=1, criterion=StubCriterion(), minimize=True)"

        self.assertEqual(actual, expected)


class PercentMissingThresholdTest(unittest.TestCase):

    def test_call(self):
        select = filters.PercentMissingThreshold(maximum=0.5)
        X = torch.tensor([
            [1, 1, float('nan')],  # pylint: disable=invalid-name
            [1, float('nan'), float('nan')]
        ])

        actual = select(X, None)
        expected = X[:, [0, 1]]

        # torch.equal doesn't work properly with nan values
        self.assertTrue(torch.isnan(actual).equal(torch.isnan(expected)))
        actual[torch.isnan(actual)] = 0
        expected[torch.isnan(expected)] = 0
        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.PercentMissingThreshold(maximum=0.5)

        actual = repr(select)
        expected = "PercentMissingThreshold(maximum=0.5)"

        self.assertEqual(actual, expected)


class VarianceThresholdTest(unittest.TestCase):

    def test_call(self):
        select = filters.VarianceThreshold(minimum=1)
        X = torch.tensor([[-1, 1], [1, 1]])  # pylint: disable=invalid-name

        actual = select(X, None)
        expected = X[:, [0]]

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.VarianceThreshold(minimum=1)

        actual = repr(select)
        expected = "VarianceThreshold(minimum=1)"

        self.assertEqual(actual, expected)


class CorrelationThresholdTest(unittest.TestCase):

    def test_call(self):
        select = filters.CorrelationThreshold(minimum=1)
        X = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0]])  # pylint: disable=invalid-name
        y = torch.tensor([1, 2, 3])  # pylint: disable=invalid-name

        actual = select(X, y)
        expected = X[:, [0]]

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.CorrelationThreshold(minimum=1)

        actual = repr(select)
        expected = "CorrelationThreshold(minimum=1)"

        self.assertEqual(actual, expected)


class CCMTest(unittest.TestCase):

    def test_call(self):
        select = filters.CCM(k=1)
        X = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0]])  # pylint: disable=invalid-name
        y = torch.tensor([1, 2, 3])  # pylint: disable=invalid-name

        actual = select(X, y)
        expected = X[:, [0]]

        self.assertTrue(actual.equal(expected))

    def test_repr(self):
        select = filters.CCM(k=1)

        actual = repr(select)
        expected = "CCM(k=1)"

        self.assertEqual(actual, expected)

    def test_repr2(self):
        select = filters.CCM(k=1, epsilon=0.1, iterations=1000)

        actual = repr(select)
        expected = "CCM(k=1, epsilon=0.1, iterations=1000)"

        self.assertEqual(actual, expected)
