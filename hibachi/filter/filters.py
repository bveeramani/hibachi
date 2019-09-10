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
"""Filter methods for feature selection."""
# pylint: disable=too-few-public-methods
import torch

from hibachi import algorithms
from hibachi.filter import criteria

__all__ = [
    "Filter", "Select", "Compose", "Threshold", "PercentMissingThreshold",
    "VarianceThreshold", "CorrelationThreshold", "CCM"
]


class Filter(algorithms.Selector):
    """Selects features according to the specified function.

    The function should have the signature: function(score) -> bool.

    Arguments:
        function (callable): A function that takes a score as an argument
            and returns true if the corresponding feature should be selected.
        criterion (Criterion): The feature scoring method.
    """

    def __init__(self, function, criterion):
        self.function = function
        self.criterion = criterion

    def __select__(self, X, y):
        """
        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector.

        Returns:
            A transformed version of X containing only the selected columns.
        """
        scores = self.criterion(X, y)
        selected = [i for i in range(len(scores)) if self.function(scores[i])]
        return X[:, selected]

    def __repr__(self):
        return self.__class__.__name__ + "(function={0}, criterion={1})".format(
            self.function, self.criterion)


class Select(algorithms.Selector):
    """Selects the k best features according to the specified criterion.

    Arguments:
        k (int): The number of features to select.
        criterion (Criterion): The feature scoring method.
        minimize (int, optional): If true, then the columns with the lowest
            scores will be selected.

    Example:
        >>> filters.Select(k=10, criterion=criteria.Correlation(square=True))
    """

    def __init__(self, k, criterion, minimize=False):
        if not k >= 0:
            raise ValueError("k cannot be negative.")

        self.k = k
        self.criterion = criterion
        self.minimize = minimize

    def __select__(self, X, y):
        """
        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector. Can be None.

        Returns:
            A transformed version of X containing only the selected columns.
        """
        scores = self.criterion(X, y)
        features = torch.argsort(scores, descending=(not self.minimize))
        selected = features[:self.k]
        return X[:, selected]

    def __repr__(self):
        string = self.__class__.__name__ + "(k={0}, criterion={1}".format(
            self.k, self.criterion)
        if self.minimize:
            string += ", minimize=True"
        return string + ")"


class Compose(algorithms.Selector):
    """Composes several filters together.

    Arguments:
        filters (list of ``Selector`` objects): list of filters to compose,

    Example:
        >>> filters.Compose([
        >>>     filters.PercentMissingThreshold(maximum=0.6),
        >>>     filters.VarianceThreshold(minimum=1.0),
        >>>     filters.CCM(epsilon=0.1)
        >>> ])
    """

    def __init__(self, filters):
        self.filters = filters

    def __select__(self, X, y):
        for select in self.filters:
            X = select(X, y)
        return X

    def __repr__(self):
        string = self.__class__.__name__ + "(["
        for i in range(len(self.filters)):
            string += "\n"
            string += "    {0}".format(self.filters[i])
            if i < len(self.filters) - 1:
                string += ","
        string += "\n])"
        return string


class Threshold(Filter):
    """Selects features whose score pass the specified cutoff.

    Arguments:
        cutoff (float): A feature is selected if and only if its score is
            greater than or equal to the cutoff.
        criterion (Criterion): The feature scoring method.
        minimize (int, optional): If true, then a feature is selected if and
            only if its score is less than or equal to the cutoff.
    """

    def __init__(self, cutoff, criterion, minimize=False):
        function = lambda score: score <= cutoff if minimize else score >= cutoff
        self.cutoff = cutoff
        self.minimize = minimize
        super().__init__(function, criterion)

    def __repr__(self):
        string = "Threshold(cutoff={0}, criterion={1}".format(
            self.cutoff, self.criterion)
        if self.minimize:
            string += ", minimize=True"
        return string + ")"


class PercentMissingThreshold(Threshold):
    """Selects features that contain less than or equal to the specified
    proportion of missing values.

    Arguments:
        maximum (float): The maximum proprtion of missing values required for
            a feature to be selected.

    Example:
        >>> X = tensor([[1., nan, nan],
        ...             [1., 1., nan]])
        >>> select = filters.PercentMissingThreshold(maximum=0.5)
        >>> select(X, None)
        tensor([[1., nan],
                [1., 1.]])
    """

    def __init__(self, maximum=0.5):
        super().__init__(maximum,
                         criterion=criteria.PercentMissingValues(),
                         minimize=True)

    def __repr__(self):
        return "PercentMissingThreshold(maximum={0})".format(self.cutoff)


class VarianceThreshold(Threshold):
    """Selects features that have a sample variance greater than the specified
    minimum.

    Arguments:
        minimum (float): The minimum variance required for a feature to be
            selected.

    Example:
        >>> select = filters.VarianceThreshold(minimum=1)
        >>> X = torch.tensor([[-1, 1],
        ...                   [ 1, 1]])
        >>> select(X, None)
        tensor([[-1],
                [ 1]])
    """

    def __init__(self, minimum=1):
        super().__init__(minimum, criterion=criteria.Variance())

    def __repr__(self):
        return "VarianceThreshold(minimum={0})".format(self.cutoff)


class CorrelationThreshold(Threshold):
    """Selects features whose correlation with the response variable is greater
    than the specified minimum.

    Arguments:
        minimum (float): The minimum correlation score required for a feature to
            be selected.

    Example:
        >>> select = filters.CorrelationThreshold(minimum=1)
        >>> X = torch.tensor([[1, 0, 0],
        ...                   [2, 0, 0],
        ...                   [3, 0, 0]])
        >>> y = torch.tensor([1, 2, 3])
        >>> select(X, y)
        tensor([[1],
                [2],
                [3]])
    """

    def __init__(self, minimum=0.5):
        super().__init__(minimum, criterion=criteria.Correlation())

    def __repr__(self):
        return "CorrelationThreshold(minimum={0})".format(self.cutoff)


class CCM(Select):
    """Selects features using the Conditional Covariance Minimization algorithm.

    Arguments:
        k (int): The number of features to select.
        epsilon (float): Use 0.001 for classificaiton and 0.1 for regression.
        iterations (int): The number of iterations to execute.

        Example:
            >>> select = filters.CCM(k=1)
            >>> X = torch.tensor([[1, 0, 0],
            ...                   [2, 0, 0],
            ...                   [3, 0, 0]])
            >>> y = torch.tensor([1, 2, 3])
            >>> select(X, y)
            tensor([[1],
                    [2],
                    [3]])
    """

    def __init__(self, k, epsilon=0.001, iterations=100):
        criterion = criteria.CCM(m=k, epsilon=epsilon, iterations=iterations)
        self.epsilon = epsilon
        self.iterations = iterations
        super().__init__(k, criterion=criterion)

    def __repr__(self):
        string = "CCM(k={0}".format(self.k)
        if self.epsilon != 0.001:
            string += ", epsilon={0}".format(self.epsilon)
        if self.iterations != 100:
            string += ", iterations={0}".format(self.iterations)
        return string + ")"
