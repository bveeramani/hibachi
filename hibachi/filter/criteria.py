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
"""Criteria for scoring features."""
import random

import torch

__all__ = [
    "Criterion", "PercentMissingValues", "Variance", "Correlation", "CCM",
    "Collinearity", "MutualInformation", "ChiSquare", "ANOVA"
]


# pylint: disable=too-few-public-methods
class Criterion:
    """Base class for all feature scoring methods."""

    # pylint: disable=invalid-name
    def __call__(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class PercentMissingValues(Criterion):
    """Calculates the proportion of missing values in each feature.

    Example:
        >>> criterion = criteria.PercentMissingValues()
        >>> X = torch.tensor([[1, 1], [float("nan"), 1]])
        >>> criterion(X, None)
        tensor([0.5000, 0.0000])
    """

    def __call__(self, X, y):
        """
        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector. Can be None.

        Returns:
            A one-dimensional vector containing the percent of missing features
            in each feature.

        Raises:
            ValueError: if X is not two-dimensional.
        """
        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))

        counts = torch.sum(torch.isnan(X), dim=0)
        return counts.type(torch.FloatTensor) / len(X)


class Variance(Criterion):
    """Calculates the variance of each feature.

    Arguments:
        unbiased (bool, optional): If true, calculate variance using Bessel's
            correction.

    Example:
        >>> criterion = criteria.PercentMissingValues()
        >>> X = torch.arange(1, 5).reshape(2, 2)
        >>> criterion(X, None)
        tensor([2., 2.])
    """

    def __init__(self, unbiased=True):
        self.unbiased = unbiased

    def __call__(self, X, y):
        """
        If X is not a float or a double, it will be cast as torch.FloatTensor.

        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector. Can be None.

        Returns:
            A one-dimensional vector containing the variance of each feature.

        Raises:
            ValueError: if X is not two-dimensional.
        """
        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))

        if not X.dtype == torch.float and not X.dtype == torch.double:
            X = X.type(torch.FloatTensor)

        return torch.var(X, dim=0, unbiased=self.unbiased)

    def __repr__(self):
        return "Variance({0})".format("" if self.unbiased else "unbiased=False")


class Correlation(Criterion):
    """Calculates the Pearson correlation coefficient for each covariate.

    Implementation is based on equation (1) in:
        G. Chandrashekar, F. Sahin. A survey on feature selection methods.

    Arguments:
        square (bool, optional): If true, correlation scores will be squared.

    Example:
        >>> criterion = criteria.Correlation()
        >>> X = torch.tensor([[0, 0], [1, 3], [2, 1], [3, 6]])
        >>> y = torch.tensor([0, 1, 2, 3])
        >>> criterion(X, y)
        tensor([1.0000, 0.6095])
    """

    def __init__(self, square=False):
        self.square = square

    def __call__(self, X, y):
        """
        If X is not a float or a double, X will be cast as torch.FloatTensor.
        If y is not a float or a double, y will be cast as torch.FloatTensor.

        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector.

        Returns:
            A one-dimensional vector containing the correlation of each feature
            to the response.

        Raises:
            ValueError: if X is not two-dimensional.
            ValueError: if y is not one-dimensional.
            ValueError: if len(X) is not equal to len(y).
        """
        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))
        if not len(y.shape) == 1:
            raise ValueError("Expected y to have one dimensions but found %d." %
                             len(y.shape))
        if not len(X) == len(y):
            raise ValueError("X and y have incompatible shapes: %s != %s." %
                             (len(X), len(y)))

        if not X.dtype == torch.float and not X.dtype == torch.double:
            X = X.type(torch.FloatTensor)
        if not y.dtype == torch.float and not y.dtype == torch.double:
            y = y.type(torch.FloatTensor)

        x_hat = torch.mean(X, dim=0)
        y_hat = torch.mean(y)

        s_xy = torch.sum((X - x_hat) * (y - y_hat).view(-1, 1), dim=0)
        s2_x = torch.sum(torch.pow(X - x_hat, 2), dim=0)
        s2_y = torch.sum(torch.pow(y - y_hat, 2)).view(-1, 1)

        scores = torch.squeeze(s_xy / torch.sqrt(s2_x * s2_y))
        return scores**2 if self.square else scores

    def __repr__(self):
        return "Correlation({0})".format("square=True" if self.square else "")


class CCM(Criterion):
    """Implements the Conditional Covariance Minimization algorithm.

    It has been proposed in Kernel Feature Selection via Conditional Covariance Minimization.
        https://arxiv.org/abs/1707.01164

    Arguments:
        m (int): The number of features to select.
        epsilon (float): Use 0.001 for classification and 0.1 for regression.
        iterations (int): The number of iterations to execute.
        lr (float): The learning rate.

    Raises:
        ValueError: if m is not a positive integer.
        ValueError: if epsilon is not a positive number.
        ValueError: if iterations is not a positive integer.
        ValueError: if lr is not a positive number.

    Example:
        >>> dataset = datasets.OrangeSkin(n=100)
        >>> X = torch.stack([x for x, y in dataset])
        >>> y = torch.stack([y for x, y in dataset])
        >>> criterion = criteria.CCM(m=4)
        >>> criterion(X, y)
        tensor([0.9658, 0.9658, 0.9658, 0.9658, 0.0000, 0.0000, 0.0000, 0.0000, 0.1366,
        0.0000])
    """

    def __init__(self, m, epsilon=0.001, iterations=100, lr=0.001):
        if m <= 0:
            raise ValueError("m must be a positive integer.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a positive number.")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer.")
        if lr <= 0:
            raise ValueError("lr must be a positive number.")

        self.m = m  # pylint: disable=invalid-name
        self.epsilon = epsilon
        self.iterations = iterations
        self.lr = lr  # pylint: disable=invalid-name

    # pylint: disable=invalid-name
    def __call__(self, X, y):
        """
        If X is not a float or a double, X will be cast as torch.FloatTensor.
        If y is not a float or a double, y will be cast as torch.FloatTensor.

        Arguments:
            X (Tensor): A two-dimensional design matrix.
            y (Tensor): A one-dimensional label vector.

        Returns:
            A one-dimensional vector containing the correlation of each feature
            to the response.

        Raises:
            ValueError: if X is not two-dimensional.
            ValueError: if y is not one-dimensional.
            ValueError: if len(X) is not equal to len(y).
        """

        def center(X):
            """Returns the centered version of the given square matrix.

            The centered square matrix is defined by the following formula:

                X - (1/n) 1 1^T X - (1/n) X 1 1^T + (1/n^2) 1 1^T X 1 1^T.

            This function uses code sourced from https://github.com/Jianbo-Lab/CCM.

            Arguments:
                X: An square tensor.

            Returns:
                The row- and column-centered version of X.
            """
            n = len(X)
            O = torch.ones(n, n)
            return X - (1 / n) * O @ X - (1 / n) * X @ O + (
                1 / pow(n, 2)) * O @ X @ O

        if not len(X.shape) == 2:
            raise ValueError("Expected X to have two dimensions but found %d." %
                             len(X.shape))
        if not len(y.shape) == 1:
            raise ValueError("Expected y to have one dimensions but found %d." %
                             len(y.shape))
        if not len(X) == len(y):
            raise ValueError("X and y have incompatible shapes: %s != %s." %
                             (len(X), len(y)))

        if not X.dtype == torch.float and not X.dtype == torch.double:
            X = X.type(torch.FloatTensor)
        if not y.dtype == torch.float and not y.dtype == torch.double:
            y = y.type(torch.FloatTensor)

        n, d = X.shape

        # Whitening transform for X
        X = (X - X.mean(dim=0)) / X.std(dim=0)
        X[torch.isnan(X) == 1] = 0

        sigma = torch.median((X[:, None] - X[None, :]).norm(2, dim=2))

        assert sigma > 0

        y = y.unsqueeze(1)
        y = y - y.mean(0)

        w = (self.m / d) * torch.ones(d)

        for _ in range(self.iterations):
            w.requires_grad_(True)

            X_w = X * w
            K_X_w = torch.exp(-torch.sum(
                (X_w.unsqueeze(1) - X_w.unsqueeze(0))**2, dim=2) /
                              (2 * sigma**2))
            G_X_w = center(K_X_w)
            G_X_w_inv = torch.inverse(G_X_w + n * self.epsilon * torch.eye(n))

            loss = torch.trace(y.transpose(0, 1) @ G_X_w_inv @ y)
            loss.backward()

            with torch.no_grad():
                w -= self.lr * w.grad
                w = w.clamp(0, 1)
                if torch.sum(w) > self.m:
                    w = project(w, self.m)

        return w

    def __repr__(self):
        string = "CCM(m={0}".format(self.m)
        if self.epsilon != 0.001:
            string += ", epsilon={0}".format(self.epsilon)
        if self.iterations != 100:
            string += ", iterations={0}".format(self.iterations)
        if self.lr != 0.001:
            string += ", lr={0}".format(self.lr)
        return string + ")"


# pylint: disable=missing-docstring
class Collinearity(Criterion):

    def __call__(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return "Collinearity()"


class MutualInformation(Criterion):

    def __call__(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return "MutualInformation()"


class ChiSquare(Criterion):

    def __call__(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return "ChiSquare()"


class ANOVA(Criterion):

    def __call__(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return "ANOVA()"


# pylint: disable=invalid-name
def project(v, z):
    """Returns the projection of the given vector onto the positive simplex.

    The positive simplex is the set defined by:

        {w : sum_i w_i = z, w_i >= 0}.

    Implements the formula specified in Figure 2 of Duchi et al. (2008).
    See http://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf.

    This function uses code sourced from https://github.com/Jianbo-Lab/CCM.

    Arguments:
        v: A one-dimensional tensor.
        z: The desired sum of the components. Must be strictly positive.

    Returns:
        The Euclidean projection of v onto the positive simplex of size z.
    """
    if len(v.shape) != 1:
        raise ValueError("v must be a one-dimensional tensor")
    if z <= 0:
        raise ValueError("z must be a strictly positive scalar")

    v = v.type(torch.FloatTensor)
    n = len(v)

    U = [i for i in range(n)]
    s = 0
    p = 0
    while U:
        k = random.choice(tuple(U))
        G = {j for j in U if v[j - 1] >= v[k - 1]}
        L = {j for j in U if v[j - 1] < v[k - 1]}
        delta_p = len(G)
        delta_s = sum([v[j - 1] for j in G])
        if (s + delta_s) - (p + delta_p) * v[k - 1] < z:
            s = s + delta_s
            p = p + delta_p
            U = L
        else:
            U = G - {k}

    theta = (s - z) / p
    w = torch.tensor([max(v[i - 1] - theta, 0) for i in range(1, n + 1)])
    return w
