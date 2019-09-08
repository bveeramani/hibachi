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
from timeit import default_timer as timer

import numpy as np
import torch


class Criterion:

    def __call__(self, X, y):
        raise NotImplementedError


class Missing(Criterion):

    def __call__(self, X, y):
        return torch.sum(torch.isnan(X), dim=0) / len(X)


class Variance(Criterion):

    def __init__(self, unbiased=True):
        self.unbiased = unbiased

    def __call__(self, X, y):
        return torch.var(X, axis=0, unbiased=self.unbiased)


class Correlation(Criterion):
    """Calculates the Pearson correlation coefficient for each covariate.

    Arguments:
        dataset: An iterable of (x, y) pairs of tensors.

    Returns:
        A one-dimensional tensor containing a Pearson correlation coefficient
        for each covariate.
    """

    def __init__(self, square=True):
        self.square = square

    def __call__(self, X, y):
        x_hat = torch.mean(x, dim=0)
        y_hat = torch.mean(y)

        s_xy = torch.sum((x - x_hat) * (y - y_hat).view(-1, 1), dim=0)
        s2_x = torch.sum(torch.pow(x - x_hat, 2), dim=0)
        s2_y = torch.sum(torch.pow(y - y_hat, 2)).view(-1, 1)

        scores = torch.squeeze(s_xy / torch.sqrt(s2_x * s2_y))
        return scores ** 2 if self.square else scores


class Colinearity(Criterion):
    pass


class MutualInformation(Criterion):
    pass


class ChiSquare(Criterion):
    pass


class ANOVA(Criterion):
    pass


class ConditionalCovariance(Criterion):

    def __init__(self, m, epsilon=0.001, iterations=100, lr=0.001):
        self.m = m
        self.epsilon = epsilon
        self.iterations = iterations
        self.lr = lr

    def __call__(self, X, y):
        n, d = X.shape

        # Whitening transform for X
        X = (X - X.mean(dim=0)) / X.std(dim=0)
        X[torch.isnan(X) == 1] = 0

        sigma = torch.median((X[:, None] - X[None, :]).norm(2, dim=2))

        assert sigma > 0

        y = y.unsqueeze(1)
        y = y - y.mean(0)

        w = (self.m / d) * torch.ones(d)

        for iteration in trange(self.iterations, leave=False):
            w.requires_grad_(True)

            X_w = X * w
            K_X_w = torch.exp(-torch.sum(
                (X_w.unsqueeze(1) - X_w.unsqueeze(0))**2, dim=2) / (2 * sigma**2))
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
    return X - (1 / n) * O @ X - (1 / n) * X @ O + (1 / pow(n, 2)) * O @ X @ O


def project(v, z):
    """Returns the projection of the given vector onto the positive simplex.

    The positive simplex is the set defined by:

        {w : \sum_i w_i = z, w_i >= 0}.

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
