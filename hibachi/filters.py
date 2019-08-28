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
"""Filter methods for feature selection.

Let d represent the total number of features. Then, a ranking r is a list
(r_0, r_1, ..., r_d-1) where 1 <= r_i <= d for all i, r_i is the rank of
feature i in ranking r, and each element of {1,..., d} appears exactly once
in r.
"""
from timeit import default_timer as timer

import numpy as np
import torch

from hibachi import criteria, utils, datasets


def correlation(dataset):
    """Calculates the Pearson correlation coefficient for each covariate.

    Arguments:
        dataset: An iterable of (x, y) pairs of tensors.

    Returns:
        A one-dimensional tensor containing a Pearson correlation coefficient
        for each covariate.
    """
    x = torch.stack([x for x, y in dataset])
    y = torch.tensor([y for x, y in dataset])

    x_hat = torch.mean(x, dim=0)
    y_hat = torch.mean(y)

    s_xy = torch.sum((x - x_hat) * (y - y_hat).view(-1, 1), dim=0)
    s2_x = torch.sum(torch.pow(x - x_hat, 2), dim=0)
    s2_y = torch.sum(torch.pow(y - y_hat, 2)).view(-1, 1)

    return torch.pow(torch.squeeze(s_xy / torch.sqrt(s2_x * s2_y)), 2)


def ccm(dataset, m=None, epsilon=0.001, num_iterations=100):
    """Computes a ranking using conditional covariance minimization.

    See https://github.com/Jianbo-Lab/CCM.

    Arguments:
        dataset: An iterable of x, y tensor pairs.

    Returns:
        A ranking over the dataset.
    """
    X = torch.stack([observation for observation, label in dataset])
    y = torch.tensor([label for observation, label in dataset])
    n, d = X.shape

    m = m or np.ceil(d / 5)

    # Whitening transform for X
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    X[torch.isnan(X) == 1] = 0

    sigma = torch.median((X[:, None] - X[None, :]).norm(2, dim=2))

    assert sigma > 0

    def kernel(x, x_tilda):
        return (1 / (2 * sigma**2)) * torch.exp(-torch.norm(x - x_tilda)**2)

    y = y.unsqueeze(1)
    y = y - y.mean(0)

    learning_rate = 0.001

    w = (m / d) * torch.ones(d)

    for iteration in trange(num_iterations, leave=False):
        w.requires_grad_(True)

        X_w = X * w
        K_X_w = torch.exp(-torch.sum(
            (X_w.unsqueeze(1) - X_w.unsqueeze(0))**2, dim=2) / (2 * sigma**2))
        G_X_w = utils.center(K_X_w)
        G_X_w_inv = torch.inverse(G_X_w + n * epsilon * torch.eye(n))

        loss = torch.trace(y.transpose(0, 1) @ G_X_w_inv @ y)
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            w = w.clamp(0, 1)
            if torch.sum(w) > m:
                w = utils.project(w, m)

    return w
