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
import numpy as np
import torch

from hibachi import criteria


def correlation(dataset):
    """Computes a ranking using Pearson's correlation coefficient.

    Arguments:
        dataset: An iterable of x, y tensor pairs.

    Returns:
        A ranking over the dataset.
    """
    scores = criteria.correlation(dataset)
    return torch.argsort(-scores) + 1


def ccm(dataset):
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

    epsilon = 0.01
    m = np.ceil(d / 5)

    # Whitening transform for X
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    sigma = torch.median((X[:, None] - X[None, :]).norm(2, dim=2))
    assert sigma > 0

    def kernel(x, x_tilda):
        return (1 / (2 * sigma**2)) * torch.exp(-torch.norm(x - x_tilda)**2)

    y = y.unsqueeze(1)
    y = y - y.mean(0)

    learning_rate = 0.1

    w = (m / d) * torch.ones(d)

    for iteration in range(100):
        w.requires_grad_(True)

        K_X_w = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                K_X_w[i, j] = kernel((X * w)[i], (X * w)[j])

        G_X_w = center(K_X_w)
        G_X_w_inv = torch.inverse(G_X_w + n * epsilon * torch.eye(n))
        loss = torch.trace(y.transpose(0, 1) @ G_X_w_inv @ y)
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            w = w.clamp(0, 1)
            if torch.sum(w) > m:
                w = project(w, m)

    # Compute rank of each feature based on weight.
    # Random permutation to avoid bias due to equal weights.
    # Code sourced from https://github.com/Jianbo-Lab/CCM
    # TODO: Rewrite this using torch
    indices = np.random.permutation(d)
    permutated_weights = w[indices]
    permutated_ranks = (-permutated_weights).argsort().argsort() + 1
    ranks = permutated_ranks[np.argsort(indices)]

    return ranks
