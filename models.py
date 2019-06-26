"""Implements various binary classification models."""
import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """Implements a simple logistic regression model"""

    def __init__(self, in_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        return torch.sigmoid(self.linear(x))
