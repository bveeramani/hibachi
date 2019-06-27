"""This module contains PyTorch models."""
import torch


class LogisticRegressionModel(torch.nn.Module):
    """A simple logistic regression model"""

    def __init__(self, in_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        return torch.sigmoid(self.linear(x))
