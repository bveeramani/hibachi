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


class SupportVectorMachine(torch.nn.Module):
    """Implements a basic linear support vector machine."""

    def __init__(self, input_size):
        raise NotImplementedError()

    def forward(self, x):  # pylint: disable=arguments-differ
        raise NotImplementedError()


class MultiLayerPreceptron(torch.nn.Module):
    """Implements a simple multi-layer preceptron with one hidden layer.

    The hidden layer contains four neurons.
    """

    def __init__(self, input_size):
        super(MultiLayerPreceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 4)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        return torch.sigmoid(output)


class BottleneckMultiLayerPreceptron(torch.nn.Module):
    """Implements a multi-layer preceptron with a bottleneck shape.

    The architecture of the model is: input_size => 32 => 64 => 32 => 1.
    """

    def __init__(self, input_size):
        super(BottleneckMultiLayerPreceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, 32)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        return torch.sigmoid(output)
