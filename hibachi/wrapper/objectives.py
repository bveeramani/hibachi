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
import torch
from torch.utils.data import DataLoader, TensorDataset

from hibachi.algorithms import Objective
from hibachi.wrapper.models import RegressionTrainingAlgorithm, ClassificationTrainingAlgorithm

__all__ = ["NegativeMSE", "ClassificationAccuracy"]


class NegativeMSE(Objective):

    def __init__(self, model_class, X, y, train=RegressionTrainingAlgorithm()):
        self.model_class = model_class
        self.X, self.y = X, y
        self.train = train

    def __call__(self, features):
        indices = [i for i in range(len(features)) if features[i] == 1]
        dataset = TensorDataset(self.X[:, indices], self.y)

        num_features = torch.sum(features)
        model = self.model_class(num_features)
        self.train(model, dataset)

        loss = torch.nn.MSELoss(reduction="sum")

        total = 0
        dataloader = DataLoader(dataset, batch_size=32)
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = torch.squeeze(outputs, dim=1)
            total += loss(predictions, labels)
        return -total / len(self.X)


class ClassificationAccuracy(Objective):

    def __init__(self, model_class, X, y, train=ClassificationTrainingAlgorithm()):
        self.model_class = model_class
        self.X, self.y = X, y
        self.train = train

    def __call__(self, features):
        indices = [i for i in range(len(features)) if features[i] == 1]
        dataset = TensorDataset(self.X[:, indices], self.y)

        num_features = torch.sum(features)
        model = self.model_class(num_features)
        self.train(model, dataset)

        num_correct = 0
        dataloader = DataLoader(dataset, batch_size=32)
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)
            num_correct += torch.sum(predicted_labels == labels)
        return num_correct / len(self.X)
