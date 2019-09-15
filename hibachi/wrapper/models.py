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
from torch.utils.data import DataLoader


class TrainingAlgorithm:

    def __init__(self,
                 loss_func,
                 batch_size=32,
                 num_epochs=100,
                 learning_rate=0.001,
                 optimizer_class=torch.optim.Adam,
                 transform_outputs=None):
        if not batch_size > 0:
            raise ValueError("Invalid batch size: {}".format(batch_size))
        if not num_epochs > 0:
            raise ValueError("Invalid epochs value: {}".format(num_epochs))

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.transform_outputs = transform_outputs

    def __call__(self, model, dataset):
        if not list(model.parameters()):
            return

        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
        optimizer = self.optimizer_class(model.parameters(),
                                         lr=self.learning_rate)
        model.train()

        for _ in range(self.num_epochs):
            for inputs, labels in dataloader:
                model.zero_grad()

                outputs = model(inputs)
                if self.transform_outputs:
                    outputs = self.transform_outputs(outputs)
                loss = self.loss_func(outputs, labels)

                loss.backward()
                optimizer.step()


class RegressionTrainingAlgorithm(TrainingAlgorithm):

    def __init__(self, batch_size=32, num_epochs=100, learning_rate=0.001):
        super().__init__(
            loss_func=torch.nn.MSELoss(),
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            transform_outputs=lambda outputs: torch.squeeze(outputs, dim=1))


class ClassificationTrainingAlgorithm(TrainingAlgorithm):

    def __init__(self, batch_size=32, num_epochs=100, learning_rate=0.001):
        super().__init__(
            loss_func=torch.nn.CrossEntropyLoss(),
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate)


class Linear(torch.nn.Module):

    def __init__(self, in_features):
        self.linear = torch.nn.Linear(in_features, 1)

    def __forward__(self, inputs):
        return self.linear(inputs)
