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
"""Wrapper methods for feature selection."""
import torch
from torch.utils.data import DataLoader

from hibachi import models, measures


class SuperGreedy:

    def __init__(self,
                 model_class=models.BinaryLogisticRegressionModel,
                 train=TrainingAlgorithm(loss_func=torch.nn.CrossEntropyLoss()),
                 measure=measures.ClassificationAccuracy()):
        self.model_class = model_class
        self.train = train
        self.measure = measure

    def __call__(self, dataset, k=10):
        if not len(dataset) > 0:
            raise ValueError("Invalid dataset size: {}".format(len(dataset)))

        scores = []
        for feature_index in range(dataset.dimensionality):
            model = self.model_class(1)
            partial_dataset = dataset.select({feature_index})
            self.train(model, partial_dataset)
            score = self.measure(model, dataset)
            scores.append(score)

        rankings = np.argsort(scores)
        return set([
            feature_index for feature_index in range(dataset.dimensionality)
            if ranking[feature_index] < k
        ])
