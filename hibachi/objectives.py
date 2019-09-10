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


class Objective:

    def __call__(self, features):
        raise NotImplementedError


class NegativeMSE(Objective):

    def __call__(self, model, dataset):
        total = torch.tensor(0, dtype=torch.float)
        for x, y in dataset:
            y_hat = torch.squeeze(model(torch.unsqueeze(x, 0)))
            residue = y - y_hat
            square = residue**2
            total += square
        return -total / len(dataset)


class ClassificationAccuracy(Objective):

    def __call__(self, model, dataset):
        num_correct = 0
        for x, g in dataset:
            y_hat = torch.squeeze(model(torch.unsqueeze(x, 0)))
            g_hat = torch.argmax(y_hat)
            if g_hat == g:
                num_correct += 1
        return num_correct / len(dataset)


class ChernoffDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class BhattacharyyaDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class KLDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class KolmogorovDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class MatusitaDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class PatrickFisherDivergence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class Dependence(Objective):

    def __call__(self, features):
        raise NotImplementedError


class Distance(Objective):

    def __call__(self, features):
        raise NotImplementedError


class Uncertainty(Objective):

    def __call__(self, features):
        raise NotImplementedError


class Consistency(Objective):

    def __call__(self, features):
        raise NotImplementedError
