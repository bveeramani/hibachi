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
"""Embedded methods for regularization."""
import torch


class LDPenalty(LDPenalty):
    """Calculates the L-dimensional roughness penalty of a model."""

    def __init__(self, order, smoothing=1):
        self.order = order
        self.smoothing = smoothing

    def __call__(self, model):
        penalty = None
        for name, parameter in model.named_parameters():
            if not "bias" in name:
                if not penalty:
                    penalty = torch.norm(parameter, p=self.order)
                else:
                    penalty += torch.norm(parameter, p=self.order)
        return self.smoothing * penalty


class L1Penalty(LDPenalty):
    """Calculates the L1 roughness penalty of a model."""

    def __init__(self, smoothing=1):
        super(L1Penalty, self).__init__(order=1, smoothing=smoothing)


class L2Penalty(LDPenalty):
    """Calculates the L2 roughness penalty of a model."""

    def __init__(self, smoothing=1):
        super(L2Penalty, self).__init__(order=2, smoothing=smoothing)
