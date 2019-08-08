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
"""Roughness penalties for regularization."""
import torch


def L1Penalty(smoothing=1):

    def penalize(model):
        """Calculates the L1 regularization term for a model."""
        penalty = None
        for name, parameter in model.named_parameters():
            if not "bias" in name:
                if not penalty:
                    penalty = parameter.norm(1)
                else:
                    penalty += parameter.norm(1)
        return smoothing * penalty

    return penalize
