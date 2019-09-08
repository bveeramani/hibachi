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

class Explainer:

    def __call__(self, model, input):
        raise NotImplementedError


class GradientStar(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class IntegratedGradient(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class LRP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class EpsilonLRP(Explainer): #????

    def __call__(self, model, input):
        raise NotImplementedError


class Occulsion(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class Saliency(Explainer): #????

    def __call__(self, model, input):
        raise NotImplementedError


class DeepLIFT(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class LIME(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class SHAP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class KernelSHAP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class MaxSHAP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class DeepSHAP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class LowOrderSHAP(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError


class L2X(Explainer):

    def __call__(self, model, input):
        raise NotImplementedError
