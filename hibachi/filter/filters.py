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
"""Filter methods for feature selection."""
from timeit import default_timer as timer

import numpy as np
import torch

from hibachi import criteria, datasets


class Filter(Selector):
    pass


class Custom(Filter):

    def __init__(self, function, criterion):
        self.function = function
        self.criterion = criterion

    def __call__(self, X, y):
        raise NotImplementedError


class Threshold(Filter):

    def __init__(self, value, criterion, minimize=False):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError


class Select(Filter):

    def __init__(self, k, criterion, minimize=False):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError


class Compose(Filter):

    def __init__(self, *filters):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError


class MissingThreshold(Filter):

    def __init__(self, cutoff=0.5):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError


class VarianceThreshold(Filter):

    def __init__(self, cutoff=0.5):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError


class CollinearityThreshold(Filter):

    def __init__(self, cutoff=0.5):
        raise NotImplementedError

    def __call__(self, X, y):
        raise NotImplementedError
