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
