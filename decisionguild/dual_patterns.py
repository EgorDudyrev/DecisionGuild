from typing import Optional

from paspailleur.pattern_structures.pattern import Pattern
from paspailleur.pattern_structures import built_in_patterns as bip


class DualPattern(Pattern):
    def __neg__(self) -> Optional['DualPattern']:
        raise NotImplementedError()

    @property
    def is_controversy(self) -> bool:
        raise NotImplementedError()


class DualIntervalPattern(DualPattern, bip.IntervalPattern):
    def __neg__(self) -> Optional['DualIntervalPattern']:
        tautology = self.get_min_pattern()
        controversy = self.get_max_pattern()
        if self == tautology:
            return controversy
        if self == controversy:
            return tautology

        if self.value[0] == tautology.value[0]:  # <x or <=x
            return self.__class__(((self.upper_bound, not self.is_upper_bound_closed), tautology.value[1]))

        if self.value[1] == tautology.value[1]:  # >x or >=x
            return self.__class__((tautology.value[0], (self.lower_bound, not self.is_lower_bound_closed),))

        return None

    @property
    def is_controversy(self) -> bool:
        return self == self.get_max_pattern()


class DualCategorySetPattern(DualPattern, bip.CategorySetPattern):
    def __neg__(self) -> Optional['DualCategorySetPattern']:
        if self.Universe is None:
            return None

        return self.__class__(self.Universe - self.value)

class DualCartesianPattern(DualPattern, bip.CartesianPattern):
    def __neg__(self) -> Optional['DualCartesianPattern']:
        if self == self.get_min_pattern():
            return self.get_max_pattern()
        if self == self.get_max_pattern():
            return self.get_min_pattern()

        if len(self.value) > 1:
            return None

        k, v = next(iter(self.value.items()))
        if not isinstance(v, DualPattern):
            return None
        if -v is None:
            return None
        return self.__class__({k: -v})

    @property
    def is_controversy(self) -> bool:
        return any(dimension.is_controversy for dimension in self.value.values())
