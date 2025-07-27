import math
from typing import Optional, Self

from frozendict import frozendict
from paspailleur.pattern_structures.pattern import Pattern
from paspailleur.pattern_structures import built_in_patterns as bip


class DualPattern(Pattern):
    def __neg__(self) -> Optional['DualPattern']:
        raise NotImplementedError()

    @property
    def is_controversy(self) -> bool:
        raise NotImplementedError()


class DualIntervalPattern(DualPattern, bip.IntervalPattern):
    MinValue = (-math.inf, True), (math.inf, True)
    MaxValue = (math.inf, True), (-math.inf, True)

    def __init__(self, value, check_input=True):
        if check_input:
            super().__init__(value)
        else:
            self._value = value

    def __and__(self, other: 'DualIntervalPattern') -> 'DualIntervalPattern':
        if self.value == self.MinValue or other.value == other.MinValue:
            return self.get_min_pattern()

        if self.value == self.MaxValue:
            return other

        if other.value == self.MaxValue:
            return self

        if self.lower_bound < other.lower_bound:
            lbound, closed_lb = self.lower_bound, self.is_lower_bound_closed
        elif other.lower_bound < self.lower_bound:
            lbound, closed_lb = other.lower_bound, other.is_lower_bound_closed
        else:  # self._lower_bound == other._lower_bound
            lbound = self.lower_bound
            closed_lb = self.is_lower_bound_closed or other.is_lower_bound_closed

        if self.upper_bound > other.upper_bound:
            ubound, closed_ub = self.upper_bound, self.is_upper_bound_closed
        elif other.upper_bound > self.upper_bound:
            ubound, closed_ub = other.upper_bound, other.is_upper_bound_closed
        else:  # self._upper_bound == other._upper_bound
            ubound = self.upper_bound
            closed_ub = self.is_upper_bound_closed or other.is_upper_bound_closed

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        return self.__class__(new_value, check_input=False)

    def __or__(self, other: 'DualIntervalPattern') -> 'DualIntervalPattern':
        if self.value == self.MaxValue or other.value == other.MaxValue:
            return self.get_max_pattern()

        if self.value == self.MinValue:
            return other
        if other.value == self.MinValue:
            return self

        if self.lower_bound < other.lower_bound:
            lbound, closed_lb = other.lower_bound, other.is_lower_bound_closed
        elif other.lower_bound < self.lower_bound:
            lbound, closed_lb = self.lower_bound, self.is_lower_bound_closed
        else:  # self._lower_bound == other._lower_bound
            lbound = self.lower_bound
            closed_lb = self.is_lower_bound_closed and other.is_lower_bound_closed

        if self.upper_bound > other.upper_bound:
            ubound, closed_ub = other.upper_bound, other.is_upper_bound_closed
        elif other.upper_bound > self.upper_bound:
            ubound, closed_ub = self.upper_bound, self.is_upper_bound_closed
        else:  # self._upper_bound == other._upper_bound
            ubound = self.upper_bound
            closed_ub = self.is_upper_bound_closed and other.is_upper_bound_closed

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        if (lbound > ubound) \
                or (lbound == ubound and not (closed_lb and closed_ub)):
            return self.get_max_pattern()
        return self.__class__(new_value, check_input=False)

    def __le__(self, other: 'DualIntervalPattern') -> bool:
        if other.lower_bound < self.lower_bound:
            return False
        # now, self.lower_bound <= other.lower_bound
        if other.lower_bound == self.lower_bound and other.is_lower_bound_closed and not self.is_lower_bound_closed:
            return False
        # now, lower_bound should be ok
        if self.upper_bound < other.upper_bound:
            return False
        # now, other.upper_bound <= self.upper_bound
        if other.upper_bound == self.upper_bound and not self.is_upper_bound_closed and other.is_upper_bound_closed:
            return False
        # now, upper_bound should be ok
        return True


    @classmethod
    def get_min_pattern(cls) -> Self:
        return cls(cls.MinValue, check_input=False)

    @classmethod
    def get_max_pattern(cls) -> Self:
        return cls(cls.MaxValue, check_input=False)

    def __neg__(self) -> Optional['DualIntervalPattern']:
        tautology = self.get_min_pattern()
        controversy = self.get_max_pattern()
        if self == tautology:
            return controversy
        if self == controversy:
            return tautology

        if self.value[0] == tautology.value[0]:  # <x or <=x
            return self.__class__(((self.upper_bound, not self.is_upper_bound_closed), tautology.value[1]), check_input=False)

        if self.value[1] == tautology.value[1]:  # >x or >=x
            return self.__class__((tautology.value[0], (self.lower_bound, not self.is_lower_bound_closed),), check_input=False)

        return None

    @property
    def is_tautology(self):
        return self.value == self.MinValue

    @property
    def is_controversy(self) -> bool:
        return self.value == self.MaxValue


class DualCategorySetPattern(DualPattern, bip.CategorySetPattern):
    def __init__(self, value, check_input: bool = True):
        if check_input:
            super().__init__(value)
        else:
            self._value = value

    def __and__(self, other: 'DualCategorySetPattern') -> 'DualCategorySetPattern':
        return self.__class__(self.value | other.value, check_input=False)

    def __or__(self, other: 'DualCategorySetPattern') -> 'DualCategorySetPattern':
        return self.__class__(self.value & other.value, check_input=False)

    def __le__(self, other: 'DualCategorySetPattern') -> bool:
        return self.value >= other.value

    def __neg__(self) -> Optional['DualCategorySetPattern']:
        if self.Universe is None:
            return None

        return self.__class__(self.Universe - self.value, check_input=False)

    @property
    def is_tautology(self):
        return self.value == self.Universe

    @classmethod
    def get_min_pattern(cls) -> Self:
        return cls(cls.Universe, check_input=False)

    @classmethod
    def get_max_pattern(cls) -> Self:
        return cls(frozenset(), check_input=False)



class DualCartesianPattern(DualPattern, bip.CartesianPattern):
    def __init__(self, value, check_input: bool = True):
        if check_input:
            super().__init__(value)
        else:
            self._value = value

    def __and__(self, other: 'DualCartesianPattern') -> 'DualCartesianPattern':
        val = dict()
        for k in set(self.value) & set(other.value):
            meet = self.value[k] & other.value[k]
            if not meet.is_tautology:
                val[k] = meet
        return self.__class__(frozendict(val), check_input=False)


    def __or__(self, other: 'DualCartesianPattern'):
        val = dict(self.value)
        for k in set(val) & set(other.value):
            val[k] = val[k] | other.value[k]
            if val[k].is_controversy:
                return self.get_max_pattern()
        for k in set(other.value) - set(val):
            val[k] = other[k]
        return self.__class__(frozendict(val), check_input=False)

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
        return self.__class__({k: -v}, check_input=False)

    @property
    def is_controversy(self) -> bool:
        return any(dimension.is_controversy for dimension in self.value.values())

    @classmethod
    def get_min_pattern(cls) -> Self:
        return cls(frozendict(), check_input=False)

    @classmethod
    def get_max_pattern(cls) -> Self:
        return cls({dim: ptype.get_max_pattern() for dim, ptype in cls.DimensionTypes.items()}, check_input=False)

    def __le__(self, other: 'DualCartesianPattern') -> bool:
        if not (set(self.value) <= set(other.value)):
            return False

        for k, v in self.value.items():
            if not (v <= other.value[k]):
                return False
        return True
