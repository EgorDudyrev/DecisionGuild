from typing import TypeVar, Optional

from paspailleur.pattern_structures.pattern import Pattern

from decisionguild.oprset import OPRSet


class LinPRSet(OPRSet):
    TargetType = TypeVar("TargetType")
    PatternType = TypeVar("PatternType", bound=Pattern)

    #############
    # Class API #
    #############
    def __init__(
            self,
            predictions: dict[PatternType, TargetType] = None,
            PatternType = None,
    ):
        priorities = {(p, p) for p in predictions}
        super().__init__(predictions, priorities, PatternType=PatternType)

    ######################
    # ML model interface #
    ######################
    def predict_single(self, x: Pattern) -> Optional[TargetType]:
        assert self._predictions is not None and self._priorities is not None,  \
            "The model is not fitted yet. Either fit it or initialise with the provided predictions and priorities."

        x = self.PatternType(x) if not isinstance(x, self.PatternType) else x

        applied_premises = {premise for premise in self.premises if premise <= x}
        predictions = [self.predictions[premise] for premise in applied_premises]
        return sum(predictions) if predictions else None

    ######################
    # Model manipulation #
    ######################
    def __add__(self, other):
        predictions = dict(self.predictions)
        for premise, other_prediction in other.predictions.items():
            if premise in predictions:
                predictions[premise] = predictions[premise] + other_prediction
            else:
                predictions[premise] = other_prediction
        return self.__class__(predictions, PatternType=self.PatternType)

    def join_closure(self) -> 'OPRSet':
        closure = super().join_closure()
        return OPRSet(closure.predictions, PatternType=closure.PatternType)
