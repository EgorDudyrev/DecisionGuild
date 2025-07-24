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
            subsumption_order: set[tuple[Pattern, Pattern]] = None,
    ):
        priorities = {(p, p) for p in predictions}
        super().__init__(predictions, priorities, PatternType=PatternType, subsumption_order=subsumption_order)

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

        subsumption_order = self._subsumption_order | other._subsumption_order
        for p in self.premises:
            for q in other.premises - self.premises:
                if p <= q:
                    subsumption_order.add((p, q))
                elif q <= p:
                    subsumption_order.add((q, p))
        return self.__class__(predictions, PatternType=self.PatternType, subsumption_order=subsumption_order)

    def scale(self, k: int) -> "LinPRSet":
        scaled_predictions = {premise: prediction * k for premise, prediction in self.predictions.items()}
        return self.__class__(scaled_predictions, PatternType=self.PatternType, subsumption_order=self._subsumption_order)

    def join_closure(self) -> 'OPRSet':
        closure = super().join_closure()
        return OPRSet(closure.predictions, PatternType=closure.PatternType, subsumption_order=closure._subsumption_order)
