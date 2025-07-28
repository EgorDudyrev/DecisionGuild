from typing import TypeVar, Optional

from bitarray.util import zeros as bazeros

from decisionguild.dual_patterns import DualPattern
from decisionguild.oprset import OPRSet


class LinPRSet(OPRSet):
    TargetType = TypeVar("TargetType")
    PatternType = TypeVar("PatternType", bound=DualPattern)

    #############
    # Class API #
    #############
    def __init__(
            self,
            predictions: dict[PatternType, TargetType] = None,
            PatternType = None,
            subsumption_order: set[tuple[DualPattern, DualPattern]] = None,
            priorities: set[tuple[DualPattern, DualPattern]] = None,  # for backwards compatibility with OPRSet. does not affect anything
            premises: list[DualPattern] = None,  # for backwards compatibility with OPRSet
            check_input: bool = True,  # for backwards compatibility with OPRSet
    ):
        if check_input:
            priorities = {(p, p) for p in predictions}
        else:
            n_premises = len(predictions)

            priorities = []
            for i in range(n_premises):
                ba = bazeros(n_premises)
                ba[i] = True
                priorities.append(ba)

        super().__init__(predictions, priorities, PatternType=PatternType, subsumption_order=subsumption_order, premises=premises, check_input=check_input)

    ######################
    # ML model interface #
    ######################
    def predict_single(self, x: DualPattern) -> Optional[TargetType]:
        assert self._predictions is not None and self._priorities is not None,  \
            "The model is not fitted yet. Either fit it or initialise with the provided predictions and priorities."

        x = self.PatternType(x) if not isinstance(x, self.PatternType) else x

        applied_premises = {premise for premise in self.premises if premise <= x}
        predictions_dict = self.predictions
        predictions = [predictions_dict[premise] for premise in applied_premises]
        return sum(predictions) if predictions else None

    ######################
    # Model manipulation #
    ######################
    def __add__(self, other):
        if other == 0:  # for the sake of using standard sum() operation
            return self

        predictions = dict(self.predictions)
        for premise, other_prediction in other.predictions.items():
            if premise in predictions:
                predictions[premise] = predictions[premise] + other_prediction
            else:
                predictions[premise] = other_prediction

        subsumption_order = self.subsumption_order | other.subsumption_order
        for p in self.premises:
            for q in other.premises - self.premises:
                if p <= q:
                    subsumption_order.add((p, q))
                elif q <= p:
                    subsumption_order.add((q, p))
        return self.__class__(predictions, PatternType=self.PatternType, subsumption_order=subsumption_order)

    def __radd__(self, other):
        return self + other
