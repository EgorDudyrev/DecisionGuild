from typing import TypeVar, Optional

import numpy as np
from paspailleur.pattern_structures.pattern import Pattern


class OPRSet:
    TargetType = TypeVar("TargetType")

    #############
    # Class API #
    #############

    def __init__(
            self,
            predictions: dict[Pattern, TargetType] = None,
            priorities: set[tuple[Pattern, Pattern]] = None
    ):
        """Initialise OrderedPatternRuleSet using `predictions` and `priorities` (if given)

        If `predictions` are given but `priorities` are not,
        then deduce priorities from the partial order on premises of predictions.

        If neither `predictions` nor `priorities` are given,
        then the model should be fitted further on.
        """
        if predictions is None:
            self._predictions, self._priorities = None, None
            return

        # `predictions` is not None
        if priorities is None:
            priorities = {(premise, other) for premise in predictions for other in predictions if premise <= other}

        assert all(premise in predictions for pair in priorities for premise in pair), \
            "Provide a prediction for every premise from `priorities` set"

        priorities = set(priorities) | {(premise, premise) for premise in predictions}  # add identity relation
        priorities = self._transitive_closure(priorities)
        assert all((b, a) not in priorities for a, b in priorities if a!=b), \
            "Priorities does not make a partial order, but rather a pseudo-order"

        self._predictions = predictions
        self._priorities = priorities

    @property
    def premises(self) -> set[Pattern]:
        return set(self.predictions)

    @property
    def predictions(self) -> dict[Pattern, TargetType]:
        return dict(self._predictions)

    @property
    def priorities(self) -> set[tuple[Pattern, Pattern]]:
        return set(self._priorities)

    ######################
    # ML model interface #
    ######################
    def predict(self, X: list[Pattern]) -> np.ndarray[Optional[TargetType]]:
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self, x: Pattern) -> Optional[TargetType]:
        assert self._predictions is not None and self._priorities is not None,  \
            "The model is not fitted yet. Either fit it or initialise with the provided predictions and priorities."
        applied_premises = {premise for premise in self.premises if premise <= x}
        maximal_premises = {premise for premise in applied_premises
                            if not any((premise, other) in self.priorities for other in applied_premises if other != premise)}
        predictions = [self.predictions[premise] for premise in maximal_premises]
        return sum(predictions)/len(predictions) if predictions else None


    #####################
    # Helping functions #
    #####################
    @staticmethod
    def _transitive_closure(relation: set[tuple[Pattern, Pattern]]) -> set[tuple[Pattern, Pattern]]:
        relation_dict = dict()  # TODO: Replace with groupby function
        for left, right in relation:
            if left not in relation_dict:
                relation_dict[left] = set()
            relation_dict[left].add(right)

        new_relation_added = True
        while new_relation_added:
            new_relation_added = False
            for left, rights in relation_dict.items():
                rights_updated = {transitive for right in rights for transitive in relation_dict[right] - rights}
                new_relation_added |= (rights_updated != set())
                relation_dict[left] |= rights_updated
        return {(a, b) for a, bs in relation_dict.items() for b in bs}
