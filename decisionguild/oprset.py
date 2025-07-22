from typing import TypeVar, Optional, Union

import numpy as np
from paspailleur.pattern_structures.pattern import Pattern

from decisionguild import converters

class OPRSet:
    TargetType = TypeVar("TargetType")
    PatternType = TypeVar("PatternType", bound=Pattern)

    #############
    # Class API #
    #############

    def __init__(
            self,
            predictions: dict[PatternType, TargetType] = None,
            priorities: set[tuple[Pattern, Pattern]] = None,
            PatternType = None,
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

        if PatternType is None:
            PatternType = next(iter(predictions.keys())).__class__
        else:
            predictions = {PatternType(k): v for k, v in predictions.items()}
            if priorities is not None:
                priorities = {(PatternType(left), PatternType(right)) for left, right in priorities}

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
        self.PatternType = PatternType

    @property
    def premises(self) -> set[Pattern]:
        return set(self.predictions)

    @property
    def predictions(self) -> dict[Pattern, TargetType]:
        return dict(self._predictions)

    @property
    def priorities(self) -> set[tuple[Pattern, Pattern]]:
        return set(self._priorities)

    def is_rule_set(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        if tautology not in self.premises:
            return False

        ruleset_priorities = {(p, p) for p in self.premises} | {(tautology, p) for p in self.premises}
        return self.priorities == ruleset_priorities

    def is_decision_list(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        if tautology not in self.premises:
            return False

        priorities_reduced = self._transitive_reduction(self.priorities)
        if any(right == tautology for _, right in priorities_reduced):
            return False
        n_premises = len(self.premises)

        unique_lefts = len({left for left, _ in priorities_reduced}) == n_premises-1
        unique_rights = len({right for right, _ in priorities_reduced}) == n_premises-1
        return unique_lefts and unique_rights

    def is_decision_tree(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        controversy = self.PatternType.get_max_pattern()
        if tautology not in self.premises:
            return False

        priorities_reduced = self._transitive_reduction(self.priorities)
        successors = {}
        for left, right in priorities_reduced:
            if left not in successors:
                successors[left] = set()
            successors[left].add(right)

        if any(len(next_nodes)!=2 for next_nodes in successors.values()):
            return False

        for premise, (next_pos, next_neg) in successors.items():
            diff_pos, diff_neg = next_pos - premise, next_neg - premise
            if diff_pos != -diff_neg:
                return False
        return True

    @classmethod
    def init_rule_set(cls, predictions: dict[Pattern, TargetType], PatternType = None) -> 'OPRSet':
        PatternType = next(iter(predictions.keys())).__class__ if PatternType is None else PatternType
        tautology = PatternType.get_min_pattern()

        priorities = {(p, p) for p in predictions} | {(tautology, p) for p in predictions}
        ruleset = cls(predictions, priorities, PatternType=PatternType)
        assert ruleset.is_rule_set()
        return ruleset

    @classmethod
    def init_decision_list(cls, predictions: list[tuple[Pattern, TargetType]], PatternType = None) -> 'OPRSet':
        premises = [premise for premise, _ in predictions]
        priorities = {(q, p) for p, q in zip(premises, premises[1:])}
        dlist = cls(dict(predictions), priorities, PatternType=PatternType)
        assert dlist.is_decision_list()
        return dlist

    @classmethod
    def init_decision_tree(cls, predictions: dict[Pattern, TargetType], PatternType = None) -> 'OPRSet':
        dtree = cls(predictions, PatternType=PatternType)
        assert dtree.is_decision_tree()
        return dtree

    @classmethod
    def from_sklearn_dtree(cls, decision_tree, feature_names: Union[list[str], list[int]] = None) -> 'OPRSet':
        return cls(*converters.from_sklearn_dtree(decision_tree, feature_names=feature_names))

    ######################
    # ML model interface #
    ######################
    def predict(self, X: list[Pattern]) -> np.ndarray[Optional[TargetType]]:
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self, x: Pattern) -> Optional[TargetType]:
        assert self._predictions is not None and self._priorities is not None,  \
            "The model is not fitted yet. Either fit it or initialise with the provided predictions and priorities."

        x = self.PatternType(x) if not isinstance(x, self.PatternType) else x

        applied_premises = {premise for premise in self.premises if premise <= x}
        maximal_premises = {premise for premise in applied_premises
                            if not any((premise, other) in self.priorities for other in applied_premises if other != premise)}
        predictions = [self.predictions[premise] for premise in maximal_premises]
        return sum(predictions)/len(predictions) if predictions else None

    ######################
    # Model manipulation #
    ######################
    def join_closure(self) -> 'OPRSet':
        controversy = self.PatternType.get_max_pattern()
        premise_list = list(self.premises)
        i = 0
        while i < len(premise_list):
            for j in range(i):
                join = premise_list[i] | premise_list[j]
                if join == controversy or join in premise_list:
                    continue
                premise_list.append(join)
            i += 1

        predictions = {premise: self.predict_single(premise) for premise in premise_list}
        return self.__class__(predictions, PatternType=self.PatternType)


    def is_join_closed(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        controversy = self.PatternType.get_max_pattern()
        if tautology not in self.premises:
            return False

        for p in self.premises:
            for q in self.premises:
                join = p | q
                if join == controversy:
                    continue

                if join not in self.premises:
                    return False
                if (p, join) not in self.priorities or (q, join) not in self.priorities:
                    return False
        return True

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

    @staticmethod
    def _transitive_reduction(relation: set[tuple[Pattern, Pattern]]) -> set[tuple[Pattern, Pattern]]:
        relation_dict = dict()
        for left, right in relation:
            if left not in relation_dict:
                relation_dict[left] = set()
            if left != right:
                relation_dict[left].add(right)

        relation_reduced = dict()
        for left, rights in relation_dict.items():
            transitives = {trans for right in rights for trans in relation_dict[right]}
            relation_reduced[left] = rights - transitives

        return {(left, right) for left, rights in relation_reduced.items() for right in rights}
