from functools import reduce
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
            subsumption_order: set[tuple[Pattern, Pattern]] = None
    ):
        """Initialise OrderedPatternRuleSet using `predictions` and `priorities` (if given)

        If `predictions` are given but `priorities` are not,
        then deduce priorities from the partial order on premises of predictions.

        If neither `predictions` nor `priorities` are given,
        then the model should be fitted further on.
        """
        if predictions is None:
            self._predictions, self._priorities = None, None
            self._subsumption_order = None
            return

        if PatternType is None:
            PatternType = next(iter(predictions.keys())).__class__
        else:
            predictions = {PatternType(k): v for k, v in predictions.items()}
            if priorities is not None:
                priorities = {(PatternType(left), PatternType(right)) for left, right in priorities}
            if subsumption_order is not None:
                subsumption_order = {(PatternType(left), PatternType(right)) for left, right in subsumption_order}

        # `predictions` is not None
        if subsumption_order is None:
            subsumption_order = {
                (premise, other) for premise in predictions for other in predictions if premise <= other
            }

        priorities = subsumption_order if priorities is None else priorities
        assert all(premise in predictions for pair in priorities for premise in pair), \
            "Provide a prediction for every premise from `priorities` set"

        priorities = set(priorities) | {(premise, premise) for premise in predictions}  # add identity relation
        priorities = self._transitive_closure(priorities)
        assert all((b, a) not in priorities for a, b in priorities if a!=b), \
            "Priorities does not make a partial order, but rather a pseudo-order"

        n_subsumed = {premise: 0 for premise in predictions}
        for _, right in subsumption_order:
            n_subsumed[right] += 1

        self._premises: list[Pattern] = sorted(predictions, key=lambda premise: n_subsumed[premise])
        self._predictions = predictions
        self._priorities = priorities
        self.PatternType = PatternType
        self._subsumption_order = subsumption_order

    @property
    def premises(self) -> set[Pattern]:
        return set(self._premises)

    @property
    def predictions(self) -> dict[Pattern, TargetType]:
        return dict(self._predictions)

    @property
    def priorities(self) -> set[tuple[Pattern, Pattern]]:
        return set(self._priorities)

    @property
    def contains_tautology(self) -> bool:
        return self._premises[0] == self.PatternType.get_min_pattern()

    def is_rule_set(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        if not self.contains_tautology:
            return False

        ruleset_priorities = {(p, p) for p in self.premises} | {(tautology, p) for p in self.premises}
        return self.priorities == ruleset_priorities

    def is_decision_list(self) -> bool:
        tautology = self.PatternType.get_min_pattern()
        if not self.contains_tautology:
            return False

        priorities_reduced = self._transitive_reduction(self.priorities)
        if any(right == tautology for _, right in priorities_reduced):
            return False
        n_premises = len(self.premises)

        unique_lefts = len({left for left, _ in priorities_reduced}) == n_premises-1
        unique_rights = len({right for right, _ in priorities_reduced}) == n_premises-1
        return unique_lefts and unique_rights

    def is_decision_tree(self) -> bool:
        if not self.contains_tautology:
            return False

        if self.priorities != self._subsumption_order:
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
    def init_rule_set(cls, predictions: dict[Pattern, TargetType], PatternType = None, subsumption_order = None) -> 'OPRSet':
        PatternType = next(iter(predictions.keys())).__class__ if PatternType is None else PatternType
        tautology = PatternType.get_min_pattern()

        priorities = {(p, p) for p in predictions} | {(tautology, p) for p in predictions}
        ruleset = cls(predictions, priorities, PatternType=PatternType, subsumption_order=subsumption_order)
        assert ruleset.is_rule_set()
        return ruleset

    @classmethod
    def init_decision_list(cls, predictions: list[tuple[Pattern, TargetType]], PatternType = None, subsumption_order = None) -> 'OPRSet':
        premises = [premise for premise, _ in predictions]
        priorities = {(q, p) for p, q in zip(premises, premises[1:])}
        dlist = cls(dict(predictions), priorities, PatternType=PatternType, subsumption_order=subsumption_order)
        assert dlist.is_decision_list()
        return dlist

    @classmethod
    def init_decision_tree(cls, predictions: dict[Pattern, TargetType], PatternType = None, subsumption_order = None) -> 'OPRSet':
        dtree = cls(predictions, PatternType=PatternType, subsumption_order=subsumption_order)
        assert dtree.is_decision_tree()
        return dtree

    @classmethod
    def from_sklearn_dtree(cls, decision_tree, feature_names: Union[list[str], list[int]] = None) -> 'OPRSet':
        predictions, priorities = converters.from_sklearn_dtree(decision_tree, feature_names=feature_names)
        return cls(predictions, priorities, subsumption_order=priorities)

    @classmethod
    def from_sklearn_rforest(cls, random_forest, feature_names: Union[list[str], list[int]] = None) -> 'OPRSet':
        estimators = [cls.from_sklearn_dtree(estimator, feature_names) for estimator in random_forest.estimators_]
        weights = [1/len(estimators) for _ in estimators]
        return cls.from_ensemble(estimators, weights)

    @classmethod
    def from_ensemble(cls, estimators: list['OPRSet'], weights: list[float]) -> 'OPRSet':
        linearised = estimators[0].linearise().scale(weights[0])
        for estimator, weight in zip(estimators[1:], weights[1:]):
            linearised += estimator.linearise().scale(weight)
        closure = linearised.join_closure()
        return cls(closure.predictions, closure.priorities, PatternType=closure.PatternType,
                   subsumption_order=closure._subsumption_order)

    ######################
    # ML model interface #
    ######################
    def predict(self, X: list[Pattern]) -> np.ndarray[Optional[TargetType]]:
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self, x: Pattern) -> Optional[TargetType]:
        assert self._predictions is not None and self._priorities is not None,  \
            "The model is not fitted yet. Either fit it or initialise with the provided predictions and priorities."

        x = self.PatternType(x) if not isinstance(x, self.PatternType) else x
        applied_premises, maximal_premises = self.traverse_single(x)

        if self.priorities != self._subsumption_order:
            maxpriority_premises = {
                premise for premise in applied_premises
                if not any((premise, other) in self.priorities for other in applied_premises if other != premise)
            }
        else:
            maxpriority_premises = maximal_premises

        predictions = [self.predictions[premise] for premise in maxpriority_premises]
        return sum(predictions)/len(predictions) if predictions else None

    ######################
    # Model manipulation #
    ######################
    def join_closure(self) -> 'OPRSet':
        premise_list = list(self.premises)
        subpremises = {premise: set() for premise in premise_list}
        for subpremise, superpremise in self._subsumption_order:
            subpremises[superpremise].add(subpremise)
        premise_list = sorted(subpremises.keys(), key=lambda premise: len(subpremises[premise]))

        controversy = self.PatternType.get_max_pattern()
        i = 0
        while i < len(premise_list):
            for j in range(i):
                p, q = premise_list[i], premise_list[j]
                if p in subpremises[q] or q in subpremises[p]:
                    continue

                join = p | q
                if join == controversy or join in subpremises:
                    continue
                premise_list.append(join)

                subpremises[join] = subpremises[p] | subpremises[q]
                for other in set(subpremises) - set(subpremises[join]):
                    if p in subpremises[other] and q in subpremises[other]:
                        subpremises[other].add(join)
                        continue
                    if join <= other:
                        subpremises[other].add(join)
                        continue
                    if other <= join:
                        subpremises[join].add(other)
            i += 1

        predictions = {premise: self.predict_single(premise) for premise in premise_list}
        subsumption_order = {(sub, superpremise) for superpremise, subs in subpremises.items() for sub in subs}
        return self.__class__(predictions, PatternType=self.PatternType, subsumption_order=subsumption_order)


    def is_join_closed(self) -> bool:
        if not self.contains_tautology:
            return False

        premises = self.premises
        controversy = self.PatternType.get_max_pattern()
        for p in premises:
            for q in premises:
                join = p | q
                if join == controversy:
                    continue

                if join not in premises:
                    return False
                if (p, join) not in self.priorities or (q, join) not in self.priorities:
                    return False
        return True

    def linearise(self) -> 'LinPRSet':
        from decisionguild.linprset import LinPRSet
        closure = self if self.is_join_closed() else self.join_closure()
        predecessors = {}
        for left, right in closure.priorities:
            if right not in predecessors:
                predecessors[right] = set()
            if left != right:
                predecessors[right].add(left)

        predictions = {}
        for premise in sorted(predecessors, key=lambda p: len(predecessors[p])):
            accumulated_prediction = sum(predictions[pred] for pred in predecessors[premise])
            predictions[premise] = closure.predictions[premise] - accumulated_prediction
        return LinPRSet(predictions, subsumption_order=closure._subsumption_order)

    def __add__(self, other: 'OPRSet') -> 'OPRSet':
        closure = (self.linearise() + other.linearise()).join_closure()
        return self.__class__(predictions=closure.predictions, priorities=closure.priorities, PatternType=closure.PatternType)

    def scale(self, k: int) -> 'OPRSet':
        scaled_predictions = {premise: prediction * k for premise, prediction in self.predictions.items()}
        return self.__class__(scaled_predictions, priorities=self.priorities, PatternType=self.PatternType, subsumption_order=self._subsumption_order)

    def traverse(self, X: list[Pattern]) -> list[tuple[set[Pattern], set[Pattern]]]:
        return [self.traverse_single(x) for x in X]

    def traverse_single(self, x: PatternType) -> tuple[set[Pattern], set[Pattern]]:
        all_successors = {premise: set() for premise in self.premises}
        for left, right in self._subsumption_order:
            if left != right:
                all_successors[left].add(right)
        direct_successors = dict()
        for premise, successors in all_successors.items():
            transitives = reduce(set.union, (all_successors[suc] for suc in successors), set())
            direct_successors[premise] = successors - set(transitives)

        top_level_premises = set(self.premises) - set(reduce(set.union, all_successors.values(), set()))

        applied_patterns = set()
        cancelled_patterns = set()
        queue = list(top_level_premises)
        while queue:
            premise = queue.pop(0)
            if premise in cancelled_patterns:
                continue
            if premise <= x:
                applied_patterns.add(premise)
                for successor in direct_successors[premise] - cancelled_patterns:
                    queue.append(successor)
            else:  # not (premise <= x)
                cancelled_patterns |= all_successors[premise]

        maximal_patterns = {premise for premise in applied_patterns
                            if direct_successors[premise].isdisjoint(applied_patterns)}
        return applied_patterns, maximal_patterns

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


class OPRClassifier(OPRSet):
    TargetType = int

    def predict_single_proba(self, x: Pattern) -> Optional[np.ndarray]:
        return super().predict_single(x)

    def predict_single(self, x: Pattern) -> Optional[TargetType]:
        return int(np.argmax(self.predict_single_proba(x)))

    def predict_proba(self, X: list[Pattern]) -> np.ndarray:
        return np.vstack([self.predict_single_proba(x) for x in X])

    @property
    def predictions(self) -> dict[Pattern, np.ndarray[float]]:
        return super().predictions

    def join_closure(self) -> 'OPRClassifier':
        closure = super().join_closure()
        predictions = {premise: self.predict_single_proba(premise) for premise in closure.premises}
        return self.__class__(predictions=predictions, priorities=closure.priorities, PatternType=closure.PatternType,
                              subsumption_order=closure._subsumption_order)
