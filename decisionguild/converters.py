import math
from typing import Union

from decisionguild.dual_patterns import DualCartesianPattern, DualIntervalPattern
from sklearn import tree


def from_sklearn_dtree(
        decision_tree: tree, feature_names: Union[list[str], list[int]] = None
) -> tuple[dict[DualCartesianPattern, float], set[tuple[DualCartesianPattern, DualCartesianPattern]]]:
    feature_names = list(range(decision_tree.n_features_in_)) if feature_names is None else feature_names

    class HyperrectanglePattern(DualCartesianPattern):
        DimensionTypes = {f: DualIntervalPattern for f in feature_names}

    added_premises = [HyperrectanglePattern({feature_names[f]: ((-math.inf, True), (t, True))})  # f <= t
                      for f, t in zip(decision_tree.tree_.feature, decision_tree.tree_.threshold)]

    premises: list[HyperrectanglePattern] = [None for _ in added_premises]
    premises[0] = HyperrectanglePattern.get_min_pattern()

    priorities = set()
    for premise_i, added_premise in enumerate(added_premises):
        child_leq = decision_tree.tree_.children_left[premise_i]
        child_gt = decision_tree.tree_.children_right[premise_i]
        if child_leq != -1:
            premises[child_leq] = premises[premise_i] | added_premise
            priorities.add((premises[premise_i], premises[child_leq]))
        if child_gt != -1:
            premises[child_gt] = premises[premise_i] | -added_premise
            priorities.add((premises[premise_i], premises[child_gt]))

    predictions = {premise: value for premise, value in zip(premises, decision_tree.tree_.value)}
    return predictions, priorities
