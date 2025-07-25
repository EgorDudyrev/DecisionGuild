import numpy as np

from decisionguild.dual_patterns import DualPattern, DualCartesianPattern, DualIntervalPattern
from decisionguild import converters

def test_from_sklearn_dtree():
    from sklearn import tree
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, Y)
    predictions, priorities = converters.from_sklearn_dtree(clf)

    class PClass(DualCartesianPattern):
        DimensionTypes = {0: DualIntervalPattern, 1: DualIntervalPattern}

    premises_true = [PClass(dict()), PClass({0: '<= 0.5'}), PClass({0: '>0.5'})]
    predictions_true = {
        premises_true[0]: np.array([[0.5, 0.5]]),
        premises_true[1]: np.array([[1, 0]]),
        premises_true[2]: np.array([[0, 1]]),
    }
    priorities_true = {
        (premises_true[0], premises_true[1]),
        (premises_true[0], premises_true[2])
    }
    assert set(predictions) == set(premises_true)
    assert set(predictions) == set(predictions_true)
    assert all((predictions[p]==predictions_true[p]).all() for p in premises_true)
    assert priorities == priorities_true
