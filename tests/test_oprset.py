import numpy as np
import pytest

from decisionguild.oprset import OPRSet
from paspailleur.pattern_structures import built_in_patterns as bip


def test_init():
    model = OPRSet()
    assert model._predictions is None
    assert model._priorities is None

    class HumanPattern(bip.CartesianPattern):
        DimensionTypes = {'age': bip.IntervalPattern, 'country': bip.CategorySetPattern}
    p = HumanPattern({'age': '>=20'})
    q = HumanPattern({'country': 'France'})
    r = HumanPattern({'country': 'France', 'age': '>=20'})
    predictions = {p: 1, q: 2, r: 3}
    model = OPRSet(predictions)
    assert model.premises == {p, q, r}
    assert model.predictions == predictions
    assert model.priorities == {(p, p), (q, q), (r, r), (p, r), (q, r)}

    with pytest.raises(AssertionError):
        model = OPRSet(predictions, {(p, q), (q, p)})

    p, q = bip.IntervalPattern('[10, 20]'), bip.IntervalPattern('[30, 40]')
    model = OPRSet({str(p): 1, str(q): 2}, PatternType=bip.IntervalPattern)
    assert model.premises == {p, q}
    assert model.predictions == {p: 1, q: 2}
    assert model.priorities == {(p, p), (q, q)}
    assert model.PatternType == bip.IntervalPattern



def test_predict_single():
    class HumanPattern(bip.CartesianPattern):
        DimensionTypes = {'age': bip.IntervalPattern, 'country': bip.CategorySetPattern}
    p = HumanPattern({'age': '>=20'})
    q = HumanPattern({'country': 'France'})
    r = HumanPattern({'country': 'France', 'age': '>=30'})
    predictions = {p: 1, q: 2, r: 3}

    model = OPRSet(predictions, PatternType=HumanPattern)
    assert model.predict_single({'age': '[30, 60]'}) == 1
    assert model.predict_single(HumanPattern({'country': 'France'})) == 2
    assert model.predict_single(HumanPattern({'age': 20, 'country': 'France'})) == 1.5
    assert model.predict_single(HumanPattern({'age': 30, 'country': 'France'})) == 3
    assert model.predict_single(HumanPattern({'age': 10})) is None


def test_predict():
    class HumanPattern(bip.CartesianPattern):
        DimensionTypes = {'age': bip.IntervalPattern, 'country': bip.CategorySetPattern}
    p = HumanPattern({'age': '>=20'})
    q = HumanPattern({'country': 'France'})
    r = HumanPattern({'country': 'France', 'age': '>=30'})
    predictions = {p: 1, q: 2, r: 3}

    model = OPRSet(predictions)
    X = [
        HumanPattern({'age': '[30, 60]'}), HumanPattern({'country': 'France'}),
        HumanPattern({'age': 20, 'country': 'France'}), HumanPattern({'age': 30, 'country': 'France'}),
        {'age': 10}  # to test if OPRSet will automatically convert the data into HumanPattern
    ]
    assert (model.predict(X) == np.array([1, 2, 1.5, 3, None])).all()
