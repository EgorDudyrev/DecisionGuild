from typing import Optional

import numpy as np
import pytest

from decisionguild.oprset import OPRSet
from decisionguild.dual_patterns import DualIntervalPattern
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


def test_is_rule_set():
    class CountryPattern(bip.CategorySetPattern):
        Universe = {'France', 'Germany', 'UK'}
    class HumanPattern(bip.CartesianPattern):
        DimensionTypes = {'age': bip.IntervalPattern, 'country': CountryPattern}
    p = HumanPattern({'age': '>=20'})
    q = HumanPattern({'country': 'France'})
    r = HumanPattern({'country': 'France', 'age': '>=30'})
    t = HumanPattern(dict())

    model = OPRSet({p: 1, q: 2, r: 3})
    assert not model.is_rule_set()
    model = OPRSet({p: 1, q: 2})
    assert not model.is_rule_set()
    model = OPRSet({p: 1, q: 2, t: 10})
    assert model.is_rule_set()
    model = OPRSet({p: 1, q: 2, r: 3, t: 10})
    assert not model.is_rule_set()


def test_is_decision_list():
    class CountryPattern(bip.CategorySetPattern):
        Universe = {'France', 'Germany', 'UK'}
    class HumanPattern(bip.CartesianPattern):
        DimensionTypes = {'age': bip.IntervalPattern, 'country': CountryPattern}

    p = HumanPattern({'age': '>=20'})
    q = HumanPattern({'country': 'France'})
    r = HumanPattern({'country': 'France', 'age': '>=30'})
    t = HumanPattern(dict())

    model = OPRSet({p: 1, q: 2, r: 3})
    assert not model.is_decision_list()
    model = OPRSet({p: 1, t: 10})
    is_dl = model.is_decision_list()
    assert is_dl
    model = OPRSet({p: 1, q: 2, r:3}, priorities={(p, q), (q, r)})
    assert not model.is_decision_list()
    model = OPRSet({p: 1, q: 2, r: 3, t: 10}, priorities={(t, r), (r, q), (q, p)})
    assert model.is_decision_list()


def test_is_decision_tree():
    t = DualIntervalPattern.get_min_pattern()
    p = DualIntervalPattern('>=20')
    q = DualIntervalPattern('<20')
    r = DualIntervalPattern('>=40')
    s = DualIntervalPattern('[20, 40)')
    model = OPRSet({p: 1, q: 2, r: 3})
    assert not model.is_decision_tree()

    model = OPRSet({t:0, p: 2, q: -2, r: 3, s: 1})
    assert model.is_decision_tree()
    model = OPRSet({p: 2, q: -2, r: 3, s: 1})
    assert not model.is_decision_tree()
    model = OPRSet({t: 0, p: 2, q: -2, r: 3})
    assert not model.is_decision_tree()
    model = OPRSet({t: 0, q: -2, r: 3, s: 1})
    assert not model.is_decision_tree()


def test_init_rule_set():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>=30')
    r = bip.IntervalPattern('>=50')
    model = OPRSet.init_rule_set({t: 0, p: 1, q: 2, r: 3})
    assert model.is_rule_set()
    with pytest.raises(AssertionError):
        model = OPRSet.init_rule_set({p: 1, q: 2})

def test_init_decision_list():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>=30')
    model = OPRSet.init_decision_list([(p, 1), (q, 2), (t, 0)])
    assert model.is_decision_list()
    with pytest.raises(AssertionError):
        OPRSet.init_decision_list([(t, 0), (p, 1), (q, 2)])

def test_init_decision_tree():
    t = DualIntervalPattern.get_min_pattern()
    p = DualIntervalPattern('<=20')
    q = DualIntervalPattern('>20')
    r = DualIntervalPattern('<=10')
    s = DualIntervalPattern('(10, 20]')
    OPRSet.init_decision_tree({t: 0, p: 2, q: -1, r: 1, s: 3})

    q2 = DualIntervalPattern('>=20')
    with pytest.raises(AssertionError):
        OPRSet.init_decision_tree({t: 0, p: 2, q2: -1, r: 1, s: 3})

    q3 = DualIntervalPattern('>=30')
    with pytest.raises(AssertionError):
        OPRSet.init_decision_tree({t: 0, p: 1, q3: 2})


def test_is_join_closed():
    t = DualIntervalPattern.get_min_pattern()
    p = DualIntervalPattern('<=20')
    q = DualIntervalPattern('>20')
    r = DualIntervalPattern('<=10')
    s = DualIntervalPattern('(10, 20]')
    model = OPRSet.init_decision_tree({t: 0, p: 2, q: -1, r: 1, s: 3})
    assert model.is_join_closed()

    model = OPRSet.init_decision_list([(p, 2), (t, 0)])
    assert model.is_join_closed()
    model = OPRSet.init_decision_list([(r, 1), (p, 2), (t, 0)])
    assert model.is_join_closed()

    model = OPRSet.init_decision_tree({t: 0, p: 2, q: -1})
    assert model.is_join_closed()

    model = OPRSet({p: 1, q:-1})
    assert not model.is_join_closed()
    model = OPRSet({p: 1, q:-1, t:0})
    assert model.is_join_closed()

    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>10')
    model = OPRSet.init_rule_set({t: 0, p: 2, q: -1})
    assert not model.is_join_closed()

def test_join_closure():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>10')
    r = bip.IntervalPattern('(10, 20]')
    model = OPRSet({t: 0, p: 2, q: 1})
    closure_true = OPRSet({t: 0, p: 2, q: 1, r: 1.5})
    closure_fact = model.join_closure()
    assert closure_fact.premises == closure_true.premises
    assert closure_fact.predictions == closure_true.predictions
    assert closure_fact.priorities == closure_true.priorities

    s = bip.IntervalPattern('>50')
    model = OPRSet({t: 0, p: 2, q: 1, s: 10})
    closure_true = OPRSet({t: 0, p: 2, q: 1, r: 1.5, s: 10})
    closure_fact = model.join_closure()
    assert closure_fact.premises == closure_true.premises
    assert closure_fact.predictions == closure_true.predictions
    assert closure_fact.priorities == closure_true.priorities

    model = OPRSet.init_decision_list([(p, 2), (q, 1), (t, 0)])
    closure_true = OPRSet({t: 0, p: 2, q: 1, r: 2})
    closure_fact = model.join_closure()
    assert closure_fact.premises == closure_true.premises
    assert closure_fact.predictions == closure_true.predictions
    assert closure_fact.priorities == closure_true.priorities
