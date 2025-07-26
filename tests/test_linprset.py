from paspailleur.pattern_structures import built_in_patterns as bip

from decisionguild.dual_patterns import DualIntervalPattern
from decisionguild.oprset import OPRSet
from decisionguild.linprset import LinPRSet

def test_init():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>10')
    r = bip.IntervalPattern('(10, 20]')
    predictions = {t: 0, p: 1, q: 2, r: 4}
    model = LinPRSet(predictions)

    assert model.premises == set(predictions)
    assert model.predictions == predictions
    assert model.priorities == {(p, p) for p in predictions}
    assert model.PatternType == bip.IntervalPattern


def test_predict_single():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>10')
    r = bip.IntervalPattern('(10, 20]')
    model = LinPRSet({t: 1, p: 2, q: 4, r: 8})

    assert model.predict_single(t) == 1
    assert model.predict_single(p) == 3
    assert model.predict_single(q) == 5
    assert model.predict_single(r) == 15
    assert model.predict_single(15) == 15
    assert model.predict_single(30) == 5
    assert model.predict_single(0) == 3


def test_add():
    t = bip.IntervalPattern.get_min_pattern()
    p = bip.IntervalPattern('<=20')
    q = bip.IntervalPattern('>10')
    r = bip.IntervalPattern('(10, 20]')
    model = LinPRSet({t: 1, p: 2, q: 4})
    model2 = LinPRSet({t: 8, q: 16, r: 32})
    model_sum_true = LinPRSet({t: 9, p: 2, q: 20, r: 32})
    model_sum = model + model2
    assert model_sum.predictions == model_sum_true.predictions
    assert model_sum.priorities == model_sum_true.priorities

def test_join_closure():
    t = DualIntervalPattern.get_min_pattern()
    p = DualIntervalPattern('<=20')
    q = DualIntervalPattern('>10')
    r = DualIntervalPattern('(10, 20]')
    model = LinPRSet({t: 1, p: 2, q: 4})
    closure_true = OPRSet({t: 1, p: 3, q: 5, r: 7})

    closure = model.join_closure()
    assert type(closure) == OPRSet
    assert closure.premises == closure_true.premises
    assert closure.predictions == closure_true.predictions
    assert closure.priorities == closure_true.priorities


