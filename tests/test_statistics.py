from histchess.paper import bootstrap_pair_ci


def synthetic(n=10):
    return [{'pair_id':str(i)} for i in range(n)]


def test_bootstrap_pair_level_bounds_determinism():
    pairs=synthetic(10)
    predictions={(str(i),v): i<7 if v=='positive' else False
                 for i in range(10) for v in ('positive','negative')}
    a=bootstrap_pair_ci(pairs,predictions,seed=19,repeats=100)
    assert a==bootstrap_pair_ci(pairs,predictions,seed=19,repeats=100)
    assert a['record_accuracy']==.85
    assert a['joint_pair_accuracy']==.7
    assert 0<=a['record_ci_low']<=a['record_accuracy']<=a['record_ci_high']<=1
    assert 0<=a['joint_ci_low']<=a['joint_pair_accuracy']<=a['joint_ci_high']<=1


def test_missing_prediction_counts_wrong():
    p=synthetic(1)
    assert bootstrap_pair_ci(p,{},seed=1,repeats=10)['record_accuracy']==0
    assert bootstrap_pair_ci(p,{},seed=1,repeats=10)['joint_pair_accuracy']==0
