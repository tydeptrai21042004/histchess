"""Diversified histories must satisfy the same exact replay constraints."""
import collections
import json

import pytest

from histchess.core import TASKS,generate,validate
from histchess.diverse import FAMILIES
from histchess.experiments import leakage_audit
from histchess.shortcuts import shortcut_experiments


@pytest.mark.parametrize('seed', [3101,3102])
def test_family_holdout_generation_replayed_and_deterministic(seed):
    pairs,_=generate(18,seed=seed,generator='diverse',split_strategy='family')
    again,_=generate(18,seed=seed,generator='diverse',split_strategy='family')
    assert json.dumps(pairs,sort_keys=True)==json.dumps(again,sort_keys=True)
    assert validate(pairs)['passed']
    assert len(pairs)==18*len(TASKS)
    audit=leakage_audit(pairs)
    assert audit['board_group_leakage_count']==0
    assert audit['full_history_leakage_count']==0
    assert audit['generator_family_across_split_count']==0
    for task in TASKS:
        subset=[p for p in pairs if p['task']==task]
        assert len({p['generator_family'] for p in subset})==3
        assert {p['split'] for p in subset}=={'train','validation','test'}
        assert all(sum(p['generator_family']==f for p in subset)==6 for f in FAMILIES[task])
        for p in subset:
            assert {p['positive']['label'],p['negative']['label']}=={0,1}


def test_original_signatures_not_perfect_on_diversified_family_test(tmp_path):
    pairs,_=generate(30,seed=3101,generator='diverse',split_strategy='family')
    metrics=shortcut_experiments(pairs,tmp_path)
    old=next(m for m in metrics if m['model']=='legacy_fixed_signatures' and m['task']=='all')
    assert old['accuracy_including_invalid']<1.0
    assert (tmp_path/'shortcut_predictions.csv').exists()
    assert (tmp_path/'shortcut_audit.json').exists()


def test_family_split_disjointness_and_per_task_balance_larger():
    pairs,_=generate(60,seed=3101,generator='diverse',split_strategy='family')
    assert validate(pairs)['passed']
    for task in TASKS:
        assert collections.Counter(p['split'] for p in pairs if p['task']==task)=={
            'train':20,'validation':20,'test':20}


def test_legacy_mode_is_backward_compatible():
    baseline,_=generate(10,seed=3101)
    explicit,_=generate(10,seed=3101,generator='legacy',split_strategy='board')
    assert baseline==explicit
