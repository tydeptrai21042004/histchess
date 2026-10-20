"""Rules are independently replayed on every counterfactual pair."""
import copy
import json
import pytest

from histchess.core import (TASKS, CAST_POS, CAST_NEG, EP_POS, EP_NEG,
                            HALF_POS, HALF_NEG, REP_POS, REP_NEG,
                            pair_from_moves, replay, observed_label, generate,
                            validate, rows_from_pairs, position_key, repetitions)

TEMPLATES = list(zip(TASKS, (CAST_POS, EP_POS, HALF_POS, REP_POS),
                       (CAST_NEG, EP_NEG, HALF_NEG, REP_NEG)))

@pytest.mark.parametrize('task,pos,neg',TEMPLATES)
def test_templates_are_legal_counterfactuals(task,pos,neg):
    p = pair_from_moves(task,pos,neg)
    assert p['positive']['label'] == 1 and p['negative']['label'] == 0
    assert p['positive']['board_observation'] == p['negative']['board_observation']
    assert p['positive']['certificate']['move_count'] == len(pos)
    assert p['negative']['certificate']['move_count'] == len(neg)
    assert bool(observed_label(replay(pos),task)) is True
    assert bool(observed_label(replay(neg),task)) is False
    fa,fb=p['positive']['fen'].split(),p['negative']['fen'].split()
    if task=='repetition': assert fa == fb
    else:
        field={'castling':2,'en_passant':3,'halfmove':4}[task]
        assert [i for i in range(6) if fa[i]!=fb[i]] == [field]


def test_castling_rights_not_current_castling_legality():
    p=pair_from_moves('castling',CAST_POS,CAST_NEG)
    assert p['positive']['white_kingside_castling']
    assert not p['negative']['white_kingside_castling']
    # Both boards have the same arrangement, independent of castling rights.
    assert p['positive']['board_observation']==p['negative']['board_observation']


def test_repetition_is_current_occurrences_not_future_claim():
    p=pair_from_moves('repetition',REP_POS,REP_NEG)
    assert p['positive']['current_position_occurrences']>=3
    assert p['negative']['current_position_occurrences']<3
    assert position_key(replay(REP_POS)) == position_key(replay(REP_NEG))
    assert len(repetitions(replay(REP_POS)))>=3


def test_en_passant_requires_legal_immediate_capture():
    pos,neg=pair_from_moves('en_passant',EP_POS,EP_NEG)['positive'],pair_from_moves('en_passant',EP_POS,EP_NEG)['negative']
    assert pos['legal_en_passant'] and not neg['legal_en_passant']

@pytest.mark.parametrize('task,pos,neg',TEMPLATES)
def test_identical_positive_and_negative_history_rejected(task,pos,neg):
    with pytest.raises(ValueError):pair_from_moves(task,pos,pos)


def test_invalid_move_and_length_rejected():
    with pytest.raises(ValueError):replay(('e2e5',))
    with pytest.raises(ValueError):pair_from_moves('castling',CAST_POS,CAST_NEG[:-1])


def test_generated_small_dataset_replay_and_reproducibility():
    a,_=generate(count_per_task=10,seed=3101)
    b,_=generate(count_per_task=10,seed=3101)
    assert len(a)==40 and json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    report=validate(a)
    assert report['passed'],report['errors']
    assert all(report['task_counts'][t]==10 for t in TASKS)
    assert len(rows_from_pairs(a))==80
    assert all({p['positive']['label'],p['negative']['label']}=={0,1} for p in a)
    assert len({p['pair_id'] for p in a})==len(a)


def test_validator_catches_tampered_label_history_and_split():
    pairs,_=generate(count_per_task=10,seed=3101)
    for attr, val in (('label',0),('certificate',{'tampered':True}),('fen','invalid')):
        clone=copy.deepcopy(pairs)
        clone[0]['positive'][attr]=val
        if attr=='label': clone[0]['positive'][attr]=2
        assert not validate(clone)['passed'],attr
    clone=copy.deepcopy(pairs)
    clone[0]['pair_id']='falsified_id'
    assert not validate(clone)['passed']


def test_generation_has_nonempty_task_splits_in_realistic_smoke():
    pairs,_=generate(count_per_task=40,seed=3101)
    counts={(task,split):sum(p['task']==task and p['split']==split for p in pairs)
            for task in TASKS for split in ('train','validation','test')}
    assert min(counts.values())>0,counts
