"""Certified 64-ply regression for every synthetic rule family.

This test MUST run on Kaggle after installing python-chess.  It must not be
silently omitted from the benchmark runner's preflight stage.
"""
import pytest
chess=pytest.importorskip('chess')
from histchess.core import (CAST_POS,CAST_NEG,EP_POS,EP_NEG,HALF_POS,
                            HALF_NEG,REP_POS,REP_NEG,pair_from_moves,validate)
from histchess.length_stress import extend_pair

@pytest.mark.parametrize('task,pos,neg',[
    ('castling',CAST_POS,CAST_NEG),
    ('en_passant',EP_POS,EP_NEG),
    ('halfmove',HALF_POS,HALF_NEG),
    ('repetition',REP_POS,REP_NEG),
])
def test_64_ply_certified_pair(task,pos,neg):
    base=pair_from_moves(task,pos,neg)
    base['generator_family']='test_fixture'
    base['split']='train'
    q=extend_pair(base,target_plies=64,seed=77,attempts=100)
    assert len(q['positive']['history_uci'])>=64
    assert len(q['positive']['history_uci'])==len(q['negative']['history_uci'])
    assert q['positive']['board_observation']==q['negative']['board_observation']
    again=pair_from_moves(task,q['positive']['history_uci'],q['negative']['history_uci'])
    assert again['positive']==q['positive']
    assert again['negative']==q['negative']
    assert validate([q])['passed']
    if task=='castling':
        first=next(i for i,(a,b) in enumerate(zip(pos,neg)) if a!=b)
        assert q['first_divergence_distance']>len(pos)-first-1


def test_all_diverse_templates_stress():
    from histchess.diverse import generate_diverse
    pairs,_=generate_diverse(count_per_task=6,seed=3101)
    output=[]
    for pair in pairs:
        q=extend_pair(pair,target_plies=64,seed=3101,attempts=100)
        output.append(q)
        assert q['positive']['label']==1
        assert q['negative']['label']==0
        assert len(q['positive']['history_uci'])>=64
    assert validate(output)['passed']
