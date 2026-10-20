import pytest

from histchess.core import TASKS, pair_from_moves, CAST_POS, CAST_NEG
from histchess.gpu_experiments import move_coords,board_features,history_features
from histchess.experiments import metrics,parse_yes_no

@pytest.mark.parametrize('value,expected',[
    ('a1h8',(1,64)),('g1f3',(7,22)),('a7a8q',(49,57)),('e2e4',(13,29))])
def test_fixed_square_vocabulary(value,expected):
    assert move_coords(value)==expected

@pytest.mark.parametrize('invalid',['','e9e4','z1e2','e2e2K','e2e4qq','e1e','e2a0','1234'])
def test_bad_uci_rejected(invalid):
    with pytest.raises(ValueError):move_coords(invalid)


def test_board_features_turn_and_empty_squares():
    key='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w'
    pieces, side=board_features(key)
    assert len(pieces)==64 and side==0
    assert len([x for x in pieces if x==0])==32
    assert board_features(key[:-1]+'b')[1]==1
    with pytest.raises(ValueError):board_features('8/8/8 w')


def test_history_zero_and_tail_preservation():
    moves=['g1f3','g8f6','f3g1','f6g8']
    assert history_features(moves,0)==([(0,0)],0)
    tail,n=history_features(moves,2)
    assert n==2 and tail==[move_coords(x) for x in moves[-2:]]
    assert history_features(moves,None,max_history=3)[1]==3


@pytest.mark.parametrize('raw,want',[
    ('YES',True),('no.',False),('Answer: Yes\n',True),('YES because',True),
    ('possibly yes',None),('I cannot decide',None),('',None),('not sure',None),
    ('No? Maybe',False)])
def test_strict_llm_output_parser(raw,want):
    assert parse_yes_no(raw) is want


def test_paired_accuracy_missing_included_as_wrong():
    p=pair_from_moves('castling',CAST_POS,CAST_NEG)
    p['split']='test'
    pid=p['pair_id']
    perfect=metrics({(pid,'positive'):True,(pid,'negative'):False},[p],'x')[0]
    assert perfect['accuracy_including_invalid']==1
    assert perfect['paired_joint_accuracy']==1
    partial=metrics({(pid,'positive'):True,(pid,'negative'):None},[p],'x')[0]
    assert partial['accuracy_including_invalid']==.5
    assert partial['paired_joint_accuracy']==0
    assert partial['invalid_or_missing']==1
    wrong=metrics({(pid,'positive'):False,(pid,'negative'):True},[p],'x')[0]
    assert wrong['accuracy_including_invalid']==0
