"""Legality regression tests. Kaggle dependency stage installs python-chess."""
import pytest

chess=pytest.importorskip("chess")
from histchess.core import (CAST_POS,CAST_NEG,EP_POS,EP_NEG,HALF_POS,
                            HALF_NEG,REP_POS,REP_NEG,pair_from_moves,validate)
from histchess.length_stress import extend_pair

@pytest.mark.parametrize("task,pos,neg",[
    ("castling",CAST_POS,CAST_NEG),
    ("en_passant",EP_POS,EP_NEG),
    ("halfmove",HALF_POS,HALF_NEG),
    ("repetition",REP_POS,REP_NEG),
])
def test_stress_preserves_labels_legality_and_observable_board(task,pos,neg):
    base=pair_from_moves(task,pos,neg)
    base["generator_family"]="test_fixture"
    base["split"]="train"
    stressed=extend_pair(base,target_plies=32,seed=77,attempts=200)
    assert len(stressed["positive"]["history_uci"])>=32
    assert stressed["positive"]["board_observation"]==stressed["negative"]["board_observation"]
    again=pair_from_moves(task,stressed["positive"]["history_uci"],
                         stressed["negative"]["history_uci"])
    assert again["positive"]==stressed["positive"]
    assert again["negative"]==stressed["negative"]
    assert validate([stressed])["passed"]
    if task=="castling":
        baseline_event_age=len(pos)-next(i for i,(a,b) in enumerate(zip(pos,neg)) if a!=b)-1
        assert stressed["first_divergence_distance"]>baseline_event_age
