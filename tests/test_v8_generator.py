"""Actual chess legality checks. Run on Kaggle with chess==1.11.2."""
import pytest


def test_v8_scenario_family_validation_and_fold_independence():
    pytest.importorskip('chess')
    from histchess.core import validate
    from histchess.research_extension import generate_scenario_pairs,dataset_report
    from histchess.architecture_benchmark import rotate_splits
    pairs,_=generate_scenario_pairs(per_family=1,seed=3101,lengths=('native',),attempts=10)
    assert len(pairs)==12
    assert validate(pairs)['passed']
    assert {p['independent_mechanism'] for p in pairs if p['task']=='castling'}=={
        'rook_g1_return','king_e2_return','rook_h2_return'}
    for fold in (0,1,2):
        rotated=rotate_splits(pairs,fold)
        for task in ('castling','en_passant','halfmove','repetition'):
            assert {p['split'] for p in rotated if p['task']==task}=={'train','validation','test'}
    assert len(dataset_report(pairs))==12


@pytest.mark.slow
def test_v8_true_64_ply_and_causal_distance():
    pytest.importorskip('chess')
    from histchess.core import validate
    from histchess.research_extension import generate_scenario_pairs
    pairs,_=generate_scenario_pairs(per_family=1,seed=3101,lengths=('64',),attempts=100)
    assert len(pairs)==12
    assert all(p['actual_plies']>=64 for p in pairs)
    assert any(p['first_divergence_distance']>=32 for p in pairs if p['task']=='castling')
    assert validate(pairs)['passed']


def test_v8_requires_independent_external_provenance(tmp_path):
    pytest.importorskip('chess')
    from histchess.research_extension import merge_external_mechanisms
    p=tmp_path/'external.jsonl'
    p.write_text('{"task":"castling","generator_family":"new","pair_id":"a"}\n')
    with pytest.raises(ValueError,match='independent_mechanism'):
        merge_external_mechanisms([],p)
