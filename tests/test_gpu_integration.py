"""Real tiny training and audits; GPU job is separately marked."""
import json
import pytest

from histchess.core import generate
from histchess.gpu_experiments import history_experiment
from histchess.experiments import baseline_experiments,leakage_audit
from histchess.paper import paper_statistics,paper_tables,make_figures
from histchess.audit import audit_final_predictions


def _run(out,require_gpu=False):
    pairs,_=generate(40,seed=3101)
    from histchess.core import validate
    assert validate(pairs)['passed']
    baseline_experiments(pairs,out)
    aud=leakage_audit(pairs)
    assert aud['board_group_leakage_count']==0
    assert aud['full_history_leakage_count']==0
    history_experiment(pairs,out,seeds=(3101,),cutoffs=(0,None),epochs=1,
                       batch_size=64,patience=1,require_gpu=require_gpu)
    paper_statistics(pairs,out,bootstrap=25,seed=3101)
    paper_tables(out);make_figures(out)
    assert (out/'gru_predictions.csv').stat().st_size>0
    assert (out/'figures'/'history_length_ablation.png').exists()
    return pairs


@pytest.mark.slow
def test_tiny_end_to_end_cpu_or_gpu(tmp_path):
    # This verifies the actual training/evaluation/export path without a model download.
    pairs=_run(tmp_path)
    assert any(p['split']=='test' for p in pairs)


@pytest.mark.gpu_integration
@pytest.mark.slow
def test_cuda_full_neural_batch(tmp_path):
    import torch
    if not torch.cuda.is_available():pytest.skip('Real CUDA GPU required')
    pairs=_run(tmp_path,require_gpu=True)
    assert pairs


def test_fail_closed_when_gpu_is_required(tmp_path):
    import torch
    if torch.cuda.is_available():pytest.skip('CUDA present')
    with pytest.raises(RuntimeError,match='CUDA'):
        history_experiment([],tmp_path,seeds=(1,),cutoffs=(None,),epochs=1,require_gpu=True)
