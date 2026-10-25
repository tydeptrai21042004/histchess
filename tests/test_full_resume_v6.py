"""Unit/integration fixtures are synthetic; not a substitute for chess replay on Kaggle."""
import json
import pytest
from histchess.architecture_benchmark import FAMILIES, TASKS, run_architecture_benchmark


def pairs_fixture():
    pairs=[]
    for i,task in enumerate(TASKS):
        for j,family in enumerate(FAMILIES[task]):
            squares=['-']*64
            squares[i*3+j]='P'
            parts=[]
            for index in range(0,64,8):
                row='';space=0
                for c in squares[index:index+8]:
                    if c=='-':space+=1
                    else:
                        if space:row+=str(space);space=0
                        row+=c
                if space:row+=str(space)
                parts.append(row)
            board='/'.join(parts)+' w'
            pairs.append({'pair_id':f'{i}_{j}','task':task,
                'generator_family':family,'board_key':board,
                'split':('train','validation','test')[j],
                'positive':{'label':1,'board_observation':board,'history_uci':['g1f3','g8f6','f3g1']},
                'negative':{'label':0,'board_observation':board,'history_uci':['g1f3','g8f6','b1c3']}})
    return pairs


def test_safe_pause_and_exact_resume(tmp_path):
    pairs=pairs_fixture()
    options=dict(architectures=('gru',),folds=(0,),seeds=(47,),cutoffs=(0,2),epochs=1,
        patience=1,batch_size=12,target_parameters=16000,max_history=16,
        position_encoding='sinusoidal')
    run_architecture_benchmark(pairs,tmp_path,max_new_runs=1,**options)
    state=json.loads((tmp_path/'architecture_status.json').read_text())
    assert state['paused'] and not state['complete'] and len(state['runs'])==1
    assert len(list((tmp_path/'architecture_run_shards').glob('*.json')))==1
    run_architecture_benchmark(pairs,tmp_path,max_new_runs=0,**options)
    state=json.loads((tmp_path/'architecture_status.json').read_text())
    assert state['complete'] and len(state['runs'])==2
    assert len(list((tmp_path/'architecture_run_shards').glob('*.json')))==2
    from pathlib import Path
    from importlib.util import spec_from_file_location, module_from_spec
    script=Path(__file__).resolve().parents[1]/'scripts/verify_architecture_results.py'
    spec=spec_from_file_location('audit_arch_v6',script)
    auditor=module_from_spec(spec);spec.loader.exec_module(auditor)
    data=tmp_path/'data';data.mkdir(exist_ok=True)
    (data/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    audit=auditor.audit(tmp_path)
    assert audit['passed'],audit['errors'][:5]
    from scripts.make_manuscript_tables import generate
    with pytest.raises(Exception,match='No such file|Incomplete experiment|Missing|450'):
        generate(tmp_path)


def test_cuda_request_fails_closed_when_unavailable(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA requested'):
        run_architecture_benchmark(pairs_fixture(),tmp_path,architectures=('gru',),
                folds=(0,),seeds=(1,),cutoffs=(0,),epochs=1,device='cuda')
