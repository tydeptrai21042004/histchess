"""Synthetic-fixture smoke checks only (not an empirical results audit)."""
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'scripts' / 'run_paper_cpu_extension.py'
spec = importlib.util.spec_from_file_location('histchess_paper_extension', source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture_pairs():
    pairs=[]
    for i,task in enumerate(module.TASKS):
        for j,fam in enumerate(module.FAMILIES[task]):
            for k in range(2):
                n=i*6+j*2+k
                ranks=['8']*8
                rank, file=divmod(n,8)
                ranks[rank] = (str(file) if file else '')+'Q'+(str(7-file) if 7-file else '')
                board='/'.join(ranks)+' w'
                pairs.append({'pair_id':f'{i}-{j}-{k}','task':task,'generator_family':fam,
                    'board_key':board,
                    'positive':{'history_uci':['g1f3','g8f6','b1c3'],'label':1},
                    'negative':{'history_uci':['b1c3','g8f6','g1f3'],'label':0}})
    return pairs

def test_cpu_non_neural_baselines():
    pairs=fixture_pairs()
    replay={(p['pair_id'],v):float(v=='positive') for p in pairs for v in ('positive','negative')}
    pred,metrics,tuning=module.train_controls(pairs,replay,256)
    assert len(pred)==1056 and len(metrics)==330 and len(tuning)==60
    assert all(r['joint_pair_accuracy']==0 for r in metrics if r['history_plies']=='0')
    assert all(r['joint_pair_accuracy']==1 for r in metrics if r['baseline']=='legal_replay_full')
    assert all(r['history_plies']=='full' for r in metrics if r['baseline']=='legal_replay_full')
