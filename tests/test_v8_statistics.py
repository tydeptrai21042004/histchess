"""No chess/torch/network needed for statistical unit tests."""
import csv
import json

import pytest

from histchess.research_statistics import _p_signflip, summarize_runs, analyze_overfitting


def _make_result(root,rep):
    root.mkdir(parents=True,exist_ok=True)
    (root/'data').mkdir(exist_ok=True)
    (root/'architecture_status.json').write_text(json.dumps({'complete':True}))
    pairs=[];records=[]
    for task in ('castling','en_passant','halfmove','repetition'):
        for i,family in enumerate(('family_a','family_b','family_c')):
            pid=f'{rep}_{task}_{i}'
            pairs.append({'pair_id':pid,'task':task,'generator_family':family,
                          'independent_mechanism':family,'replicate_id':rep,
                          'positive':{'label':1},'negative':{'label':0}})
            for model in ('better','worse'):
                for variant,label in (('positive',1),('negative',0)):
                    predicted=(label if model=='better' else 0)
                    records.append({'architecture':model,'fold':(i+1)%3,
                                    'seed':3101,'history_plies':'full',
                                    'pair_id':pid,'variant':variant,
                                    'label':label,'prediction':predicted})
    (root/'data'/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    with (root/'architecture_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def test_repetition_cluster_bootstrap_and_holm(tmp_path):
    folders=[]
    for rep in range(5):
        root=tmp_path/f'replication_{rep}'
        _make_result(root,rep)
        folders.append(root)
    stats=summarize_runs(folders,tmp_path/'stats',bootstrap=200,seed=7)
    overall=[r for r in stats if r['task']=='all' and r['history_plies']=='full']
    assert len(overall)==1
    row=overall[0]
    assert row['generator_replications']==5
    assert row['inference_status']=='conditional_replication_inference'
    assert row['mean_accuracy_difference']==pytest.approx(0.5)
    assert row['replicate_bootstrap_ci_95_low']==pytest.approx(0.5)
    assert 0<=row['paired_signflip_p_unadjusted']<=row['holm_p_adjusted']<=1
    assert (tmp_path/'stats'/'statistics_protocol.json').is_file()


def test_statistics_withhold_pvalues_for_only_one_generator_seed(tmp_path):
    root=tmp_path/'only_replication'
    _make_result(root,0)
    stats=summarize_runs([root],tmp_path/'stats',bootstrap=200)
    assert stats and all(x['inference_status']=='descriptive_only' for x in stats)
    assert all(x['paired_signflip_p_unadjusted'] is None for x in stats)
    assert _p_signflip([1,1,1,1,1])==0.0625


def test_overfit_diagnostic_detects_degradation(tmp_path):
    root=tmp_path/'result';root.mkdir()
    with (root/'architecture_metrics.csv').open('w',newline='') as f:
        data=[{'architecture':'transformer','fold':0,'seed':1,'history_plies':'full',
               'task':'all','best_epoch':1,'selected_grid_trial':0}]
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    with (root/'architecture_learning_curves.csv').open('w',newline='') as f:
        data=[{'architecture':'transformer','fold':0,'seed':1,'history':'None',
               'epoch':1,'train_loss':0.5,'validation_loss':0.6,'grid_trial':0},
              {'architecture':'transformer','fold':0,'seed':1,'history':'None',
               'epoch':2,'train_loss':0.1,'validation_loss':1.6,'grid_trial':0}]
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    res=analyze_overfitting(root,root/'diagnostics.csv')
    assert len(res)==1
    assert res[0]['validation_degradation_from_best']==pytest.approx(1.)
    assert res[0]['last_generalization_gap']==pytest.approx(1.5)
