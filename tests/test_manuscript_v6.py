"""Tests only publication formatting and fail-closed behavior, not scientific scores."""
import csv
import json
import pytest
from scripts.make_manuscript_tables import generate


def test_manuscript_requires_complete_audited_protocol(tmp_path):
    (tmp_path/'architecture_status.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(FileNotFoundError):generate(tmp_path)


def test_manuscript_table_formatter_with_synthetic_fixture(tmp_path):
    # The fixture is artificial by design; it does NOT represent experiment results.
    models=('elman_rnn','lstm','gru','causal_transformer','linear_attention','diagonal_ssm')
    histories=('0','4','8','16','full')
    config={'models':{m:{} for m in models},'folds':[0,1,2],
        'seeds':[3101,3102,3103,3104,3105],'cutoffs':list(histories),
        'protocol_version':'histchess-v6-publication-full-2026-10',
        'train_protocol':{'device':'cpu','epochs':1}}
    runs=[{'model':m,'fold':f,'seed':sd,'history':h} for m in models for f in range(3)
          for sd in config['seeds'] for h in histories]
    (tmp_path/'architecture_status.json').write_text(json.dumps({'complete':True,'runs':runs}))
    (tmp_path/'pipeline_status.json').write_text(json.dumps({'complete':False,'paused':False}))
    (tmp_path/'comparison_status.json').write_text(json.dumps({'complete':True}))
    (tmp_path/'architecture_config.json').write_text(json.dumps(config))
    for name in ('independent_integrity_audit.json','comparison_integrity_audit.json'):
        (tmp_path/name).write_text(json.dumps({'passed':True}))
    neural=[]
    for x in runs:
        for task in ('all','castling','en_passant','halfmove','repetition'):
            neural.append({'architecture':x['model'],'history_plies':x['history'],
                'fold':x['fold'],'seed':x['seed'],'task':task,'record_accuracy':0.5,
                'joint_pair_accuracy':0,'pair_ranking_accuracy':0.5})
    def writecsv(name, rows):
        with (tmp_path/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    writecsv('architecture_metrics.csv',neural)
    cls=[{'baseline':m,'history_plies':h,'task':'all','fold':f,'record_accuracy':.5,
          'joint_pair_accuracy':0} for m in ('board_only','move_bag','move_ngrams','privileged_replay_oracle')
         for h in histories for f in range(3)]
    writecsv('comparison_metrics.csv',cls)
    folder=tmp_path/'data';folder.mkdir()
    pairs=[{'pair_id':f'p{i}', 'task':task,'generator_family':'toy',
            'positive':{'history_uci':['a2a4']},'negative':{'history_uci':['b2b4']}}
           for i,task in enumerate(('castling','en_passant','halfmove','repetition'))]
    (folder/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    output=generate(tmp_path)
    assert output['architecture_runs']==450
    assert (tmp_path/'manuscript_neural_table.tex').is_file()
    assert (tmp_path/'manuscript_methods_manifest.json').is_file()
    assert len((tmp_path/'manuscript_task_breakdown.csv').read_text().splitlines())==6*5*5+1
    assert 'Causal Transformer' in (tmp_path/'manuscript_results.md').read_text()
