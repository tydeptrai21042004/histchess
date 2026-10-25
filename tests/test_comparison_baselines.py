"""Classical baseline tests, no python-chess needed for generated fixtures."""
import csv
import json

import pytest

from histchess.comparison_baselines import BASELINES, features, run
from histchess.architecture_benchmark import FAMILIES, TASKS


def fixture_pairs(n=2):
    pairs=[]
    for i,task in enumerate(TASKS):
        for j,family in enumerate(FAMILIES[task]):
            for k in range(n):
                # Unique valid board encodings and paired observations.
                square=i*12+j*n+k
                if square>=64:raise AssertionError('Test fixture too big')
                squares=['-']*64;squares[square]='P'
                ranks=[]
                for rank in range(8):
                    out='';blank=0
                    for piece in squares[rank*8:(rank+1)*8]:
                        if piece=='-':blank+=1
                        else:
                            if blank:out+=str(blank);blank=0
                            out+=piece
                    if blank:out+=str(blank)
                    ranks.append(out)
                board='/'.join(ranks)+' w'
                pairs.append({'pair_id':f'p{i}-{j}-{k}', 'task':task,
                    'board_key':board,'generator_family':family,'split':('train','validation','test')[j],
                    'positive':{'board_observation':board,'label':1,'history_uci':['g1f3','g8f6','b1c3']},
                    'negative':{'board_observation':board,'label':0,'history_uci':['b1c3','g8f6','g1f3']}})
    return pairs


def test_all_classical_comparisons_complete(tmp_path):
    pairs=fixture_pairs()
    metrics, pred=run(pairs,tmp_path,folds=(0,1,2),cutoffs=(0,3,None),max_history=16)
    assert len(metrics)==3*3*len(BASELINES)*5
    assert len(pred)==3*3*len(BASELINES)*2*8
    for row in metrics:
        if row['baseline']=='privileged_replay_oracle':
            assert row['record_accuracy']==row['joint_pair_accuracy']==1
        if row['history_plies']=='0' and row['baseline']!='privileged_replay_oracle':
            assert row['record_accuracy']==.5 and row['joint_pair_accuracy']==0
    status=json.loads((tmp_path/'comparison_status.json').read_text())
    assert status['complete'] and status['prediction_rows']==len(pred)
    again=run(pairs,tmp_path,folds=(0,1,2),cutoffs=(0,3,None),max_history=16)
    assert again==(metrics,pred)
    changed=[dict(x) for x in pairs]
    changed[0]=dict(changed[0]);changed[0]['positive']=dict(changed[0]['positive']);changed[0]['positive']['history_uci']=['g1f3','g8f6','a2a4']
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        run(changed,tmp_path,folds=(0,1,2),cutoffs=(0,3,None),max_history=16)


def test_order_sensitivity_and_no_private_fen():
    a={'task':'repetition','board_key':'8/8/8/8/8/8/8/8 w',
       'history_uci':['g1f3','g8f6','f3g1'],'fen':'private', 'label':1}
    b={**a,'history_uci':list(reversed(a['history_uci']))}
    assert features(a,'move_bag',None,32)==features(b,'move_bag',None,32)
    assert features(a,'move_ngrams',None,32)!=features(b,'move_ngrams',None,32)
    assert 'private' not in str(features(a,'move_ngrams',None,32))
