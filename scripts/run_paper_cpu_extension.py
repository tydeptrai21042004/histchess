#!/usr/bin/env python3
"""HistChess v6 publication companion: CPU-only, NO NEURAL TRAINING.

Requires the *actual completed* v6 result folder. Preserves it unchanged.
Trains only small sklearn logistic-regression diagnostic controls on its existing pairs.
Runs exact python-chess legal replay as an algorithmic full-history reference.
All numerical values are computed from saved predictions; no fabricated values.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, statistics
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

TASKS=('castling','en_passant','halfmove','repetition')
FAMILIES={
'castling':('rook_g1_return','king_e2_return','rook_h2_return'),
'en_passant':('quiet_vs_double','two_black_doubles','black_captures_white_double'),
'halfmove':('both_pawns_last','white_pawn_penultimate','black_pawn_last'),
'repetition':('legacy_two_cycles','interleaved_reverse','interleaved_forward'),
}
CUTS=('0','4','8','16','full')
METHODS=('suffix_1','suffix_2','suffix_4','prefix_4')
FIXED='legacy_fixed_signatures'

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path, rows):
    if not rows: raise ValueError('No rows to write: '+str(path))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def visible(history, cutoff, max_history):
    s=list(history)[-max_history:]
    return s if cutoff=='full' else s[-int(cutoff):] if int(cutoff)>0 else []
def board_features(key):
    placement, side=key.split()
    board=[]
    for ch in placement.replace('/',''):
        if ch.isdigit():board.extend(['-']*int(ch))
        else:board.append(ch)
    if len(board)!=64 or side not in ('w','b'): raise ValueError('Invalid observable board: '+key)
    return {**{f'square_{i}={piece}':1 for i,piece in enumerate(board) if piece!='-'},'turn='+side:1}
def features(row,method,cutoff,max_history):
    seq=visible(row['history_uci'],cutoff,max_history)
    selected=seq[-int(method.split('_')[1]):] if method.startswith('suffix') else seq[:int(method.split('_')[1])]
    v={'task='+row['task']:1,**board_features(row['board_key'])}
    for i,token in enumerate(selected):
        position= len(selected)-1-i if method.startswith('suffix') else i
        v[f'{method}_{position}={token}']=1
    return v

def split_rows(pairs,fold):
    outputs={'train':[],'validation':[],'test':[]}
    for p in pairs:
        fam=FAMILIES[p['task']]
        if p['generator_family'] not in fam:raise ValueError('Unexpected family '+p['generator_family'])
        which=('train','validation','test')[(fam.index(p['generator_family'])-fold)%3]
        for variant in ('positive','negative'):
            r=p[variant]
            outputs[which].append({'pair_id':p['pair_id'],'variant':variant,'task':p['task'],
                'family':p['generator_family'],'board_key':p['board_key'],
                'history_uci':r['history_uci'],'label':int(r['label'])})
    return outputs

def fit_baseline(split,method,cutoff,max_history):
    test=split['test']
    if cutoff=='0':return [0.5]*len(test), {'C':'NA','validation_logloss':math.log(2)}
    vec=DictVectorizer(sparse=True)
    x=vec.fit_transform([features(r,method,cutoff,max_history) for r in split['train']])
    xv=vec.transform([features(r,method,cutoff,max_history) for r in split['validation']])
    xt=vec.transform([features(r,method,cutoff,max_history) for r in test])
    y=np.asarray([r['label'] for r in split['train']]); yv=np.asarray([r['label'] for r in split['validation']])
    if set(y)!={0,1} or set(yv)!={0,1}:raise RuntimeError('Both labels required in train/validation')
    chosen=None
    for c in (0.1,1.0,10.0):
        model=LogisticRegression(C=c,solver='liblinear',max_iter=1000,random_state=3101)
        model.fit(x,y)
        loss=float(log_loss(yv,model.predict_proba(xv)[:,1],labels=[0,1]))
        if chosen is None or loss<chosen[0]-1e-12:chosen=(loss,c,model)
    return chosen[2].predict_proba(xt)[:,1].tolist(),{'C':chosen[1],
        'validation_logloss':chosen[0], 'train_features':x.shape[1]}

def verify_replay(pairs):
    try:import chess
    except ImportError as e:raise RuntimeError('Install python-chess (pip install chess==1.11.2)') from e
    result={};bad=[]
    for p in pairs:
        labels={}
        for variant in ('positive','negative'):
            b=chess.Board()
            for index,code in enumerate(p[variant]['history_uci'],1):
                mv=chess.Move.from_uci(code)
                if mv not in b.legal_moves:raise RuntimeError(f'Illegal history {p["pair_id"]}/{variant} ply {index}: {code}')
                b.push(mv)
            obs=b.board_fen()+' '+('w' if b.turn else 'b')
            if obs!=p['board_key']:raise RuntimeError('Board mismatch '+p['pair_id'])
            task=p['task']
            value={'castling':lambda:b.has_kingside_castling_rights(chess.WHITE),
                   'en_passant':lambda:b.has_legal_en_passant(),
                   'halfmove':lambda:b.halfmove_clock>=2,
                   'repetition':lambda:b.is_repetition(3)}[task]()
            labels[variant]=int(value)
            if int(value)!=int(p[variant]['label']):bad.append((p['pair_id'],variant))
            result[(p['pair_id'],variant)]=float(value)
        if labels!={'positive':1,'negative':0}:raise RuntimeError('Counterfactual label mismatch '+p['pair_id'])
    if bad:raise RuntimeError('Rule-replay label mismatch: '+str(bad[:5]))
    return result

def metric_rows(records,method,fold,cutoff):
    out=[]
    for task in ('all',)+TASKS:
        sub=[r for r in records if task=='all' or r['task']==task]
        if not sub:raise RuntimeError('Missing task '+task)
        pairs=collections.defaultdict(dict)
        for r in sub:
            k=r['pair_id'];v=r['variant']
            if v in pairs[k]:raise RuntimeError('duplicate pair prediction')
            pairs[k][v]=r
        for pp in pairs.values():
            if set(pp)!={'positive','negative'}:raise RuntimeError('Incomplete pair')
        accuracy=sum(int(r['prediction']==r['label']) for r in sub)/len(sub)
        joint=sum(int(pp['positive']['prediction']==1 and pp['negative']['prediction']==0) for pp in pairs.values())/len(pairs)
        ranking=sum((1 if pp['positive']['score']>pp['negative']['score'] else
                     .5 if pp['positive']['score']==pp['negative']['score'] else 0) for pp in pairs.values())/len(pairs)
        out.append({'baseline':method,'fold':fold,'history_plies':cutoff,'task':task,
            'test_pairs':len(pairs),'test_records':len(sub),
            'record_accuracy':accuracy,'joint_pair_accuracy':joint,'pair_ranking_accuracy':ranking})
    return out

def original_fixed_shortcut(task, history):
    """Fixed, documented pre-v6 template signatures (NOT tuned on this test set)."""
    if task=='castling':return int('h1g1' not in history)
    if task=='en_passant':return int(bool(history) and history[-1]=='d7d5')
    if task=='halfmove':return int(bool(history) and history[-1]=='f6g8')
    if task=='repetition':return int(len(history)>=4 and history[-4]=='g1f3')
    raise ValueError(task)

def train_controls(pairs, replay, max_history):
    preds=[];metrics=[];tuning=[]
    for fold in (0,1,2):
        splits=split_rows(pairs,fold)
        if any(not v for v in splits.values()):raise RuntimeError('Incomplete fold')
        for cutoff in CUTS:
            for method in METHODS:
                probs,meta=fit_baseline(splits,method,cutoff,max_history)
                rows=[]
                for x,p in zip(splits['test'],probs):
                    r={'baseline':method,'fold':fold,'history_plies':cutoff,
                        'pair_id':x['pair_id'],'variant':x['variant'],'task':x['task'],'family':x['family'],
                        'label':x['label'],'prediction':int(p>=.5),'score':float(p)}
                    rows.append(r)
                preds.extend(rows);metrics.extend(metric_rows(rows,method,fold,cutoff))
                tuning.append({'baseline':method,'fold':fold,'history_plies':cutoff,**meta})
        fixed=[]
        for x in splits['test']:
            guess=original_fixed_shortcut(x['task'],x['history_uci'][-max_history:])
            fixed.append({'baseline':FIXED,'fold':fold,'history_plies':'full',
                'pair_id':x['pair_id'],'variant':x['variant'],'task':x['task'],'family':x['family'],
                'label':x['label'],'prediction':guess,'score':float(guess)})
        preds.extend(fixed);metrics.extend(metric_rows(fixed,FIXED,fold,'full'))
        method='legal_replay_full'
        rows=[]
        for x in splits['test']:
            p=replay[x['pair_id'],x['variant']]
            rows.append({'baseline':method,'fold':fold,'history_plies':'full',
                'pair_id':x['pair_id'],'variant':x['variant'],'task':x['task'],'family':x['family'],
                'label':x['label'],'prediction':int(p>=.5),'score':p})
        preds.extend(rows);metrics.extend(metric_rows(rows,method,fold,'full'))
    return preds,metrics,tuning

def describe(pairs):
    rows=[]
    for p in pairs:
        a=p['positive']['history_uci'];b=p['negative']['history_uci']
        prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),len(a))
        suffix=next((i for i,(x,y) in enumerate(zip(reversed(a),reversed(b))) if x!=y),len(a))
        rows.append({'pair_id':p['pair_id'],'task':p['task'],'family':p['generator_family'],
          'history_length':len(a),'shared_prefix':prefix,'shared_suffix':suffix,
          'same_move_bag':int(collections.Counter(a)==collections.Counter(b)),
          'last_move_equal':int(a[-1]==b[-1])})
    return rows

def make_figures(root,out,all_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Use publication-readable plots that never re-estimate neural scores.
    neural=read_csv(root/'architecture_metrics.csv')
    old=read_csv(root/'comparison_metrics.csv')
    rank={x:i for i,x in enumerate(CUTS)}
    fig,ax=plt.subplots(figsize=(8.3,4.8))
    models=sorted({r['architecture'] for r in neural})
    for model in models:
        values=[]
        for context in CUTS:
            v=[float(r['joint_pair_accuracy']) for r in neural if r['architecture']==model and r['history_plies']==context and r['task']=='all']
            if not v:raise RuntimeError('Missing neural results for '+model+'/'+context)
            values.append(statistics.mean(v))
        ax.plot(range(len(CUTS)),values,marker='o',markersize=3,label=model.replace('_',' '),lw=1.35)
    for method,style in [('suffix_1','--'),('suffix_4',':')]:
        vals=[statistics.mean(float(r['joint_pair_accuracy']) for r in all_metrics if r['baseline']==method and r['history_plies']==c and r['task']=='all') for c in CUTS]
        ax.plot(range(len(CUTS)),vals,linestyle=style,linewidth=2,label=method.replace('_',' '))
    ax.set_xticks(range(len(CUTS)),CUTS);ax.set_xlabel('Allowed recent history (plies)')
    ax.set_ylabel('Strict joint-pair accuracy');ax.set_ylim(-.03,1.03)
    ax.set_title('History visibility and pairwise success (descriptive)')
    ax.grid(alpha=.16);ax.legend(fontsize=7,ncol=2,loc='best')
    fig.tight_layout();fig.savefig(out/'figure_1_joint_accuracy.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True)
    for ax,task in zip(axes.flat,TASKS):
        names=['suffix_1','suffix_2','suffix_4','prefix_4','move_bag','move_ngrams',FIXED]
        scores=[]
        for n in names:
            source=old if n in ('move_bag','move_ngrams') else all_metrics
            subset=[float(x['joint_pair_accuracy']) for x in source if x['baseline']==n and x['history_plies']=='full' and x['task']==task]
            if not subset:raise RuntimeError('Missing '+n+'/'+task)
            scores.append(statistics.mean(subset))
        ax.bar(range(len(names)),scores)
        ax.set_xticks(range(len(names)),['last1','last2','last4','first4','bag','n-gram','fixed'],rotation=25)
        ax.set_ylim(0,1.02);ax.set_title(task.replace('_',' '));ax.grid(axis='y',alpha=.14)
    fig.supylabel('Strict joint-pair accuracy');fig.suptitle('How far can simple history shortcuts go?')
    fig.tight_layout();fig.savefig(out/'figure_2_shortcut_diagnostics.pdf',bbox_inches='tight');plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True,help='EXISTING fully completed v6 output directory')
    p.add_argument('--out',required=True,help='SEPARATE new folder, never use --results')
    args=p.parse_args(argv)
    root=Path(args.results).resolve();out=Path(args.out).resolve()
    if out==root or root in out.parents:raise RuntimeError('Output must be OUTSIDE original results directory')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output dir must be new/empty to avoid mixing results')
    required=['pipeline_status.json','architecture_status.json','architecture_config.json',
        'comparison_status.json','independent_integrity_audit.json','comparison_integrity_audit.json',
        'data/pairs.jsonl','architecture_metrics.csv','architecture_predictions.csv','comparison_metrics.csv']
    for f in required:
        if not (root/f).is_file():raise RuntimeError('Required original result missing: '+f)
    for f in ('pipeline_status.json','architecture_status.json','comparison_status.json'):
        if not json.loads((root/f).read_text()).get('complete'):raise RuntimeError('Existing v6 output incomplete: '+f)
    for f in ('independent_integrity_audit.json','comparison_integrity_audit.json'):
        if not json.loads((root/f).read_text()).get('passed'):raise RuntimeError('Existing v6 audit failed: '+f)
    cfg=json.loads((root/'architecture_config.json').read_text());max_history=int(cfg['max_history'])
    if cfg['protocol_version']!='histchess-v6-publication-full-2026-10':
        raise RuntimeError('Wrong experiment protocol: required HistChess v6')
    pairs=[json.loads(line) for line in (root/'data/pairs.jsonl').read_text().splitlines() if line.strip()]
    if len(pairs)!=1200 or len({x['pair_id'] for x in pairs})!=len(pairs):
        raise RuntimeError('Expected 1200 unique v6 pairs')
    if any(p['positive']['board_observation']!=p['negative']['board_observation'] or p['board_key']!=p['positive']['board_observation'] for p in pairs):
        raise RuntimeError('Pair board mismatch')
    print('Verifying all histories by legal replay...',flush=True)
    replay=verify_replay(pairs)
    print('Training small CPU baselines (no Torch, no GPU)...',flush=True)
    pred,metrics,tuning=train_controls(pairs,replay,max_history)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'cpu_baseline_predictions.csv',pred)
    write_csv(out/'cpu_baseline_metrics.csv',metrics)
    write_csv(out/'cpu_baseline_validation_selection.csv',tuning)
    detail=describe(pairs)
    write_csv(out/'dataset_pair_diagnostics.csv',detail)
    detail_summary=[]
    for task in ('all',)+TASKS:
        sub=[r for r in detail if task=='all' or r['task']==task]
        detail_summary.append({'task':task,'pairs':len(sub),'history_min':min(x['history_length'] for x in sub),
           'history_median':statistics.median(x['history_length'] for x in sub),
           'history_max':max(x['history_length'] for x in sub),
           'same_move_bag_fraction':statistics.mean(x['same_move_bag'] for x in sub),
           'identical_last_move_fraction':statistics.mean(x['last_move_equal'] for x in sub),
           'shared_suffix_median':statistics.median(x['shared_suffix'] for x in sub)})
    write_csv(out/'dataset_summary_by_task.csv',detail_summary)
    summary=[]
    for method in METHODS+(FIXED,'legal_replay_full'):
        for task in ('all',)+TASKS:
            vals=[r for r in metrics if r['baseline']==method and r['task']==task and r['history_plies']=='full']
            summary.append({'baseline':method,'task':task,'folds':len(vals),
              'mean_record_accuracy':statistics.mean(float(r['record_accuracy']) for r in vals),
              'mean_joint_pair_accuracy':statistics.mean(float(r['joint_pair_accuracy']) for r in vals),
              'interpretation':'Three correlated held-out family rotations; not an external-validity CI.'})
    write_csv(out/'cpu_baseline_summary.csv',summary)
    make_figures(root,out,metrics)
    manifest={'source_results':str(root),'pair_count':len(pairs),'all_legal_replay_labels_verified':True,
      'original_prediction_sha256':sha256(root/'architecture_predictions.csv'),
      'original_pairs_sha256':sha256(root/'data/pairs.jsonl'),
      'original_architecture_metrics_sha256':sha256(root/'architecture_metrics.csv'),
      'CPU_only':True,'neural_training_runs_added':0,'new_cpu_baselines':list(METHODS)+[FIXED],
      'rule_replay':'Full-history deterministic legal reconstruction; NOT a learned model or LLM.',
      'note':'This extension is based on the existing v6 test set; it is not an external dataset. Do not treat folds as independent.'}
    (out/'paper_cpu_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('CPU PAPER EXTENSION COMPLETE:',out,flush=True)
    print('Prediction rows:',len(pred),'metric rows:',len(metrics),flush=True)

if __name__=='__main__':main()
