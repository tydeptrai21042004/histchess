"""Resumable CPU/GPU HistChess neural + classical comparison pipeline.

Usage: python scripts/run_research_pipeline.py --profile pilot --out results
All generated scores originate in the executed models and validated chess pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROFILES={
    'pilot':{'models':['gru','causal_transformer'],'folds':[0], 'seeds':[3101],
             'cutoffs':['0','16'], 'epochs':2,'pairs':30,'target_parameters':150000},
    'full':{'models':['elman_rnn','lstm','gru','causal_transformer','linear_attention','diagonal_ssm'],
            'folds':[0,1,2], 'seeds':[3101,3102,3103,3104,3105],
            'cutoffs':['0','4','8','16','full'],'epochs':50,'pairs':300,'target_parameters':150000},
}

def call(label,command,logdir):
    print(f'\n==== {label} ====\n$ ' + ' '.join(map(str,command)),flush=True)
    with (logdir/(label+'.log')).open('w',encoding='utf8') as f:
        proc=subprocess.Popen(list(map(str,command)),cwd=ROOT,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,
             env=os.environ.copy())
        for line in proc.stdout:
            sys.stdout.write(line);sys.stdout.flush();f.write(line)
        rc=proc.wait()
    if rc: raise RuntimeError(f'{label} exit code {rc}; check {logdir/(label+".log")}')

def run_pipeline(argv=None):
    ap=argparse.ArgumentParser(description='Validated HistChess architecture and classical comparison')
    ap.add_argument('--profile',choices=tuple(PROFILES),default='pilot')
    ap.add_argument('--out',required=True)
    ap.add_argument('--long-target-plies',type=int,default=0,
                    help='EXPERIMENTAL: optional 64-ply stress; NOT in validated default')
    ap.add_argument('--pairs-jsonl',help='Optional existing dataset; checked via legal replay')
    ap.add_argument('--skip-tests',action='store_true',help='Skip only preflight tests; audits remain mandatory')
    ap.add_argument('--max-history',type=int,default=256)
    ap.add_argument('--epochs',type=int,help='Override max epochs')
    ap.add_argument('--pairs-per-task',type=int,help='Override new pairs per task')
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--max-new-runs',type=int,default=0,help='Pause after N new neural configurations; 0 means no run-count limit')
    ap.add_argument('--max-session-seconds',type=int,default=0,help='Pause between runs when time budget is reached')
    ap.add_argument('--patience',type=int,default=7,help='Validation-loss early stopping patience')
    args=ap.parse_args(argv)
    if args.long_target_plies not in (0,64):
        raise ValueError('Supported stress setting is 64 plies, or 0 for validated normal pipeline')
    cfg=dict(PROFILES[args.profile])
    if args.epochs is not None:cfg['epochs']=args.epochs
    if args.pairs_per_task is not None:cfg['pairs']=args.pairs_per_task
    if cfg['epochs']<1 or cfg['pairs']<3 or args.patience<1:raise ValueError('Invalid epoch, patience or pair count')
    import torch
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    logs=out/'pipeline_logs';logs.mkdir(exist_ok=True)
    status={'complete':False,'profile':args.profile,'stages':[],
            'settings':cfg,'long_target_plies':args.long_target_plies,
            'device':device,'patience':args.patience,
            'expected_neural_runs':len(cfg['models'])*len(cfg['folds'])*len(cfg['seeds'])*len(cfg['cutoffs']),
            'note':'Classical benchmark is same-pair, NOT external or pretrained.'}
    status_file=out/'pipeline_status.json'
    def persist():status_file.write_text(json.dumps(status,indent=2,default=str),encoding='utf8')
    persist()
    def step(label, cmd):
        start=time.monotonic()
        call(label,cmd,logs)
        status['stages'].append({'name':label,'success':True,'seconds':round(time.monotonic()-start,2)})
        persist()
    try:
        if not args.skip_tests:
            step('01_tests',[sys.executable,'-m','pytest','-q','tests',
                              '-m','not gpu_integration and not slow',
                              '--ignore=tests/test_length_stress.py'])
            if args.long_target_plies:
                step('02_stress_tests',[sys.executable,'-m','pytest','-q','tests/test_length_stress.py'])
        cmd=[sys.executable,'scripts/run_architecture_benchmark.py','--out',out,
             '--architectures',*cfg['models'],'--folds',*map(str,cfg['folds']),
             '--seeds',*map(str,cfg['seeds']),'--cutoffs',*cfg['cutoffs'],
             '--epochs',str(cfg['epochs']),'--patience',str(args.patience),'--batch-size','64',
             '--max-history',str(args.max_history),'--target-parameters',str(cfg['target_parameters']),
             '--device',device,'--position-encoding','sinusoidal',
             '--length-eval-cutoffs','16','32','64','full',
             '--length-eval-train-cutoffs','16','--generator-seed','3101']
        if args.max_new_runs:cmd+=['--max-new-runs',str(args.max_new_runs)]
        if args.max_session_seconds:cmd+=['--max-session-seconds',str(args.max_session_seconds)]
        if args.long_target_plies:cmd+=['--long-target-plies',str(args.long_target_plies)]
        if args.pairs_jsonl:cmd+=['--pairs-jsonl',args.pairs_jsonl]
        else:cmd+=['--pairs-per-task',str(cfg['pairs'])]
        step('03_neural',cmd)
        interim=json.loads((out/'architecture_status.json').read_text(encoding='utf8'))
        if not interim.get('complete'):
            if not interim.get('paused'):
                raise RuntimeError('Neural benchmark incomplete without a safe pause')
            status['paused']=True
            status['neural_runs_completed']=len(interim.get('runs',[]))
            status['remaining_runs']=status['expected_neural_runs']-status['neural_runs_completed']
            print('SAFE PAUSE:',status['neural_runs_completed'],'of',status['expected_neural_runs'],
                  'neural configurations. Attach output ZIP to next Kaggle session.',flush=True)
            return
        step('04_neural_audit',[sys.executable,'scripts/verify_architecture_results.py',out,
                                '--write',out/'independent_integrity_audit.json'])
        compare=[sys.executable,'scripts/run_comparison_baselines.py',
                 '--pairs-jsonl',out/'data/pairs.jsonl','--out',out,
                 '--folds',*map(str,cfg['folds']),'--cutoffs',*cfg['cutoffs'],
                 '--max-history',str(args.max_history)]
        step('05_classical_same_pairs',compare)
        step('06_classical_audit',[sys.executable,'scripts/verify_comparison_results.py',out,
                                   '--write',out/'comparison_integrity_audit.json'])
        step('07_matched_table',[sys.executable,'scripts/make_comparison_table.py',out])
        arch=json.loads((out/'architecture_status.json').read_text())
        cls=json.loads((out/'comparison_status.json').read_text())
        n=len(cfg['models'])*len(cfg['folds'])*len(cfg['seeds'])*len(cfg['cutoffs'])
        if not arch['complete'] or len(arch['runs'])!=n or not cls['complete']:
            raise RuntimeError('Completion manifest failed verification')
        for p in ['independent_integrity_audit.json','comparison_integrity_audit.json']:
            if not json.loads((out/p).read_text())['passed']:raise RuntimeError(f'Incomplete integrity audit: {p}')
        if args.profile=='full':
            step('08_manuscript_tables',[sys.executable,'scripts/make_manuscript_tables.py',out])
        status['complete']=True;status['neural_runs']=n
        print('ALL STAGES AND INDEPENDENT AUDITS COMPLETE:',n,'neural runs',flush=True)
    except BaseException as exc:
        status['error']=repr(exc);status['traceback']=traceback.format_exc()
        raise
    finally:
        persist()

if __name__=='__main__':run_pipeline()
