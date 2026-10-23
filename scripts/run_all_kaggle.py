#!/usr/bin/env python3
"""End-to-end reproducible HistChess Kaggle GPU paper pipeline.

Every required experiment is fail-loud; a partial output archive is still saved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histchess.core import TASKS,generate,validate,write_jsonl
from histchess.experiments import baseline_experiments, leakage_audit, llm_experiment, save_json
from histchess.gpu_experiments import history_experiment
from histchess.paper import paper_statistics, paper_tables, make_figures
from histchess.audit import audit_final_predictions
from histchess.shortcuts import shortcut_experiments


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='/kaggle/working/histchess_results' if Path('/kaggle/working').exists() else 'histchess_results')
    ap.add_argument('--pairs-per-task',type=int,default=240)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--llm-batch-size',type=int,default=8)
    ap.add_argument('--seed',type=int,default=3101)
    ap.add_argument('--train-seeds',type=int,default=2)
    ap.add_argument('--llm-pairs-per-task',type=int,default=20)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--max-history',type=int,default=128)
    ap.add_argument('--generator',choices=('diverse','legacy'),default='diverse',
                    help='diverse = 3 mechanism families per task; legacy = original release')
    ap.add_argument('--split-strategy',choices=('family','board'),default=None,
                    help='family holds out entire mechanisms; board holds out positions only')
    ap.add_argument('--models',nargs='+',default=['HuggingFaceTB/SmolLM2-135M-Instruct',
                                                'HuggingFaceTB/SmolLM2-360M-Instruct'])
    ap.add_argument('--smoke',action='store_true',help='Fast sanity run only. Not paper complete.')
    ap.add_argument('--skip-llm',action='store_true',help='Skip pretrained comparisons; NOT full paper.')
    ap.add_argument('--allow-cpu',action='store_true',help='Only for local debugging; full Kaggle mode requires GPU.')
    args=ap.parse_args()
    if args.split_strategy is None:
        args.split_strategy = 'family' if args.generator == 'diverse' else 'board'
    if args.generator == 'legacy' and args.split_strategy != 'board':
        ap.error('Legacy generator has no independent families; use --split-strategy board')
    if args.smoke:
        args.pairs_per_task=40
        args.epochs=2
        args.train_seeds=1
        args.llm_pairs_per_task=2
        args.bootstrap=200
    if args.train_seeds<1 or args.pairs_per_task<10 or args.llm_pairs_per_task<1:
        ap.error('Invalid repetitions or dataset size')
    out=Path(args.out).resolve()
    if out==Path('/') or out==Path.home().resolve():
        ap.error('Unsafe output directory')
    if out.exists():
        print('Removing previous result directory to prevent stale-result contamination:',out,flush=True)
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    gpu=torch.cuda.get_device_name(0) if device=='cuda' else None
    status={"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "python":sys.version,"platform":platform.platform(),
            "torch":torch.__version__,"cuda_version":torch.version.cuda,
            "device":device,"gpu":gpu,"parameters":vars(args),
            "experiments":{},"completed":False,"paper_complete":False}
    print('HistChess device:',device,'GPU:',gpu,flush=True)
    def step(name, fn):
        tick=time.time();print(f'\n=== {name} ===',flush=True)
        try:
            result=fn()
            status['experiments'][name]={"status":"PASS","seconds":round(time.time()-tick,2)}
            save_json(out/'run_status.json',status)
            return result
        except Exception as exc:
            status['experiments'][name]={"status":"FAILED","seconds":round(time.time()-tick,2),
                                         "error":repr(exc),"traceback":traceback.format_exc()}
            save_json(out/'run_status.json',status)
            print(traceback.format_exc(),flush=True)
            raise
    required=['environment_preflight','dataset_generation','independent_replay_validation',
              'board_FEN_and_oracle_controls','leakage_audit','shortcut_baselines',
              'gpu_train_and_retrained_history_ablations',
              'pretrained_llm_comparison','paired_statistical_analysis','paper_tables_and_figures','final_output_integrity_audit']
    try:
        def preflight():
            if not args.allow_cpu and device!='cuda':
                raise RuntimeError('No CUDA GPU. In Kaggle notebook Settings choose Accelerator GPU (T4/P100).')
            if args.skip_llm and not args.smoke:
                print('Explicit --skip-llm: this run cannot be marked complete.',flush=True)
            if not args.skip_llm:
                import transformers
                print('Transformers:',transformers.__version__,flush=True)
            import chess
            print('python-chess:',chess.__version__,flush=True)
            torch.manual_seed(args.seed)
            if device=='cuda': torch.cuda.manual_seed_all(args.seed)
            random.seed(args.seed)
            return True
        step('environment_preflight',preflight)
        pairs,rejections=step('dataset_generation',lambda:generate(
            args.pairs_per_task,args.seed,generator=args.generator,
            split_strategy=args.split_strategy))
        write_jsonl(out/'data'/'pairs.jsonl',pairs)
        save_json(out/'data'/'generation_rejections.json',rejections)
        validation=step('independent_replay_validation',lambda:validate(pairs))
        save_json(out/'validation.json',validation)
        if not validation['passed']:
            raise RuntimeError('Rule replay/certificate validation failed')
        task_split={t:{s:sum(p['task']==t and p['split']==s for p in pairs)
                       for s in ('train','validation','test')} for t in TASKS}
        save_json(out/'dataset_split_summary.json',task_split)
        if any(min(cnt.values()) == 0 for cnt in task_split.values()):
            raise RuntimeError(f'An empty task split: {task_split}')
        baseline=step('board_FEN_and_oracle_controls',lambda:baseline_experiments(pairs,out))
        audit=step('leakage_audit',lambda:leakage_audit(pairs))
        save_json(out/'leakage_audit.json',audit)
        bad=[k for k in ('board_group_leakage_count','full_history_leakage_count',
                        'full_FEN_across_split_count') if audit[k]]
        if bad: raise RuntimeError(f'Cross-split leakage detected: {bad}')
        if args.split_strategy == 'family' and audit['generator_family_across_split_count']:
            raise RuntimeError('Generator family leakage across splits')
        step('shortcut_baselines',lambda:shortcut_experiments(pairs,out))
        seeds=[args.seed+i for i in range(args.train_seeds)]
        step('gpu_train_and_retrained_history_ablations', lambda:history_experiment(
            pairs,out,seeds=seeds,cutoffs=(0,4,8,16,32,None),epochs=args.epochs,
            patience=args.patience,batch_size=args.batch_size,
            max_history=args.max_history,require_gpu=not args.allow_cpu))
        if args.skip_llm:
            status['experiments']['pretrained_llm_comparison']={"status":"SKIPPED_EXPLICITLY"}
        else:
            step('pretrained_llm_comparison',lambda:llm_experiment(
                pairs,out,models=args.models,pairs_per_task=args.llm_pairs_per_task,
                seed=args.seed,batch_size=args.llm_batch_size,
                require_gpu=not args.allow_cpu))
        step('paired_statistical_analysis',lambda:paper_statistics(
            pairs,out,bootstrap=args.bootstrap,seed=args.seed))
        step('paper_tables_and_figures',lambda:(paper_tables(out),make_figures(out)))
        final_audit=step('final_output_integrity_audit',lambda:audit_final_predictions(
            pairs,out,seeds,(0,4,8,16,32,None),llm_models=args.models,
            llm_pairs_per_task=args.llm_pairs_per_task,skip_llm=args.skip_llm))
        save_json(out/'final_output_audit.json',final_audit)
        status['completed']=all(status['experiments'].get(x,{}).get('status')=='PASS'
                                for x in required)
        # 'completed' means the code ran; these tests do NOT establish the
        # validity of generalization claims or reproduce an unseen manuscript.
        # Compatibility with the existing artifact verifier: means only that
        # this repository's requested stages completed, NOT journal readiness.
        status['paper_complete']=status['completed'] and not args.smoke
        status['claim_scope']='synthetic_generator_family_holdout' if args.split_strategy=='family' else 'synthetic_board_group_holdout'
        status['submission_ready_assessed']=False
        return 0 if (status['completed'] or args.smoke or args.skip_llm) else 1
    except Exception:
        return 1
    finally:
        status['total_seconds']=round(time.time()-start,2)
        status['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(out/'run_status.json',status)
        data_path=out/'data'/'pairs.jsonl'
        save_json(out/'manifest.json',{
            'version':'HistChess diversified synthetic pipeline v3',
            'status':'COMPLETE' if status['completed'] else 'INCOMPLETE',
            'gpu':gpu,'parameters':vars(args),
            'pairs_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.exists() else None,
            'paper_complete':status['paper_complete'],
            'limitations':['All samples remain synthetic and must not be described as independent real games.',
                           'Generator-family holdout tests three defined families per task, not general chess understanding.',
                           'LLMs are evaluated on a fixed subset, not necessarily the full GRU test set.',
                           'Full FEN is privileged information and is not a fair history-only comparator.',
                           'Successful execution does not certify journal readiness.']})
        archive=out.parent/(out.name+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():z.write(path,arcname=str(Path(out.name)/path.relative_to(out)))
        print('\nRESULT ARCHIVE:',archive,flush=True)
        print('ALL PIPELINE EXPERIMENTS COMPLETE:',status['completed'],flush=True)
        print('Check run_status.json; incomplete steps are NOT reported as completed.',flush=True)

if __name__=='__main__':sys.exit(main())
