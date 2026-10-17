#!/usr/bin/env python3
"""One-command, fail-loudly HistChess reproduction pipeline for Kaggle CPU."""
import argparse
import json
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histchess.core import generate,write_jsonl,validate
from histchess.experiments import baseline_experiments,leakage_audit,llm_experiment,save_json,sequence_experiment,write_csv


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/kaggle/working/histchess_results' if Path('/kaggle/working').exists() else 'histchess_results')
    ap.add_argument('--pairs-per-task',type=int,default=80)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=3101)
    ap.add_argument('--llm-pairs-per-task',type=int,default=2)
    ap.add_argument('--models',nargs='*',default=['HuggingFaceTB/SmolLM2-135M-Instruct','HuggingFaceTB/SmolLM2-360M-Instruct'])
    ap.add_argument('--skip-llm',action='store_true',help='Smoke testing only; NOT all 8 experiments')
    ap.add_argument('--strict',action='store_true',help='Exit nonzero if ANY requested experiment fails')
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    status={"started_utc":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            "python":sys.version,"platform":platform.platform(),"parameters":vars(args),"experiments":{},"completed":False}
    def step(name, fn):
        ts=time.time();print(f'\n=== {name} ===',flush=True)
        try:
            result=fn()
            status['experiments'][name]={"status":"PASS","seconds":round(time.time()-ts,2)}
            save_json(out/'run_status.json',status)
            return result
        except Exception as exc:
            status['experiments'][name]={"status":"FAILED","seconds":round(time.time()-ts,2),"error":repr(exc),"traceback":traceback.format_exc()}
            save_json(out/'run_status.json',status)
            print(traceback.format_exc(),flush=True)
            if args.strict:
                raise
            return None
    try:
        pairs=step('dataset_generation', lambda: generate(args.pairs_per_task,args.seed))
        if pairs is None:
            return 2
        pairs,rejections=pairs
        write_jsonl(out/'data'/'pairs.jsonl',pairs)
        save_json(out/'data'/'generation_rejections.json',rejections)
        validation=step('full_history_rule_oracle_and_validation',lambda: validate(pairs))
        if not validation or not validation['passed']:
            raise RuntimeError('Failed dataset validation; no experimental results will be accepted')
        save_json(out/'validation.json',validation)
        def baseline():
            return baseline_experiments(pairs,out)
        base=step('board_only_and_full_FEN_baselines',baseline)
        if base is None:
            raise RuntimeError('Failed the required baselines')
        audit=step('leakage_and_duplicate_audit',lambda: leakage_audit(pairs))
        if audit is None or audit['board_group_leakage_count'] or audit['full_history_leakage_count'] or audit['full_FEN_across_split_count']:
            raise RuntimeError('Leakage detected or audit failed')
        save_json(out/'leakage_audit.json',audit)
        grus=step('small_sequence_model_and_history_ablation',lambda: sequence_experiment(pairs,out,args.epochs,seed=args.seed))
        if grus is not None:
            write_csv(out/'gru_metrics.csv',grus)
        if args.skip_llm:
            status['experiments']['two_pretrained_LLMs']={"status":"SKIPPED_BY_USER","note":"Core workflow only; all-eight requirements NOT fulfilled"}
        else:
            lm=step('two_pretrained_LLMs',lambda: llm_experiment(pairs,out,args.models,args.llm_pairs_per_task,args.seed))
        # Build compact reproducible summary with no fabricated metrics.
        required=['dataset_generation','full_history_rule_oracle_and_validation','board_only_and_full_FEN_baselines','leakage_and_duplicate_audit','small_sequence_model_and_history_ablation','two_pretrained_LLMs']
        status['completed']=all(status['experiments'].get(name,{}).get('status')=='PASS' for name in required)
        status['total_seconds']=round(time.time()-started,2)
        save_json(out/'run_status.json',status)
        save_json(out/'manifest.json',{"required_steps":required,"all_pass":status['completed'],"pairs":len(pairs),"records":2*len(pairs),"notes":["No LLM result means no claim of completing all experiments.","LLM diagnostics use a small balanced sample; increase --llm-pairs-per-task for publishable model comparisons.","Board-only and FEN are symbolic feature-availability baselines, not trained language models."]})
        print('\nALL REQUESTED EXPERIMENTS COMPLETE:',status['completed'],flush=True)
        return 0 if status['completed'] or (args.skip_llm and grus is not None) else 1
    finally:
        # Always archive results (including failures) rather than producing oversized caches.
        archive=out.parent/(out.name+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    z.write(path,arcname=str(Path(out.name)/path.relative_to(out)))
        print('RESULT_ARCHIVE:',archive,flush=True)

if __name__=='__main__':
    sys.exit(main())
