#!/usr/bin/env python3
"""Resumable HistChess V8: length stress + validation tuning + independent-seed replication.

This is a new SYNTHETIC experimental protocol. It never modifies completed
legacy v6/v7 folders. Mandatory chess replay validation gates every stage.

Example (GPU): python scripts/run_v8_research.py --out v8_output \\
    --replications 5 --per-family 8 --levels native 32 64 \\
    --architectures elman_original lstm gru_original causal_transformer_postnorm \\
                     linear_attention_multihead s4d_lin \\
    --cutoffs 0 16 32 64 full --grid-lrs .0003 .001 \\
    --grid-dropouts .1 .3 --max-new-runs 25

Repeat with the exact flags/output directory to resume. Runs LLMs only if
--run-llm is enabled (requires HF weights, Internet and sufficient GPU memory).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histchess.research_extension import (generate_scenario_pairs,write_scenario_dataset,
                                         merge_external_mechanisms,LEVELS)
from histchess.core import read_jsonl,validate
from histchess.architecture_benchmark import ARCHITECTURES,rotate_splits
from histchess.research_statistics import summarize_runs,analyze_overfitting


def run(args):
    output=Path(args.out).resolve()
    if output in (Path('/'),Path.home().resolve(),ROOT.resolve()):
        raise ValueError('Unsafe output directory')
    output.mkdir(parents=True,exist_ok=True)
    params={key:value for key,value in vars(args).items() if key not in ('out','max_new_runs','run_llm','llm_models','llm_pairs_per_task','llm_batch_size','allow_cpu_llm','bootstrap')}
    config_file=output/'v8_config.json'
    if config_file.exists():
        previous=json.loads(config_file.read_text())
        if previous!=params:raise RuntimeError('Configuration differs from checkpoint; choose another output directory')
    else:
        config_file.write_text(json.dumps(params,indent=2,sort_keys=True))
    status={'protocol':'histchess-v8-conditional-synthetic',
            'model_benchmark_complete':False,'all_replications_complete':False,
            'statistics_complete':False,'llm_complete':False,'runs':{},
            'limitations':['V8 scenarios are derived from three underlying motifs per rule, not nine independent mechanisms.',
                           'Distant decisive event certified only for castling.',
                           'Replications quantify conditional synthetic generator variability, not real-game external validity.']}
    completed=[]
    def stage(argv):
        print('$',' '.join(map(str,argv)),flush=True)
        subprocess.run(list(map(str,argv)),cwd=ROOT,check=True)
    try:
        for rep in range(args.replications):
            folder=output/f'replication_{rep:02d}'
            data=folder/'dataset'
            file=data/'pairs.jsonl'
            if file.exists():
                pairs=read_jsonl(file)
                manifest=json.loads((data/'protocol_manifest.json').read_text())
                if hashlib.sha256(file.read_bytes()).hexdigest()!=manifest['dataset_sha256']:
                    raise RuntimeError('Checkpoint dataset checksum mismatch: '+str(file))
            else:
                print(f'Building verified replica {rep+1}/{args.replications}',flush=True)
                pairs,rejections=generate_scenario_pairs(
                    args.per_family,args.generator_seed,tuple(args.levels),
                    args.long_attempts,rep)
                if args.external_pairs_jsonl:
                    external=str(args.external_pairs_jsonl)
                    if args.replications>1 and '{rep}' not in external:
                        raise ValueError('For multiple replicas, external source must be a distinct '
                                         'dataset per replica, e.g. /path/source_{rep}.jsonl')
                    external=Path(external.replace('{rep}',str(rep)))
                    if not external.is_file():
                        raise FileNotFoundError(external)
                    pairs=merge_external_mechanisms(pairs,external)
                    for pair in pairs:pair['replicate_id']=rep
                write_scenario_dataset(data,pairs,rejections)
            report=validate(pairs)
            if not report['passed']:
                raise RuntimeError('Chess replay validation failed: '+str(report['errors'][:3]))
            if any(not any(p['task']==task and p['scenario_context']==length for p in pairs)
                   for task in ('castling','en_passant','halfmove','repetition') for length in args.levels):
                raise RuntimeError('Missing task/length cohort')
            arch=folder/'neural'
            current=arch/'architecture_status.json'
            if not (current.exists() and json.loads(current.read_text()).get('complete')):
                argv=[sys.executable,'scripts/run_architecture_benchmark.py',
                      '--pairs-jsonl',file,'--out',arch,
                      '--architectures',*args.architectures,
                      '--folds','0','1','2','--seeds',*map(str,args.seeds),
                      '--cutoffs',*args.cutoffs,
                      '--epochs',str(args.epochs),'--patience',str(args.patience),
                      '--batch-size',str(args.batch_size),'--max-history',str(args.max_history),
                      '--target-parameters',str(args.target_parameters),
                      '--device',args.device,'--position-encoding','sinusoidal',
                      '--length-eval-cutoffs',*args.length_eval_cutoffs,
                      '--length-eval-train-cutoffs',*args.length_eval_train_cutoffs,
                      '--grid-lrs',*map(str,args.grid_lrs),
                      '--grid-weight-decays',*map(str,args.grid_weight_decays),
                      '--grid-dropouts',*map(str,args.grid_dropouts)]
                if args.max_new_runs:argv+=['--max-new-runs',str(args.max_new_runs)]
                stage(argv)
            current_status=json.loads(current.read_text())
            if not current_status.get('complete'):
                status['runs'][str(rep)]={'completed':False,
                    'configuration_runs':len(current_status.get('runs',[])),
                    'expected_configuration_runs':len(args.architectures)*3*len(args.seeds)*len(args.cutoffs)}
                print('Pausing at neural run boundary. Archive output and resume with identical flags.',flush=True)
                break
            audit=arch/'independent_integrity_audit.json'
            if not (audit.exists() and json.loads(audit.read_text()).get('passed')):
                stage([sys.executable,'scripts/verify_architecture_results.py',arch,'--write',audit])
            if not json.loads(audit.read_text()).get('passed'):
                raise RuntimeError('Independent neural audit failed')
            analyze_overfitting(arch,folder/'overfitting_diagnostics.csv')
            if args.length_eval_cutoffs:
                transfer=arch/'architecture_length_ood_metrics.csv'
                if not transfer.is_file() or transfer.stat().st_size==0:
                    raise RuntimeError('Missing true-context length-shift evaluations')
                with transfer.open(newline='') as f:
                    rows=list(csv.DictReader(f))
                actual=[r for r in rows if r['task']=='all' and
                        r['training_history_plies']=='16' and
                        r['evaluation_history_plies'] in ('32','64','full')]
                if not actual or not any(int(r['test_records_with_changed_context'])>0 for r in actual):
                    raise RuntimeError('Requested length shift changes no held-out input; not genuine long-context evidence')
            # Same-pair classical controls for this particular dataset/replicate.
            cls=arch/'comparison_status.json'
            if not (cls.exists() and json.loads(cls.read_text()).get('complete')):
                stage([sys.executable,'scripts/run_comparison_baselines.py',
                       '--pairs-jsonl',arch/'data/pairs.jsonl','--out',arch,
                       '--folds','0','1','2','--cutoffs',*args.cutoffs,
                       '--max-history',str(args.max_history)])
            cla=arch/'comparison_integrity_audit.json'
            if not (cla.exists() and json.loads(cla.read_text()).get('passed')):
                stage([sys.executable,'scripts/verify_comparison_results.py',arch,'--write',cla])
            if not json.loads(cla.read_text()).get('passed'):
                raise RuntimeError('Independent classical-control audit failed')
            if not (arch/'neural_vs_classical_matched.csv').is_file():
                stage([sys.executable,'scripts/make_comparison_table.py',arch])
            status['runs'][str(rep)]={'completed':True,'pairs':len(pairs),
                    'neural_audit':True,'classical_audit':True}
            completed.append(arch)
            if args.run_llm:
                llmdir=folder/'pretrained_llm'
                metric_file=llmdir/'llm_metrics.csv'
                if not metric_file.exists():
                    from histchess.experiments import llm_experiment
                    fold_zero=rotate_splits(pairs,0)
                    llm_experiment(fold_zero,llmdir,args.llm_models,
                                   pairs_per_task=args.llm_pairs_per_task,
                                   seed=args.generator_seed+rep*100003,
                                   batch_size=args.llm_batch_size,
                                   require_gpu=not args.allow_cpu_llm)
                revisions=llmdir/'pretrained_model_revisions.json'
                if not metric_file.is_file() or not metric_file.stat().st_size or not revisions.is_file():
                    raise RuntimeError('Incomplete LLM predictions/metrics/model provenance')
                versions=json.loads(revisions.read_text())
                if [x['model'] for x in versions]!=args.llm_models:
                    raise RuntimeError('Existing LLM results use different model identifiers')
                with metric_file.open(newline='') as f:llm_rows=list(csv.DictReader(f))
                if len(llm_rows)!=5*len(args.llm_models):
                    raise RuntimeError('LLM metrics do not cover all requested tasks and models')
                with (llmdir/'llm_predictions.csv').open(newline='') as f:
                    llm_prediction_rows=list(csv.DictReader(f))
                if len(llm_prediction_rows)!=2*4*args.llm_pairs_per_task*len(args.llm_models):
                    raise RuntimeError('LLM predictions are incomplete or sample size differs')
        else:
            status['all_replications_complete']=True
        status['model_benchmark_complete']=status['all_replications_complete']
        if status['all_replications_complete']:
            summarize_runs(completed,output/'statistics',bootstrap=args.bootstrap,
                           min_replications=5)
            status['statistics_complete']=True
            status['llm_complete']=args.run_llm
        else:
            status['llm_complete']=False
    finally:
        (output/'v8_status.json').write_text(json.dumps(status,indent=2))
    print('V8 PIPELINE STATUS:',json.dumps(status,indent=2))
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--per-family',type=int,default=8)
    p.add_argument('--replications',type=int,default=5)
    p.add_argument('--generator-seed',type=int,default=3101)
    p.add_argument('--levels',nargs='+',choices=LEVELS,default=['native','32','64'])
    p.add_argument('--long-attempts',type=int,default=100)
    p.add_argument('--external-pairs-jsonl',default='',
                   help='Optional INDEPENDENT new chess mechanisms, provenance and legal replay required. '
                        'For multiple replications use a filename containing {rep}.')
    p.add_argument('--architectures',nargs='+',choices=list(ARCHITECTURES),
                   default=['elman_original','lstm','gru_original','causal_transformer_postnorm',
                            'linear_attention_multihead','s4d_lin'])
    p.add_argument('--cutoffs',nargs='+',default=['0','16','32','64','full'])
    p.add_argument('--length-eval-cutoffs',nargs='+',default=['32','64','full'],
                   help='Evaluate trained 16-ply checkpoint on longer actual histories')
    p.add_argument('--length-eval-train-cutoffs',nargs='+',default=['16'])
    p.add_argument('--seeds',type=int,nargs='+',default=[3101,3102])
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--max-history',type=int,default=256)
    p.add_argument('--target-parameters',type=int,default=150000)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--grid-lrs',type=float,nargs='+',default=[0.0003,0.001])
    p.add_argument('--grid-weight-decays',type=float,nargs='+',default=[0.001])
    p.add_argument('--grid-dropouts',type=float,nargs='+',default=[0.1,0.3])
    p.add_argument('--max-new-runs',type=int,default=0)
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--run-llm',action='store_true')
    p.add_argument('--llm-models',nargs='+',default=['HuggingFaceTB/SmolLM2-135M-Instruct',
                                                  'HuggingFaceTB/SmolLM2-360M-Instruct'])
    p.add_argument('--llm-pairs-per-task',type=int,default=4)
    p.add_argument('--llm-batch-size',type=int,default=4)
    p.add_argument('--allow-cpu-llm',action='store_true')
    a=p.parse_args()
    if any(int(c)>a.max_history for c in a.cutoffs if c!='full'):
        p.error('Some history cutoffs exceed max-history')
    if a.length_eval_cutoffs and not set(a.length_eval_train_cutoffs)<=set(a.cutoffs):
        p.error('Length-shift source cutoffs must also be training cutoffs')
    if any(int(c)>a.max_history for c in a.levels if c!='native'):
        p.error('Some target history lengths exceed max-history')
    if '256' in a.levels and a.max_history<288:
        p.error('256-ply minimum may overshoot: set --max-history 288 or greater')
    if not any(c in a.levels for c in ('32','64','128','256')):
        p.error('Genuine long-history cohort required')
    if len(set(a.seeds))!=len(a.seeds) or a.replications<1 or a.per_family<1:
        p.error('Invalid repetitions or per-family sample count')
    run(a)

if __name__=='__main__':main()
