#!/usr/bin/env python3
"""Multi-seed generated-data stress suite; does not need any downloaded LLM."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import TASKS,generate,validate
from histchess.experiments import leakage_audit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs-per-task',type=int,default=60)
    ap.add_argument('--seeds',type=int,nargs='+',default=[3101,3102,3103])
    ap.add_argument('--generator',choices=('diverse','legacy'),default='diverse')
    ap.add_argument('--split-strategy',choices=('family','board'),default=None)
    ap.add_argument('--out',default='stress_test_report.json')
    args=ap.parse_args()
    if args.pairs_per_task<10:ap.error('Require at least 10 pairs per task')
    results=[]
    for seed in args.seeds:
        start=time.time()
        pairs,rejections=generate(args.pairs_per_task,seed=seed,
            generator=args.generator,split_strategy=args.split_strategy or
            ('family' if args.generator=='diverse' else 'board'))
        val=validate(pairs)
        leakage=leakage_audit(pairs)
        split={f'{task}:{sp}':sum(p['task']==task and p['split']==sp for p in pairs)
               for task in TASKS for sp in ('train','validation','test')}
        row={'seed':seed,'passed':val['passed'] and min(split.values())>0
             and leakage['board_group_leakage_count']==0
             and leakage['full_history_leakage_count']==0
             and ((args.split_strategy or 'family') != 'family' or
                  leakage['generator_family_across_split_count']==0),
             'n_pairs':len(pairs),'duration_seconds':round(time.time()-start,2),
             'min_split':min(split.values()),'split_counts':split,
             'invalid_pairs':len(val['errors']),
             'rejections':rejections}
        print(json.dumps(row,indent=2),flush=True)
        results.append(row)
    report={'passed':all(x['passed'] for x in results),'runs':results}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
