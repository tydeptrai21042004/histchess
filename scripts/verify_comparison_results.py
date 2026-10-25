"""Independent structural/numerical audit of the classical comparison results."""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

BASELINES={"board_only","move_bag","move_ngrams","privileged_replay_oracle"}
TASKS={"castling","en_passant","halfmove","repetition"}

def csv_rows(path):
    with path.open(newline='',encoding='utf8') as fh:return list(csv.DictReader(fh))

def audit(outdir):
    out=Path(outdir); errors=[]
    def check(cond,msg):
        if not cond:errors.append(msg)
    required=['comparison_status.json','comparison_config.json','comparison_metrics.csv',
              'comparison_predictions.csv','data/pairs.jsonl']
    missing=[x for x in required if not (out/x).is_file()]
    if missing:return {'passed':False,'errors':['Missing '+x for x in missing]}
    config=json.loads((out/'comparison_config.json').read_text())
    status=json.loads((out/'comparison_status.json').read_text())
    pairs=[json.loads(s) for s in (out/'data/pairs.jsonl').read_text().splitlines() if s.strip()]
    metrics=csv_rows(out/'comparison_metrics.csv')
    preds=csv_rows(out/'comparison_predictions.csv')
    pmap={p['pair_id']:p for p in pairs}
    check(len(pmap)==len(pairs),'duplicate pair IDs')
    check(status.get('complete') is True,'comparison stage incomplete')
    from histchess.architecture_benchmark import FAMILIES
    expected={}
    for fold in config['folds']:
        for pid,p in pmap.items():
            idx=FAMILIES[p['task']].index(p['generator_family'])
            if ('train','validation','test')[(idx-int(fold))%3]=='test':
                expected.setdefault(int(fold),set()).add(pid)
    group=collections.defaultdict(dict)
    for r in preds:
        b=r['baseline'];fold=int(r['fold']);context=r['history_plies']; pid=r['pair_id'];v=r['variant']
        check(b in BASELINES,f'unknown baseline {b}')
        check(pid in expected.get(fold,set()),f'wrong test pair {fold}/{pid}')
        check(v in ('positive','negative'),f'invalid variant {v}')
        if pid in pmap:
            check(r['task']==pmap[pid]['task'],'task mismatch')
            check(int(r['label'])==int(pmap[pid][v]['label']),'label mismatch')
        score=float(r['score']); guessed=int(r['prediction'])
        check(math.isfinite(score) and 0<=score<=1,'invalid score')
        check(guessed==int(score>=0.5),'wrong hard threshold')
        if b=='privileged_replay_oracle':check(guessed==int(r['label']),'oracle incorrect')
        k=(b,fold,context);pk=(pid,v)
        check(pk not in group[k],f'duplicate prediction {k}/{pk}')
        group[k][pk]=r
    configurations={(b,int(f),str(h)) for b in BASELINES for f in config['folds'] for h in config['cutoffs']}
    check(set(group)==configurations,'baseline run configurations incomplete')
    for (b,f,h),sample in group.items():
        wanted={(pid,v) for pid in expected[f] for v in ('positive','negative')}
        check(set(sample)==wanted,f'test prediction coverage mismatch {b}/{f}/{h}')
    metric_map={}
    for m in metrics:
        key=(m['baseline'],int(m['fold']),m['history_plies'],m['task'])
        check(key not in metric_map,f'duplicate metric {key}')
        metric_map[key]=m
    expected_m={(b,f,h,t) for b,f,h in configurations for t in ('all',)+tuple(sorted(TASKS))}
    check(set(metric_map)==expected_m,'baseline metrics incomplete')
    for (b,f,h),sample in group.items():
        for task in ('all',)+tuple(sorted(TASKS)):
            records=[r for r in sample.values() if task=='all' or r['task']==task]
            if not records:continue
            n=len(records); correct=sum(int(int(r['prediction'])==int(r['label'])) for r in records)
            ids={r['pair_id'] for r in records}
            joint=sum(int(int(sample[(pid,'positive')]['prediction'])==1 and
                          int(sample[(pid,'negative')]['prediction'])==0) for pid in ids)
            m=metric_map.get((b,f,h,task))
            if not m:continue
            check(int(m['test_records'])==n and int(m['test_pairs'])==len(ids),'pair count mismatch')
            check(abs(float(m['record_accuracy'])-correct/n)<1e-10,'incorrect record accuracy')
            check(abs(float(m['joint_pair_accuracy'])-joint/len(ids))<1e-10,'incorrect joint accuracy')
            if h=='0' and b!='privileged_replay_oracle':
                check(abs(correct/n-.5)<1e-10 and joint==0,f'zero-history invariant violated {b}/{f}')
    check(len(preds)==int(status.get('prediction_rows',-1)),'status prediction count mismatch')
    check(len(metrics)==int(status.get('metric_rows',-1)),'status metric count mismatch')
    return {'passed':not errors,'errors':errors[:100],'configuration_count':len(group),
            'prediction_rows':len(preds),'metric_rows':len(metrics)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('outdir');ap.add_argument('--write')
    args=ap.parse_args();report=audit(args.outdir)
    if args.write:Path(args.write).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
