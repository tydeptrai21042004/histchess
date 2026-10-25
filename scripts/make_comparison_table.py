"""Exact same-fold, same-pair, same-cutoff architecture vs classical comparisons."""
from __future__ import annotations
import argparse
import collections
import csv
from pathlib import Path

def read(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def write(outdir):
    root=Path(outdir)
    a=read(root/'architecture_predictions.csv')
    b=read(root/'comparison_predictions.csv')
    bm=collections.defaultdict(dict)
    for x in b:
        key=(int(x['fold']),x['history_plies'],x['baseline'])
        bm[key][(x['pair_id'],x['variant'])]=x
    am=collections.defaultdict(dict)
    for x in a:
        key=(x['architecture'],int(x['fold']),int(x['seed']),x['history_plies'])
        am[key][(x['pair_id'],x['variant'])]=x
    out=[]
    for (arch,fold,seed,h),data in sorted(am.items()):
        for baseline in ('board_only','move_bag','move_ngrams'):
            other=bm.get((fold,h,baseline))
            if other is None or set(data)!=set(other):raise RuntimeError('Test samples do not match')
            a_correct=sum(int(int(x['prediction'])==int(x['label'])) for x in data.values())
            b_correct=sum(int(int(x['prediction'])==int(x['label'])) for x in other.values())
            ids={pid for pid,v in data}
            def joint(data):return sum(int(int(data[(pid,'positive')]['prediction'])==1 and int(data[(pid,'negative')]['prediction'])==0) for pid in ids)
            for key in data:
                if int(data[key]['label'])!=int(other[key]['label']):raise RuntimeError('Mismatched held-out labels')
            out.append({'architecture':arch,'baseline':baseline,'fold':fold,'seed':seed,'history_plies':h,
                'same_test_pairs':len(ids),'neural_record_accuracy':a_correct/len(data),
                'classical_record_accuracy':b_correct/len(data),
                'neural_minus_classical_record_accuracy':(a_correct-b_correct)/len(data),
                'neural_joint_pair_accuracy':joint(data)/len(ids),
                'classical_joint_pair_accuracy':joint(other)/len(ids),
                'descriptive_only':True})
    with (root/'neural_vs_classical_matched.csv').open('w',newline='',encoding='utf8') as f:
        if not out:raise RuntimeError('No matched comparison records')
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    return len(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('outdir');args=p.parse_args()
    print('MATCHED ARCHITECTURE VS CLASSICAL ROWS:',write(args.outdir),flush=True)
if __name__=='__main__':main()
