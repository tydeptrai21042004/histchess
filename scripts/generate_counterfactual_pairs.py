#!/usr/bin/env python3
"""Create canonical, checked counterfactual history pairs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import generate,validate,write_jsonl

p=argparse.ArgumentParser()
p.add_argument('--out',default='data/pairs.jsonl')
p.add_argument('--pairs-per-task',type=int,default=300)
p.add_argument('--seed',type=int,default=3101)
a=p.parse_args()
pairs,rejections=generate(a.pairs_per_task,a.seed)
report=validate(pairs)
if not report['passed']:
    raise RuntimeError(json.dumps(report,indent=2))
write_jsonl(a.out,pairs)
print(json.dumps({'data':a.out,'validation':report,'generation_rejections':rejections},indent=2))
