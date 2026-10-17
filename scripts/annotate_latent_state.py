#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import read_jsonl,rows_from_pairs,write_jsonl
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--out',default='data/flat_records.jsonl');a=p.parse_args()
rows=rows_from_pairs(read_jsonl(a.dataset));write_jsonl(a.out,rows)
print(json.dumps({'flat_records':len(rows),'path':a.out}))
