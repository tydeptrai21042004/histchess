#!/usr/bin/env python3
"""Descriptive task and history-length distribution (no external data)."""
import argparse,collections,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import read_jsonl
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--out',default='data/motifs.json');a=p.parse_args()
pairs=read_jsonl(a.dataset)
summary={}
for task in sorted({r['task'] for r in pairs}):
    part=[r for r in pairs if r['task']==task]
    lengths=[len(r['positive']['history_uci']) for r in part]
    summary[task]={'pairs':len(part),'min_plies':min(lengths),'median_plies':statistics.median(lengths),'max_plies':max(lengths),'split_counts':dict(collections.Counter(r['split'] for r in part))}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
