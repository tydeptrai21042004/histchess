#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import read_jsonl,validate
p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',default='data/pairs.jsonl');a=p.parse_args()
report=validate(read_jsonl(a.dataset))
print(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
