#!/usr/bin/env python3
"""Package a validated dataset and reproducibility metadata separately from model cache."""
import argparse,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import read_jsonl,validate
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--out',default='histchess_dataset_release.zip');a=p.parse_args()
report=validate(read_jsonl(a.dataset))
if not report['passed']:
    raise RuntimeError('Unvalidated dataset cannot be packaged: '+str(report['errors'][:2]))
with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(a.dataset,arcname='pairs.jsonl')
    z.writestr('validation.json',json.dumps(report,indent=2))
    z.writestr('README.txt','HistChess generated legal chess histories. All labels validated by python-chess replay. See project README for caveats.\n')
print(a.out)
