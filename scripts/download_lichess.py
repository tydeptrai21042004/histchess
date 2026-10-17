#!/usr/bin/env python3
"""OPTIONAL bounded HTTP downloader; not needed for all-eight experiments."""
import argparse,shutil,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',required=True);p.add_argument('--max-mb',type=int,default=100)
a=p.parse_args()
if not a.url.startswith('https://'):raise ValueError('HTTPS URL required')
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
limit=a.max_mb*1024*1024
with urllib.request.urlopen(a.url,timeout=30) as resp, out.open('wb') as dst:
    done=0
    while True:
        block=resp.read(min(1024*1024,limit-done+1))
        if not block:break
        done+=len(block)
        if done>limit:
            out.unlink(missing_ok=True)
            raise RuntimeError('Download exceeded byte limit; choose smaller PGN sample')
        dst.write(block)
print(f'Downloaded {done} bytes to {out}')
