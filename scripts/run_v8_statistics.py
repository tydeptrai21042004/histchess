#!/usr/bin/env python3
"""Standalone conditional replication inference. Requires completed run dirs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.research_statistics import summarize_runs,analyze_overfitting
p=argparse.ArgumentParser()
p.add_argument('--runs',nargs='+',required=True)
p.add_argument('--out',required=True)
p.add_argument('--bootstrap',type=int,default=5000)
p.add_argument('--min-replications',type=int,default=5)
a=p.parse_args()
for folder in a.runs:
    analyze_overfitting(folder,Path(folder)/'v8_overfitting_diagnostics.csv')
result=summarize_runs(a.runs,a.out,bootstrap=a.bootstrap,min_replications=a.min_replications)
print(f'Completed {len(result)} multiplicity-controlled comparisons; inference requires '
      f'{a.min_replications} independent generated replicas.')
