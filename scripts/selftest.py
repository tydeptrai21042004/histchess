#!/usr/bin/env python3
"""Fast chess-specific invariants to run BEFORE the full Kaggle experiments."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import TASKS,CAST_POS,CAST_NEG,EP_POS,EP_NEG,HALF_POS,HALF_NEG,REP_POS,REP_NEG,pair_from_moves,generate,validate

def main():
    for task, pos, neg in zip(TASKS,(CAST_POS,EP_POS,HALF_POS,REP_POS),(CAST_NEG,EP_NEG,HALF_NEG,REP_NEG)):
        p=pair_from_moves(task,pos,neg)
        assert p['positive']['label']==1 and p['negative']['label']==0
        print(f'PASS template {task}: {p["positive"]["fen"]}')
    pairs,rejected=generate(count_per_task=5,seed=3101)
    report=validate(pairs)
    print(json.dumps({'report':report,'rejection_classes':list(rejected)},indent=2))
    assert report['passed'] and len(pairs)==20
if __name__=='__main__':main()
