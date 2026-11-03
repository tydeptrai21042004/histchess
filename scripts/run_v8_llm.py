#!/usr/bin/env python3
"""Run actual pretrained HF language models on the exact held-out chess pairs.

No synthetic responses or substituted tiny random models. Hugging Face weights
must load; when unavailable the command fails without claiming LLM evidence.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from histchess.core import read_jsonl,validate
from histchess.architecture_benchmark import rotate_splits
from histchess.experiments import llm_experiment
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pairs-jsonl',required=True)
p.add_argument('--out',required=True)
p.add_argument('--fold',type=int,choices=(0,1,2),default=0)
p.add_argument('--models',nargs='+',default=['HuggingFaceTB/SmolLM2-135M-Instruct',
                                          'HuggingFaceTB/SmolLM2-360M-Instruct'])
p.add_argument('--pairs-per-task',type=int,default=8)
p.add_argument('--batch-size',type=int,default=4)
p.add_argument('--max-new-tokens',type=int,default=6)
p.add_argument('--seed',type=int,default=3101)
p.add_argument('--allow-cpu',action='store_true',help='Debug only; CPU inference can be slow')
a=p.parse_args()
if len(set(a.models))!=len(a.models) or a.pairs_per_task<1 or a.batch_size<1 or a.max_new_tokens<1:
    p.error('Invalid models or count')
pairs=read_jsonl(a.pairs_jsonl)
report=validate(pairs)
if not report['passed']:raise RuntimeError('Chess validation failed: '+str(report['errors'][:3]))
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
fold=rotate_splits(pairs,a.fold)
manifest={'dataset_sha256':hashlib.sha256(Path(a.pairs_jsonl).read_bytes()).hexdigest(),
          'fold':a.fold,'models':a.models,'pairs_per_task':a.pairs_per_task,
          'source':'verified matched held-out chess histories',
          'claim_scope':'zero-shot deterministic inference of real pretrained HF causal LMs'}
path=out/'pretrained_protocol.json'
if path.exists() and json.loads(path.read_text())!=manifest:
    raise ValueError('Existing LLM evaluation has different protocol settings')
path.write_text(json.dumps(manifest,indent=2))
llm_experiment(fold,out,a.models,pairs_per_task=a.pairs_per_task,seed=a.seed,
               batch_size=a.batch_size,require_gpu=not a.allow_cpu,
               max_new_tokens=a.max_new_tokens)
print('Pretrained LLM matched-pair predictions complete:',out)
