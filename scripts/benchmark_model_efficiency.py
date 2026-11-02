"""Synthetic tensor-only architecture latency benchmark (not chess accuracy).

Example:
python scripts/benchmark_model_efficiency.py --architectures gru_original s4d_lin \
  --lengths 16 64 128 --batch-size 16 --device cpu --out efficiency.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from histchess.architecture_benchmark import ARCHITECTURES, make_budgeted_model, num_parameters


def run(args):
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    results=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    for model_name in args.architectures:
        for length in args.lengths:
            model=make_budgeted_model(model_name,target=args.target_parameters,
                max_history=max(args.lengths),position_encoding='sinusoidal').to(device).eval()
            board=torch.randint(0,13,(args.batch_size,64),device=device)
            side=torch.randint(0,2,(args.batch_size,),device=device)
            task=torch.randint(0,4,(args.batch_size,),device=device)
            moves=torch.empty(args.batch_size,length,3,device=device,dtype=torch.long)
            moves[...,0:2]=torch.randint(1,65,(args.batch_size,length,2),device=device)
            moves[...,2]=0
            lengths=torch.full((args.batch_size,),length,device=device,dtype=torch.long)
            batch=(board,side,task,moves,lengths)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                for _ in range(args.warmup):_ = model(*batch)
                sync();t0=time.perf_counter()
                for _ in range(args.steps):_ = model(*batch)
                sync();elapsed=(time.perf_counter()-t0)*1000/args.steps
            results.append(dict(architecture=model_name,history_length=length,
                batch_size=args.batch_size,parameters=num_parameters(model),device=str(device),
                mean_forward_ms_per_batch=elapsed,
                examples_per_second=args.batch_size/(elapsed/1000),
                peak_cuda_memory_bytes=(torch.cuda.max_memory_allocated(device)
                     if device.type=='cuda' else ''),warmup=args.warmup,steps=args.steps,
                note='Synthetic forward-pass performance only; not chess rule accuracy'))
            print(model_name, length, f'{elapsed:.2f} ms',flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf8',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (out.with_suffix('.metadata.json')).write_text(json.dumps({
        'torch':torch.__version__,'device':str(device),'seed':args.seed,
        'warmup':args.warmup,'steps':args.steps,'comparability':
        'Same machine / device / batch / lengths required. CPU peak memory not measured.'},indent=2))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--architectures',nargs='+',choices=list(ARCHITECTURES),
                   default=['gru_original','causal_transformer_postnorm','linear_attention_multihead','s4d_lin'])
    p.add_argument('--lengths',nargs='+',type=int,default=[16,32,64,128,256])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--seed',type=int,default=3101)
    p.add_argument('--target-parameters',type=int,default=150000)
    p.add_argument('--out',default='model_efficiency.csv')
    a=p.parse_args()
    if not a.lengths or min(a.lengths)<1 or a.steps<1 or a.batch_size<1 or a.warmup<0:
        raise ValueError('Positive lengths, batch and steps required')
    run(a)

if __name__=='__main__':main()
