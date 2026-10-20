#!/usr/bin/env python3
"""Independent fail-closed validation of a Kaggle HistChess result ZIP.

Usage: python scripts/verify_artifacts.py /kaggle/working/histchess_results.zip
Prints a JSON report; nonzero exit on missing, stale, invalid or incomplete output.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

REQUIRED={
    'run_status.json','manifest.json','final_output_audit.json',
    'data/pairs.jsonl','validation.json','baseline_metrics.csv',
    'leakage_audit.json','gru_predictions.csv','gru_metrics.csv',
    'gru_learning_curve.csv','history_ablation.csv','gru_model_config.json',
    'history_gru_best.pt','paired_confidence_intervals.csv',
    'paired_significance.csv','paper_tables.md','paper_tables.tex',
    'figures/history_length_ablation.png','figures/validation_loss.png',
}
LLM_FILES={'llm_predictions.csv','llm_metrics.csv',
           'matched_sample_comparison.csv','llm_sample.json'}


def verify(archive,require_complete=True):
    path=Path(archive)
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC error: '+bad)
        roots={p.split('/')[0] for p in z.namelist() if p.strip('/')}
        if len(roots)!=1:raise ValueError(f'Unexpected output roots: {roots}')
        root=next(iter(roots))
        names={p[len(root)+1:] for p in z.namelist() if p.startswith(root+'/')}
        missing=REQUIRED-names
        if require_complete:missing|=LLM_FILES-names
        if missing:raise ValueError(f'Missing report outputs: {sorted(missing)}')
        def load(p):return json.loads(z.read(root+'/'+p))
        status=load('run_status.json')
        manifest=load('manifest.json')
        aud=load('final_output_audit.json')
        if not aud.get('passed'):raise ValueError('Final prediction integrity audit did not pass')
        if require_complete:
            if not status.get('completed') or not status.get('paper_complete'):
                raise ValueError('Experiment incomplete: run_status.json says not complete')
            if manifest.get('status')!='COMPLETE' or not manifest.get('paper_complete'):
                raise ValueError('Experiment incomplete: manifest.json says not complete')
        source_hash=hashlib.sha256(z.read(root+'/data/pairs.jsonl')).hexdigest()
        if source_hash!=manifest.get('pairs_sha256'):
            raise ValueError('SHA256 mismatch for pairs.jsonl')
        if not load('validation.json').get('passed'):
            raise ValueError('Dataset replay certificate validation failed')
        records=sum(1 for line in z.read(root+'/data/pairs.jsonl').splitlines() if line.strip())
        if records <= aud.get('test_pairs',0):
            raise ValueError('Dataset smaller than test set')
        if len([x for x in names if x.startswith('figures/') and x.endswith('.png')])<2:
            raise ValueError('Publication figures missing')
        for figure in ('figures/history_length_ablation.png','figures/validation_loss.png'):
            if z.read(root+'/'+figure)[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Not a valid PNG signature: '+figure)
        for p in REQUIRED| (LLM_FILES if require_complete else set()):
            if not z.getinfo(root+'/'+p).file_size:
                raise ValueError('Empty required artifact: '+p)
        return {'passed':True,'zip':str(path),'files':len(names),
                'dataset_pairs':records,'test_pairs':aud['test_pairs'],
                'gru_predictions':aud['gru_predictions_checked'],
                'complete':bool(status.get('completed')),
                'paper_complete':bool(status.get('paper_complete'))}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('archive')
    ap.add_argument('--allow-smoke',action='store_true')
    args=ap.parse_args()
    try: print(json.dumps(verify(args.archive,not args.allow_smoke),indent=2))
    except Exception as exc:
        print(f'VERIFICATION FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
