"""Fail-closed integrity audit of the outputs of all experiments."""
from __future__ import annotations

import math
from collections import Counter,defaultdict
from pathlib import Path

from .core import TASKS
from .paper import read_csv


def audit_final_predictions(pairs,outdir,seeds,cutoffs,llm_models=None,llm_pairs_per_task=0,skip_llm=False):
    outdir=Path(outdir)
    test=[p for p in pairs if p['split']=='test']
    ids={(p['pair_id'],v) for p in test for v in ('positive','negative')}
    per_task=Counter(p['task'] for p in test)
    train_test={p['pair_id']:p for p in test}
    records=read_csv(outdir/'gru_predictions.csv')
    all_keys=set()
    for r in records:
        key=(int(r['seed']),r['history_moves'],r['pair_id'],r['variant'])
        if key in all_keys:raise ValueError(f'Duplicate GRU prediction: {key}')
        all_keys.add(key)
        if (r['pair_id'],r['variant']) not in ids:
            raise ValueError('GRU prediction outside TEST split')
        p=train_test[r['pair_id']]
        if r['task']!=p['task'] or int(r['label'])!=int(p[r['variant']]['label']):
            raise ValueError(f'Corrupted GRU prediction label {key}')
        prob=float(r['score'])
        if not math.isfinite(prob) or not 0<=prob<=1:
            raise ValueError(f'Bad GRU predicted probability: {key}')
        if int(r['prediction'])!=int(prob>=.5):
            raise ValueError(f'Prediction threshold mismatch: {key}')
    expected={(seed,'full' if keep is None else str(keep),pairid,variant)
              for seed in seeds for keep in cutoffs for pairid,variant in ids}
    if all_keys != expected:
        missing=expected-all_keys
        extra=all_keys-expected
        raise ValueError(f'GRU coverage mismatch: {len(missing)} missing, {len(extra)} extra')
    # A board+task-only model sees identical input on both members of a pair.
    for seed in seeds:
        board_only={(r['pair_id'],r['variant']):r['prediction'] for r in records
                    if int(r['seed'])==seed and r['history_moves']=='0'}
        for p in test:
            if board_only[(p['pair_id'],'positive')]!=board_only[(p['pair_id'],'negative')]:
                raise ValueError('Board-only model distinguished same-board histories')
    llm_details={}
    if not skip_llm:
        llm=read_csv(outdir/'llm_predictions.csv')
        samples=__import__('json').loads((outdir/'llm_sample.json').read_text())
        sample_ids=set(samples['pair_ids'])
        if len(sample_ids)!=llm_pairs_per_task*len(TASKS):
            raise ValueError('LLM selection has duplicate/insufficient pair IDs')
        if not sample_ids.issubset(train_test):
            raise ValueError('LLM sample contains non-test pair IDs')
        if Counter(train_test[i]['task'] for i in sample_ids)!=Counter({task:llm_pairs_per_task for task in TASKS}):
            raise ValueError('LLM sample not balanced per task')
        expected_llm={(model,pairid,variant) for model in llm_models
                      for pairid in sample_ids for variant in ('positive','negative')}
        actual_llm={(r['model'],r['pair_id'],r['variant']) for r in llm}
        if len(actual_llm)!=len(llm) or actual_llm!=expected_llm:
            raise ValueError('LLM missing/duplicate model predictions or unequal sampled IDs')
        for r in llm:
            p=train_test[r['pair_id']]
            if r['task']!=p['task'] or int(r['label'])!=p[r['variant']]['label']:
                raise ValueError('LLM prediction label mismatch')
            if r['valid'].lower()=='true':
                if r['prediction'] not in ('0','1'):
                    raise ValueError('Valid LLM prediction must be 0 or 1')
            elif r['prediction']:
                raise ValueError('Invalid LLM must not have a numeric prediction')
        llm_details={'models':len(llm_models),'sampled_test_pairs':len(sample_ids),
                     'evaluated_records':len(llm)}
    for filename in ('validation.json','leakage_audit.json','baseline_metrics.csv',
                     'shortcut_metrics.csv','shortcut_predictions.csv','shortcut_audit.json',
                     'gru_learning_curve.csv','history_ablation.csv',
                     'paired_confidence_intervals.csv','paired_significance.csv',
                     'paper_tables.md','paper_tables.tex','history_gru_best.pt'):
        if not (outdir/filename).is_file() or (outdir/filename).stat().st_size==0:
            raise ValueError('Missing/empty output artifact: '+filename)
    if not skip_llm and not (outdir/'matched_sample_comparison.csv').is_file():
        raise ValueError('Missing fair LLM-versus-GRU matched-sample table')
    if not list((outdir/'figures').glob('*.png')):
        raise ValueError('No generated figures')
    return {'passed':True,'test_pairs':len(test),'test_pairs_by_task':dict(per_task),
            'gru_predictions_checked':len(records),
            'gru_conditions_checked':len(seeds)*len(cutoffs),
            'board_only_matched_input_invariance':'PASS',
            'llm':llm_details}
