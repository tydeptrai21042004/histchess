"""Conditional out-of-family inference for independently re-generated datasets.

A training seed is never considered an independent dataset; folds revisit the
same set of pairs. Inference draws independent *generator replications* as
clusters, and only describes variation conditional on the specified generator
mechanisms. No population-level or real-game claim is made.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('');return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def _mean(v):
    return sum(v)/len(v)


def _p_signflip(diffs, *, seed=4101, draws=40000):
    """Two-sided paired sign-flip at the independently sampled replicate level."""
    n=len(diffs)
    if n<2:raise ValueError('Independent repetitions are needed')
    actual=abs(_mean(diffs))
    rng=random.Random(seed)
    if n<=18:
        sums=(abs(sum(a*x for a,x in zip(signs,diffs))/n)
              for signs in itertools.product((-1,1),repeat=n))
        count=sum(x>=actual-1e-14 for x in sums)
        return count/(1<<n)
    count=0
    for _ in range(draws):
        stat=abs(sum((1 if rng.getrandbits(1) else -1)*d for d in diffs)/n)
        count+=int(stat>=actual-1e-14)
    return (count+1)/(draws+1)


def _cluster_bootstrap(diffs, repeats,seed):
    rng=random.Random(seed);n=len(diffs)
    boots=sorted(_mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(repeats))
    return boots[int(.025*(len(boots)-1))],boots[int(.975*(len(boots)-1))]


def summarize_runs(result_directories,outdir,bootstrap=5000,seed=3101,min_replications=5):
    """Compute paired model differences without seed/fold pseudo-replication.

    Rejected unless all models have the same held-out datasets (per replication)
    and prediction keys, so the word 'paired' means the exact same records.
    """
    if bootstrap<100 or min_replications<2:
        raise ValueError('Insufficient bootstrap or replicate threshold')
    runs=[];used=set()
    for source in result_directories:
        folder=Path(source)
        status=json.loads((folder/'architecture_status.json').read_text())
        if not status.get('complete'):
            raise RuntimeError(f'Incomplete neural benchmark: {folder}')
        pairs=[json.loads(x) for x in (folder/'data/pairs.jsonl').read_text().splitlines() if x.strip()]
        ids={p['pair_id']:p for p in pairs}
        reps={p.get('replicate_id',0) for p in pairs}
        if len(reps)!=1:
            raise ValueError('A run must have one generator replication')
        rep=next(iter(reps))
        if rep in used:raise ValueError(f'Duplicate generator replication ID: {rep}')
        used.add(rep)
        predictions=read_csv(folder/'architecture_predictions.csv')
        classical=folder/'comparison_predictions.csv'
        if classical.is_file():
            for row in read_csv(classical):
                if row['baseline']=='privileged_replay_oracle':
                    continue  # privileged deterministic reference, never a learned comparator
                predictions.append({**row,
                    'architecture':'baseline:'+row['baseline'],
                    'seed':'classical'})
        runs.append((rep,ids,predictions,folder))
    if not runs:raise ValueError('No run directories supplied')
    model_scores=[]
    for rep,pairs,preds,folder in runs:
        by_config=collections.defaultdict(dict)
        for r in preds:
            key=(r['architecture'],r['history_plies'],r['fold'],r['seed'])
            item=(r['pair_id'],r['variant'])
            if item in by_config[key]:raise ValueError('Duplicate raw prediction')
            if item[0] not in pairs:raise ValueError('Unknown pair in predictions')
            by_config[key][item]=r
        # Strict same-record comparison for every architecture and classical
        # method at a fixed fold/context. Do not call missing rows paired.
        memberships=collections.defaultdict(dict)
        for (arch,hist,fold,train_seed),values in by_config.items():
            group=(hist,fold)
            member_set=frozenset(values)
            memberships[group].setdefault(arch,member_set)
            if memberships[group][arch]!=member_set:
                raise ValueError(f'Seed changed held-out membership: {group}/{arch}')
        for group,member_by_arch in memberships.items():
            if len(set(member_by_arch.values()))!=1:
                raise ValueError(f'Unmatched paired model/classical samples: {group}')
        # Collapse seeds and fold rotations to a single accuracy per model/task,
        # then macro-average underlying source mechanisms, not scenario counts.
        collected=collections.defaultdict(lambda:collections.defaultdict(list))
        for (arch,hist,fold,train_seed),records in by_config.items():
            for (pid,variant),r in records.items():
                pair=pairs[pid]
                mechanism=pair.get('independent_mechanism',pair['generator_family'])
                correct=int(int(r['prediction'])==int(pair[variant]['label']))
                for task in ('all',pair['task']):
                    collected[(arch,hist,task)][mechanism].append(correct)
        for (arch,hist,task),mechanisms in sorted(collected.items()):
            if not mechanisms:raise ValueError('Empty mechanism group')
            score=_mean([_mean(x) for x in mechanisms.values()])
            model_scores.append({'generator_replication':rep,'architecture':arch,
                                 'history_plies':hist,'task':task,
                                 'macro_record_accuracy':score,
                                 'independent_base_mechanisms':len(mechanisms),
                                 'source':str(folder)})
    by_rep=collections.defaultdict(dict)
    for r in model_scores:
        key=(r['generator_replication'],r['history_plies'],r['task'])
        by_rep[key][r['architecture']]=float(r['macro_record_accuracy'])
    groups=collections.defaultdict(list)
    for (rep,hist,task),by_model in by_rep.items():
        for a,b in itertools.combinations(sorted(by_model),2):
            groups[(hist,task,a,b)].append((rep,by_model[a]-by_model[b]))
    summaries=[]
    for (hist,task,a,b),vals in sorted(groups.items()):
        if {r for r,_ in vals}!=used:raise ValueError('Unmatched independent replications')
        d=[x for _,x in sorted(vals)]
        enough=len(d)>=min_replications
        low,high=_cluster_bootstrap(d,bootstrap,seed) if enough else (None,None)
        pvalue=_p_signflip(d,seed=seed) if enough else None
        summaries.append({'history_plies':hist,'task':task,'model_a':a,'model_b':b,
                          'generator_replications':len(d),'mean_accuracy_difference':_mean(d),
                          'replicate_bootstrap_ci_95_low':low,
                          'replicate_bootstrap_ci_95_high':high,
                          'paired_signflip_p_unadjusted':pvalue,
                          'holm_p_adjusted':None,
                          'inference_status':'conditional_replication_inference' if enough else 'descriptive_only',
                          'scope':'Fixed three underlying synthetic mechanisms; no external-validity inference'})
    eligible=sorted((i for i,r in enumerate(summaries) if r['paired_signflip_p_unadjusted'] is not None),
                    key=lambda i:summaries[i]['paired_signflip_p_unadjusted'])
    previous=0.0;count=len(eligible)
    for rank,index in enumerate(eligible):
        adjusted=min(1.,(count-rank)*summaries[index]['paired_signflip_p_unadjusted'])
        previous=max(previous,adjusted)
        summaries[index]['holm_p_adjusted']=previous
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'replicate_model_scores.csv',model_scores)
    write_csv(out/'paired_cluster_statistics.csv',summaries)
    protocol={'generator_replications':sorted(used),
              'minimum_independent_replications_for_inference':min_replications,
              'confidence_interval_method':'nonparametric bootstrap over generator replications',
              'hypothesis_test':'paired sign flip over generator replications',
              'multiplicity':'Holm across all model pairs / tasks / cutoffs in the supplied experiment',
              'cluster_independence_assumption':'independently re-seeded generation conditional on fixed motifs',
              'important':'Inference is conditional on synthetic mechanisms, NOT independent new generator families or real chess games.'}
    (out/'statistics_protocol.json').write_text(json.dumps(protocol,indent=2))
    return summaries


def analyze_overfitting(result_directory,out_csv):
    folder=Path(result_directory)
    curves=read_csv(folder/'architecture_learning_curves.csv')
    metrics=read_csv(folder/'architecture_metrics.csv')
    choices={}
    for m in metrics:
        if m['task']=='all':
            choices[(m['architecture'],m['fold'],m['seed'],m['history_plies'])]=m
    grouped=collections.defaultdict(list)
    for r in curves:
        history='full' if r['history'] in ('None','full') else r['history']
        key=(r['architecture'],r['fold'],r['seed'],history)
        if key not in choices:continue
        wanted=int(choices[key].get('selected_grid_trial',0))
        if int(r.get('grid_trial',0))!=wanted:continue
        grouped[key].append(r)
    diagnostics=[]
    for key,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:int(r['epoch']))
        chosen=choices[key]
        best=int(chosen['best_epoch'])
        best_row=next(x for x in rows if int(x['epoch'])==best)
        final=rows[-1]
        diagnostics.append({'architecture':key[0],'fold':key[1],'seed':key[2],
                            'history_plies':key[3],'selected_lr':chosen.get('selected_lr',''),
                            'selected_dropout':chosen.get('selected_dropout',''),
                            'selected_weight_decay':chosen.get('selected_weight_decay',''),
                            'selected_grid_trial':chosen.get('selected_grid_trial',0),
                            'best_epoch':best,'last_epoch':int(final['epoch']),
                            'train_loss_at_best':float(best_row['train_loss']),
                            'validation_loss_at_best':float(best_row['validation_loss']),
                            'train_loss_last':float(final['train_loss']),
                            'validation_loss_last':float(final['validation_loss']),
                            'last_generalization_gap':float(final['validation_loss'])-float(final['train_loss']),
                            'validation_degradation_from_best':float(final['validation_loss'])-float(best_row['validation_loss']),
                            'note':'Descriptive learning dynamics; do not infer universal architecture limitations'})
    write_csv(out_csv,diagnostics)
    return diagnostics
