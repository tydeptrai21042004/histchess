"""V8 held-out scenario and genuine-length dataset generator.

Nine scenario groups per chess task (three original *mechanisms* times
three length regimes). IMPORTANT: this does NOT create nine independently
invented chess-rule mechanisms; provenance records the underlying three
motifs. Independently re-seeded replications measure generator randomness
conditional on those motifs, not transfer to unconstrained real games.

Every example is chess-verified; requested history lengths are minimums,
not fabricated exact lengths. For EP, halfmove and repetition the extension
usually precedes the short decisive suffix: these are long inputs but are NOT
certified long causal-memory problems. Castling uses a common continuation
AFTER the decisive event, with its distance measured explicitly.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from .core import TASKS, validate, write_jsonl
from .diverse import FAMILIES, generate_diverse
from .length_stress import extend_pair

LEVELS = ("native", "32", "64", "128", "256")
SCENARIO_FAMILIES = {task: tuple(f"{mechanism}__{level}"
                                   for mechanism in FAMILIES[task] for level in LEVELS)
                           for task in TASKS}


def generate_scenario_pairs(per_family=4, seed=3101, lengths=("native", "32", "64"),
                            attempts=100, generator_replication=0):
    """Return verified pairs without changing existing v6/v7 generators.

    Each task/motif/length scenario gets exactly ``per_family`` pairs.
    ``lengths`` draws from LEVELS. If a target is impossible, FAIL CLOSED:
    never substitute shorter histories or silently omit a difficult task.
    """
    if not isinstance(per_family,int) or per_family<1:
        raise ValueError('per_family must be a positive integer')
    if not lengths or len(set(lengths))!=len(lengths) or any(x not in LEVELS for x in lengths):
        raise ValueError(f'lengths must be distinct values from {LEVELS}')
    if attempts<1:raise ValueError('attempts must be positive')
    # Generate extra bases so a later context extension can be retried without
    # reusing an observed board. Source mechanism diversity remains THREE.
    required=per_family*len(lengths)
    pool_per_family=max(required+max(6,per_family),2*required)
    base,_=generate_diverse(count_per_task=3*pool_per_family,
                            seed=seed+generator_replication*100003,
                            split_strategy='family')
    indexed=collections.defaultdict(list)
    for pair in base:indexed[(pair['task'],pair['generator_family'])].append(pair)
    result=[];seen_boards=set();rejections=collections.Counter()
    for task in TASKS:
        for group,mechanism in enumerate(FAMILIES[task]):
            candidates=iter(indexed[(task,mechanism)])
            for level in lengths:
                made=0
                for candidate in candidates:
                    if made>=per_family:break
                    try:
                        if level=='native':
                            p=dict(candidate)
                            p['first_divergence_distance']=len(p['positive']['history_uci'])-1-next(
                                (i for i,(a,b) in enumerate(zip(
                                  p['positive']['history_uci'],p['negative']['history_uci'])) if a!=b),
                                len(p['positive']['history_uci']))
                            p['stress_protocol']='original_verified_short_history'
                        else:
                            p=extend_pair(candidate,int(level),
                                          seed=seed+generator_replication*100003+len(result)*23,
                                          attempts=attempts)
                        if p['board_key'] in seen_boards:
                            rejections[f'{task}:{mechanism}:{level}:duplicate_board']+=1
                            continue
                        seen_boards.add(p['board_key'])
                        p['generator_family']=f'{mechanism}__{level}'
                        p['independent_mechanism']=mechanism
                        p['scenario_context']=level
                        p['replicate_id']=generator_replication
                        p['split']=('train','validation','test')[group]
                        p['requested_minimum_plies']=0 if level=='native' else int(level)
                        p['actual_plies']=len(p['positive']['history_uci'])
                        p['long_causal_distance_certified']=(
                            task=='castling' and p['first_divergence_distance']>=32)
                        result.append(p);made+=1
                    except (ValueError,AssertionError,RuntimeError) as exc:
                        rejections[f'{task}:{mechanism}:{level}:{type(exc).__name__}']+=1
                if made!=per_family:
                    raise RuntimeError(f'Insufficient independently validated pairs: '
                        f'{task}/{mechanism}/{level}: {made}/{per_family}; {dict(rejections)}')
    check=validate(result)
    if not check['passed']:
        raise RuntimeError('Independent chess legality audit failed: '+json.dumps(check['errors'][:5]))
    for task in TASKS:
        for level in lengths:
            q=[p for p in result if p['task']==task and p['scenario_context']==level]
            assert len(q)==3*per_family
            if level!='native':
                assert all(p['actual_plies']>=int(level) for p in q)
    return result,dict(rejections)


def dataset_report(pairs, *, require_long_causal=32):
    rows=[]
    for p in pairs:
        length=len(p['positive']['history_uci'])
        assert length==len(p['negative']['history_uci'])
        rows.append({'pair_id':p['pair_id'],'task':p['task'],
                     'scenario_family':p['generator_family'],
                     'independent_mechanism':p.get('independent_mechanism',p['generator_family']),
                     'generator_replication':p.get('replicate_id',0),
                     'requested_minimum_plies':p.get('requested_minimum_plies',0),
                     'actual_history_plies':length,
                     'first_divergence_distance':p.get('first_divergence_distance',''),
                     'certified_long_dependency':int(
                       p['task']=='castling' and
                       p.get('first_divergence_distance',-1)>=require_long_causal),
                     'explanation':('Long causal gap in castling-rights history' if p['task']=='castling'
                                    else 'Long input only; decisive event generally near final moves')})
    return rows


def write_scenario_dataset(out,pairs,rejections=None):
    from .architecture_benchmark import _write_csv
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    write_jsonl(path/'pairs.jsonl',pairs)
    _write_csv(path/'length_and_dependency_audit.csv',dataset_report(pairs))
    content=(path/'pairs.jsonl').read_bytes()
    (path/'dataset_sha256.txt').write_text(hashlib.sha256(content).hexdigest()+'\n')
    manifest={'protocol':'histchess-v8-scenario-conditional-replications',
              'dataset_sha256':hashlib.sha256(content).hexdigest(),
              'pair_count':len(pairs),'generator_rejections':rejections or {},
              'independent_base_mechanisms_per_task':{
                  task:len({p.get('independent_mechanism',p['generator_family'])
                            for p in pairs if p['task']==task}) for task in TASKS},
              'scenario_family_count_per_task':{
                  task:len({p['generator_family'] for p in pairs if p['task']==task})
                  for task in TASKS},
              'limitations':[
                 'Nine scenario groups can arise from three underlying motif families; do not call them nine independent generator mechanisms.',
                 'Generator-seed replications estimate conditional synthetic variability, not population-of-chess uncertainty.',
                 'Only castling can be certified as a distant causal-event experiment with this stress extension.',
                 'Chess-verified synthetic trajectories are not independently sampled real-game trajectories.']}
    (path/'protocol_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def merge_external_mechanisms(built_in,external_path):
    """Add truly separate, user-supplied counterfactual generation sources.

    Each external pair must contain independent_mechanism and source_provenance;
    these are declarations, NOT external-authenticity proofs. Every pair is
    independently replayed, with board and ID disjointness enforced.
    External generators are assigned three disjoint fold groups by TASK.
    """
    from .core import read_jsonl
    source=read_jsonl(external_path)
    if not source:raise ValueError('External source has no pairs')
    for p in source:
        if p.get('task') not in TASKS:raise ValueError('Invalid external task')
        if not p.get('independent_mechanism') or not p.get('source_provenance'):
            raise ValueError('External pair requires independent_mechanism and source_provenance')
        if not p.get('generator_family'):
            raise ValueError('External pair requires generator_family')
    buckets={}
    for task in TASKS:
        names=sorted({p['independent_mechanism'] for p in source if p['task']==task})
        buckets[task]={name:i%3 for i,name in enumerate(names)}
    existing_mechanisms={(p['task'],p['independent_mechanism']) for p in built_in}
    for p in source:
        if (p['task'],p['independent_mechanism']) in existing_mechanisms:
            raise ValueError('External mechanism name collides with built-in name')
        group=buckets[p['task']][p['independent_mechanism']]
        p['mechanism_fold_group']=group
        p['split']=('train','validation','test')[group]
    all_pairs=list(built_in)+source
    boards=[p['board_key'] for p in all_pairs]
    if len(set(boards))!=len(boards):
        raise ValueError('External source duplicates an observable board')
    report=validate(all_pairs)
    if not report['passed']:
        raise RuntimeError('External source failed replay: '+json.dumps(report['errors'][:5]))
    return all_pairs
