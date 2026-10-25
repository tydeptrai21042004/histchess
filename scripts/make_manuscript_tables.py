"""Generate descriptive manuscript tables ONLY from fully audited completed v6 data.

This is formatting, not inferential statistics.  Do not cite the generated tables
as evidence of external validity or architectural impossibility.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


def load_rows(filename):
    with Path(filename).open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition: raise RuntimeError(message)


def pretty(name):
    return {'elman_rnn':'Elman RNN','lstm':'LSTM','gru':'GRU',
            'causal_transformer':'Causal Transformer',
            'linear_attention':'Linear Attention',
            'diagonal_ssm':'Real Diagonal SSM',
            'board_only':'Board-only null','move_bag':'Move bag + LR',
            'move_ngrams':'Move n-grams + LR'}.get(name,name)


def generate(root):
    root=Path(root)
    ast=json.loads((root/'architecture_status.json').read_text())
    pst=json.loads((root/'pipeline_status.json').read_text())
    cls=json.loads((root/'comparison_status.json').read_text())
    require(ast.get('complete') and cls.get('complete'),'Incomplete experiment')
    # The pipeline status is not marked complete until AFTER table generation.
    require(not pst.get('paused'),'Cannot publish partial results')
    for audit in ('independent_integrity_audit.json','comparison_integrity_audit.json'):
        require(json.loads((root/audit).read_text()).get('passed'),f'Audit failed: {audit}')
    config=json.loads((root/'architecture_config.json').read_text())
    require(config.get('protocol_version')=='histchess-v6-publication-full-2026-10',
            'Unexpected protocol version')
    require(len(ast['runs'])==len(config['models'])*len(config['folds'])*len(config['seeds'])*len(config['cutoffs']),
            'Missing neural configurations')
    require(len(ast['runs'])==450 and len(config['models'])==6 and len(config['folds'])==3
            and len(config['seeds'])==5 and len(config['cutoffs'])==5,
            'Manuscript export requires ALL 450 neural configurations')
    neural=load_rows(root/'architecture_metrics.csv')
    classical=load_rows(root/'comparison_metrics.csv')
    pairs=[json.loads(x) for x in (root/'data/pairs.jsonl').read_text().splitlines() if x.strip()]
    grouped=collections.defaultdict(list)
    for item in neural:
        grouped[(item['architecture'],item['history_plies'],item['task'])].append(item)
    outrows=[]
    for (model,cutoff,task),rows in sorted(grouped.items()):
        r=[float(x['record_accuracy']) for x in rows]
        j=[float(x['joint_pair_accuracy']) for x in rows]
        q=[float(x['pair_ranking_accuracy']) for x in rows]
        require(len(rows)==len(config['folds'])*len(config['seeds']),f'Incomplete {model}/{cutoff}/{task}')
        outrows.append({'architecture':model,'history_plies':cutoff,'task':task,
             'fold_seed_runs':len(rows),'record_accuracy_mean':statistics.mean(r),
             'record_accuracy_sd_runs':statistics.pstdev(r),
             'joint_pair_accuracy_mean':statistics.mean(j),
             'ranking_accuracy_mean':statistics.mean(q),
             'interpretation':'descriptive, folds and seeds correlated'})
    with (root/'manuscript_task_breakdown.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
    tbl={ (x['architecture'],x['history_plies'],x['task']):x for x in outrows}
    models=list(config['models'])
    histories=[str(x) for x in config['cutoffs']]
    lines=['# HistChess v6 — measured manuscript tables','',
           '> Descriptive only. These are measured values, NOT independent-generator confidence intervals.',
           '> The data are synthetic and all learned models were trained from scratch.','',
           '## Neural architectures (all four tasks pooled)','',
           '| Architecture | History (plies) | Record accuracy (mean ± SD) | Joint-pair accuracy | Runs |',
           '|---|---:|---:|---:|---:|']
    latex=[r'% Auto-generated ONLY after independent neural and classical audits passed.',
           r'% Fold and seed results are correlated. This table is descriptive.',
           r'\begin{table}[t]',r'\centering',
           r'\caption{HistChess synthetic counterfactual benchmark: descriptive neural results. Fold-seed runs are correlated; SD is not an independent-family uncertainty estimate.}',
           r'\begin{tabular}{llrrr}',r'\hline',
           r'Architecture & History & Record accuracy & Joint-pair accuracy & Runs \\',r'\hline']
    for m in models:
        for h in histories:
            r=tbl[m,h,'all']; acc=float(r['record_accuracy_mean']);sd=float(r['record_accuracy_sd_runs']);joint=float(r['joint_pair_accuracy_mean']);n=r['fold_seed_runs']
            lines.append(f'| {pretty(m)} | {h} | {acc:.4f} ± {sd:.4f} | {joint:.4f} | {n} |')
            latex.append(f'{pretty(m).replace("_",r"\_")} & {h} & {acc:.4f} $\\pm$ {sd:.4f} & {joint:.4f} & {n} \\\\')
    latex.extend([r'\hline',r'\end{tabular}',r'\end{table}'])
    lines+=['','## Classical controls (same held-out pairs)','',
            '| Baseline | History | Record accuracy | Joint-pair accuracy | Folds |',
            '|---|---:|---:|---:|---:|']
    classgroups=collections.defaultdict(list)
    for r in classical:
        if r['task']=='all':classgroups[(r['baseline'],r['history_plies'])].append(r)
    for (name,h),records in sorted(classgroups.items()):
        if name=='privileged_replay_oracle':continue
        lines.append(f"| {pretty(name)} | {h} | {statistics.mean(float(x['record_accuracy']) for x in records):.4f} | {statistics.mean(float(x['joint_pair_accuracy']) for x in records):.4f} | {len(records)} |")
    lengths=[len(p[v]['history_uci']) for p in pairs for v in ('positive','negative')]
    task_counts=collections.Counter(p['task'] for p in pairs)
    families={task:sorted({p['generator_family'] for p in pairs if p['task']==task}) for task in sorted(task_counts)}
    sha=hashlib.sha256((root/'data/pairs.jsonl').read_bytes()).hexdigest()
    manifest={'protocol':config['protocol_version'],'architecture_runs':len(ast['runs']),
              'models':models,'folds':config['folds'],'seeds':config['seeds'],
              'cutoffs':histories,'train_options':config['train_protocol'],
              'total_counterfactual_pairs':len(pairs),'pairs_per_task':dict(task_counts),
              'generator_families_per_task':families,
              'history_lengths':{'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)},
              'dataset_file_sha256':sha,'neural_audit_passed':True,'classical_audit_passed':True,
              'limitations':['Synthetic counterfactual trajectories, not external real-game evaluation',
                             'Only three mechanisms per task',
                             'Repeated folds and random seeds are correlated',
                             'History-length cutoff reassessment is not train-short/test-long on new long histories',
                             'No pretrained large language models, no full published S4D implementation',
                             'Descriptive effects, not corrected confirmatory inference']}
    lines+=['','## Data and reporting restrictions','',
            f'- Counterfactual pairs: {len(pairs)}; history lengths (plies): min {min(lengths)}, median {statistics.median(lengths)}, max {max(lengths)}.',
            f'- Neural runs audited: {len(ast["runs"])}. Folds: {config["folds"]}; seeds: {config["seeds"]}.',
            '- Privileged replay oracle excluded from fair baseline table.',
            '- Length-cutoff reassessment reuses the same examples; it is not independent long-sequence generalization.',
            '- Mean ± SD values describe correlated fold-seed runs; not a confidence interval over new generator families.',
            '- Inspect learning curves and validation losses before asserting convergence.',
            '- Do not call this an external LLM benchmark.']
    (root/'manuscript_results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (root/'manuscript_neural_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf8')
    (root/'manuscript_methods_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('MANUSCRIPT TABLES WRITTEN:',len(outrows),'per-task rows,',len(ast['runs']),'audited neural runs',flush=True)
    return manifest

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: python scripts/make_manuscript_tables.py <output-dir>')
    generate(sys.argv[1])
