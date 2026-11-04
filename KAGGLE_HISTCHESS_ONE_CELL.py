# HistChess — ONE paper protocol, ONE Kaggle cell, ONE resumable results archive.
# Kaggle Settings: Internet ON; GPU T4 (or better) ON.
# To CONTINUE, attach HistChess_Study_Results.zip as Kaggle Input and run this SAME cell.
# The supplied starter ZIP already contains 462 of 900 audited training configurations.
# On the first run WITHOUT a checkpoint attached, clone GitHub and train from ZERO.
# Subsequent sessions can attach the latest results ZIP to resume automatically.
#
# Scientific protocol (never silently reduced):
#   5 independently seeded synthetic datasets; 6 architectures; 3 folds;
#   2 training seeds; 5 context lengths = 900 neural configurations,
#   4 validation-selected optimizer trials per configuration;
#   matching classical baselines, real length-shift tests, independent
#   integrity checks, replication-level inference, and two pretrained LLMs.
#
# This launcher changes scheduling/checkpoint packaging plus an inference-only
# interruption guard. Scientific neural models, data and fingerprints are unchanged.

from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

REPOSITORY = 'https://github.com/tydeptrai21042004/histchess.git'
BRANCH = 'main'
WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
SOURCE = WORK / 'histchess_source'
ROOT = WORK / 'HistChess_Study_Results'
STUDY = ROOT / 'study'
ARCHIVE = WORK / 'HistChess_Study_Results.zip'

# Kaggle's practical session limit may vary. Save before the limit, even when
# 900 configurations cannot fit in a single GPU session.
SESSION_COMPUTE_MINUTES = 500
START_RESERVE_SECONDS = 720
CHUNK_NEW_TRAINING_CONFIGS = 12
EXPECTED_REPLICATIONS = 5
EXPECTED_CONFIGS_PER_REPLICATION = 180
TOTAL_CONFIGS = 900
ARCHITECTURES = [
    'elman_original', 'lstm', 'gru_original',
    'causal_transformer_postnorm', 'linear_attention_multihead', 's4d_lin'
]
SEEDS = ['3101', '3102']
CONTEXT_CUTOFFS = ['0', '16', '32', '64', 'full']
LLM_MODELS = [
    'HuggingFaceTB/SmolLM2-135M-Instruct',
    'HuggingFaceTB/SmolLM2-360M-Instruct'
]

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _check_member(info: zipfile.ZipInfo, prefix: str) -> Path:
    name = info.filename
    if not name.startswith(prefix) or '\\' in name:
        raise RuntimeError('Incorrect checkpoint member: ' + name)
    relative = name[len(prefix):]
    pieces = Path(relative).parts
    if not relative or any(p in ('.', '..') for p in pieces) or Path(relative).is_absolute():
        raise RuntimeError('Unsafe checkpoint path: ' + name)
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError('Symlink not permitted in checkpoint: ' + name)
    return Path(relative)


def restore_checkpoint(path: Path, root: Path = ROOT):
    """Restores ONLY the single selected experiment, no unrelated old tracks."""
    if any(root.iterdir()) if root.exists() else False:
        raise RuntimeError('Output folder is not empty; cannot overwrite an in-progress study')
    with zipfile.ZipFile(path) as z:
        corrupt = z.testzip()
        if corrupt:
            raise RuntimeError('Corrupt archive member: ' + corrupt)
        prefix = root.name + '/'
        for info in z.infolist():
            if info.is_dir():
                continue
            relative = _check_member(info, prefix)
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe output location in checkpoint')
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    if not (root/'study_manifest.json').is_file():
        raise RuntimeError('Missing pinned source manifest in restored study')
    if not (root/'study'/'v8_config.json').is_file():
        raise RuntimeError('Missing immutable scientific protocol in restored study')


def pack_checkpoint(root: Path = ROOT, archive: Path = ARCHIVE):
    if not root.is_dir():
        return
    partial = archive.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file():
                z.write(f, root.name + '/' + f.relative_to(root).as_posix())
    with zipfile.ZipFile(partial) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Failed ZIP CRC validation: ' + bad)
    partial.replace(archive)
    print('CHECKPOINT:', archive, 'MiB:', round(archive.stat().st_size/1048576, 2), flush=True)


def run_command(argv, *, cwd=None, time_budget=None):
    argv = [str(x) for x in argv]
    print('\n$', ' '.join(argv), flush=True)
    proc = subprocess.Popen(argv, cwd=cwd, start_new_session=True)
    try:
        code = proc.wait(timeout=time_budget)
    except subprocess.TimeoutExpired:
        print('Session time budget reached; preserving completed training shards.', flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return False
    if code != 0:
        raise RuntimeError(f'Process exit {code}: {" ".join(argv)}')
    return True


def _study_cmd(include_llm: bool):
    cmd = [sys.executable, 'scripts/run_v8_research.py',
        '--out', STUDY, '--replications', '5',
        '--per-family', '8', '--levels', 'native', '32', '64',
        '--long-attempts', '100', '--architectures', *ARCHITECTURES,
        '--seeds', *SEEDS, '--cutoffs', *CONTEXT_CUTOFFS,
        '--length-eval-cutoffs', '32', '64', 'full',
        '--length-eval-train-cutoffs', '16', '--max-history', '256',
        '--epochs', '50', '--patience', '7', '--batch-size', '64',
        '--target-parameters', '150000', '--device', 'cuda',
        '--grid-lrs', '0.0003', '0.001',
        '--grid-weight-decays', '0.001', '--grid-dropouts', '0.1', '0.3',
        '--max-new-runs', str(CHUNK_NEW_TRAINING_CONFIGS),
        '--bootstrap', '5000']
    if include_llm:
        cmd += ['--run-llm', '--llm-models', *LLM_MODELS,
                '--llm-pairs-per-task', '4', '--llm-batch-size', '4']
    return cmd


def count_progress(root: Path = ROOT):
    neural = 0
    rep_status = []
    audited = True
    for rep in range(EXPECTED_REPLICATIONS):
        directory = root / 'study' / f'replication_{rep:02d}' / 'neural'
        state = read_json(directory / 'architecture_status.json')
        count = len(state.get('runs', []))
        complete = bool(state.get('complete') and count == EXPECTED_CONFIGS_PER_REPLICATION)
        audits_ok = (
            read_json(directory / 'independent_integrity_audit.json').get('passed') is True and
            read_json(directory / 'comparison_integrity_audit.json').get('passed') is True
        )
        neural += count
        rep_status.append({'replication': rep + 1, 'configs': count,
                           'expected_configs': EXPECTED_CONFIGS_PER_REPLICATION,
                           'training_complete': complete,
                           'audits_passed': audits_ok})
        audited &= complete and audits_ok
    original = read_json(root / 'study' / 'v8_status.json')
    stats = (root / 'study' / 'statistics' / 'paired_cluster_statistics.csv')
    statistics_ok = bool(original.get('statistics_complete') and stats.is_file() and stats.stat().st_size > 100)
    llms_ok = bool(original.get('llm_complete'))
    if llms_ok:
        for rep in range(EXPECTED_REPLICATIONS):
            d = root / 'study' / f'replication_{rep:02d}' / 'pretrained_llm'
            llms_ok &= (d / 'llm_metrics.csv').is_file() and (d / 'llm_predictions.csv').is_file()
    return {'study': 'HistChess Long-History Counterfactual Study',
            'neural_configs_completed': neural,
            'neural_configs_expected': TOTAL_CONFIGS,
            'neural_configs_remaining': max(0, TOTAL_CONFIGS - neural),
            'replications': rep_status,
            'neural_and_baseline_audits_passed': bool(audited),
            'statistics_complete': statistics_ok,
            'pretrained_llms_complete': bool(llms_ok),
            'all_complete': bool(audited and statistics_ok and llms_ok)}


def save_progress(root: Path = ROOT):
    progress = count_progress(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / 'study_progress.partial'
    tmp.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    tmp.replace(root/'study_progress.json')
    print('ONE STUDY PROGRESS:', json.dumps({
        k:v for k,v in progress.items() if k != 'replications'}, indent=2), flush=True)
    return progress


def _apply_parser_test_fix(source: Path):
    """Only align two stale tests with the unchanged strict LLM parser."""
    file = source / 'tests' / 'test_encoding_metrics.py'
    if not file.is_file():
        raise RuntimeError('Expected parser test file is absent')
    code = file.read_text(encoding='utf-8')
    for old, new in (("('YES because',True)", "('YES because',None)"),
                     ("('No? Maybe',False)", "('No? Maybe',None)")):
        if old in code:
            if code.count(old) != 1:
                raise RuntimeError('Parser expectations do not match audited version')
            code = code.replace(old, new)
        elif new not in code:
            raise RuntimeError('Parser expectations changed; independent review required')
    file.write_text(code, encoding='utf-8')
    parser = (source/'histchess'/'experiments.py').read_text(encoding='utf-8')
    if r're.fullmatch(r"(?:answer\s*:\s*)?(yes|no)[.!]?",s)' not in parser:
        raise RuntimeError('Production LLM parser changed; cannot assume prior protocol')


def _apply_interrupted_llm_fix(source: Path):
    """Fix completed-but-partial LLM files on this verified source version.

    Idempotent: if GitHub already contains the fix, leave it untouched.
    This only updates the LLM output recovery in the orchestration script;
    it does not change architecture/model/dataset modules.
    """
    path = source / 'scripts' / 'run_v8_research.py'
    code = path.read_text(encoding='utf-8')
    old = "                if not metric_file.exists():\n                    from histchess.experiments import llm_experiment\n                    fold_zero=rotate_splits(pairs,0)\n                    llm_experiment(fold_zero,llmdir,args.llm_models,\n                                   pairs_per_task=args.llm_pairs_per_task,\n                                   seed=args.generator_seed+rep*100003,\n                                   batch_size=args.llm_batch_size,\n                                   require_gpu=not args.allow_cpu_llm)\n                revisions=llmdir/'pretrained_model_revisions.json'\n"
    new = "                # An interrupted inference may leave only some output files.\n                # Resume by recomputing that replica, never by trusting a lone metrics file.\n                def llm_artifacts_complete():\n                    revisions=llmdir/'pretrained_model_revisions.json'\n                    predictions=llmdir/'llm_predictions.csv'\n                    if not all(f.is_file() for f in (metric_file,revisions,predictions)):\n                        return False\n                    try:\n                        versions=json.loads(revisions.read_text())\n                        with metric_file.open(newline='') as f:\n                            metrics=list(csv.DictReader(f))\n                        with predictions.open(newline='') as f:\n                            rows=list(csv.DictReader(f))\n                    except (OSError,ValueError,KeyError):\n                        return False\n                    if [x.get('model') for x in versions]!=args.llm_models:\n                        raise RuntimeError('Existing LLM outputs have different model IDs; '\n                                           'do not mix incompatible evaluations')\n                    return (len(metrics)==5*len(args.llm_models) and\n                            len(rows)==2*4*args.llm_pairs_per_task*len(args.llm_models))\n\n                if not llm_artifacts_complete():\n                    # The model outputs are written at the end of inference.\n                    # Delete any incomplete output set before recomputing it.\n                    for fname in ('llm_predictions.csv','llm_metrics.csv',\n                                  'pretrained_model_revisions.json'):\n                        (llmdir/fname).unlink(missing_ok=True)\n                    from histchess.experiments import llm_experiment\n                    fold_zero=rotate_splits(pairs,0)\n                    llm_experiment(fold_zero,llmdir,args.llm_models,\n                                   pairs_per_task=args.llm_pairs_per_task,\n                                   seed=args.generator_seed+rep*100003,\n                                   batch_size=args.llm_batch_size,\n                                   require_gpu=not args.allow_cpu_llm)\n                if not llm_artifacts_complete():\n                    raise RuntimeError('Pretrained LLM outputs remain incomplete')\n                revisions=llmdir/'pretrained_model_revisions.json'\n"
    if old in code:
        if code.count(old) != 1:
            raise RuntimeError('Unexpected duplicates in LLM recovery block')
        path.write_text(code.replace(old, new), encoding='utf-8')
        print('Applied interrupted-LLM-output recovery guard', flush=True)
    elif new in code:
        print('Interrupted-LLM-output recovery guard already present', flush=True)
    else:
        raise RuntimeError('Unexpected upstream run_v8_research.py; refusing unsafe patch')


def setup_source(root: Path = ROOT, source: Path = SOURCE):
    if source.exists():
        shutil.rmtree(source)
    run_command(['git', 'clone', '--branch', BRANCH, REPOSITORY, source])
    manifest = read_json(root/'study_manifest.json')
    pinned = manifest.get('source_commit')
    if pinned:
        run_command(['git', 'checkout', pinned], cwd=source)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if pinned and commit != pinned:
        raise RuntimeError('Source commit mismatch; will not mix experiments')
    needed = ['scripts/run_v8_research.py', 'scripts/verify_architecture_results.py',
              'scripts/run_comparison_baselines.py', 'histchess/architecture_benchmark.py']
    if any(not (source/n).is_file() for n in needed):
        raise RuntimeError('Missing required scientific implementation at source commit')
    _apply_parser_test_fix(source)
    _apply_interrupted_llm_fix(source)
    return commit


def main():
    assert WORK.is_dir() and INPUT.is_dir(), 'Run this cell in Kaggle; enable Internet + GPU.'
    started = time.monotonic()
    deadline = started + SESSION_COMPUTE_MINUTES * 60
    previous = list(INPUT.rglob(ARCHIVE.name))
    if len(previous) > 1:
        raise RuntimeError('Attach just ONE latest HistChess_Study_Results.zip checkpoint.')
    if previous and not (ROOT/'study_manifest.json').exists():
        print('Restoring the selected HistChess study from:', previous[0], flush=True)
        restore_checkpoint(previous[0])
    commit = setup_source()
    run_command([sys.executable, '-m', 'pip', 'install', '-q',
                 'chess==1.11.2', 'scikit-learn>=1.3,<2', 'pytest>=7,<10',
                 'transformers>=4.45,<5', 'safetensors>=0.4', 'matplotlib>=3.7,<4'])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU accelerator is required. No CPU fallback.')
    manifest = read_json(ROOT/'study_manifest.json')
    if manifest:
        if manifest['source_commit'] != commit:
            raise RuntimeError('Pinned GitHub commit changed; abort before touching results')
        if manifest['torch_version'] != torch.__version__:
            raise RuntimeError('PyTorch version differs from saved shard fingerprints. '
                               f'Required: {manifest["torch_version"]}; available: {torch.__version__}. '
                               'Use the original Kaggle runtime; never relabel/reuse incompatible shards.')
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'study_manifest.json').write_text(json.dumps({
            'study': 'HistChess Long-History Counterfactual Study',
            'source_commit': commit, 'repository': REPOSITORY,
            'torch_version': torch.__version__,
            'experiment_parameters': {
                'replications': 5, 'architectures': ARCHITECTURES,
                'folds': [0, 1, 2], 'seeds': SEEDS, 'context_cutoffs': CONTEXT_CUTOFFS,
                'pairs_per_mechanism_and_length': 8, 'lengths': ['native','32','64'],
                'epochs': 50, 'patience': 7, 'optimizer_trials': 4,
                'trainable_parameter_target': 150000, 'bootstrap': 5000,
                'pretrained_llms': LLM_MODELS}}, indent=2))
    print('PINNED SOURCE:', commit, '| CUDA:', torch.cuda.get_device_name(0), flush=True)
    if not read_json(ROOT/'preflight_passed.json').get('commit') == commit:
        run_command([sys.executable, '-m', 'pytest', '-q', 'tests', '-m',
                     'not slow and not gpu_integration',
                     '--ignore=tests/test_length_stress.py'], cwd=SOURCE)
        run_command([sys.executable, '-m', 'pytest', '-q',
                     'tests/test_v8_generator.py', '-m', 'slow'], cwd=SOURCE)
        (ROOT/'preflight_passed.json').write_text(json.dumps({'commit':commit,'passed':True},indent=2))
    info = save_progress()
    while not info['neural_and_baseline_audits_passed']:
        remaining = deadline - time.monotonic()
        if remaining < START_RESERVE_SECONDS:
            print('SESSION CHECKPOINT: continue next Kaggle session with saved ZIP.', flush=True)
            break
        old = info['neural_configs_completed']
        # During the neural phase, DO NOT load/download LLMs. They cannot block
        # completion of the neural experiments or their matched classical controls.
        success = run_command(_study_cmd(include_llm=False), cwd=SOURCE,
                              time_budget=remaining - 150)
        info = save_progress()
        if not success:
            break
        if not info['neural_and_baseline_audits_passed'] and info['neural_configs_completed'] <= old:
            raise RuntimeError('The benchmark made no progress; inspect the checkpoint error.')
    # Only after ALL independent neural and classical audits pass do LLMs run.
    # This keeps one study, with two execution phases, not multiple proposals.
    if info['neural_and_baseline_audits_passed'] and not info['all_complete']:
        remaining = deadline - time.monotonic()
        if remaining >= START_RESERVE_SECONDS:
            run_command(_study_cmd(include_llm=True), cwd=SOURCE,
                        time_budget=remaining - 150)
        else:
            print('The pretrained evaluations will resume next Kaggle session.', flush=True)
        info = save_progress()
    if info['all_complete']:
        print('COMPLETE: the single HistChess study and all its audits are complete.',flush=True)
    else:
        print('INCOMPLETE (VALID CHECKPOINT): attach the new ZIP and rerun SAME cell.',flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        ROOT.mkdir(parents=True,exist_ok=True)
        (ROOT/'launcher_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        traceback.print_exc()
        raise
    finally:
        # Archive even on a recoverable timeout/failure; previous shards are retained.
        save_progress()
        pack_checkpoint()
