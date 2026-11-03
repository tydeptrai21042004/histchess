# HISTCHESS: COMPLETE GITHUB -> KAGGLE RESULTS, ONE PYTHON CELL
# Kaggle Settings: Internet = ON, Accelerator = GPU (T4/P100 recommended).
# First session: run this cell with NO previous-results input.
# Later session: attach the most recent HistChess_GitHub_ALL_Results.zip as
# a Kaggle Input and run this EXACT SAME cell. Do not change its settings.
# Parser unit-test expectations are corrected after clone; model code unchanged.
# Every track starts from scratch when there is no checkpoint. Completed
# training configurations are reused only on an explicitly resumed run.

import csv
import hashlib
import json
import os
import shutil
import signal
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
SOURCE = WORK / 'histchess_github'
OUT = WORK / 'HistChess_GitHub_ALL_Results'
ARCHIVE = WORK / 'HistChess_GitHub_ALL_Results.zip'

# All result-producing tracks are ON by default.
RUN_V8 = True                       # genuine 32/64-ply, 5 replications, HPO, stats, LLMs
RUN_V7_FOUNDATION_FULL = True       # original 450 full foundation-model configs
RUN_V6_LEGACY_FULL = True           # original 450 full legacy-model configs
RUN_V6_CPU_EXTENSION = True         # shortcut and legal-replay controls
RUN_MODEL_EFFICIENCY = True         # eleven architectures, length 16..256
RUN_PRETRAINED_LLMS = True          # real SmolLM2-135M/360M checkpoints on v8 pairs

# True paper protocol: DO NOT reduce these if reporting full results.
V8_REPLICATIONS = 5
V8_PAIRS_PER_FAMILY = 8
V8_LEVELS = ['native', '32', '64']
V8_MODELS = [
    'elman_original', 'lstm', 'gru_original',
    'causal_transformer_postnorm', 'linear_attention_multihead', 's4d_lin',
]
V8_SEEDS = ['3101', '3102']
V8_CUTOFFS = ['0', '16', '32', '64', 'full']
LLM_MODELS = ['HuggingFaceTB/SmolLM2-135M-Instruct',
              'HuggingFaceTB/SmolLM2-360M-Instruct']

# Chunked, automatic checkpointing (NOT a cap on total runs).
# Each invocation loops until all tracks finish or the session budget expires.
V8_NEW_CONFIGS_PER_CHUNK = 6
LEGACY_NEW_CONFIGS_PER_CHUNK = 20
SESSION_MINUTES_FOR_COMPUTE = 505  # save time before Kaggle's usual session cap
MINIMUM_START_SECONDS = 600        # do not start a new training chunk near deadline

assert WORK.is_dir() and INPUT.is_dir(), 'Run this in Kaggle, not locally.'
started = time.monotonic()
compute_deadline = started + SESSION_MINUTES_FOR_COMPUTE * 60


def seconds_left():
    return max(0, int(compute_deadline - time.monotonic()))


def run_cmd(argv, *, cwd=None, timed=True):
    argv = [str(s) for s in argv]
    print('\n$', ' '.join(argv), flush=True)
    # A process group prevents child trainings continuing after a forced timeout.
    proc = subprocess.Popen(argv, cwd=cwd, start_new_session=True)
    try:
        remaining = seconds_left() if timed else None
        if timed and remaining < MINIMUM_START_SECONDS:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            return False
        rc = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        print('TIME BUDGET EXPIRED: stopping this chunk; saved run shards will resume.', flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return False
    if rc:
        raise RuntimeError(f'Command failed (exit {rc}): {" ".join(argv)}')
    return True


def extract_results(source):
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Corrupt checkpoint member: ' + bad)
        prefix = OUT.name + '/'
        for entry in z.infolist():
            if entry.is_dir():
                continue
            if not entry.filename.startswith(prefix) or '\\' in entry.filename:
                raise RuntimeError('Unsafe or incorrect checkpoint layout: ' + entry.filename)
            relative = entry.filename[len(prefix):]
            dest = (OUT / relative).resolve()
            if not dest.is_relative_to(OUT.resolve()):
                raise RuntimeError('Unsafe checkpoint path: ' + entry.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)


def pack_results():
    if not OUT.is_dir():
        return
    tmp = ARCHIVE.with_suffix('.partial')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for file in sorted(OUT.rglob('*')):
            if file.is_file():
                z.write(file, OUT.name + '/' + file.relative_to(OUT).as_posix())
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP integrity error: ' + bad)
    tmp.replace(ARCHIVE)
    print('\nCHECKPOINT / DOWNLOAD:', ARCHIVE, flush=True)
    print('Archive size (MiB):', round(ARCHIVE.stat().st_size / 1048576, 2), flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def legacy_complete(profile):
    root = OUT / profile
    status = read_json(root / 'pipeline_status.json')
    if not status.get('complete'):
        return False
    for name in ('independent_integrity_audit.json', 'comparison_integrity_audit.json'):
        if read_json(root / name).get('passed') is not True:
            raise RuntimeError(f'Incorrect completed status: {root / name}')
    return True


def v8_complete():
    status = read_json(OUT / 'v8' / 'v8_status.json')
    return bool(status.get('all_replications_complete') and
                status.get('statistics_complete') and
                (status.get('llm_complete') if RUN_PRETRAINED_LLMS else True))


def progress():
    v8_runs = sum(len(read_json(folder / 'neural' / 'architecture_status.json').get('runs', []))
                  for folder in (OUT / 'v8').glob('replication_*')) if (OUT / 'v8').exists() else 0
    return {
        'v8_neural_configs_done': v8_runs,
        'v8_expected_configs': V8_REPLICATIONS * len(V8_MODELS) * 3 * len(V8_SEEDS) * len(V8_CUTOFFS),
        'v8_complete': v8_complete(),
        'v7_neural_configs_done': len(read_json(OUT / 'foundation_full' / 'architecture_status.json').get('runs', [])),
        'v7_complete': legacy_complete('foundation_full'),
        'v6_neural_configs_done': len(read_json(OUT / 'full' / 'architecture_status.json').get('runs', [])),
        'v6_complete': legacy_complete('full'),
        'cpu_extension_complete': (OUT / 'legacy_cpu_extension' / 'paper_cpu_manifest.json').is_file(),
        'efficiency_complete': (OUT / 'efficiency' / 'model_efficiency.csv').is_file(),
    }


def next_v8_chunk():
    cmd = [sys.executable, 'scripts/run_v8_research.py',
        '--out', OUT / 'v8', '--replications', str(V8_REPLICATIONS),
        '--per-family', str(V8_PAIRS_PER_FAMILY), '--levels', *V8_LEVELS,
        '--long-attempts', '100', '--architectures', *V8_MODELS,
        '--seeds', *V8_SEEDS, '--cutoffs', *V8_CUTOFFS,
        '--length-eval-cutoffs', '32', '64', 'full',
        '--length-eval-train-cutoffs', '16',
        '--max-history', '256', '--epochs', '50', '--patience', '7',
        '--batch-size', '64', '--device', 'cuda', '--grid-lrs', '0.0003', '0.001',
        '--grid-weight-decays', '0.001', '--grid-dropouts', '0.1', '0.3',
        '--max-new-runs', str(V8_NEW_CONFIGS_PER_CHUNK), '--bootstrap', '5000']
    if RUN_PRETRAINED_LLMS:
        cmd += ['--run-llm', '--llm-models', *LLM_MODELS,
                '--llm-pairs-per-task', '4', '--llm-batch-size', '4']
    return run_cmd(cmd, cwd=SOURCE)


def next_legacy_chunk(profile):
    cmd = [sys.executable, 'scripts/run_research_pipeline.py',
        '--profile', profile, '--out', OUT / profile, '--device', 'cuda',
        '--epochs', '50', '--patience', '7', '--skip-tests',
        '--max-new-runs', str(LEGACY_NEW_CONFIGS_PER_CHUNK),
        '--max-session-seconds', str(max(120, seconds_left() - 120))]
    # Same verified generated pairs for original v6 and foundation-aligned v7.
    if profile == 'full' and RUN_V7_FOUNDATION_FULL:
        shared = OUT / 'foundation_full' / 'data' / 'pairs.jsonl'
        if not shared.is_file():
            return True  # v7 dataset has not been generated yet
        cmd += ['--pairs-jsonl', shared]
    return run_cmd(cmd, cwd=SOURCE)


def all_requested_done():
    return (not RUN_V8 or v8_complete()) and \
           (not RUN_V7_FOUNDATION_FULL or legacy_complete('foundation_full')) and \
           (not RUN_V6_LEGACY_FULL or legacy_complete('full')) and \
           (not RUN_V6_CPU_EXTENSION or (OUT / 'legacy_cpu_extension' / 'paper_cpu_manifest.json').is_file()) and \
           (not RUN_MODEL_EFFICIENCY or (OUT / 'efficiency' / 'model_efficiency.csv').is_file())


try:
    previous = list(INPUT.rglob(ARCHIVE.name))
    if len(previous) > 1:
        raise RuntimeError('Attach only one previous checkpoint ZIP')
    if not (OUT / 'run_manifest.json').is_file() and previous:
        print('RESTORING checkpoint:', previous[0], flush=True)
        extract_results(previous[0])

    # Clone latest main ONLY for a new experiment. Resume exact original commit.
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    run_cmd(['git', 'clone', '--branch', BRANCH, REPOSITORY, SOURCE], timed=False)
    old_manifest = read_json(OUT / 'run_manifest.json')
    if old_manifest:
        run_cmd(['git', 'checkout', old_manifest['commit']], cwd=SOURCE, timed=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE,
                                     text=True).strip()
    print('HISTCHESS SOURCE COMMIT:', commit, flush=True)
    needed = ['scripts/run_v8_research.py', 'scripts/run_v8_llm.py',
              'scripts/run_research_pipeline.py',
              'scripts/run_paper_cpu_extension.py',
              'scripts/benchmark_model_efficiency.py']
    missing = [f for f in needed if not (SOURCE / f).is_file()]
    if missing:
        raise RuntimeError('GitHub main does not include expected v8 scripts: ' + repr(missing))

    # GitHub commit 58993ba has two stale parser-test expectations.
    # Preserve the production parser: ambiguous/explanatory LLM replies remain INVALID.
    # Patch ONLY those two test assertions when this exact mismatch exists.
    test_path = SOURCE / 'tests' / 'test_encoding_metrics.py'
    tests_text = test_path.read_text(encoding='utf-8')
    pairs = [
        ("('YES because',True)", "('YES because',None)"),
        ("('No? Maybe',False)", "('No? Maybe',None)"),
    ]
    for before, after in pairs:
        if before in tests_text:
            if tests_text.count(before) != 1:
                raise RuntimeError('Unexpected duplicated parser test: ' + before)
            tests_text = tests_text.replace(before, after)
            print('Corrected stale parser-test expectation:', before, '=>', after, flush=True)
        elif after not in tests_text:
            raise RuntimeError('Parser test changed upstream; inspect before proceeding: ' + before)
    test_path.write_text(tests_text, encoding='utf-8')

    # Ensure we did not accidentally modify the scientific LLM parser.
    parser_source = (SOURCE / 'histchess' / 'experiments.py').read_text(encoding='utf-8')
    if r're.fullmatch(r"(?:answer\s*:\s*)?(yes|no)[.!]?",s)' not in parser_source:
        raise RuntimeError('LLM parser changed in GitHub; review its strict protocol first')

    # Keep Kaggle's existing CUDA-enabled PyTorch; do not replace it with CPU torch.
    run_cmd([sys.executable, '-m', 'pip', 'install', '-q',
             'chess==1.11.2', 'scikit-learn>=1.3,<2', 'pytest>=7,<10',
             'transformers>=4.45,<5', 'safetensors>=0.4',
             'matplotlib>=3.7,<4'], timed=False)
    import torch
    print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator in Kaggle Settings to run all results.')

    settings = {
        'commit': commit, 'repository': REPOSITORY,
        'torch_version': torch.__version__,
        'enabled_tracks': [RUN_V8, RUN_V7_FOUNDATION_FULL, RUN_V6_LEGACY_FULL,
                           RUN_V6_CPU_EXTENSION, RUN_MODEL_EFFICIENCY, RUN_PRETRAINED_LLMS],
        'v8': {'replications': V8_REPLICATIONS, 'per_family': V8_PAIRS_PER_FAMILY,
               'levels': V8_LEVELS, 'models': V8_MODELS, 'seeds': V8_SEEDS,
               'cutoffs': V8_CUTOFFS, 'llm_models': LLM_MODELS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if old_manifest and old_manifest != settings:
        raise RuntimeError('Resume settings/torch version differ from saved archive. '
                           'Use the original Kaggle configuration and cell.')
    (OUT / 'run_manifest.json').write_text(json.dumps(settings, indent=2))

    gate = OUT / 'preflight_passed.json'
    if read_json(gate).get('commit') != commit:
        # Cheap tests + the truly long 64-ply generator smoke test.
        if not run_cmd([sys.executable, '-m', 'pytest', '-q', 'tests', '-m',
                        'not slow and not gpu_integration',
                        '--ignore=tests/test_length_stress.py'], cwd=SOURCE):
            raise RuntimeError('Preflight did not finish; resume next Kaggle session')
        if not run_cmd([sys.executable, '-m', 'pytest', '-q',
                        'tests/test_v8_generator.py', '-m', 'slow'], cwd=SOURCE):
            raise RuntimeError('64-ply legality smoke test did not finish; resume next session')
        gate.write_text(json.dumps({'commit': commit, 'passed': True}, indent=2))

    # Efficiency is independent of training completion, so run it once up front.
    efficiency = OUT / 'efficiency' / 'model_efficiency.csv'
    if RUN_MODEL_EFFICIENCY and not efficiency.is_file() and seconds_left() > MINIMUM_START_SECONDS:
        models = ['elman_rnn', 'lstm', 'gru', 'causal_transformer',
                  'linear_attention', 'diagonal_ssm', 'elman_original',
                  'gru_original', 'causal_transformer_postnorm',
                  'linear_attention_multihead', 's4d_lin']
        run_cmd([sys.executable, 'scripts/benchmark_model_efficiency.py',
                 '--architectures', *models, '--lengths', '16', '32', '64',
                 '128', '256', '--device', 'cuda', '--out', efficiency], cwd=SOURCE)

    cycle = 0
    while seconds_left() >= MINIMUM_START_SECONDS and not all_requested_done():
        cycle += 1
        before = progress()
        print('\n========== RESEARCH CYCLE', cycle, '==========', flush=True)
        print('Progress:', json.dumps(before, indent=2), flush=True)
        for name in ('v8', 'foundation_full', 'full'):
            if seconds_left() < MINIMUM_START_SECONDS:
                break
            if name == 'v8' and RUN_V8 and not v8_complete():
                if not next_v8_chunk():
                    break
            elif name == 'foundation_full' and RUN_V7_FOUNDATION_FULL and not legacy_complete(name):
                if not next_legacy_chunk(name):
                    break
            elif name == 'full' and RUN_V6_LEGACY_FULL and not legacy_complete(name):
                if not next_legacy_chunk(name):
                    break
        if (RUN_V6_CPU_EXTENSION and legacy_complete('full') and
            not (OUT / 'legacy_cpu_extension' / 'paper_cpu_manifest.json').is_file() and
            seconds_left() >= MINIMUM_START_SECONDS):
            cpu = OUT / 'legacy_cpu_extension'
            # This extension has no shard resume: discard incomplete outputs and
            # rerun rather than presenting partial controls as complete.
            if cpu.is_dir() and any(cpu.iterdir()):
                print('Discarding incomplete CPU-extension outputs before retry.', flush=True)
                shutil.rmtree(cpu)
            run_cmd([sys.executable, 'scripts/run_paper_cpu_extension.py',
                     '--results', OUT / 'full', '--out', cpu], cwd=SOURCE)
        after = progress()
        (OUT / 'stage_summary.json').write_text(json.dumps(after, indent=2))
        if after == before and not all_requested_done():
            if seconds_left() >= MINIMUM_START_SECONDS:
                raise RuntimeError('No stage made progress; inspect Kaggle logs and status files.')
            break

    state = progress()
    state['all_requested_stages_complete'] = all_requested_done()
    state['source_commit'] = commit
    (OUT / 'stage_summary.json').write_text(json.dumps(state, indent=2))
    print('\nFINAL STAGE SUMMARY:', json.dumps(state, indent=2), flush=True)
    if not state['all_requested_stages_complete']:
        print('PARTIAL: save the ZIP, attach it to a new Kaggle session, rerun same cell.', flush=True)
    else:
        print('COMPLETE: all enabled results and audits finished.', flush=True)
except BaseException:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'launcher_error.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    pack_results()
