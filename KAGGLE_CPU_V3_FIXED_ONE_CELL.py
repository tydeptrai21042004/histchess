# HistChess v3 — Fixed CPU-only Kaggle ONE CELL
# Notebook settings: Accelerator=None, Internet=ON.
# Uses the strengthened GitHub repository; also accepts the original patch ZIP
# from Kaggle Inputs when those strengthened source files are not yet on GitHub.
import os, sys, json, shutil, zipfile, subprocess, traceback, hashlib, platform
from pathlib import Path

REPO_URL = 'https://github.com/tydeptrai21042004/histchess.git'
PROFILE = 'full'  # 'full' or 'cpu_quick'
CFG = {
    'full': dict(pairs=300, epochs=14, seeds=2, llm_pairs=24, bootstrap=2000),
    'cpu_quick': dict(pairs=90, epochs=5, seeds=2, llm_pairs=8, bootstrap=500),
}[PROFILE]

W = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
REPO = W / 'histchess'
OUT = W / 'histchess_results_v3'
DIAG = W / 'histchess_v3_fixed_diagnostics'
# Remove stale outputs from earlier notebooks so a failed run cannot
# accidentally package or present old results as the current run.
for directory in (OUT, W / 'histchess_smoke_v3', DIAG):
    if directory.exists():
        shutil.rmtree(directory)
DIAG.mkdir(parents=True)
for name in ('histchess_results_v3.zip', 'histchess_dataset_v3.zip',
             'histchess_smoke_v3.zip', 'histchess_v3_fixed_diagnostics.zip'):
    (W / name).unlink(missing_ok=True)
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_XET': '1',
    'USE_TF': '0', 'USE_FLAX': '0', 'PYTHONHASHSEED': '0',
    'HF_HOME': str(W / '.histchess_model_cache'),
})

def run(name, *args, cwd=REPO):
    args = [str(x) for x in args]
    print('\n===', name, '===\n$', ' '.join(args), flush=True)
    with (DIAG / f'{name}.log').open('w', encoding='utf-8') as logfile:
        with subprocess.Popen(args, cwd=str(cwd), env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                logfile.write(line)
            code = proc.wait()
    if code:
        raise RuntimeError(f'{name} failed (exit {code}). See {DIAG / (name + ".log")}')

def zip_tree(directory, target):
    if not directory.is_dir():
        return
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(directory.rglob('*')):
            if p.is_file():
                z.write(p, arcname=str(Path(directory.name) / p.relative_to(directory)))
    with zipfile.ZipFile(target) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'Bad ZIP content: {bad}')
    print('SAVED:', target, flush=True)

def ensure_v3_source():
    if (REPO / 'histchess/diverse.py').is_file() and (REPO / 'histchess/shortcuts.py').is_file():
        return 'Strengthened GitHub repository'
    paths = [W / 'histchess_strengthened_code_patch.zip']
    if Path('/kaggle/input').exists():
        paths.extend(sorted(Path('/kaggle/input').rglob('histchess_strengthened_code_patch.zip')))
    patch = next((p for p in paths if p.is_file()), None)
    if patch is None:
        raise RuntimeError('Strengthened source not on GitHub. Attach the earlier '
                           'histchess_strengthened_code_patch.zip to Kaggle Inputs.')
    with zipfile.ZipFile(patch) as z:
        if z.testzip():
            raise RuntimeError('Bad strengthened patch ZIP')
        members = z.namelist()
        if not {'histchess/diverse.py', 'histchess/shortcuts.py'}.issubset(members):
            raise RuntimeError('Unexpected patch ZIP contents')
        for info in z.infolist():
            if info.is_dir():
                continue
            path = (REPO / info.filename).resolve()
            if not path.is_relative_to(REPO.resolve()):
                raise RuntimeError('Unsafe ZIP path: ' + info.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(info))
    return f'GitHub + {patch.name}'

def apply_family_split_fix():
    """Fix cross-task board collisions in the previous strengthened generator."""
    target = REPO / 'histchess/diverse.py'
    before = target.read_text(encoding='utf-8')
    old = 'novelty = (task, pair["board_key"])'
    new = 'novelty = pair["board_key"]'
    if old in before:
        if before.count(old) != 1:
            raise RuntimeError('Unexpected multiple uniqueness checks: refusing ambiguous patch')
        target.write_text(before.replace(old, new), encoding='utf-8')
        result = 'Applied GLOBAL board-key uniqueness fix'
    elif new in before:
        result = 'Global board-key uniqueness fix is already present'
    else:
        raise RuntimeError('Unknown diverse.py version: fix cannot be applied safely')
    (DIAG / 'family_split_fix.json').write_text(json.dumps({
        'status': result, 'source_sha256_before': hashlib.sha256(before.encode()).hexdigest(),
        'source_sha256_after': hashlib.sha256(target.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')
    print(result, flush=True)

# The preflight reproduces the original failing size and checks the full size.
PROBE = r'''
import json
from histchess.core import TASKS, generate, validate
from histchess.experiments import leakage_audit
for size in (60, FULL_SIZE):
    pairs, rejects = generate(size, seed=3101, generator='diverse', split_strategy='family')
    validation = validate(pairs)
    leakage = leakage_audit(pairs)
    check = {
        'pairs_per_task': size,
        'total_pairs': len(pairs),
        'validation_passed': validation['passed'],
        'errors': validation['errors'],
        'board_group_leakage_count': leakage['board_group_leakage_count'],
        'full_history_leakage_count': leakage['full_history_leakage_count'],
        'full_FEN_across_split_count': leakage['full_FEN_across_split_count'],
        'generator_family_across_split_count': leakage['generator_family_across_split_count'],
        'unique_boards': len({p['board_key'] for p in pairs}),
        'task_split_counts': {task: {split: sum(p['task']==task and p['split']==split for p in pairs)
                                     for split in ('train', 'validation', 'test')} for task in TASKS},
    }
    print(json.dumps(check, indent=2), flush=True)
    assert validation['passed'], validation['errors']
    assert check['unique_boards'] == len(pairs), 'Cross-task board duplicates remain'
    assert not any(check[k] for k in (
        'board_group_leakage_count', 'full_history_leakage_count',
        'full_FEN_across_split_count', 'generator_family_across_split_count'))
'''.replace('FULL_SIZE', str(CFG['pairs']))

try:
    if REPO.exists():
        shutil.rmtree(REPO)
    run('01_clone', 'git', 'clone', '--depth', '1', REPO_URL, REPO, cwd=W)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    source = ensure_v3_source()
    apply_family_split_fix()
    (DIAG / 'source_info.json').write_text(json.dumps({
        'commit': commit, 'source': source, 'profile': PROFILE, 'configuration': CFG,
        'python': sys.version, 'platform': platform.platform(), 'device': 'cpu',
    }, indent=2), encoding='utf-8')
    (DIAG / 'local_source_diff.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--', 'histchess/diverse.py'], cwd=REPO, text=True))

    run('02_install', sys.executable, '-m', 'pip', 'install', '-q',
        'chess==1.11.2', 'transformers>=4.45,<5', 'safetensors>=0.4',
        'huggingface-hub>=0.25,<1', 'pytest>=7,<10')
    import torch
    assert not torch.cuda.is_available(), 'CPU-only mode was not respected'
    print('CPU verified:', torch.__version__, flush=True)

    run('03_reproduce_and_validate', sys.executable, '-c', PROBE)
    run('04_selftest', sys.executable, 'scripts/selftest.py')
    run('05_pytest', sys.executable, '-m', 'pytest', '-q',
        'tests', '-m', 'not gpu_integration')
    run('06_stress', sys.executable, 'scripts/stress_test.py',
        '--generator', 'diverse', '--split-strategy', 'family',
        '--pairs-per-task', '60', '--seeds', '3101', '3102', '3103',
        '--out', DIAG / 'stress_report.json')

    run('07_smoke', sys.executable, 'scripts/run_all_kaggle.py',
        '--out', W / 'histchess_smoke_v3', '--allow-cpu', '--smoke',
        '--generator', 'diverse', '--split-strategy', 'family', '--skip-llm')
    run('08_verify_smoke', sys.executable, 'scripts/verify_artifacts.py',
        W / 'histchess_smoke_v3.zip', '--allow-smoke')

    run('09_full_cpu_experiments', sys.executable, 'scripts/run_all_kaggle.py',
        '--out', OUT, '--allow-cpu', '--generator', 'diverse', '--split-strategy', 'family',
        '--pairs-per-task', CFG['pairs'], '--epochs', CFG['epochs'],
        '--train-seeds', CFG['seeds'], '--llm-pairs-per-task', CFG['llm_pairs'],
        '--llm-batch-size', '2', '--bootstrap', CFG['bootstrap'], '--batch-size', '128')
    run('10_verify_results', sys.executable, 'scripts/verify_artifacts.py',
        W / 'histchess_results_v3.zip')
    pairs = OUT / 'data' / 'pairs.jsonl'
    run('11_validate_dataset', sys.executable, 'scripts/validate_dataset.py', pairs)
    run('12_package_dataset', sys.executable, 'scripts/package_release.py',
        pairs, '--out', W / 'histchess_dataset_v3.zip')

    status = json.loads((OUT / 'run_status.json').read_text(encoding='utf-8'))
    if not (status.get('completed') and status.get('paper_complete')):
        raise RuntimeError('Full run_status.json reports incomplete experiments')
    if status.get('parameters', {}).get('split_strategy') != 'family':
        raise RuntimeError('Incorrect split strategy in output metadata')
    print('\nSUCCESS: strengthened CPU pipeline completed and verified!', flush=True)
    print('Paper tables:', OUT / 'paper_tables.md', flush=True)
    print('Shortcut baseline:', OUT / 'shortcut_metrics.csv', flush=True)

except BaseException:
    (DIAG / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    print('\nFAILED: check diagnostics and any partial result artifacts.', flush=True)
    raise
finally:
    if OUT.is_dir() and not (W / 'histchess_results_v3.zip').exists():
        zip_tree(OUT, W / 'histchess_results_v3_PARTIAL.zip')
    zip_tree(DIAG, W / 'histchess_v3_fixed_diagnostics.zip')
    print('\nAVAILABLE KAGGLE OUTPUTS:')
    for name in ('histchess_results_v3.zip', 'histchess_results_v3_PARTIAL.zip',
                 'histchess_dataset_v3.zip', 'histchess_smoke_v3.zip',
                 'histchess_v3_fixed_diagnostics.zip'):
        p = W / name
        if p.is_file():
            print(p, f'({p.stat().st_size:,} bytes)', flush=True)
