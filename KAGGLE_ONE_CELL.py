# HistChess | ONE CELL Kaggle GPU runner, including fail-safe diagnostics.
# Kaggle Notebook Settings: GPU T4 or P100; Internet ON for chess and HF models.
# Attach histchess_gpu_verified_full.zip in the notebook's Input data, or set REPO_URL.
# The first mode to run is 'smoke'. Use 'full' after tests pass.
import os, sys, zipfile, shutil, subprocess, json, traceback, platform, time
from pathlib import Path

REPO_URL = ''   # Optional: URL of your UPDATED GitHub repository.
MODE = 'full'  # 'smoke': small CUDA run (skips LLM); 'full': complete evaluation.
PAIRS_PER_TASK = 300
GRU_EPOCHS = 14
TRAIN_SEEDS = 2
LLM_PAIRS_PER_TASK = 24
GRU_BATCH_SIZE = 128
LLM_BATCH_SIZE = 8
RUN_CUDA_SMOKE_FIRST = True

WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
INPUT = Path('/kaggle/input')
WORK.mkdir(parents=True,exist_ok=True)
PROJECT = WORK / 'histchess-main'
OUT = WORK / 'histchess_results'
DIAG = WORK / 'histchess_diagnostics'
DIAG.mkdir(exist_ok=True)
os.environ.update({
    'HF_HOME':str(WORK/'.hf_cache'),
    'TOKENIZERS_PARALLELISM':'false',
    'PYTHONHASHSEED':'0',
    'OMP_NUM_THREADS':'4',
    'CUBLAS_WORKSPACE_CONFIG':':4096:8',
    'HF_HUB_DISABLE_XET':'1',
})

# Every bootstrap step, including errors before the research runner, has a log.
def execute(argv,cwd=None,label=None):
    argv=[str(x) for x in argv]
    label = label or Path(argv[0]).name
    logfile = DIAG / (label.replace('/','_')+'.log')
    print('\n>>>',' '.join(argv),'\nLog:',logfile,flush=True)
    with logfile.open('w',encoding='utf8') as f:
        proc=subprocess.Popen(argv,cwd=cwd,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,bufsize=1,
                              errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True)
            f.write(line)
        code=proc.wait()
    if code:
        raise subprocess.CalledProcessError(code,argv)


def zip_directory(path, archive):
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(path.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path(path.name)/p.relative_to(path)))
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('Output archive corrupted: '+bad)
    print('ARCHIVE:',archive,'size:',archive.stat().st_size,flush=True)


try:
    if MODE not in ('smoke','full'):
        raise ValueError('Set MODE to smoke or full')
    import torch
    (DIAG/'environment.json').write_text(json.dumps({
        'python':sys.version,'platform':platform.platform(),
        'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Kaggle notebook > Settings > Accelerator > GPU T4/P100.')
    print('Kaggle CUDA detected:',torch.cuda.get_device_name(0),flush=True)
    if PROJECT.exists():shutil.rmtree(PROJECT)
    if REPO_URL.strip():
        execute(['git','clone','--depth','1',REPO_URL.strip(),PROJECT],label='clone')
    else:
        candidates=(sorted(INPUT.rglob('histchess_gpu_verified_full.zip')) if INPUT.exists() else [])
        if not candidates:
            # Accept the former full archive name as well, but never a patch-only ZIP.
            candidates=sorted(INPUT.rglob('histchess_gpu_full.zip')) if INPUT.exists() else []
        if not candidates:
            raise FileNotFoundError('Attach histchess_gpu_verified_full.zip under Kaggle Input, or set REPO_URL')
        source=WORK/'histchess_source';shutil.rmtree(source,ignore_errors=True);source.mkdir()
        print('Using attached archive:',candidates[0],flush=True)
        with zipfile.ZipFile(candidates[0]) as z:
            if z.testzip():raise ValueError('Source ZIP CRC failed')
            for member in z.infolist():
                member_path=(source/member.filename).resolve()
                if not member_path.is_relative_to(source.resolve()):
                    raise ValueError('Unsafe path in ZIP: '+member.filename)
                if member.is_dir():continue
                # No symlinks or inherited file permissions from user-supplied ZIP.
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlink in source ZIP not allowed: '+member.filename)
            z.extractall(source)
        roots=list(source.rglob('scripts/run_all_kaggle.py'))
        if len(roots)!=1:raise RuntimeError('Source archive must contain exactly one Kaggle project')
        shutil.copytree(roots[0].parent.parent,PROJECT)
    if not (PROJECT/'tests'/'test_rules.py').is_file():
        raise RuntimeError('Wrong or older source version: required regression tests are missing')
    if not (PROJECT/'histchess'/'gpu_experiments.py').is_file():
        raise RuntimeError('Wrong repository version: GPU experiments module missing')

    # Keep the Kaggle precompiled CUDA build of torch, never install a new torch.
    execute([sys.executable,'-m','pip','install','-q','--disable-pip-version-check',
             '--retries','4','--timeout','40','chess==1.11.2',
             'transformers>=4.45,<5','safetensors>=0.4','huggingface-hub>=0.25,<1',
             'pytest>=7,<10'],label='install_dependencies')
    execute([sys.executable,'scripts/selftest.py'],cwd=PROJECT,label='chess_rules_selftest')
    execute([sys.executable,'-m','pytest','-q','tests',
             '-m','not slow and not gpu_integration'],cwd=PROJECT,label='fast_regression_tests')

    # End-to-end smoke: dataset -> CUDA GRU -> statistics -> figures -> audit.
    # This is much smaller than the main run and never claims paper completion.
    if RUN_CUDA_SMOKE_FIRST or MODE=='smoke':
        execute([sys.executable,'scripts/run_all_kaggle.py',
                 '--out',WORK/'histchess_smoke', '--smoke','--skip-llm'],
                cwd=PROJECT,label='cuda_pipeline_smoke')
        smoke_report=json.loads((WORK/'histchess_smoke'/'final_output_audit.json').read_text())
        assert smoke_report['passed'], 'Full smoke integrity audit failed'
        (DIAG/'smoke_validation.json').write_text(json.dumps(smoke_report,indent=2))

    if MODE=='full':
        execute([sys.executable,'scripts/run_all_kaggle.py','--out',OUT,
                 '--pairs-per-task',str(PAIRS_PER_TASK),
                 '--epochs',str(GRU_EPOCHS),
                 '--train-seeds',str(TRAIN_SEEDS),
                 '--batch-size',str(GRU_BATCH_SIZE),
                 '--llm-batch-size',str(LLM_BATCH_SIZE),
                 '--llm-pairs-per-task',str(LLM_PAIRS_PER_TASK)],
                cwd=PROJECT,label='complete_paper_pipeline')
        execute([sys.executable,'scripts/verify_artifacts.py',WORK/'histchess_results.zip'],
                cwd=PROJECT,label='verify_result_archive')
        status=json.loads((OUT/'run_status.json').read_text())
        if not status['completed'] or not status['paper_complete']:
            raise RuntimeError('Runner returned but paper experiment was not marked COMPLETE')
        print('\nSUCCESS: all required pipeline stages marked PASS.',flush=True)
        print('Download:',WORK/'histchess_results.zip',flush=True)
    else:
        print('\nSMOKE SUCCESS. To run the full experiment change MODE to full.',flush=True)
        print('Download:',WORK/'histchess_smoke.zip',flush=True)

except BaseException as exc:
    import traceback
    (DIAG/'bootstrap_failure.txt').write_text(traceback.format_exc(),encoding='utf8')
    print('\nPIPELINE FAILED. The diagnosis ZIP contains logs and the exact error.',flush=True)
    print(traceback.format_exc(),flush=True)
    raise
finally:
    # Always keep a useful downloadable diagnostics archive, even when pip/selftests fail.
    zip_directory(DIAG,WORK/'histchess_diagnostics.zip')
    for f in (WORK/'histchess_results.zip',WORK/'histchess_smoke.zip'):
        if f.exists():print('AVAILABLE:',f,flush=True)
