# Paste this entire file into ONE Kaggle Notebook Python cell.
# Settings: Accelerator=None; Internet=ON. Upload the patch to GitHub first.
import os, sys, subprocess, shutil, zipfile, json, traceback
from pathlib import Path

REPO_URL = "https://github.com/tydeptrai21042004/histchess.git"
PROFILE = "full"  # or "cpu_quick"; quick runs all components with smaller samples
SETTINGS = {
    "full": dict(pairs=300,epochs=14,seeds=2,llm_pairs=24,bootstrap=2000),
    "cpu_quick": dict(pairs=90,epochs=5,seeds=2,llm_pairs=8,bootstrap=500),
}
if PROFILE not in SETTINGS: raise ValueError(PROFILE)
S=SETTINGS[PROFILE]
WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO=WORK/'histchess'
OUT=WORK/'histchess_results_v3'
DIAG=WORK/'histchess_v3_diagnostics'
DIAG.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='-1',TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_XET='1')
os.environ['HF_HOME']=str(WORK/'.histchess_model_cache')

def run(label,*args,cwd=None):
    args=list(map(str,args))
    print('\n==',label,'==\n$', ' '.join(args),flush=True)
    log=DIAG/(label+'.log')
    with log.open('w',encoding='utf8') as f:
        p=subprocess.Popen(args,cwd=str(cwd) if cwd else None,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,errors='replace',bufsize=1,env=os.environ.copy())
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line)
        code=p.wait()
    if code:raise RuntimeError(f'{label} failed: exit {code}; log: {log}')

def save_dir_zip(path):
    dest=WORK/(path.name+'.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(path.rglob('*')):
            if f.is_file():z.write(f,f'{path.name}/{f.relative_to(path)}')
    with zipfile.ZipFile(dest) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Archive CRC failed: '+bad)
    print('ARTIFACT:',dest,flush=True)

try:
    if REPO.exists():shutil.rmtree(REPO)
    run('01_clone','git','clone','--depth','1',REPO_URL,REPO)
    if not (REPO/'histchess/diverse.py').exists():
        raise RuntimeError('GitHub does not contain strengthened patch. Upload ZIP code files first.')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    (DIAG/'github_commit.txt').write_text(sha+'\n')
    run('02_install',sys.executable,'-m','pip','install','-q',
        'chess==1.11.2','transformers>=4.45,<5','safetensors>=0.4',
        'huggingface-hub>=0.25,<1','pytest>=7,<10')
    run('03_rules',sys.executable,'scripts/selftest.py',cwd=REPO)
    run('04_tests',sys.executable,'-m','pytest','-q','tests','-m','not gpu_integration',cwd=REPO)
    run('05_diverse_stress',sys.executable,'scripts/stress_test.py',
        '--pairs-per-task','60','--seeds','3101','3102','3103',
        '--generator','diverse','--split-strategy','family',
        '--out',DIAG/'diverse_stress.json',cwd=REPO)
    run('06_smoke',sys.executable,'scripts/run_all_kaggle.py',
        '--out',WORK/'histchess_smoke_v3','--allow-cpu','--smoke',
        '--generator','diverse','--split-strategy','family','--skip-llm',cwd=REPO)
    run('07_verify_smoke',sys.executable,'scripts/verify_artifacts.py',
        WORK/'histchess_smoke_v3.zip','--allow-smoke',cwd=REPO)
    run('08_full_cpu',sys.executable,'scripts/run_all_kaggle.py',
        '--out',OUT,'--allow-cpu','--pairs-per-task',S['pairs'],
        '--epochs',S['epochs'],'--train-seeds',S['seeds'],
        '--llm-pairs-per-task',S['llm_pairs'],'--llm-batch-size','2',
        '--bootstrap',S['bootstrap'],'--batch-size','128',
        '--generator','diverse','--split-strategy','family',cwd=REPO)
    run('09_verify_results',sys.executable,'scripts/verify_artifacts.py',
        WORK/'histchess_results_v3.zip',cwd=REPO)
    pairs=OUT/'data/pairs.jsonl'
    run('10_validate',sys.executable,'scripts/validate_dataset.py',pairs,cwd=REPO)
    run('11_package',sys.executable,'scripts/package_release.py',pairs,
        '--out',WORK/'histchess_dataset_v3.zip',cwd=REPO)
    status=json.loads((OUT/'run_status.json').read_text())
    if not status.get('completed'):raise RuntimeError('Core experiments not complete')
    print('\nSUCCESS | source commit:',sha,
          '\nEXPERIMENTS:',WORK/'histchess_results_v3.zip',
          '\nDATASET:',WORK/'histchess_dataset_v3.zip',
          '\nSHORTCUT REPORT:',OUT/'shortcut_metrics.csv',
          '\nFAMILY AUDIT:',OUT/'leakage_audit.json',flush=True)
except BaseException:
    (DIAG/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    save_dir_zip(DIAG)
    for x in sorted(WORK.glob('histchess*zip')):print('AVAILABLE:',x,flush=True)
