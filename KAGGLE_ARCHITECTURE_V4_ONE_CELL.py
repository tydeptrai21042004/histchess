# HistChess v4 research benchmark -- one CPU-only Kaggle cell.
# Kaggle settings: Accelerator=None, Internet=On.
# Upload HistChess_v4_code_patch.zip as input, OR push the contents of that ZIP
# (preserving paths) into https://github.com/tydeptrai21042004/histchess.
# Input archive from v3 is never resumed into v4; the new fingerprint forbids it.
import hashlib, json, os, shutil, subprocess, sys, traceback, zipfile
from pathlib import Path

REPO_URL='https://github.com/tydeptrai21042004/histchess.git'
PROFILE='pilot'    # set 'full' for all 6 models, 3 folds, 5 seeds
LONG_TARGET=64     # 0 disables synthetic neutral-history stress. 64 needs python-chess validation.
USE_ATTACHED_BASE_PAIRS=False   # true: load an external v3 pairs.jsonl and extend it, if selected
RESUME_V4=True
PROFILES={
    'pilot':dict(folds=(0,),seeds=(3101,),cutoffs=('0','16'),epochs=2,pairs=30,
                 models=('gru','causal_transformer')),
    'full':dict(folds=(0,1,2),seeds=(3101,3102,3103,3104,3105),
                cutoffs=('0','4','8','16','full'),epochs=50,pairs=300,
                models=('elman_rnn','lstm','gru','causal_transformer','linear_attention','diagonal_ssm')),
}
CFG=PROFILES[PROFILE]
WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Run this cell in Kaggle.'
REPO=WORK/'histchess'
NAME=f'histchess_architecture_v4_{PROFILE}_long{LONG_TARGET}'
OUT=WORK/NAME
LOGS=WORK/(NAME+'_diagnostics')
LOGS.mkdir(parents=True,exist_ok=True)
os.environ.update({'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                   'TOKENIZERS_PARALLELISM':'false','USE_TF':'0','USE_FLAX':'0'})


def execute(label,args,cwd=REPO):
    print('\n==',label,'==',' '.join(map(str,args)),flush=True)
    with (LOGS/(label+'.log')).open('w',encoding='utf8') as f:
        proc=subprocess.Popen(list(map(str,args)),cwd=str(cwd),stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True)
            f.write(line)
        if proc.wait():
            raise RuntimeError(f'{label} failed; see {LOGS/(label+".log")}')


def unzip_safe(path,where,prefix=None):
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise RuntimeError(f'ZIP CRC failure: {path}')
        files=0
        for member in z.infolist():
            if member.is_dir():continue
            name=member.filename
            if prefix and not name.startswith(prefix):
                raise RuntimeError(f'Unexpected ZIP entry: {name}')
            dest=(where/name).resolve()
            if not dest.is_relative_to(where.resolve()):
                raise RuntimeError(f'Path traversal in ZIP: {name}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(member))
            files+=1
    return files


def package(folder,destination):
    if not folder.exists():return
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file():z.write(f,f'{folder.name}/{f.relative_to(folder)}')
    with zipfile.ZipFile(destination) as z:
        if z.testzip():raise RuntimeError('Output ZIP failed CRC verification')
    print('SAVED:',destination,destination.stat().st_size,flush=True)


def attach_patch():
    patches=sorted(Path('/kaggle/input').rglob('HistChess_v4_code_patch.zip'))
    if not patches:
        return 'GitHub source used (no patch attached)'
    p=patches[0]
    count=unzip_safe(p,REPO)
    print('APPLIED SOURCE PATCH:',p,count,'files',flush=True)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def get_v3_pairs():
    if not USE_ATTACHED_BASE_PAIRS:return None
    for name in ('histchess_results_v3.zip','histchess_dataset_v3.zip'):
        for archive in sorted(Path('/kaggle/input').rglob(name)):
            with zipfile.ZipFile(archive) as z:
                names=[x for x in z.namelist() if x.endswith('/data/pairs.jsonl') or x=='pairs.jsonl']
                if len(names)!=1:continue
                dest=WORK/'histchess_v4_source_pairs.jsonl'
                dest.write_bytes(z.read(names[0]))
                return dest
    raise RuntimeError('USE_ATTACHED_BASE_PAIRS=True but no valid v3 dataset ZIP found')


def restore():
    if not RESUME_V4 or OUT.exists():return
    files=sorted(Path('/kaggle/input').rglob(NAME+'.zip'))
    if files:
        count=unzip_safe(files[0],WORK,NAME+'/')
        print('RESTORED',count,'result files. Fingerprint check still applies.',flush=True)


try:
    restore()
    if REPO.exists():shutil.rmtree(REPO)
    execute('01_clone',['git','clone','--depth','1',REPO_URL,REPO],WORK)
    patch_sha=attach_patch()
    source=(REPO/'histchess/architecture_benchmark.py').read_text(encoding='utf8')
    if 'histchess-v4-evaluation-2026-10' not in source or not (REPO/'histchess/length_stress.py').exists():
        raise RuntimeError('Repository is not HistChess v4. Push the patch files or attach HistChess_v4_code_patch.zip.')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    (LOGS/'source.json').write_text(json.dumps({'git_sha':sha,'patch_sha':patch_sha,'profile':PROFILE,
        'long_target_plies':LONG_TARGET,'config':CFG},indent=2),encoding='utf8')
    execute('02_dependencies',[sys.executable,'-m','pip','install','-q',
                                'chess==1.11.2','pytest>=7,<10'],WORK)
    import torch
    if torch.cuda.is_available():raise RuntimeError('Use CPU-only Kaggle setting (Accelerator=None).')
    execute('03_tests',[sys.executable,'-m','pytest','-q',
                        'tests/test_architecture_benchmark.py','-m','not gpu_integration'])
    cmd=[sys.executable,'scripts/run_architecture_benchmark.py','--out',OUT,
         '--architectures',*CFG['models'],'--folds',*CFG['folds'],'--seeds',*CFG['seeds'],
         '--cutoffs',*CFG['cutoffs'],'--epochs',CFG['epochs'],'--patience',7,
         '--batch-size',64,'--max-history',256,'--target-parameters',150000,
         '--device','cpu','--position-encoding','sinusoidal',
         '--length-eval-cutoffs','16','32','64','full',
         '--length-eval-train-cutoffs','16','--generator-seed',3101]
    if LONG_TARGET:cmd+=['--long-target-plies',LONG_TARGET]
    dataset=get_v3_pairs()
    if dataset:cmd+=['--pairs-jsonl',dataset]
    else:cmd+=['--pairs-per-task',CFG['pairs']]
    execute('04_experiments',cmd)
    execute('05_independent_audit',[sys.executable,'scripts/verify_architecture_results.py',OUT,
                                    '--write',OUT/'independent_integrity_audit.json'])
    status=json.loads((OUT/'architecture_status.json').read_text(encoding='utf8'))
    expected=len(CFG['models'])*len(CFG['folds'])*len(CFG['seeds'])*len(CFG['cutoffs'])
    if not status['complete'] or len(status['runs'])!=expected:
        raise RuntimeError(f'Incomplete experiment: expected {expected} runs')
    print('\nCOMPLETE:',expected,'HistChess v4 runs audited.',flush=True)
except BaseException:
    (LOGS/'failure.txt').write_text(traceback.format_exc(),encoding='utf8')
    print('FAILED; valid completed shards remain available for resume.',flush=True)
    raise
finally:
    package(OUT,WORK/(NAME+'.zip'))
    package(LOGS,WORK/(NAME+'_diagnostics.zip'))
