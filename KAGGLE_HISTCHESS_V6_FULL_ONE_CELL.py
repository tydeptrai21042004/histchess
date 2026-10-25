# HistChess v6: paste this WHOLE file as ONE Kaggle Python cell.
# Kaggle settings: Internet=On; Accelerator=GPU (recommended) or None (CPU).
# Add HistChess_v6_full_patch.zip in Kaggle Inputs, OR commit its contents to GitHub.
# For resume: add the latest histchess_v6_full_<cpu|cuda>_stress0.zip to Kaggle Inputs.
# Do not change DEVICE/PATIENCE/EPOCHS or other scientific settings between resumes.
import hashlib, json, os, shutil, subprocess, sys, traceback, zipfile
from pathlib import Path

PROFILE = 'full'            # 6 models, 3 folds, 5 seeds, 5 cutoffs = 450 neural runs
DEVICE = 'auto'             # 'auto', 'cuda' or 'cpu'; CUDA must be enabled in Kaggle Settings
LONG_TARGET = 0            # 64 remains experimental; default 0 uses validated short histories
MAX_NEW_RUNS = 0           # 0 = no count limit; set e.g. 25 for short deliberate chunks
MAX_SESSION_MINUTES = 600  # 10h safe-pause between runs, leaving time to ZIP; Kaggle limits vary
EPOCHS = 50                # upper bound; validation early stopping uses patience=7
PATIENCE = 7
RESTORE_RESULT_ZIP = True
REPO_URL = 'https://github.com/tydeptrai21042004/histchess.git'

WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Use Kaggle Notebook with Internet enabled.'
REPO=WORK/'histchess'
DIAG=WORK/'histchess_v6_full_launcher_diagnostics'
DIAG.mkdir(exist_ok=True)
os.environ.update({'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                   'TOKENIZERS_PARALLELISM':'false','USE_TF':'0','USE_FLAX':'0',
                   'PYTHONHASHSEED':'0'})

def run(label,args,cwd=REPO):
    print('\n====',label,'====\n$',' '.join(map(str,args)),flush=True)
    with (DIAG/(label+'.log')).open('w',encoding='utf-8') as f:
        p=subprocess.Popen(list(map(str,args)),cwd=str(cwd),
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,errors='replace',bufsize=1,env=os.environ.copy())
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line)
        rc=p.wait()
    if rc:raise RuntimeError(f'{label} failed ({rc}); inspect {DIAG/(label+".log")}')

def safe_unzip(archive, dest, prefix=None):
    with zipfile.ZipFile(archive) as z:
        invalid=z.testzip()
        if invalid:raise RuntimeError(f'Corrupt ZIP member {invalid}')
        count=0
        for member in z.infolist():
            if member.is_dir():continue
            name=member.filename
            if prefix and not name.startswith(prefix):raise RuntimeError(f'Unexpected member: {name}')
            if name.startswith('/') or '\\' in name:raise RuntimeError(f'Unsafe ZIP path: {name}')
            destfile=(dest/name).resolve()
            if not destfile.is_relative_to(dest.resolve()):raise RuntimeError('ZIP path traversal')
            destfile.parent.mkdir(parents=True,exist_ok=True)
            destfile.write_bytes(z.read(member));count+=1
    return count

def make_zip(folder):
    if not folder.is_dir():return
    dest=WORK/(folder.name+'.zip')
    tmp=dest.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file():z.write(f,f'{folder.name}/{f.relative_to(folder)}')
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():raise RuntimeError('Output ZIP failed integrity check')
    tmp.replace(dest)
    print('DOWNLOAD FROM KAGGLE OUTPUT:',dest,dest.stat().st_size,'bytes',flush=True)

OUT=None
try:
    if REPO.exists():shutil.rmtree(REPO)
    run('01_clone',['git','clone','--depth','1',REPO_URL,REPO],WORK)
    patches=sorted(Path('/kaggle/input').rglob('HistChess_v6_full_patch.zip'))
    if len(patches)>1:raise RuntimeError('Attach only one v6 patch ZIP')
    patch_sha=None
    if patches:
        patch_sha=hashlib.sha256(patches[0].read_bytes()).hexdigest()
        print('APPLYING V6 PATCH:',patches[0],safe_unzip(patches[0],REPO))
    main_source=REPO/'histchess/architecture_benchmark.py'
    if not main_source.is_file() or 'histchess-v6-publication-full-2026-10' not in main_source.read_text():
        raise RuntimeError('HistChess v6 not found. Attach HistChess_v6_full_patch.zip or commit it to GitHub.')
    if not (REPO/'scripts/make_manuscript_tables.py').is_file():
        raise RuntimeError('Missing manuscript table script from v6 patch')
    run('02_install',[sys.executable,'-m','pip','install','-q','chess==1.11.2',
        'scikit-learn>=1.3,<2','pytest>=7,<10'],WORK)
    import torch
    chosen=('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE=='auto' else DEVICE
    if chosen=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable: enable GPU Accelerator in Kaggle settings')
    print('TORCH:',torch.__version__,'DEVICE:',chosen,
          'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',flush=True)
    NAME=f'histchess_v6_full_{chosen}_stress{LONG_TARGET}'
    OUT=WORK/NAME
    if RESTORE_RESULT_ZIP and not OUT.exists():
        saved=sorted(Path('/kaggle/input').rglob(NAME+'.zip'))
        if len(saved)>1:raise RuntimeError('More than one resume archive found; remove older copies')
        if saved:
            print('RESUMING FROM:',saved[0],safe_unzip(saved[0],WORK,NAME+'/'),'files',flush=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    (DIAG/'source.json').write_text(json.dumps({'github_sha':sha,'patch_sha256':patch_sha,
         'torch_version':torch.__version__,'device':chosen,'profile':PROFILE,
         'epochs':EPOCHS,'patience':PATIENCE,'max_new_runs':MAX_NEW_RUNS,
         'max_session_minutes':MAX_SESSION_MINUTES,'long_target':LONG_TARGET},indent=2))
    command=[sys.executable,'scripts/run_research_pipeline.py',
             '--profile',PROFILE,'--long-target-plies',str(LONG_TARGET),'--out',OUT,
             '--device',chosen,'--epochs',str(EPOCHS),'--patience',str(PATIENCE),
             '--max-session-seconds',str(MAX_SESSION_MINUTES*60)]
    if MAX_NEW_RUNS:command+=['--max-new-runs',str(MAX_NEW_RUNS)]
    run('03_full_pipeline',command)
    status=json.loads((OUT/'pipeline_status.json').read_text())
    if status.get('complete'):
        print('\nSUCCESS: ALL 450 neural runs AND classical comparisons audited.',flush=True)
        print('PAPER TABLES:',OUT/'manuscript_results.md',OUT/'manuscript_neural_table.tex',flush=True)
    elif status.get('paused'):
        print('\nSAFE RESUME POINT:',status['neural_runs_completed'],'of',status['expected_neural_runs'],flush=True)
        print('Add the latest',NAME+'.zip','as the input of your next Kaggle session, then rerun this cell.',flush=True)
        print('Do not report publication results before full completion and both audits.',flush=True)
    else:
        raise RuntimeError('Unexpected incomplete pipeline status')
except BaseException:
    (DIAG/'failure.txt').write_text(traceback.format_exc(),encoding='utf8')
    print('FAILED. Existing partial results, if any, will still be packaged.',flush=True)
    raise
finally:
    if OUT is not None:make_zip(OUT)
    make_zip(DIAG)
