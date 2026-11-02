# Paste this entire file into ONE Kaggle Python cell.
# Upload HistChess_Foundation_Aligned_Code_v7.zip to Kaggle Inputs first.
# Kaggle: Internet ON (pip dependency install); GPU optional.
# Results are NEW v7 results, NOT the completed v6 manuscript measurements.
import hashlib, json, os, subprocess, sys, traceback, zipfile
from pathlib import Path

PROFILE='foundation_pilot'      # switch to 'foundation_full' only after pilot passes
DEVICE='auto'                  # auto/cpu/cuda
LONG_TARGET_PLIES=0           # optional 64; synthetic stress, not new external data
MAX_NEW_RUNS=0               # optionally limit number of newly trained runs
RESTORE_RESULT_ZIP=True     # attach the previous output ZIP for a new Kaggle session
MAX_SESSION_MINUTES=540
SOURCE_ZIP_NAME='HistChess_Foundation_Aligned_Code_v7.zip'
WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Run inside a Kaggle notebook'
REPO=WORK/'histchess-foundation-v7'
OUT=WORK/f'histchess_{PROFILE}_{LONG_TARGET_PLIES}ply_{DEVICE}'
LOGS=WORK/'histchess_foundation_launcher_logs'
LOGS.mkdir(exist_ok=True)


def command(name,args,cwd):
    print('\n===',name,'===',flush=True)
    with (LOGS/(name+'.log')).open('w',encoding='utf8') as f:
        proc=subprocess.Popen([str(x) for x in args],cwd=cwd,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True);f.write(line)
        if proc.wait()!=0:raise RuntimeError('Command failed: '+name)


def safe_extract(archive,destination,strip_prefix):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Corrupted ZIP entry: '+bad)
        for member in z.infolist():
            if member.is_dir():continue
            raw=member.filename
            if not raw.startswith(strip_prefix) or '\\' in raw:
                raise RuntimeError('Unsafe/unexpected source ZIP member '+raw)
            relative=raw[len(strip_prefix):]
            if not relative:continue
            path=(destination/relative).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise RuntimeError('ZIP path traversal')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(member))


def zip_results(folder):
    if not folder.exists():return
    output=WORK/(folder.name+'.zip')
    temp=output.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for file in sorted(folder.rglob('*')):
            if file.is_file():z.write(file,f'{folder.name}/{file.relative_to(folder)}')
    with zipfile.ZipFile(temp) as z:
        if z.testzip():raise RuntimeError('Results ZIP failed CRC')
    temp.replace(output)
    print('KAGGLE OUTPUT ZIP:',output,flush=True)


try:
    sources=list(Path('/kaggle/input').rglob(SOURCE_ZIP_NAME))
    if len(sources)!=1:raise RuntimeError(f'Expected 1 attached {SOURCE_ZIP_NAME}, found {len(sources)}')
    archive=sources[0]
    source_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    if REPO.exists():
        import shutil
        shutil.rmtree(REPO)
    safe_extract(archive,REPO,'histchess-main/')
    assert (REPO/'histchess/foundation_models.py').is_file()
    assert (REPO/'FOUNDATION_V7_README.md').is_file()
    command('install_dependencies',[sys.executable,'-m','pip','install','-q',
        'chess==1.11.2','scikit-learn>=1.3,<2','pytest>=7,<10'],WORK)
    import torch
    device='cuda' if (DEVICE=='auto' and torch.cuda.is_available()) else ('cpu' if DEVICE=='auto' else DEVICE)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    (LOGS/'reproduction_source.json').write_text(json.dumps({
        'source_zip_sha256':source_hash,'torch_version':torch.__version__,
        'device':device,'profile':PROFILE,'long_target_plies':LONG_TARGET_PLIES},indent=2))
    if RESTORE_RESULT_ZIP and not OUT.exists():
        restores=list(Path('/kaggle/input').rglob(OUT.name+'.zip'))
        if len(restores)>1:raise RuntimeError('Multiple previous result ZIPs attached')
        if restores:
            safe_extract(restores[0],OUT,OUT.name+'/')
            print('RESTORED previous result shards from:',restores[0],flush=True)
    invocation=[sys.executable,'scripts/run_research_pipeline.py','--profile',PROFILE,
        '--out',OUT,'--device',device,'--long-target-plies',str(LONG_TARGET_PLIES),
        '--max-session-seconds',str(MAX_SESSION_MINUTES*60)]
    if MAX_NEW_RUNS:invocation+=['--max-new-runs',str(MAX_NEW_RUNS)]
    command('run_pipeline',invocation,REPO)
    status=json.loads((OUT/'pipeline_status.json').read_text())
    if status.get('complete'):
        print('ALL TRAINING AND NEURAL/CLASSICAL AUDITS COMPLETE',flush=True)
    elif status.get('paused'):
        print('PARTIAL RUN. Do not report publication scores. Resume with same output directory and unchanged source.',flush=True)
    else:raise RuntimeError('Pipeline did not complete or safely pause')
except BaseException:
    (LOGS/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    zip_results(OUT)
    zip_results(LOGS)
