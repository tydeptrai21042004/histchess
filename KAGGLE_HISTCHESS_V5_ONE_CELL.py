# HistChess v5: paste this WHOLE file as ONE Kaggle Python cell.
# Kaggle: Accelerator=None, Internet=On.
# Either commit v5 patch contents to GitHub OR attach HistChess_v5_code_patch.zip as Kaggle Input.
# The source ZIP produced below is NOT needed if the GitHub repository already contains v5.
import os,sys,subprocess,json,zipfile,shutil,hashlib,traceback
from pathlib import Path

PROFILE='pilot'   # switch to 'full' ONLY after pilot passes (450 neural runs)
LONG_TARGET=0     # 0 is reliable default; 64 is experimental and explicitly fail-closed
REPO_URL='https://github.com/tydeptrai21042004/histchess.git'
RESTORE_RESULT_ZIP=True
WORK=Path('/kaggle/working'); assert WORK.is_dir(), 'Must run on Kaggle'
REPO=WORK/'histchess'
NAME=f'histchess_v5_{PROFILE}_stress{LONG_TARGET}'
OUT=WORK/NAME
DIAG=WORK/(NAME+'_diagnostics');DIAG.mkdir(exist_ok=True)
os.environ.update({'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                   'TOKENIZERS_PARALLELISM':'false','USE_TF':'0','USE_FLAX':'0','PYTHONHASHSEED':'0'})

def package(folder):
    if not folder.is_dir():return
    target=WORK/(folder.name+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,f'{folder.name}/{p.relative_to(folder)}')
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise RuntimeError('Bad archive CRC: '+str(target))
    print('OUTPUT READY:',target,target.stat().st_size,'bytes',flush=True)

def safe_extract(zip_path,dest,required_prefix=None):
    count=0
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip():raise RuntimeError('Bad input ZIP: '+str(zip_path))
        for member in z.infolist():
            if member.is_dir():continue
            if required_prefix and not member.filename.startswith(required_prefix):
                raise ValueError('Unexpected restore directory: '+member.filename)
            if member.filename.startswith('/') or '\\' in member.filename:
                raise ValueError('Unsafe ZIP member: '+member.filename)
            where=(dest/member.filename).resolve()
            if not where.is_relative_to(dest.resolve()):raise RuntimeError('Unsafe ZIP path')
            where.parent.mkdir(parents=True,exist_ok=True)
            where.write_bytes(z.read(member));count+=1
    return count

def cmd(label,args,cwd=REPO):
    print('\n==== '+label+' ====\n$ '+' '.join(map(str,args)),flush=True)
    with (DIAG/(label+'.log')).open('w',encoding='utf8') as f:
        p=subprocess.Popen([str(x) for x in args],cwd=str(cwd),
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line)
        rc=p.wait()
    if rc:raise RuntimeError(f'{label} exited {rc}: '+str(DIAG/(label+'.log')))

try:
    if RESTORE_RESULT_ZIP and not OUT.exists():
        matches=sorted(Path('/kaggle/input').rglob(NAME+'.zip'))
        if matches:
            restored=safe_extract(matches[0],WORK,NAME+'/')
            print('RESTORED EXISTING RESULT FILES:',restored,flush=True)
    if REPO.exists():shutil.rmtree(REPO)
    cmd('01_clone',['git','clone','--depth','1',REPO_URL,REPO],WORK)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    patches=sorted(Path('/kaggle/input').rglob('HistChess_v5_code_patch.zip'))
    patch_hash=None
    if patches:
        patch=patches[0]; patch_hash=hashlib.sha256(patch.read_bytes()).hexdigest()
        print('PATCH:',patch,'SHA256:',patch_hash)
        print('PATCHED FILES:',safe_extract(patch,REPO))
    required=('scripts/run_research_pipeline.py','scripts/run_comparison_baselines.py',
              'scripts/verify_comparison_results.py','histchess/comparison_baselines.py')
    if any(not (REPO/x).is_file() for x in required):
        raise RuntimeError('HistChess v5 missing. Commit patch contents to GitHub or attach HistChess_v5_code_patch.zip. Do NOT attach the old v4 patch.')
    (DIAG/'source.json').write_text(json.dumps({'git_sha':sha,'v5_patch_sha256':patch_hash,
                             'profile':PROFILE,'long_target':LONG_TARGET},indent=2))
    cmd('02_small_dependencies',[sys.executable,'-m','pip','install','-q',
        'chess==1.11.2','scikit-learn>=1.3,<2','pytest>=7,<10'],WORK)
    import torch
    print('PyTorch CPU:',torch.__version__,flush=True)
    if torch.cuda.is_available():raise RuntimeError('CPU-only required. Set Accelerator=None')
    cmd('03_complete_research_pipeline',[sys.executable,'scripts/run_research_pipeline.py',
        '--profile',PROFILE,'--long-target-plies',str(LONG_TARGET),'--out',OUT])
    final=json.loads((OUT/'pipeline_status.json').read_text(encoding='utf8'))
    if not final['complete']:raise RuntimeError('Pipeline did not complete successfully')
    print('COMPLETE: neural training + same-pair classical benchmark + BOTH independent audits')
except BaseException:
    (DIAG/'failure.txt').write_text(traceback.format_exc(),encoding='utf8')
    print('FAILED. Partial files and diagnostics will be packaged; no false completion.',flush=True)
    raise
finally:
    package(OUT)
    package(DIAG)
