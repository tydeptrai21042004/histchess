# HistChess Kaggle: paste this ENTIRE file into one Kaggle Python cell.
# Notebook settings: Internet = ON, Accelerator = None (CPU).
# OPTION A (recommended): add histchess_kaggle_patch.zip as a Kaggle input Dataset.
# OPTION B: set REPO_URL to a repo that ALREADY includes the patch changes.
import os, sys, subprocess, zipfile, shutil
from pathlib import Path

REPO_URL = ""  # optional: e.g. https://github.com/YOUR_USER/histchess.git
PAIRS_PER_TASK = 80
GRU_EPOCHS = 3
LLM_PAIRS_PER_TASK = 2  # diagnostic only; raise to >=20 for a meaningful model assessment
OUT = Path('/kaggle/working/histchess_results')
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['HF_HUB_DISABLE_XET']='1'

def run(cmd, cwd=None):
    print('\n>>>',' '.join(map(str,cmd)),flush=True)
    subprocess.run([str(x) for x in cmd],cwd=str(cwd) if cwd else None,check=True)

patches = list(Path('/kaggle/input').rglob('histchess_kaggle_patch.zip')) if Path('/kaggle/input').exists() else []
project = WORK/'histchess-main'
if REPO_URL:
    if project.exists(): shutil.rmtree(project)
    run(['git','clone','--depth','1',REPO_URL,str(project)])
if patches:
    print('Using patch:',patches[0])
    with zipfile.ZipFile(patches[0]) as archive:
        # Reject path traversal entries.
        for name in archive.namelist():
            dest=(WORK/name).resolve()
            if not dest.is_relative_to(WORK.resolve()): raise ValueError('Bad zip archive path')
        archive.extractall(WORK)
if not (project/'scripts'/'run_all_kaggle.py').is_file():
    raise FileNotFoundError('Project source missing: attach histchess_kaggle_patch.zip or set REPO_URL to updated repository')

# Kaggle already includes torch, numpy and pandas; avoid reinstalling large GPU wheels.
run([sys.executable,'-m','pip','install','-q','python-chess==1.999','transformers>=4.44,<5','safetensors>=0.4'])
run([sys.executable,'scripts/selftest.py'],cwd=project)
run([sys.executable,'scripts/run_all_kaggle.py',
     '--out',str(OUT), '--pairs-per-task',str(PAIRS_PER_TASK),
     '--epochs',str(GRU_EPOCHS), '--llm-pairs-per-task',str(LLM_PAIRS_PER_TASK),
     '--strict'],cwd=project)
print('\nResults:',str(OUT)+'.zip')
print('Check run_status.json for ALL REQUESTED EXPERIMENTS COMPLETE = True')
