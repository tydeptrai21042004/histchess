# HistChess architectural benchmark — Kaggle ONE CPU cell.
# Settings: Accelerator=None; Internet=On.
# Either push patch ZIP CONTENTS to GitHub or attach HistChess_architecture_patch.zip in Kaggle input.
# No pretrained LLM runs. These are small, research-backed causal architectures.
import os,sys,json,zipfile,shutil,subprocess,traceback,time,hashlib
from pathlib import Path

REPO_URL='https://github.com/tydeptrai21042004/histchess.git'
PROFILE='full'  # 'pilot' tests the entire execution chain; 'full' runs all 3 family folds and 5 seeds.
USE_ATTACHED_V3_PAIRS=True   # Reuse precisely the uploaded v3 dataset if added to Kaggle Inputs.
RESUME_FROM_ATTACHED_RESULTS=True  # Add the earlier partial architecture ZIP to Kaggle Inputs to resume a new session.

PROFILES={
 'pilot':dict(folds=(0,),seeds=(3101,),cutoffs=('0','8','full'),epochs=2,pairs=60),
 'full': dict(folds=(0,1,2),seeds=(3101,3102,3103,3104,3105),
              cutoffs=('0','4','8','16','full'),epochs=14,pairs=300),
}
CFG=PROFILES[PROFILE]
W=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO=W/'histchess'
OUT=W/f'histchess_architecture_{PROFILE}'
LOGS=W/f'histchess_architecture_{PROFILE}_diagnostics'
LOGS.mkdir(exist_ok=True)
os.environ.update({'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'4',
                   'MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
                   'USE_TF':'0','USE_FLAX':'0','PYTHONHASHSEED':'0'})

def run(label,cmd,cwd=REPO):
 print('\n==',label,'==\n$',' '.join(map(str,cmd)),flush=True)
 with (LOGS/f'{label}.log').open('w',encoding='utf8') as fh:
  proc=subprocess.Popen([str(x) for x in cmd],cwd=str(cwd),env=os.environ.copy(),
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,errors='replace',bufsize=1)
  for line in proc.stdout:print(line,end='',flush=True);fh.write(line)
  rc=proc.wait()
 if rc:raise RuntimeError(f'{label} failed with exit code {rc}; see {LOGS/f"{label}.log"}')

def package(directory,dest):
 if not directory.exists():return
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(directory.rglob('*')):
   if f.is_file():z.write(f,arcname=f'{directory.name}/{f.relative_to(directory)}')
 with zipfile.ZipFile(dest) as z:
  if z.testzip():raise RuntimeError('Bad ZIP integrity: '+str(dest))
 print('SAVED:',dest,flush=True)

def apply_patch():
 if (REPO/'histchess/architecture_benchmark.py').exists():
  return 'Architecture benchmark source already on GitHub'
 candidates=[W/'HistChess_architecture_patch.zip']
 if Path('/kaggle/input').exists():candidates+=list(Path('/kaggle/input').rglob('HistChess_architecture_patch.zip'))
 patch=next((p for p in candidates if p.is_file()),None)
 if patch is None:raise RuntimeError('Push the new patch to GitHub, or attach HistChess_architecture_patch.zip as Kaggle Input')
 with zipfile.ZipFile(patch) as z:
  if z.testzip():raise RuntimeError('Patch CRC failure')
  for member in z.infolist():
   if member.is_dir():continue
   dest=(REPO/member.filename).resolve()
   if not dest.is_relative_to(REPO.resolve()):raise RuntimeError('Unsafe patch ZIP path')
   dest.parent.mkdir(parents=True,exist_ok=True)
   dest.write_bytes(z.read(member))
 return 'Applied attached patch '+hashlib.sha256(patch.read_bytes()).hexdigest()

def dataset_from_input():
 if not USE_ATTACHED_V3_PAIRS or not Path('/kaggle/input').exists():return None
 for name in ('histchess_results_v3.zip','histchess_dataset_v3.zip'):
  for path in sorted(Path('/kaggle/input').rglob(name)):
   with zipfile.ZipFile(path) as z:
    options=[a for a in z.namelist() if a.endswith('/data/pairs.jsonl') or a=='pairs.jsonl']
    if len(options)==1:
     target=W/'histchess_architecture_input_pairs.jsonl'
     target.write_bytes(z.read(options[0]))
     return target
 return None

def restore_previous_result():
 if not RESUME_FROM_ATTACHED_RESULTS or OUT.exists() or not Path('/kaggle/input').exists():return
 name=f'histchess_architecture_{PROFILE}.zip'
 archives=sorted(Path('/kaggle/input').rglob(name))
 if not archives:return
 archive=archives[0]
 print('RESUME: restoring saved architecture shards from',archive,flush=True)
 with zipfile.ZipFile(archive) as z:
  if z.testzip():raise RuntimeError('Resume archive CRC error')
  expected=OUT.name+'/'
  for member in z.infolist():
   if member.is_dir():continue
   if not member.filename.startswith(expected):raise RuntimeError('Resume ZIP has unexpected directory')
   dest=(W/member.filename).resolve()
   if not dest.is_relative_to(W.resolve()):raise RuntimeError('Unsafe result archive member')
   dest.parent.mkdir(parents=True,exist_ok=True)
   dest.write_bytes(z.read(member))

try:
 restore_previous_result()
 if REPO.exists():shutil.rmtree(REPO)
 run('01_clone',['git','clone','--depth','1',REPO_URL,REPO],W)
 patch_info=apply_patch()
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 (LOGS/'source.json').write_text(json.dumps({'commit':sha,'patch':patch_info,'profile':PROFILE,'config':CFG},indent=2))
 print('SOURCE:',patch_info,'SHA:',sha)
 run('02_dependencies',[sys.executable,'-m','pip','install','-q','chess==1.11.2','pytest>=7,<10'])
 import torch
 if torch.cuda.is_available():raise RuntimeError('Expected accelerator None / CPU-only')
 print('CPU:',torch.__version__)
 run('03_core_tests',[sys.executable,'-m','pytest','-q','tests','-m','not gpu_integration'])
 import chess
 print('CHESS:',chess.__version__)
 args=[sys.executable,'scripts/run_architecture_benchmark.py','--out',OUT,
       '--architectures','elman_rnn','lstm','gru','causal_transformer','linear_attention','diagonal_ssm',
       '--folds',*CFG['folds'],'--seeds',*CFG['seeds'],'--cutoffs',*CFG['cutoffs'],
       '--epochs',CFG['epochs'],'--patience',4,'--batch-size',64,
       '--max-history',256,'--target-parameters',150000,'--device','cpu']
 source_pairs=dataset_from_input()
 if source_pairs:
  print('DATASET: using uploaded v3:',source_pairs)
  args+=['--pairs-jsonl',source_pairs]
 else:
  print('DATASET: generating a new legal v3 dataset from source code')
  args+=['--pairs-per-task',CFG['pairs'],'--generator-seed',3101]
 run('04_architecture_experiments',args)
 status=json.loads((OUT/'architecture_status.json').read_text())
 if not status['complete']:raise RuntimeError('Incomplete architecture benchmark')
 folds=len(CFG['folds']);seeds=len(CFG['seeds']);cuts=len(CFG['cutoffs'])
 expected=6*folds*seeds*cuts
 if len(status['runs'])!=expected:raise RuntimeError(f'Expected {expected} completed training runs; got {len(status["runs"])}')
 print('\nCOMPLETE: all',expected,'causal architecture runs finished.')
 print('Summary:',OUT/'architecture_summary.csv')
 print('Predictions:',OUT/'architecture_predictions.csv')
except BaseException:
 (LOGS/'failure.txt').write_text(traceback.format_exc())
 print('FAILED; results from completed architecture shards are preserved for resume.')
 raise
finally:
 package(OUT,W/f'histchess_architecture_{PROFILE}.zip')
 package(LOGS,W/f'histchess_architecture_{PROFILE}_diagnostics.zip')
 for n in (f'histchess_architecture_{PROFILE}.zip',f'histchess_architecture_{PROFILE}_diagnostics.zip'):
  f=W/n
  if f.exists():print('AVAILABLE:',f, f.stat().st_size,'bytes')
