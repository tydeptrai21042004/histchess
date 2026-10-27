# HistChess v6: CPU-only paper extension (NO GPU and NO neural training).
# Kaggle: Accelerator=None. Add BOTH as Kaggle inputs:
#   1) HistChess_paper_cpu_patch.zip (this archive)
#   2) histchess_v6_full_cpu_stress0.zip (your actual completed v6 results)
# If your original result is CUDA-trained, use histchess_v6_full_cuda_stress0.zip instead.
import os,sys,subprocess,zipfile,shutil
from pathlib import Path
WORK=Path('/kaggle/working');INPUT=Path('/kaggle/input')
assert WORK.is_dir() and INPUT.is_dir(), 'Run on Kaggle Notebook.'
patches=list(INPUT.rglob('HistChess_paper_cpu_patch.zip'))
results=list(INPUT.rglob('histchess_v6_full_*_stress0.zip'))
if len(patches)!=1:raise RuntimeError('Attach EXACTLY one HistChess_paper_cpu_patch.zip to Kaggle Inputs')
if len(results)!=1:raise RuntimeError('Attach EXACTLY one completed v6 result ZIP (CPU or CUDA)')
def unpack(zpath,dest):
    with zipfile.ZipFile(zpath) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP corruption: '+str(bad))
        for x in z.infolist():
            if x.is_dir():continue
            p=(dest/x.filename).resolve()
            if x.filename.startswith('/') or '\\' in x.filename or not p.is_relative_to(dest.resolve()):
                raise RuntimeError('Unsafe ZIP entry: '+x.filename)
            p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(x) as f, p.open('wb') as o:shutil.copyfileobj(f,o)
subprocess.run([sys.executable,'-m','pip','install','-q','chess==1.11.2','scikit-learn>=1.3,<2','matplotlib>=3.7,<4'],check=True)
unpack(patches[0],WORK/'hc_paper_tools')
unpack(results[0],WORK)
source=WORK/results[0].stem
if not (source/'pipeline_status.json').exists():
    raise RuntimeError('Original ZIP must contain the matching top-level v6 result directory.')
out=WORK/'histchess_v6_paper_cpu_extension'
if out.exists():shutil.rmtree(out)
subprocess.run([sys.executable,str(WORK/'hc_paper_tools'/'scripts'/'run_paper_cpu_extension.py'),
                '--results',str(source),'--out',str(out)],check=True)
zipout=WORK/'histchess_v6_paper_cpu_extension.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(out.rglob('*')):
        if f.is_file():z.write(f,f.relative_to(WORK))
print('DOWNLOAD:',zipout,zipout.stat().st_size,'bytes')
