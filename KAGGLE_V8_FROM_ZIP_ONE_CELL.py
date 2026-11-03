# HistChess v8 -- paste into ONE Kaggle Python cell.
# Kaggle Settings: Internet ON; Accelerator GPU (recommended).
# Upload HistChess_V8_Code.zip as a Kaggle Input dataset.
# Optionally attach previous HistChess_V8_Results.zip as another Kaggle Input.
# Do not use the unmodified GitHub main: it may not contain these new changes.
import json,os,sys,time,subprocess,zipfile,shutil,traceback
from pathlib import Path
WORK=Path('/kaggle/working'); INPUT=Path('/kaggle/input')
CODE=WORK/'histchess_v8_code'; RESULTS=WORK/'HistChess_V8_Results'
RESULT_ZIP=WORK/'HistChess_V8_Results.zip'
PER_SESSION_NEW_RUNS=25
RUN_PRETRAINED_LLMS=True
# Reduce models/cutoffs/replications ONLY for a smoke run, not a paper-ready result.
ARCHITECTURES=['elman_original','lstm','gru_original','causal_transformer_postnorm',
               'linear_attention_multihead','s4d_lin']
SEEDS=['3101','3102']; REPLICATIONS=5; PAIRS_PER_SCENARIO=8
LEVELS=['native','32','64']


def extract_scoped(src,destination,prefix):
    with zipfile.ZipFile(src) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP entry: {bad}')
        for member in z.infolist():
            if member.is_dir():continue
            name=member.filename
            if not name.startswith(prefix) or '\\' in name:
                raise ValueError(f'Unexpected ZIP path: {name}')
            relative=name[len(prefix):]
            dest=(destination/relative).resolve()
            if not dest.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe ZIP entry: {name}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as fin,dest.open('wb') as fout:shutil.copyfileobj(fin,fout)


def zip_results():
    if not RESULTS.is_dir():return
    temporary=RESULT_ZIP.with_suffix('.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for f in sorted(RESULTS.rglob('*')):
            if f.is_file():z.write(f,f'{RESULTS.name}/{f.relative_to(RESULTS)}')
    with zipfile.ZipFile(temporary) as z:
        if z.testzip():raise RuntimeError('Damaged results ZIP')
    temporary.replace(RESULT_ZIP)
    print('RESULT ZIP:',RESULT_ZIP,'MB:',round(RESULT_ZIP.stat().st_size/1048576,1))


try:
    code_zip=list(INPUT.rglob('HistChess_V8_Code.zip'))
    if len(code_zip)!=1:raise RuntimeError('Attach exactly one HistChess_V8_Code.zip input')
    if CODE.exists():shutil.rmtree(CODE)
    extract_scoped(code_zip[0],CODE,'histchess-main/')
    assert (CODE/'scripts/run_v8_research.py').exists()
    previous=list(INPUT.rglob('HistChess_V8_Results.zip'))
    if len(previous)>1:raise RuntimeError('Multiple checkpoint ZIPs attached')
    if previous and not (RESULTS/'v8_config.json').exists():
        extract_scoped(previous[0],RESULTS,'HistChess_V8_Results/')
        print('Restored prior audited checkpoint:',previous[0])
    RESULTS.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','pip','install','-q','chess==1.11.2','scikit-learn>=1.3,<2',
         'pytest>=7,<10','transformers>=4.45,<5','safetensors>=0.4']
    subprocess.run(cmd,check=True)
    import torch
    print('Torch',torch.__version__,'CUDA available',torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError('V8 full protocol requires Kaggle GPU. Enable the Accelerator first.')
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_v8_statistics.py',
                    'tests/test_v8_generator.py','-m','not slow'],cwd=CODE,check=True)
    options=[sys.executable,'scripts/run_v8_research.py',
             '--out',RESULTS,'--replications',str(REPLICATIONS),
             '--per-family',str(PAIRS_PER_SCENARIO),
             '--levels',*LEVELS,'--architectures',*ARCHITECTURES,
             '--seeds',*SEEDS,'--cutoffs','0','16','32','64','full',
             '--length-eval-cutoffs','32','64','full',
             '--length-eval-train-cutoffs','16',
             '--epochs','50','--patience','7','--device','cuda',
             '--grid-lrs','0.0003','0.001',
             '--grid-weight-decays','0.001',
             '--grid-dropouts','0.1','0.3',
             '--max-new-runs',str(PER_SESSION_NEW_RUNS)]
    if RUN_PRETRAINED_LLMS:options+=['--run-llm']
    print('Running v8 paper protocol; checkpointing each neural configuration...',flush=True)
    subprocess.run(list(map(str,options)),cwd=CODE,check=True)
    status=json.loads((RESULTS/'v8_status.json').read_text())
    print('All requested stages complete:',status['all_replications_complete'],
          'Statistics complete:',status['statistics_complete'],
          'LLM complete:',status['llm_complete'])
except BaseException:
    print(traceback.format_exc())
    raise
finally:
    zip_results()
