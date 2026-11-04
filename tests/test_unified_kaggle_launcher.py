"""Lightweight stdlib tests for the one-study Kaggle orchestration layer."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

FILE=Path(__file__).resolve().parents[1]/'KAGGLE_HISTCHESS_ONE_CELL.py'
spec=importlib.util.spec_from_file_location('histchess_unified_launcher', FILE)
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_one_protocol_config_is_900_not_1800():
    assert launcher.TOTAL_CONFIGS == 900
    assert len(launcher.ARCHITECTURES) == 6
    assert len(launcher.SEEDS) == 2
    assert len(launcher.CONTEXT_CUTOFFS) == 5
    assert launcher.EXPECTED_REPLICATIONS == 5
    cmd=launcher._study_cmd(include_llm=False)
    assert '--run-llm' not in cmd
    assert '--max-new-runs' in cmd
    assert cmd[cmd.index('--out')+1] == launcher.STUDY
    assert '--epochs' in cmd and cmd[cmd.index('--epochs')+1] == '50'
    assert '--run-llm' in launcher._study_cmd(include_llm=True)


def test_checkpoint_roundtrip_and_no_extra_tracks(tmp_path):
    root=tmp_path/'HistChess_Study_Results'
    (root/'study').mkdir(parents=True)
    (root/'study'/'v8_config.json').write_text('{}')
    (root/'study_manifest.json').write_text('{"source_commit":"abcdef"}')
    archive=tmp_path/'test.zip'
    launcher.pack_checkpoint(root,archive)
    recovered=tmp_path/'recovered'/'HistChess_Study_Results'
    launcher.restore_checkpoint(archive,recovered)
    assert json.loads((recovered/'study_manifest.json').read_text())['source_commit']=='abcdef'
    assert launcher.count_progress(recovered)['neural_configs_completed']==0
    assert launcher.count_progress(recovered)['all_complete'] is False


def test_zip_traversal_is_rejected(tmp_path):
    root=tmp_path/'HistChess_Study_Results'
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('HistChess_Study_Results/../../bad.txt','bad')
    with pytest.raises(RuntimeError,match='Unsafe checkpoint'):
        launcher.restore_checkpoint(archive,root)


def test_llm_recovery_patch_is_idempotent(tmp_path):
    src=tmp_path/'source'
    script=src/'scripts'/'run_v8_research.py'
    script.parent.mkdir(parents=True)
    # Build a stub with the upstream interruption-sensitive block.
    old_block=launcher._apply_interrupted_llm_fix.__code__.co_consts
    original=next(s for s in old_block if isinstance(s,str) and 'if not metric_file.exists():' in s)
    script.write_text('prefix\n'+original+'suffix\n')
    launcher._apply_interrupted_llm_fix(src)
    corrected=script.read_text()
    assert 'def llm_artifacts_complete():' in corrected
    launcher._apply_interrupted_llm_fix(src)
    assert script.read_text()==corrected


def test_new_study_without_checkpoint_has_zero_training_configs(tmp_path):
    root=tmp_path/'HistChess_Study_Results'
    root.mkdir()
    progress=launcher.count_progress(root)
    assert progress['neural_configs_completed']==0
    assert progress['neural_configs_remaining']==900
    assert not progress['all_complete']
