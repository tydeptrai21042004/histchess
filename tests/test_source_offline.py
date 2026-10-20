"""Tests executable without python-chess/HuggingFace to identify basic packaging faults."""
import ast
import pathlib
import pytest

ROOT=pathlib.Path(__file__).resolve().parents[1]


def test_all_python_files_syntax_valid():
    files=sorted(ROOT.rglob('*.py'))
    assert len(files)>=20
    for p in files:
        ast.parse(p.read_text(encoding='utf8'),filename=str(p))


def test_kaggle_cell_is_standalone_and_contains_diagnostics():
    text=(ROOT/'KAGGLE_ONE_CELL.py').read_text()
    assert 'histchess_diagnostics.zip' in text
    assert 'torch.cuda.is_available()' in text
    assert 'pytest' in text
    assert 'verify_artifacts.py' in text
    assert 'pip' in text
    assert 'pip\',\'install\',\'-q\'' in text
    assert "'torch>=" not in text


def test_all_expected_source_files_exist():
    files=['histchess/core.py','histchess/gpu_experiments.py',
           'histchess/paper.py','histchess/audit.py',
           'scripts/run_all_kaggle.py','scripts/stress_test.py',
           'scripts/verify_artifacts.py','scripts/selftest.py',
           'tests/test_rules.py','KAGGLE_ONE_CELL.py']
    for f in files:assert (ROOT/f).is_file(),f
