# HistChess v3 — strengthened synthetic benchmark patch

**Scope.** This ZIP includes only new and modified code plus patch instructions. It does **not** contain newly computed scientific results. Apply at the root of `tydeptrai21042004/histchess`, keeping directory structure. Commit and push before running the Kaggle cell.

## Changes

- New `histchess/diverse.py`: **3 generation mechanisms for each of 4 rule tasks**, each independently passed through legal-move replay and same-board/FEN controls. Legacy generator remains fully available.
- New **mechanism-held-out** split (`--generator diverse --split-strategy family`): complete generation families withheld from training/validation; every task has train, validation, and test pairs.
- New `histchess/shortcuts.py`: records legacy fixed-rule shortcut performance and prefix/suffix n-gram memorization classifiers trained **on train only**, scored on held-out test only. Raw predictions and signature coverage exported.
- Expanded `leakage_audit.json`: per-task family/split counts and cross-split family overlap.
- Paired bootstrap confidence intervals and automatically generated `paper_tables.md` now include shortcut models; no fabricated numbers.
- Fail-closed full pipeline audit and artifact checks for the new reports.
- `tests/test_diverse_generation.py` tests deterministic replay, family-level holdout, non-perfect original signatures and compatibility with the old generator.
- `KAGGLE_CPU_STRENGTHENED_ONE_CELL.py` starts from GitHub clone and produces separate v3 result + dataset + diagnostics archives.

## Run (Kaggle, CPU, Internet ON)

After uploading these files to GitHub, copy the contents of `KAGGLE_CPU_STRENGTHENED_ONE_CELL.py` into **one** Kaggle cell. Select **Accelerator=None**. The cell runs unit/integration tests, stress cases, smoke, the complete GRU/LLM pipeline, validation and ZIP verification. It preserves partial outputs on error.

Or run locally:

```bash
pip install -r requirements.txt
python -m pytest -q tests -m 'not gpu_integration'
python scripts/stress_test.py --generator diverse --split-strategy family --pairs-per-task 60
python scripts/run_all_kaggle.py --out histchess_results_v3 --allow-cpu --generator diverse --split-strategy family --pairs-per-task 300 --epochs 14 --train-seeds 2 --llm-pairs-per-task 24
python scripts/verify_artifacts.py histchess_results_v3.zip
```

## Required paper wording / limitations

This remains a **synthetic** counterfactual dataset, with **three defined recipe families per task**. Family holdout evaluates **inter-mechanism transfer** only; it is not equivalent to unseen natural games, independent labeling rules engines, new LLMs, or true out-of-distribution chess generalization. Keep the original scores as **v2 in-distribution** results; never merge v2 and v3 prediction CSVs, since the data and test distributions differ. Report task-wise accuracy and paired joint accuracy, held-out family names, nonparametric paired CIs, the prefix/suffix shortcut coverage, and exact tested Git SHA.

The `paper_complete` field in the legacy output contract means that the **configured code stages ran**; it is not a claim that manuscript peer-review requirements are fulfilled. Newly generated results are only valid when all new chess-rule tests, `validation.json`, `leakage_audit.json`, `shortcut_audit.json`, and artifact verification pass on the actual Kaggle execution.
