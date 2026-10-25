# HistChess v5 local verification record

Date: 2026-10-25. Based on user-uploaded `histchess-main(3).zip`.

## Executed and passed

- `python -m pytest -q tests/test_comparison_baselines.py tests/test_architecture_benchmark.py` — **11 passed** in the local CPU environment.
- `python -m compileall -q histchess scripts tests KAGGLE_HISTCHESS_V5_ONE_CELL.py` — no syntax errors.
- Neural + classical *synthetic-fixture* integration: trained GRU at two cutoffs; exported predictions/metrics, ran `scripts/verify_architecture_results.py` and `scripts/verify_comparison_results.py` — **both audits passed**; generated `neural_vs_classical_matched.csv` (6 rows). This is a code-level smoke test only, **not legal-chess experimental evidence**.
- Verified CLI help for research pipeline and standalone comparison entrypoints.

## Not executed / not certified here

- Full legal-chess `python-chess` tests: unavailable in this container; network access prevented installing `chess==1.11.2`.
- Kaggle live run, all six neural architectures at 450 configurations, actual 64-ply synthetic stress, external-real-game evaluation: **not executed**.

## Kaggle requirements

- Accelerator None, Internet On; dependency installation of `chess==1.11.2`, `scikit-learn>=1.3,<2`, `pytest>=7,<10`.
- Run `PROFILE='pilot'`, `LONG_TARGET=0` first; the runner executes the local suite and independent audits on actual legal chess pairs.
- Only if the pilot passes, change `PROFILE='full'` and retain source SHA, dataset SHA and completed shards.
- `LONG_TARGET=64` is an experimental add-on; do not claim a successful stress run without its own passing test and legal replay.
