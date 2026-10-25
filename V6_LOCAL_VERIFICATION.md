# HistChess v6 — Verification record (not research results)

Source: updated from HistChess v5 repository.

Verified locally on Python 3.13, CPU-only PyTorch 2.10.0:

- Python compilation: updated architecture engine, pipeline, Kaggle launcher, manuscript exporter passed.
- `tests/test_full_resume_v6.py`, `test_manuscript_v6.py`, `test_architecture_benchmark.py`, `test_comparison_baselines.py`: **15 tests passed**.
- Safe-pause/resume integration: 1 newly trained configuration saved, then remaining configuration trained with the original resumed shard reused.
- Independent neural audit on synthetic unit-test fixtures: passed.
- Manuscript formatting test using **artificial fixtures** (not empirical results): produced LaTeX/Markdown/CSV/manifest.
- Device fail-closed: CUDA requested without CUDA raises an explicit error.

Not verified in this execution environment:

- Real chess legality suite (`python-chess` is not installed locally; network-disabled pip install failed).
- Real Kaggle GPU execution or relative GPU/CPU throughput.
- All 450 final configurations; **no empirical manuscript results have been generated here**.

Before claiming full results, execute the supplied Kaggle cell, require `pipeline_status.json: complete=true`, both independently audited status JSONs showing `passed=true`, and inspect learning curves / data length distribution. This is a three-family-per-task **synthetic** benchmark, not external pretrained LLM testing.
