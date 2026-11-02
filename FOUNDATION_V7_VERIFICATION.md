# Local verification report — 2026-11-02

Verified with Python 3.13 and PyTorch 2.10.0+cpu; all commands operated on the supplied source plus v7 changes. The original repository revision was `c0d2fa579a3a53f9408a8e90f70df90686a40e85` as recorded in the attached archive.

**Passed**

- Python `compileall` for HistChess modules and scripts.
- 23 numerical/architecture tests passed using `python -m pytest -q tests/test_foundation_models.py tests/test_architecture_benchmark.py`.
- The GRU reference equations, multi-head ELU+1 attention against explicit causal quadratic scores, linear-attention parallel/recurrent outputs and gradients, S4D-Lin FFT convolution/recurrent outputs and gradients, causality, masks, finite gradients and parameter budgets were tested.
- Synthetic-tensor efficiency-script smoke run: GRU-Original and S4D-Lin at lengths 8/16 produced finite latency measurements.
- **10 training configurations** on a *synthetic fixture (not legally verified chess)* passed run initialization, one epoch of optimization, prediction file creation, and the independent neural results audit (`passed: true`, `errors: []`). This is integration verification only, not measured chess accuracy.

**Not verified here**

- Full chess game generation, `python-chess` replay, end-to-end v7 pilot/full pipeline and classical comparisons. The local runtime did not contain the `chess` package and could not install it due to unavailable package-network access. Running the full suite directly produced import errors from the missing dependency. Install `chess==1.11.2` and rerun on your Kaggle or connected environment.
- The complete new 450-run foundation experiment, model performance improvements, GPU memory-speed measurements and independent real-game transfer have **not** been executed and are not claimed.

**Method boundaries**: S4D-Lin is the official minimal diagonal-ZOH mathematical variant, not the S4D-LegS reference; the classifier block/FFN and causal Transformer adaptations remain specific to this benchmark. See `FOUNDATION_V7_README.md`.
