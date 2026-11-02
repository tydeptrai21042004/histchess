# HistChess v7 — Foundation-aligned code upgrade

**Status:** Source implementation and numerical reference tests are supplied. No v7 training accuracy has been claimed or fabricated. HistChess v6 experiment names and implementations remain available, and v6 manuscript numbers must not be overwritten with v7 outputs.

## What changed

1. **Five additional explicitly named model variants** in `histchess/foundation_models.py`:
   - `elman_original`: one-layer tanh Elman recurrence (vs. original v6 two layers).
   - `gru_original`: Cho et al. (2014) equations (5)–(8), including reset before the hidden candidate projection and original update-gate convention. Original optimized PyTorch GRU remains `gru`.
   - `causal_transformer_postnorm`: post-LayerNorm masked self-attention variant (original v6 `causal_transformer` is pre-LN). Causal masking and binary classification head are HistChess adaptations, not a complete encoder–decoder reproduction.
   - `linear_attention_multihead`: four-head, ELU+1 causal feature-map attention; recurrent and parallel implementations are both available for numerical equivalence checks. The original `linear_attention` remains.
   - `s4d_lin`: **minimal S4D-Lin** complex conjugate diagonal kernel with ZOH discretization, trainable time steps, FFT causal convolution and reference-style GLU mixing; original real-only `diagonal_ssm` remains a named ablation. This is S4D-Lin, not S4D-LegS/HiPPO, nor a reproduction of all full S4D training configurations. The additional residual FFN is a HistChess adaptation.

2. `tests/test_foundation_models.py`: exact reference-gate equations, gradient checks, attention against quadratic causal kernel, linear attention parallel/recurrent matching, S4D-Lin ZOH kernel vs. recurrent state steps, future-token non-leakage, trainable parameter budgets and history-cutoff alias diagnostics.
3. `history_cutoff_coverage.csv` is written with neural runs; flags when different nominal cutoffs expose identical actual histories. The existing test-time length reassessment is not a new out-of-distribution set.
4. `scripts/run_research_pipeline.py` accepts `--profile foundation_pilot` and `--profile foundation_full`. The original `pilot` / `full` remain. Use distinct output paths for v6 and v7.
5. `scripts/benchmark_model_efficiency.py` provides inference-time/throughput measurements on controlled synthetic tensor inputs, reporting GPU peak allocated memory only on CUDA.
6. Audited manuscript-table export now handles both legacy and foundation variants, and labels the protocol in the output.
7. AdamW automatically excludes the S4D-Lin continuous-state log-timescale and diagonal eigenvalue parameters from weight decay, as in the minimal official reference.

## Installation and checks

Install dependencies (use your Python environment with network access):

```bash
python -m pip install -r requirements-v5-cpu.txt
python -m pytest -q tests/test_foundation_models.py tests/test_architecture_benchmark.py
```

Run the full original dependency-bearing test suite before publishing results:

```bash
python -m pytest -q tests -m 'not slow and not gpu_integration'
```

## Run neural + classical evaluation

```bash
# fast integration experiment; trained predictions, not previously published scores
python scripts/run_research_pipeline.py --profile foundation_pilot --out results/foundation_pilot --device cpu

# full 6-model, 450-run comparison with 3 family rotations and 5 seeds
python scripts/run_research_pipeline.py --profile foundation_full --out results/foundation_full --device auto

# Optional existing 64-ply synthetic stress. May be substantially slower and
# is NOT an independent long-horizon benchmark for en passant / halfmove.
python scripts/run_research_pipeline.py --profile foundation_pilot --long-target-plies 64 --out results/foundation_stress64 --device auto

# Efficiency on your hardware; comparison needs same device and run settings.
python scripts/benchmark_model_efficiency.py --architectures gru_original causal_transformer_postnorm linear_attention_multihead s4d_lin --lengths 16 32 64 128 --out results/efficiency.csv
```

### Results to inspect

- `architecture_metrics.csv`, `architecture_predictions.csv`, `architecture_learning_curves.csv`
- `observability_bounds.csv` and **`history_cutoff_coverage.csv`** (cutoff aliases)
- `independent_integrity_audit.json`, `comparison_integrity_audit.json`
- `neural_vs_classical_matched.csv`, `manuscript_results.md` (for completed full profiles only)

**Integrity warning:** Old v6 training shards cannot be loaded into a v7 folder. The fingerprint now includes `foundation_models.py` and the protocol; a changed source, dataset, environment, or model set requires a new output directory.

## Scientific limitations that remain

- The dataset still has **three generator mechanisms per task** and at most 24 plies in default generation. This patch does not claim to have implemented the previously proposed 8–12 new independent mechanisms per task, natural PGN-derived challenge set, or 128–256-ply certified long-history test. Those require *new data generation and independent audits*, not just longer padding.
- `--long-target-plies 64` uses the existing synthetic length-stress generator: for castling it extends a shared suffix after the decisive event; for other tasks it often prepends neutral moves. Long sequence length should not be confused with long causal-event distance.
- Neural networks are trained from scratch; they are not full reproductions of the original foundational papers' empirical tasks or pretrained models. LSTM remains PyTorch's **modern LSTM**, not an exact reproduction of the original 1997 gate design. The Transformer retains a task-adapted causal encoder stack.
- Training defaults remain shared rather than architecture-specific tuned budgets. Do not declare architecture superiority from a single default optimizer setting; use validation-only hyperparameter tuning as a separate experiment.
- The correctness suite tests numerical mechanisms; it does not certify chess legality in the absence of `python-chess`, independent chess engines, or natural-game evaluation.

## Foundational code and papers

- Cho et al. (2014), EMNLP: https://aclanthology.org/D14-1179/ (GRU equations (5)–(8)).
- Vaswani et al. (2017): https://arxiv.org/abs/1706.03762 (post-norm multi-head scaled dot-product attention).
- Katharopoulos et al. (2020): https://proceedings.mlr.press/v119/katharopoulos20a.html (causal feature-map linear attention).
- Gu et al. (2022): https://arxiv.org/abs/2206.11893 (S4D parameterization/initialization).
- Official minimal S4D-Lin reference: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py (minimal S4D-Lin with ZOH). The new implementation follows the same mathematical core, without importing/copied external code.

No novel model-accuracy measurements are included in this code archive. Run the experiments and audit the raw predictions before revising the manuscript results.

## Kaggle run directly from this ZIP (no GitHub clone)

The ZIP includes `KAGGLE_FOUNDATION_V7_FROM_ZIP_ONE_CELL.py`. Upload the **full source ZIP** as a Kaggle Input, paste the runner into one notebook cell and execute. The runner extracts the exact uploaded source, installs runtime dependencies, checks its SHA-256, runs `foundation_pilot` by default and archives results. Set `PROFILE='foundation_full'` after checking pilot outputs. To resume a new session, attach the previous result ZIP as a second Kaggle input and leave `RESTORE_RESULT_ZIP=True` with unchanged experimental settings. Source archive integrity and model training fingerprint are checked; publication results require both independent audits.
