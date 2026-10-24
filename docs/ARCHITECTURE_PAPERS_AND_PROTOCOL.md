# HistChess: Paper-supported sequence architecture comparison

This is a **new** experiment independent of the legacy bidirectional GRU and the two SmolLM2 prompt tests. All models here are **causal, trained from scratch, and given identical observable features**. The benchmark does not evaluate a commercial LLM or reproduce numerical results claimed by other papers.

## Architectures and peer-reviewed sources

| CLI name | Mechanism | Supporting paper | What is and is not replicated |
|---|---|---|---|
| `elman_rnn` | Two-layer tanh RNN; recurrent hidden state | J. L. Elman, *Finding Structure in Time*, Cognitive Science 14 (1990), 179–211. [DOI](https://doi.org/10.1207/s15516709cog1402_1) | Core Elman recurrence; not the original paper's dataset/network hyperparameters. |
| `lstm` | Two-layer gated recurrent cell | S. Hochreiter and J. Schmidhuber, *Long Short-Term Memory*, Neural Computation 9 (1997), 1735–1780. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735) | Standard PyTorch LSTM, not a published pretrained checkpoint. |
| `gru` | Two-layer gated recurrent unit | K. Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder*, EMNLP (2014). [DOI](https://doi.org/10.3115/v1/D14-1179) | Unidirectional GRU; unlike the project's old bidirectional GRU. |
| `causal_transformer` | Two-layer causal softmax multi-head self-attention | A. Vaswani et al., *Attention Is All You Need*, NeurIPS (2017). [Paper](https://arxiv.org/abs/1706.03762) | Decoder-style **causal attention** on history with learned absolute positions, but not GPT pretraining or a full language model. |
| `linear_attention` | Two-layer ELU+1 feature-map causal attention, cumulative key/value state | A. Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML (2020). [Paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) | Correct causal linear-attention mechanism; this is a small research implementation, not their architecture/checkpoint. |
| `diagonal_ssm` | Two-layer **real** diagonal linear recurrent state transition | A. Gu et al., *On the Parameterization and Initialization of Diagonal State Space Models* (2022). [Paper](https://arxiv.org/abs/2206.11893) | **S4D-inspired ablation only**. It is NOT a faithful S4D implementation and does not support claims about a canonical S4D/Mamba checkpoint. |

Additional state-tracking context: S. Toshniwal et al., *Chess as a Testbed for Language Model State Tracking*, AAAI (2022), [DOI](https://doi.org/10.1609/aaai.v36i10.21390); B. Walker and T. Lyons, *Chess-World-Model: A 10M-Game Benchmark for Exact State Tracking from Chess Move Sequences* (2026), [paper](https://arxiv.org/abs/2605.30100); W. Merrill, J. Petty, A. Sabharwal, *The Illusion of State in State-Space Models*, ICML (2024), [paper](https://arxiv.org/abs/2404.08819).

## What the code actually tests

1. **Causality**: no model can attend to unseen future moves; transformer uses an upper-triangular attention mask, linear attention uses prefix key/value accumulators, recurrent models process left-to-right.
2. **Matching**: same board piece placement, side-to-move, task identity, complete UCI coordinates and promotion code; same train/validation/test IDs within each fold; hidden rule-state FEN fields excluded; model widths picked *before* fitting to get close to a 150k trainable-parameter budget (exact counts reported).
3. **Family holdout**: three rotations of the three built-in generator families per task. Each family becomes test exactly once. These rotations are **correlated views of the same synthetic dataset**, not independent real-game tests.
4. **History**: each model is **retrained** under its assigned history cutoff. The result isolates effect of training with the specified observable input, not post hoc test-only truncation.
5. **Seeds and validation**: identical seed list and optimizer/loss; checkpoint chosen by validation BCE, never test accuracy.
6. **Counterfactual metrics**: record accuracy and joint-pair accuracy, all raw predictions, family-level metrics, paired comparisons, leakage-safe split details, and exact observability bounds.
7. **Resume**: completed individual runs saved in `architecture_run_shards/` and bound to a fingerprint of data and run configuration. Interrupted jobs can resume without silently combining mismatched files.

### Mathematical bound

For balanced pairs with identical board and identical visible history suffix but opposite labels, no deterministic classifier based solely on those observations can exceed 50% record accuracy for **that pair** or successfully classify both members. Across a mixture with fraction `d` of distinguishable pairs, the *pairwise-information bound* is at most `0.5 + d/2` on record accuracy and `d` on joint-pair accuracy. These are loose ceilings, not model scores and not proof of architecture-specific impossibility.

### Limitations, required before an architecture-level paper

- The current v3 histories are generally short (up to 24 plies); `full` and >=32 plies may be equivalent. Not a long-memory stress test.
- The current halfmove label is `halfmove_clock >= 2`, not the actual 50- or 75-move chess rules.
- There are only three generator families per chess task and documented shortcut templates remain. Rotate test family and retain shortcut diagnostics, but add independent mechanisms and realistic longer trajectories before claiming a general inductive-bias weakness.
- Parameter matching is approximate; two-layer networks have different computational cost. Report exact parameters, FLOPs/timing, training-set size, best validation epoch. Differences are conditional on budget/hyperparameters; no single failure proves a flaw in all LLMs.
- `causal_transformer` is a small model **trained from scratch on labeled chess histories**; it is evidence about self-attention's inductive bias under this setting, **not about GPT/LLM pretraining in general**.
- `diagonal_ssm` is a transparent simplified ablation. For claims about exact S4D/Mamba, adapt the authors' own implementations and conduct a separately labeled experiment.

## Output files

- `architecture_config.json`: paper, mechanism, feature permissions, architecture budget.
- `dataset_validation.json`: independent chess-rule replay from `python-chess`.
- `architecture_metrics.csv`: per-model, per-fold, per-seed, per-history, per-task record and joint-pair accuracies.
- `architecture_predictions.csv`: every raw test prediction and score.
- `architecture_summary.csv`: descriptive fold/seed summaries, including min/max over runs.
- `architecture_matched_comparisons.csv`: matched pair-bootstrap differences, *not* adjusted for multiplicity or independent generator sampling.
- `observability_bounds.csv`: exact information-limited ceilings.
- `architecture_learning_curves.csv`: training and validation BCE.
- `architecture_status.json`: strict completion record.
- `architecture_run_shards/`: individually cached experiments to resume safely.

Use `python scripts/run_architecture_benchmark.py --help` for options.

## v4 optional methodology additions (source patch October 2026)

The v3 release, including its existing 450 results, is preserved. Running v4
creates a different fingerprint and must use a **new** output directory. These
features are controls, not findings about real large language models.

- `--position-encoding sinusoidal`: fixed (rather than randomly initialized
  unseen learned) position signals. Does not guarantee length generalization.
- `--length-eval-cutoffs 16 32 64 full --length-eval-train-cutoffs 16`:
  evaluates the SAME trained 16-ply checkpoint using different observed
  context windows. `architecture_length_ood_metrics.csv` explicitly lists the
  fraction whose context **actually changed**. A nominal cutoff > example
  length is not a longer example.
- `--long-target-plies 64`: opt-in synthetic reversible knight excursions,
  never silently treating shortened examples as successful. **Castling**
  excursions follow the distinguishing event and thus stress its retention;
  the other three tasks place neutral moves BEFORE their decisive suffix and
  mainly test long-input processing, not long retention. This is NOT realistic
  chess-game distribution or a benchmark of thousands of independent histories.
  All histories are rechecked using python-chess including automatic terminal
  outcomes. `tests/test_length_stress.py` runs in Kaggle with chess installed.
- `--lr`, `--weight-decay`, and `--dropout` expose optimizer controls for
  separate, validation-selected sweeps; no test-set hyperparameter tuning.
- `observability_bounds.csv` now includes an **empirical observation-consistency
  ceiling** computed from identical observable input signatures across the
  held-out dataset, in addition to the looser pairwise ceiling.
- `architecture_metrics.csv` adds calibrated Brier score and tie-aware
  pair-ranking accuracy. The original record and strict joint-pair metrics
  remain unchanged.
- Resume fingerprints include complete pair data and source hashes for the
  generator, model, and rule checker. Shards from older revisions intentionally
  cannot be reused in the new experiment.
- `scripts/verify_architecture_results.py` recomputes recorded metrics from
  raw predictions and checks labels, membership, completeness, score ranges,
  and the zero-history invariant. It has no pretrained-LLM dependency.

**Still needed before a broad architecture claim:** independent new generator
families (beyond the existing three), realistic game distributions, faithfully
implemented S4D/Mamba baselines, actual long-dependency tasks beyond castling,
validation-selected optimization, systematic parameter scale ablations, and
independent-family uncertainty estimates. No code patch can substitute for
these empirical controls, and no result by itself proves an LLM impossibility.
