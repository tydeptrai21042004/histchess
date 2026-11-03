# HistChess v8 — new research protocol (separate from completed v6/v7)

This package **adds source code; it does not claim new experimental accuracy results**. Original v6/v7 source, results and metrics remain separate. The v8 protocol requires a genuine execution on Kaggle or an equivalent Python environment.

## Five research weaknesses addressed

| Gap | Implemented code | Proof / remaining boundary |
|---|---|---|
| Histories were only 4–24 plies | `histchess/research_extension.py` builds **genuine** legal trajectories at *minimum* 32 and 64 plies by default; optional 128/256, subject to legal generation | `dataset/length_and_dependency_audit.csv`, `architecture_length_ood_metrics.csv`, `history_cutoff_coverage.csv`, validation assertions. **Only castling** has a certified distant causal event; long en-passant, halfmove and repetition inputs can remain short-dependency tasks. |
| Transformer/attention overfitting | `--grid-lrs`, `--grid-weight-decays`, `--grid-dropouts` select training hyperparameters using **validation loss only**. All candidate learning curves and final generalization gaps are saved. | `architecture_learning_curves.csv`, `overfitting_diagnostics.csv`, chosen hyperparameters and best epoch in `architecture_metrics.csv`. Training budget and positional encoding remain visible. This is a controlled mitigation, not a guarantee that overfitting disappears. |
| Only three generator mechanisms | Adds 9+ auditable **scenario groups** (3 source motifs × context regimes), plus `--external-pairs-jsonl` for genuine new, provenance-labeled independent mechanisms | **The built-in source still has THREE independent motifs per rule**. More scenarios do NOT equal new independent mechanisms. New externally sourced mechanisms must be legal, matched, board-disjoint and attributable to their source. Without these, the fundamental diversity limitation remains. |
| Correlated folds and seeds | `histchess/research_statistics.py` collapses training seeds and fold rotations, compares the same held-out records, performs replicate-cluster bootstrap, paired sign-flip and Holm corrections | At least five independently seeded **dataset replications** for conditional confidence intervals; otherwise all comparisons remain descriptive. Even with five, inference is conditional on the three fixed synthetic motif families. |
| No pretrained LLMs | `scripts/run_v8_llm.py`, or `scripts/run_v8_research.py --run-llm` loads genuine pretrained Hugging Face weights (SmolLM2-135M-Instruct and 360M-Instruct by default), exact matched held-out pairs and deterministic prompts | Records raw predictions, invalid answers, test pair IDs, model weight revision metadata and paired accuracy; refuses to invent outputs. GPU/internet/model access are required for practical execution. |

Additional repair: `histchess/architecture_benchmark.py` now finds v7 `gru_original` and `causal_transformer_postnorm` reference names, so the neural matched-comparison file is no longer silently empty.

## Kaggle use

1. Upload **HistChess_V8_Code.zip** as a Kaggle dataset. In Kaggle Settings, enable **Internet ON** and **GPU**.
2. Paste the complete contents of `KAGGLE_V8_FROM_ZIP_ONE_CELL.py` into **one** Kaggle code cell and run.
3. Each session writes `HistChess_V8_Results.zip` even if interrupted by a run error. Attach that ZIP as a Kaggle input in a new session to resume **with identical settings**. The runner pins existing data through SHA-256 and completed training shards through their experiment fingerprints.
4. A status value of `all_replications_complete: false` means **unfinished**. Do not report unfinished configurations or treat skipped LLMs as evaluated.

The Kaggle launcher can run many different configurations, so it intentionally pauses at a configurable run boundary, `PER_SESSION_NEW_RUNS=25`. Increase it if your session capacity permits, or decrease it for shorter sessions.

## Direct command-line use

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/test_v8_statistics.py tests/test_v8_generator.py
python scripts/run_v8_research.py --out /kaggle/working/HistChess_V8_Results \
  --replications 5 --per-family 8 --levels native 32 64 \
  --max-new-runs 25 --run-llm
```

The full defaults request six architectures, 3 fold rotations, 2 training seeds, five cutoffs, five generated datasets and 4 validation hyperparameter candidates per neural configuration. Neural runs are **900 configurations × 4 validation candidates**; they may need repeated Kaggle sessions. Pretrained LM inferences are separate, and executed only when enabled. Do not count test-only history changes as retraining or independently generated data.

### Optional certified 128 and 256 plies

```bash
python scripts/run_v8_research.py --out output_128 --levels native 32 64 128 \
  --cutoffs 0 16 32 64 128 full --max-history 256
```

For minimum 256 plies use `--max-history 288` or more to allow generator overshoot and adjust all cutoffs. Length generation **fails explicitly** if no legal match can be constructed; it never fabricates or silently shortens an example. Larger length cohorts need independent confirmation in the target Kaggle environment.

### Additional truly independent generator mechanisms

Export JSONL rows in the same schema as verified `pairs.jsonl` with `generator_family`, `independent_mechanism`, and `source_provenance`. Every row must contain a verified positive/negative matched pair derived from the standard chess initial position. Unique source mechanisms are assigned to disjoint fold groups and checked with `core.validate` using legal chess replay.

```bash
python scripts/run_v8_research.py --out results_external --replications 1 \
  --external-pairs-jsonl /kaggle/input/new_motifs/external.jsonl
```

For more than one replication, provide separate files, e.g. `external_{rep}.jsonl`, to avoid pretending the same examples are independent. Source provenance is a declaration, not a certificate that a purported independent mechanism was constructed independently.

### Optional standalone pretrained LLM comparison

```bash
python scripts/run_v8_llm.py --pairs-jsonl results/replication_00/dataset/pairs.jsonl \
   --out results/replication_00/pretrained_llm --fold 0 \
   --models HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct
```

These are **pretrained small language models**; results must not be generalized to GPT-6 or to all large language models. Longer contexts can exceed the model's context window: the runner raises rather than silently truncating a causal move.

## Review boundaries still open

- The new built-in generator adds scenario variance but does not produce six new *independent* chess-state mechanisms. Add genuinely new generators, independently authored puzzles, or a validated real-game-prefix counterfactual benchmark to make a broad external-validity claim.
- Long castling histories use common continuation after the rights-changing event. Other tasks use legal prehistory before short suffixes. Do not use overall context length as a substitute for distance to the decisive event.
- Replication clusters are independent by re-seeding but share motif definitions. A Holm-adjusted p-value here is conditional on the synthetic data distribution, not a general chess-population theorem.
- Validation tuning can reduce optimization confounds; it does not imply a faithful reproduction of the original S4D training recipe or a universal architecture ranking.
