# HistChess — Counterfactual Chess-History Architecture Benchmark

**Release: v6 full-run protocol (October 2026).** HistChess tests whether learning systems recover **history-dependent chess state** from a visible board, side-to-move and legal move history. It generates matched positive/negative histories that end at the **same observable board**, but disagree on a precisely specified hidden chess-rule label. A new **classical comparison track** uses the *same held-out counterfactual pair IDs* as six causal neural architectures. Results are only produced by executing code; there are **no prefilled model scores**.

> **Scientific scope:** This is a synthetic, legality-verified benchmark, not a pretrained-LLM experiment, real-game evaluation, or proof of fundamental impossibility for Transformers/LLMs. The diagonal SSM is an **S4D-inspired real-valued ablation**, not the authors' S4D. The new classical comparison is **within-dataset**, not an external public benchmark.

## 1. Tasks and input permissions

| Task | Positive label means | What is deliberately withheld from trainable models |
| --- | --- | --- |
| `castling` | White retains **kingside castling rights** (not necessarily a legal castle move now) | Castling-right FEN flags |
| `en_passant` | Side to move has a **legal en-passant capture** | En-passant FEN square |
| `halfmove` | `halfmove_clock >= 2` (not 50/75-move draw) | Halfmove clock |
| `repetition` | Current position has occurred **at least three times, including now** | Occurrence history and count |

Neural and classical learned models get the same general observable information: current piece placement, side to move, task and the allowed UCI history suffix. They **never** receive canonical full FEN, hidden labels, or chess-rule metadata as features. The `privileged_replay_oracle` intentionally uses ground truth and **must not be reported as a fair learned-model competitor**.

Every counterfactual pair contains equally long, individually replayable legal histories with opposite labels. The generator and checker use `python-chess`; they are not two independent rule engines.

## 2. What is compared?

| Track | Models | Interpretation |
| --- | --- | --- |
| Neural architectures | `elman_rnn`, `lstm`, `gru`, `causal_transformer`, `linear_attention`, `diagonal_ssm` | Trained-from-scratch, approximately matched parameter budgets and causal histories |
| Classical benchmark | `board_only`, `move_bag`, `move_ngrams` | Null and regularized logistic-regression controls using the exact same test pair IDs |
| Analytical reference | `privileged_replay_oracle` | Perfect reference deliberately using privileged target information; **not a fair baseline** |

`move_bag` uses frequency of observed UCI moves without order. `move_ngrams` also includes adjacent move 2/3-grams and last-four-move identity features. For these two models, logistic-regression regularization is selected using **validation BCE/log-loss only** from `C in {0.1, 1, 10}`. Each neural architecture selects its best checkpoint by validation BCE. No reported test scores are used to select settings.

**Matched comparison:** `neural_vs_classical_matched.csv` calculates differences within the same fold, cutoff, and held-out pair IDs. It does not treat seeds/folds as independent new chess generators.

## 3. Kaggle full benchmark: CPU or GPU, resumable

Use **`KAGGLE_HISTCHESS_V6_FULL_ONE_CELL.py`**. Kaggle settings: **Internet=On**, and either **Accelerator=GPU** (recommended for the larger neural study) or **Accelerator=None** (CPU is mathematically valid). Select from Settings, then run the entire Python file in **one notebook cell**. The code detects the selected PyTorch device and logs the actual device. GPU is not guaranteed to outperform CPU for these small models and short sequences; measure throughput on your hardware.

1. Commit `HistChess_v6_full_patch.zip` contents over v5 in the root of GitHub **or** attach that exact ZIP as a Kaggle Input. The runner uses current GitHub `tydeptrai21042004/histchess` and applies an attached v6 patch when present.
2. Run the full one-cell script. The default is `PROFILE='full'`, `EPOCHS=50`, `PATIENCE=7`, `LONG_TARGET=0`, and `MAX_SESSION_MINUTES=600`. The 50-epoch limit is **not a guarantee of convergence**: early stopping selects the best checkpoint according to validation BCE.
3. The pipeline saves **one fingerprinted result shard after each neural configuration**. If Kaggle prints `SAFE RESUME POINT`, download the latest output archive, add it to your next Kaggle session as an Input, and execute the **same v6 one-cell runner with identical scientific settings**. Run counts and data fingerprints are checked. Do not mix CPU and GPU shards or modify the dataset/code/settings between sessions.
4. Only after all **450 neural runs** finish does the pipeline execute and audit the within-HistChess classical baselines, compute same-pair differences and generate manuscript tables. `pipeline_status.json` must say `complete: true`; both integrity audits must say `passed: true`.
5. Download `manuscript_neural_table.tex`, `manuscript_results.md`, `manuscript_task_breakdown.csv`, and `manuscript_methods_manifest.json` from the finished output ZIP. Copy measured tables into your manuscript and report the specific dataset/architecture restrictions below.

A full run covers **6 architectures × 3 generator-family folds × 5 independent initialization seeds × 5 history cutoffs = 450 neural training configurations**. The classical comparison covers **4 baseline methods × 3 folds × 5 cutoffs = 60 configurations**, of which one is a privileged label oracle and **not** a fair learned competitor. Original v3/v4/v5 result shards are deliberately incompatible with v6.

### Direct command-line mode

```bash
python -m pip install -r requirements-v5-cpu.txt
python scripts/run_research_pipeline.py \
  --profile full --device auto --epochs 50 --patience 7 \
  --max-session-seconds 36000 \
  --out results/histchess_v6_full
```

The runner uses an installed CUDA-capable PyTorch if one exists; otherwise, it runs on CPU. `--device cuda` fails explicitly if CUDA is unavailable. When the session budget is reached, an incomplete status is correctly marked as **paused**, never as completed. Restart the command on the same output folder to resume. `--max-new-runs 25` optionally limits each session to 25 newly trained configurations. Use a fresh output folder if any scientific setting changes.

### Full benchmark profiles

| Factor | v5 pilot (smoke test) | v6 full research experiment |
| --- | --- | --- |
| Architectures | GRU, Transformer | All 6 architectures |
| Rotated generator-family folds | 1 | 3 |
| Initialization seeds | 1 | 5 |
| History cutoffs | 0, 16 | 0, 4, 8, 16, full |
| Maximum epochs / patience | 2 / 7 | **50 / 7** |
| Counterfactual pairs per task | 30 | 300 |
| Neural configurations | 4 | **450** |
| Device | CPU or CUDA | CPU or CUDA; fixed throughout resume |
| Long-history generator | Disabled | Disabled pending certification |

**Manuscript validity:** v6 is a stronger *execution and reporting protocol*, not a new external benchmark. HistChess uses three synthetic generator mechanisms per rule; fold rotations reuse the same underlying pairs. Five seeds and three folds do not constitute 15 independent chess populations. The full-history category often contains short sequences, so alternate cutoffs on the same records are context sensitivity, **not** an independent long-history generalization experiment. Check `history_length_profile.csv` and `manuscript_methods_manifest.json` before any long-memory claim.

### Optional long-history generator — experimental, NOT the default

The source includes `--long-target-plies 64` as an **opt-in synthetic stress generator**, not a validated, published long-context result. The earlier v4 generator failed on castling and v4.1 was not yet fully verified against the complete chess-specific suite. This release **does not silently fall back** to shorter sequences: an opt-in failure stops with diagnostics. Do not enable it for the first pilot.

```bash
# Only after the default pilot succeeds, use a NEW output directory:
python scripts/run_research_pipeline.py --profile pilot \
  --long-target-plies 64 --out results/histchess_v5_stress_experimental
```

Under this generator, castling may involve a long gap after the critical move; other tasks commonly prepend neutral moves and therefore do **not** establish long-distance dependency retention. See `docs/ARCHITECTURE_PAPERS_AND_PROTOCOL.md`.

## 5. Auditing outputs

All results are stored under the selected output directory. Key files:

| Path | Contents |
| --- | --- |
| `pipeline_status.json`, `pipeline_logs/*.log` | Stage success/failure, stage output and settings |
| `data/pairs.jsonl`, `dataset_validation.json`, `history_length_profile.csv` | Full replay-verifiable data, rule checks, length distribution |
| `architecture_metrics.csv`, `architecture_predictions.csv` | Neural metrics and every held-out raw prediction |
| `architecture_summary.csv`, `architecture_length_ood_metrics.csv` | Descriptive aggregates and fixed-checkpoint context evaluations |
| `comparison_metrics.csv`, `comparison_predictions.csv` | Non-neural controls on the same test pairs |
| `comparison_summary.csv`, `comparison_validation_selection.csv` | Classical fold summaries and validation-selected regularization |
| `neural_vs_classical_matched.csv` | Exact matched fold/seed/cutoff comparative scores |
| `independent_integrity_audit.json`, `comparison_integrity_audit.json` | Independent raw-score audits; both must say `passed: true` |
| `manuscript_results.md`, `manuscript_neural_table.tex` | Generated **only after 450 configurations and both audits pass** |
| `manuscript_task_breakdown.csv`, `manuscript_methods_manifest.json` | Per-task descriptive metrics, protocol and dataset profile |

Run both checks manually at any time:

```bash
python scripts/verify_architecture_results.py results/histchess_v5_pilot
python scripts/verify_comparison_results.py results/histchess_v5_pilot
```

**Zero-context invariants:** If both histories in a balanced pair are identical to the model at cutoff zero, any deterministic classifier must score exactly 50% record accuracy and **0% strict joint-pair accuracy** on those pairs. The audits enforce this. The privileged oracle is excluded because it has target information.

## 6. Methodology boundaries for publications

- Three generator families per task are rotated among train, validation and test. These are **correlated synthetic splits**, not independent real-game samples. No external chess-PGN benchmark is executed by the new classical track.
- A 150k-parameter recurrent or attention network is not an off-the-shelf LLM. The experiment cannot show that large language models have a mathematical flaw.
- `diagonal_ssm` is **not** an implementation of the full published S4D or Mamba method.
- Most ordinary synthetic histories are **short**. Testing nominal 64/256 cutoffs on 24-ply examples is not a real long-context evaluation.
- Equal parameter counts do not equal computational cost or optimization quality. Training epochs and optimizer budgets can affect outcomes.
- The pairwise observability bound separates unavailable information from learnable information; a residual neural gap alone does **not** prove an architectural impossibility.
- Holdout-family standard deviations are descriptive only; they are not population-level confidence intervals over new generator mechanisms.
- Rule replay uses `python-chess` plus schema checks; results need independent implementations and more realistic data before broad claims.

## 7. Existing workflows and research references

The legacy GPU/SmolLM2 and dataset/Snakemake workflows are preserved in the repository and are **separate** from the CPU-first v5 architecture comparison. The old `KAGGLE_ONE_CELL.py` and `scripts/run_all_kaggle.py` are not the v5 launcher. Do **not** merge their numerical outputs without a protocol reconciliation.

Original architecture models and citations are documented in [`docs/ARCHITECTURE_PAPERS_AND_PROTOCOL.md`](docs/ARCHITECTURE_PAPERS_AND_PROTOCOL.md). The results are not reproduced numerical scores from those publications. Key sources include Elman (1990), Hochreiter and Schmidhuber (1997), Cho et al. (2014), Vaswani et al. (2017), Katharopoulos et al. (2020), and Gu et al. (2022).

**Repository rule:** Commit code, not experimental claims. Release a result only after `pipeline_status.json` reports `complete: true`, both independent audit JSONs pass, and the dataset, model version, and exact run configuration are archived.
