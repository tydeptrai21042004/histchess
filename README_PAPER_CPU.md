# HistChess v6 — CPU publication extension (additive only)

This companion does **not** modify the original repository, its training configuration, pretrained models, or original Kaggle v6 results. It produces a **separate ZIP**.

## Inputs

- `HistChess_paper_cpu_patch.zip` (this package).
- The **actual original result ZIP** `histchess_v6_full_cpu_stress0.zip` or `histchess_v6_full_cuda_stress0.zip` (around 25 MB for the successful CPU job). The tiny diagnostics ZIP is not sufficient.

## Kaggle

- Kaggle Settings: `Accelerator=None`. Internet ON only if `chess`, scikit-learn or matplotlib need installation.
- Add BOTH archives to Kaggle Inputs.
- Paste the entire content of `KAGGLE_PAPER_CPU_ONE_CELL.py` into ONE cell, and run.
- Download `histchess_v6_paper_cpu_extension.zip` from Kaggle Output.

## Local

```bash
pip install chess==1.11.2 'scikit-learn>=1.3,<2' 'matplotlib>=3.7,<4'
python scripts/run_paper_cpu_extension.py --results PATH_TO_ORIGINAL_V6_OUTPUT_DIRECTORY --out ./histchess_v6_paper_cpu_extension
```

## Actual new baselines (CPU only)

Train/val/test splits exactly match v6 3-fold family rotations. Per fold, validation chooses C from {0.1,1,10}. All learned baseline results use *only observable* final board, side-to-move, task and restricted visible move history. They never read the target metadata or test labels for prediction.

- `suffix_1`: last visible ply + board/side/task.
- `suffix_2`: last two visible plies + board/side/task.
- `suffix_4`: last four visible plies + board/side/task.
- `prefix_4`: first four plies OF THE VISIBLE WINDOW + board/side/task (not full-history prefix under truncated conditions).
- `legacy_fixed_signatures`: historical rule-of-thumb from the original `shortcuts.py`, evaluated once on the full-history test splits without tuning; useful to expose template leakage.
- `legal_replay_full`: deterministic rule-based full-history replay using `python-chess`. This is an algorithmic reference, **not a trained neural baseline**. It actually computes the true rule label and checks legal moves, unlike the existing v6 `privileged_replay_oracle` that reads labels. It is run **ONLY at full-history visibility**, because a cropped suffix cannot generally be replayed from the initial board.

The original v6 board-only null, move-bag and move-ngrams remain untouched and are shown alongside CPU baselines in the generated plots.

## Output

- `cpu_baseline_predictions.csv`: every held-out prediction, including pair_id, fold, task, variant, label, score.
- `cpu_baseline_metrics.csv`: record, strict joint-pair and ranking accuracies for each fold/task/history.
- `cpu_baseline_validation_selection.csv`: CPU model selection evidence; no test-set tuning.
- `cpu_baseline_summary.csv`: full-history descriptive means.
- `dataset_pair_diagnostics.csv` and `dataset_summary_by_task.csv`: history lengths, shared prefix/suffix, same move-bag indicator, equal last move, and per-task aggregate.
- `paper_cpu_manifest.json`: original file SHA256 provenance and explicit 0 new neural training runs.
- `figure_1_joint_accuracy.pdf`: actual saved six-architecture scores vs two cheap shortcut baselines.
- `figure_2_shortcut_diagnostics.pdf`: short-window shortcut controls for each task.

## Required manuscript caveats

- This is an additional **within-HistChess baseline analysis**, not an external chess benchmark.
- The six neural architectures were trained from scratch, not pretrained LLMs; no claim about pretrained LLM failure follows directly.
- Pair/fold/seed reuse is correlated. Do not treat 15 fold-seed configurations as independent datasets.
- En passant needs the final move; halfmove >=2 needs the last two relevant plies. They are NOT long-memory tasks.
- Full legal replay is a task-specific symbolic upper reference and is not a fair neural model comparison.
- Retain all original v6 results and compare by exact test pair IDs, never mix datasets or silently rerun neural models.
- The extension cannot determine paper acceptance or confirm model weaknesses without inspecting the actual numerical outputs.
