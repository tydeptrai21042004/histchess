# HistChess v4 code-change-only patch

Apply the ZIP entries directly at the root of `tydeptrai21042004/histchess`.
The ZIP includes only changed/new files (not a full replacement repository).
Do **not** extract the ZIP to a new nested directory inside the repository.

## Important compatibility constraints

- Your already finished 450-run architecture archive remains a separate v3
  result. Do not overwrite or reuse the v3 experiment folder as v4.
- Resume fingerprints intentionally changed to bind model/generator source,
  optimizer settings, input boards, labels, and dataset annotations.
- `KAGGLE_ARCHITECTURE_V4_ONE_CELL.py` clones the GitHub repository. It always
  overlays `HistChess_v4_code_patch.zip` if you attach it to Kaggle Inputs. If
  no ZIP is attached, it requires the new v4 code to be on GitHub.
- The Kaggle runner **defaults to a pilot**, not a full paper experiment.
  Select `PROFILE='full'` once the pilot passes. It runs CPU only.
- The new long-history generator is a **synthetic stress control**. Castling
  has a long post-event delay; the other three tasks have long *preambles*.
  Do not cite their performance as evidence of long-range rule-state memory.
- Full verified long-history datasets for 256/512 plies are not implemented.
  The 75-move/fivefold auto-draw conditions limit naive reversible loops.
  This patch does not claim to add realistic historical game trajectories or
  independent fourth-to-eighth generator mechanisms.

## Existing dataset / baseline audit

If you have Kaggle's original `histchess_architecture_full.zip`, unpack it
and run (in a Python environment):

```bash
python scripts/verify_architecture_results.py histchess_architecture_full \
  --write histchess_architecture_full/independent_integrity_audit.json
```

The audit is purely an integrity check, not an external-validity certification.

## Local v4 experiment

```bash
python -m pip install chess==1.11.2 torch pytest
python -m pytest -q tests/test_architecture_benchmark.py tests/test_length_stress.py
python scripts/run_architecture_benchmark.py \
    --out outputs/histchess_v4_pilot \
    --pairs-per-task 30 --generator-seed 3101 \
    --long-target-plies 64 --architectures gru causal_transformer \
    --folds 0 --seeds 3101 --cutoffs 0 16 \
    --epochs 2 --patience 2 --batch-size 64 \
    --max-history 256 --position-encoding sinusoidal \
    --length-eval-cutoffs 16 32 64 full \
    --length-eval-train-cutoffs 16 --device cpu
python scripts/verify_architecture_results.py outputs/histchess_v4_pilot
```

If the long-history generator cannot meet its requested target while keeping
legal, non-terminal moves, it raises an error. Do not silently lower the
requested length or substitute unvalidated records. For a non-long-history
pipeline smoke test only, use `--long-target-plies 0` or omit it.


## v4.1 correction — certified long-history generator

The initial v4 synthetic history extension could exhaust 100 attempts with
`No reversible knight move available`, and long knight-only walks were also
vulnerable to automatic fivefold repetition. v4.1 replaces them with:

- A **common legal continuation after castling-right divergence**, maintaining
  identical visible boards while preserving distinct castling rights.
- For other rules, **short reversible knight cycles interleaved with legal
  irreversible pawn blocks** before the decisive suffix. This makes the input
  long; it does NOT create a long causal dependency for these other rules.
- Explicit replay of every intermediate position, automatic game-outcome
  checks, and fully rebuilt labels/certificates.
- A 64-ply regression covering all four rules and a small multi-family sample,
  called by the Kaggle preflight tests before model training.

Old v4 datasets and shards must not be resumed into v4.1. The runner uses
separate `histchess_architecture_v4_1_*` result directories. Attach
`HistChess_v4_1_code_patch.zip`, not the former v4 ZIP.
