# HistChess architecture benchmark — additive GitHub patch

This ZIP contains **new files only**, so it does not replace the v3 generator, baseline
results or manuscript. Extract these paths into the root of
`https://github.com/tydeptrai21042004/histchess` and commit/push.

## Added files

- `histchess/architecture_benchmark.py` — six research-backed causal sequence-model variants; 3-fold generator-family rotation; train and test metrics; promotion-safe UCI input; theoretical observability bounds; resumable runs; source paper metadata.
- `scripts/run_architecture_benchmark.py` — one command for new experiments.
- `tests/test_architecture_benchmark.py` — CPU gradient, causality, budget, split, promotion and observability regression tests.
- `docs/ARCHITECTURE_PAPERS_AND_PROTOCOL.md` — authoritative citations, model details, evaluation protocol and known limitations.
- `KAGGLE_ARCHITECTURE_CPU_ONE_CELL.py` — CPU-only Kaggle launcher; can apply this ZIP from a Kaggle Input if it is not on GitHub.

## How to run

In Kaggle, set Accelerator=None and Internet=On. Copy the entire contents
of `KAGGLE_ARCHITECTURE_CPU_ONE_CELL.py` into a single Python cell.

Either:
1. Extract these 5 new files into your GitHub repository, then push, **or**
2. Attach the exact `HistChess_architecture_patch.zip` file as a Kaggle Input.

Set `PROFILE='pilot'` to test; `PROFILE='full'` runs 6 architectures x 3
independent-family rotations x 5 model training seeds x 5 history lengths =
**450 neural training jobs**. This setting can be resource intensive on CPU.
Each run writes a bound checkpoint shard. To resume within a session,
rerun without deleting the output folder. To resume on a new Kaggle session,
attach the previously exported `histchess_architecture_full.zip` as an Input.

If you already have `histchess_results_v3.zip`, attach it to Kaggle Inputs
as well so the architecture study uses *exactly the previously evaluated*
1200 paired examples; otherwise the runner will regenerate and validate them.

Outputs contain neural predictions and descriptive statistical comparisons,
not invented LLM scores, claims of a general LLM impossibility, or unsupported
replications of pretrained paper results.

## Local quick command

```
python -m pip install chess==1.11.2 pytest torch
python -m pytest -q tests/test_architecture_benchmark.py
python scripts/run_architecture_benchmark.py --pairs-per-task 60 \
 --architectures elman_rnn lstm gru causal_transformer linear_attention diagonal_ssm \
 --folds 0 --seeds 3101 --cutoffs 0 8 full --epochs 2 --device cpu \
 --out ./histchess_architecture_pilot
```

**Validation status:** pure-PyTorch unit tests passed and a 54-run CPU
smoke-training matrix was executed on the attached v3 dataset. Full legal-
move replay and 450-run complete benchmark require Kaggle's `python-chess`
and full execution; no performance numbers should be treated as final until then.
