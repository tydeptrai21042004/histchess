# HistChess — minimal strict-parser preflight patch + clone-to-Kaggle launcher

Applies to the supplied HistChess main-source snapshot, Git commit `58993ba80237a75ff2bec497cacfac8d2cce04bd`.

## Fixed failure
`pytest` had 87 passing, 2 failing tests in `tests/test_encoding_metrics.py`. The production `parse_yes_no` function intentionally returns `None` (invalid) for non-strict replies. Corrected ONLY the two stale test expectations:

- `('YES because', True)` → `('YES because', None)`
- `('No? Maybe', False)` → `('No? Maybe', None)`

No model, parser, architecture, experimental protocol, or evaluation metrics were changed.

## ZIP contents
- `tests/test_encoding_metrics.py` — corrected test file; copy into GitHub and commit, OR simply use the Kaggle cell to apply the patch after cloning
- `HistChess_strict_parser_test_fix.patch` — exact unified diff; from repository root: `git apply HistChess_strict_parser_test_fix.patch`
- `KAGGLE_HISTCHESS_GITHUB_ALL_RESULTS_FIXED_ONE_CELL.py` — self-contained ONE Python cell; clones GitHub, applies patch idempotently, tests, runs all V8/V7/V6/efficiency/LLM tracks with saving and resuming
- `README.md` — this document

## Kaggle execution
Turn Internet ON and enable a GPU. Open `KAGGLE_HISTCHESS_GITHUB_ALL_RESULTS_FIXED_ONE_CELL.py`, copy its entire code into ONE Kaggle notebook cell, and execute. You do NOT need to upload this ZIP to Kaggle or update GitHub first.

The launcher produces `/kaggle/working/HistChess_GitHub_ALL_Results.zip`. For a second session, attach the most recent results ZIP as Kaggle Input and rerun exactly the same cell. The workload is very large (V8 900, V7 450, V6 450 neural configurations plus other stages) and cannot be assumed to fit in one session.

This patch has been checked for test expectation consistency, parser behavior, syntax, and ZIP integrity locally. Training, GPU and complete preflight have NOT been run here.
