# HistChess v3 — generator-family split validation fix

## Observed failure
Kaggle commit `41c51f62bdfea95a1574b728e5c46e9a68bb524d`:
`tests/test_diverse_generation.py::test_family_split_disjointness_and_per_task_balance_larger`
failed at `assert validate(pairs)["passed"]` (56 tests passed, 1 failed).

## Root cause and fix
The diversified generator tracked uniqueness by `(task, board_key)`, allowing the
same observable chess board to be assigned to different generator-family splits
across different tasks. `validate()` correctly prohibits that cross-split board
leakage. The generator now tracks `board_key` **globally** across all tasks,
resampling colliding candidates while retaining the same rule oracle, family
assignments and target counts. Regression assertions check global board uniqueness.

## Usage
1. To permanently update GitHub, extract this archive into the root of the
   strengthened `histchess` repository, preserving directory paths; commit and push.
2. To run immediately, execute `KAGGLE_CPU_V3_FIXED_ONE_CELL.py` in a Kaggle notebook
   (Accelerator=None, Internet=On). That cell clones GitHub and automatically
   applies this precise fix if the previous version is still committed.

The fix has passed Python syntax checks and ZIP CRC verification locally.
The end-to-end chess replay tests must run in Kaggle; this sandbox does not have
`python-chess` installed and cannot download it. No model accuracy claim is made.
