# HistChess: GPU single-session paper reproduction

**New version:** An end-to-end experiment runner for a synthetic matched-counterfactual chess-history benchmark. **All results must be computed by execution. This ZIP does not contain measured paper numbers or the manuscript.**

## Important chess rule definition

Four binary tasks are generated from legal histories starting at the conventional initial chess position:

- **White kingside castling rights** (not immediate legality).
- **Legal en-passant capture** for the side to move.
- **Halfmove clock >= 2** (not the fifty-move draw).
- **Current position appears >=3 times**, including now (not prospective claimability).

The castling example labels in the earlier scaffold were **inverted**; this is fixed. The positive history leaves the white h1 rook unmoved. The negative history moves it out and back, permanently losing castling rights. The pair is otherwise identical in observable piece placement and side to move. Each pair is independently reconstructed and audited with `chess.Board()` and legal move replay. Repetition pairs have *identical full FENs* but different occurrence counts. Castling, EP and halfmove pairs differ in one canonical FEN field each.

The rules for current-position repetition in `python-chess` deliberately differ from `can_claim_threefold_repetition()` (possible future moves): see <https://python-chess.readthedocs.io/en/latest/core.html>.

## Kaggle full GPU run from zero

1. Create a Kaggle Notebook. Turn on **Internet** (to download open-weight models) and select **GPU T4/P100**. A CPU runtime is rejected by default.
2. Upload `histchess_gpu_full.zip` as a Kaggle Input Dataset, **or** push its contents to your GitHub repository and supply `REPO_URL` in the bootstrap.
3. Paste the entire `KAGGLE_ONE_CELL.py` into one Python cell and run. It installs only small missing packages; **it never replaces Kaggle's preinstalled GPU PyTorch**.
4. Download `/kaggle/working/histchess_results.zip` from Kaggle Outputs; check `run_status.json` and `manifest.json`. Only report rows actually computed by this run.

Default full mode: **300 pairs per task** (2,400 position-history records), 2 GPU GRU training seeds, independently retrained 0/4/8/16/32/full-ply models, 14 maximum epochs with best validation selection, and two pretrained SmolLM2 language models on the *same fixed held-out matched sample* of 24 pairs per task (48 records per task). Generation/split/test sizes vary if you edit the settings; model comparisons are strictly sampled from held-out positions.

To test the environment without claiming completed results, change `MODE='smoke'`. For larger paper experiments, raise `PAIRS_PER_TASK`, `TRAIN_SEEDS`, `GRU_EPOCHS`, and `LLM_PAIRS_PER_TASK` gradually, ensuring each task has enough held-out pairs. No result is fabricated on failure: partial outputs are zipped and `run_status.json` documents the error.

## Included evaluations

1. Four-task synthetic paired data with replay certificates and legal move checks.
2. Fresh replay-validation of labels, same-board constraint, single-FEN-field checks, and repetition same-FEN checks.
3. Privileged rule oracle, board-only balanced majority, privileged full-FEN analytical baseline.
4. Trainable **board-only neural control** (`history_moves=0`) and 4/8/16/32/full histories, retrained for each cutoff and GPU seed. All share the same board observation and task; never access true FEN metadata.
5. Strict joint paired accuracy; missing or invalid model replies count as wrong in reported record accuracy.
6. Board/FEN/history split overlap counts; synthetic-prefix overlap disclosure.
7. Multiple model seeds, early stopping on validation BCE, saved full-history checkpoint, learning curves, per-record test predictions.
8. Batched two-pretrained-model GPU inference (history-only prompts; no canonical full FEN). Fixed sample manifest and raw replies retained.
9. Pair-resampling 95% confidence intervals, paired sign-flip comparisons against balanced board-majority (unadjusted exploratory p-values), plus **fair neural-versus-LLM matched-sample metrics** calculated using exactly the same test pair IDs.
10. Paper-ready `.md` and `.tex` tables, two PNG figures, integrity audit, diagnostics, manifest and a compact ZIP (pretrained weights excluded).

## Limitations you must state in the manuscript

- This is **synthetic, template-family-controlled data**. Group-disjoint-by-board splits do not provide independent opening, game, or template family splits. Increased *quantity* alone does not demonstrate real-game generalization.
- There are few underlying counterfactual mechanisms; models may exploit generator cues. More independently sourced counterfactual motifs or an entirely new held-out generator are recommended for a stronger journal submission.
- The full-FEN analytical reference directly reveals three labels and cannot be treated as a fair history-only neural comparison. Repetition is *not* available in a single FEN.
- Oracle validation depends on one implementation, `python-chess`, plus checked algebraic invariants. It is not an independent chess rules engine.
- Each neural history-length cutoff is *retrained*, not a drop-history-at-inference-only experiment. "Full" retains the last `MAX_HISTORY=128` plies, which covers supplied synthetic generators but should be reported accurately.
- LLM results come from a **fixed held-out diagnostic subset**. Direct LLM/neural comparisons must use identical test pairs; `matched_sample_comparison.csv` explicitly performs that restriction.
- Multi-seed variance is not a substitute for independent dataset families. Bootstrap intervals describe sampled test pairs under the generating mechanism.
- A successful run does **not** imply the unspecified manuscript's every experiment has been reproduced: the paper itself was not attached.

## Commands

```bash
pip install -r requirements.txt
python scripts/selftest.py
python scripts/run_all_kaggle.py --out results --allow-cpu --smoke
# Kaggle GPU full:
python scripts/run_all_kaggle.py --out /kaggle/working/histchess_results --pairs-per-task 300 --epochs 14 --train-seeds 2 --llm-pairs-per-task 24
```

Output paths: `data/pairs.jsonl`, `data/generation_rejections.json`, `validation.json`, `dataset_split_summary.json`, `baseline_metrics.csv`, `leakage_audit.json`, `gru_metrics.csv`, `gru_predictions.csv`, `gru_learning_curve.csv`, `history_ablation.csv`, `gru_model_config.json`, `history_gru_best.pt`, `llm_sample.json`, `llm_predictions.csv`, `llm_metrics.csv`, `paired_confidence_intervals.csv`, `matched_sample_comparison.csv`, `paired_significance.csv`, `paper_tables.md`, `paper_tables.tex`, `figures/*.png`, `final_output_audit.json`, `run_status.json`, `manifest.json`.

## Reliability upgrade (20 October 2026)

This revision adds focused pytest suites (replayed chess laws, illegal moves,
FEN invariants, pairing certificates, deterministic synthetic generation,
negative/corrupted data tests, train/test overlap, encoding, LLM parsing,
paired bootstrap, corrupted prediction audit, ZIP release validation), a
small real CUDA neural-training integration test, and an offline mocked LLM
batching/invalid-reply integration test. A `scripts/stress_test.py` command
validates generation across multiple random seeds. The notebook now creates
`histchess_diagnostics.zip` even if installation, tests, or the training run
fails before normal packaging. Diagnostics retain Kaggle logs and the exact
bootstrap exception. No SVG files are generated; publication figures are PNG.

### Suggested validation sequence

```bash
# First attach the updated ZIP as a Kaggle Input, set accelerator GPU and Internet ON.
# The full one-cell bootstrap automatically runs:
python scripts/selftest.py
python -m pytest -q tests -m "not slow and not gpu_integration"
python scripts/run_all_kaggle.py --smoke --skip-llm

# Optional extra stress/integration runs:
python -m pytest -q tests -m slow
python -m pytest -q tests/test_gpu_integration.py -m gpu_integration
python scripts/stress_test.py --pairs-per-task 60 --seeds 3101 3102 3103

# After a complete run:
python scripts/verify_artifacts.py /kaggle/working/histchess_results.zip
```

**Environment limitations:** The source build/ZIP checks can run without a GPU.
Chess replay, synthetic generation and all live CUDA/LLM tests require
Kaggle dependencies and a GPU. Tests that use mocked model inference validate
only the caller's logic; they do not establish inference correctness for a
real Hugging Face model. No numerical experimental results are pre-populated
in the code archive.
