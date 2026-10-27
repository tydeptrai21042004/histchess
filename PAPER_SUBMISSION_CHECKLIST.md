# HistChess v6: conservative paper-readiness checklist

## Main contribution (recommended)

*HistChess: A Counterfactual, Observability-Aware Benchmark for History-Dependent Chess State Tracking*

Do not call the six small architectures 'LLMs'; these networks were trained from scratch. Claims about pretrained LLM failure need actual pretrained-LM evaluation and are not provided by the 450-run benchmark.

## Essential (no additional neural runs)

1. Archive the original fully completed v6 result ZIP and confirm SHA256 of the pair file and prediction file. Never overwrite them.
2. Run the CPU companion script. Report six neural models as the original experiment and the new diagnostic baselines as a separate, post-hoc pre-specified CPU evaluation.
3. Obtain the original results archive. Without it, numerical metrics cannot be inspected and no model-performance claim should be written.
4. Report separate results for castling, en passant, halfmove, and repetition. En passant (last move) and halfmove >= 2 (recent reset events) are not long-distance memory tasks.
5. Report record accuracy, strict joint-pair accuracy, pair ranking, and observability ceiling per cutoff. Never use indistinguishability-induced low scores as evidence of model failure.
6. Compare board-only, move bag, move n-grams, last 1/2/4 plies, first 4 visible plies, and full-history deterministic legal replay. Identify which are trainable learned baselines and which are algorithmic references.
7. Measure generator family differences and seed variability. The three families/task are NOT sufficient for independent-family population confidence intervals. Bootstrap over repeated pairs does not solve this.
8. Release exact JSONL pairs, generator code, verifier, schemas, task definitions, experiment provenance, license and data card. Include the historical distinction between `board_fen` and full FEN hidden rights.
9. Clearly document `is_repetition(3)` as *current-position threefold occurrence* (not prospective claimability) and castling as *White kingside right*, not generic castling legality.
10. Review the 2026 NeurIPS E&D dataset hosting and Croissant RAI metadata requirements, if targeting that specific venue.

## Paper sections

1. Introduction: observational aliasing and the distinction between missing information and learning failure.
2. Related work: Toshniwal et al. AAAI 2022; Harang et al. 2025; Chess-World-Model 2026; MET-Bench ICML 2026. Do not claim the first chess state-tracking benchmark.
3. Dataset and hidden-state labels: rule definitions, six-field FEN isolation, legality certificates, generation mechanisms.
4. Counterfactual protocol: pairwise same observation, opposite target label; family-held-out rotation; truncated-history visibility.
5. Information analysis: exact empirical observable-signature ceiling and per-pair distinguishability bound, including their limitations.
6. Experimental protocol: six architectures and hyperparameters, early stopping, five seeds, three correlated folds, cutoffs; board-only/move bag/shortcuts/replay controls.
7. Main findings: figures, full per-task results, paired matched comparisons, performance to observability ceiling.
8. Failures and limitations: synthetic generator, three families, 1,200 pairs, short histories, no pretrained LLM, model parameter budget, no independent long-distance experiment.
9. Ethics and responsible use: no personal data; dual-use risks minimal; generalization claims limited to synthetic rules.
10. Data/software availability: immutable release, code and metadata.

## Suggested main manuscript tables

- Dataset composition: counts by task/family; history min/median/max; same move-bag/last-move statistics.
- Full-history model comparison: 6 original models + 3 classical controls + CPU local-window baselines; keep legal replay in a **separate reference** row.
- Context cutoffs: 0/4/8/16/full, full observed or exact ceiling by task; don't infer long-range memory without genuinely long trajectories.
- Family-held-out generalization: separate 3 folds instead of collapsing everything to a single mean.
- Complexity/integrity: parameter counts, training protocol, 70 test passes, 450 audited runs, verification counts and source hashes.

## Strong claim / weak claim boundary

Can claim (if supported by actual saved metrics): relative model and shortcut performance on the existing controlled synthetic benchmark; formally verified board/label aliasing; reproducibility of dataset generation and experiments.

Cannot claim from the current v6 experiment alone: pretrained LLM weakness; Transformers cannot track state generally; real-game generalization; full S4D/Mamba comparison; model scores on external chess benchmarks; near-human or game-playing skill.
