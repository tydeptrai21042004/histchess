"""History-only shortcut diagnostics and train-fitted n-gram baselines.

A perfect score on a small synthetic grammar is NOT evidence of generalizing
chess state tracking.  These baselines are deliberately simple and are compared
on exactly the same held-out matched pairs as the learned models.
"""
from __future__ import annotations

import collections
from pathlib import Path

from .core import TASKS, rows_from_pairs
from .experiments import metrics, write_csv, save_json


def original_shortcut(task, moves):
    """Four leakage-like signatures discovered in the original 2026 release.

    They are fixed before seeing the new dataset. They are not optimized
    against the new held-out test split.
    """
    if task == "castling":
        return "h1g1" not in moves
    if task == "en_passant":
        return bool(moves and moves[-1] == "d7d5")
    if task == "halfmove":
        return bool(moves and moves[-1] == "f6g8")
    if task == "repetition":
        return len(moves) >= 4 and moves[-4] == "g1f3"
    raise ValueError(task)


def ngram_feature(row, n, mode):
    seq = row["history_uci"]
    if mode == "suffix":
        return tuple(seq[-n:])
    if mode == "prefix":
        return tuple(seq[:n])
    raise ValueError(mode)


def _learn(train, n, mode):
    buckets = collections.defaultdict(lambda: [0,0])
    for row in train:
        buckets[(row["task"],ngram_feature(row,n,mode))][int(row["label"])] += 1
    return {key: (int(v[1] > v[0]), sum(v)) for key, v in buckets.items()}


def shortcut_experiments(pairs, outdir):
    """Save all test predictions and non-tuned shortcut diagnostic results.

    N-gram vote is trained solely on TRAIN; ties / unseen inputs output 0.
    Coverage is reported because majority fallback is not learned evidence.
    """
    rows = rows_from_pairs(pairs)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise ValueError("Shortcut evaluation requires train and test samples")
    predictions = []
    summary = []
    baselines = [("legacy_fixed_signatures",None,None)]
    baselines += [(f"{mode}_{n}_memorization",n,mode)
                  for mode in ("suffix","prefix") for n in (1,2,4)]
    for name,n,mode in baselines:
        fitted = None if mode is None else _learn(train,n,mode)
        pred = {}
        known = collections.Counter()
        total = collections.Counter()
        for r in test:
            key = (r["pair_id"],r["variant"])
            if fitted is None:
                guess = int(original_shortcut(r["task"],r["history_uci"]))
                covered = True
            else:
                lookup = fitted.get((r["task"],ngram_feature(r,n,mode)))
                guess = lookup[0] if lookup else 0
                covered = lookup is not None
            pred[key] = bool(guess)
            total[r["task"]] += 1
            total["all"] += 1
            known[r["task"]] += int(covered)
            known["all"] += int(covered)
            predictions.append({"model":name,"task":r["task"],"pair_id":r["pair_id"],
                                "variant":r["variant"],"label":r["label"],
                                "prediction":guess,"seen_train_signature":int(covered)})
        for row in metrics(pred,pairs,name):
            task = row["task"]
            row["train_signature_coverage"] = round(known[task]/total[task],6)
            row["comparison"] = "same_test_pairs"
            summary.append(row)
    outdir = Path(outdir)
    write_csv(outdir/"shortcut_metrics.csv",summary)
    write_csv(outdir/"shortcut_predictions.csv",predictions)
    legacy = next(r for r in summary if r["model"] == "legacy_fixed_signatures" and r["task"] == "all")
    save_json(outdir/"shortcut_audit.json",{
        "original_fixed_signature_accuracy":legacy["accuracy_including_invalid"],
        "original_fixed_signature_joint_accuracy":legacy["paired_joint_accuracy"],
        "old_signatures_have_perfect_test_accuracy":legacy["accuracy_including_invalid"] == 1.,
        "interpretation":"History-only heuristic probes; high scores indicate template shortcuts. "
                         "Family holdout tests only withheld built-in mechanisms, not real-game OOD.",
        "n_train_records":len(train),"n_test_records":len(test),
        "baseline_count":len(baselines)})
    return summary
