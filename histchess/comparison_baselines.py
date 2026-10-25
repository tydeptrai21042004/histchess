"""Auditable classical comparison on EXACT HistChess pair IDs and fold rotations.

This is a non-neural, within-dataset comparison, NOT an external benchmark.
No FEN metadata or privileged rules enter any trainable baseline.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .architecture_benchmark import (
    _check_pair_integrity, _metrics, _write_csv, rotate_splits, rows_from_pairs,
)

BASELINES = ("board_only", "move_bag", "move_ngrams", "privileged_replay_oracle")


def suffix(history, cutoff, max_history):
    xs = list(history)[-max_history:]
    return xs if cutoff is None else xs[-cutoff:] if cutoff else []


def features(row, baseline, cutoff, max_history):
    if baseline not in BASELINES[:3]:
        raise ValueError(f"Not a trainable observation baseline: {baseline}")
    v = {f"task={row['task']}": 1.0}
    if baseline == "board_only":
        # Baseline is intentionally uninformative WITHIN matched pairs.
        # The neural board-only control can still be trained separately.
        v[f"board={row['board_key']}"] = 1.0
        return v
    seq = suffix(row["history_uci"], cutoff, max_history)
    # The same board/task feature permissions as neural, no hidden FEN fields.
    v[f"board={row['board_key']}"] = 1.0
    v["length"] = len(seq) / max_history
    for token, count in collections.Counter(seq).items():
        v[f"move={token}"] = float(count) / max(1, len(seq))
    if baseline == "move_ngrams":
        for n in (2, 3):
            for gram, count in collections.Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1)).items():
                v[f"{n}gram={'/'.join(gram)}"] = float(count) / max(1, len(seq)-n+1)
        for i, token in enumerate(seq[-4:]):
            v[f"last4={len(seq[-4:])-i}:{token}"] = 1.0
    return v


def _predict_probabilities(train, validation, test, baseline, cutoff, max_history):
    if baseline == "board_only":
        # Exactly balanced labels per matched pair: this is the fixed 0.5 null.
        return [0.5] * len(test), {"C": "NA", "validation_logloss": 0.6931471805599453}
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform([features(r, baseline, cutoff, max_history) for r in train])
    Xv = vectorizer.transform([features(r, baseline, cutoff, max_history) for r in validation])
    Xt = vectorizer.transform([features(r, baseline, cutoff, max_history) for r in test])
    y = np.asarray([int(r["label"]) for r in train], dtype=int)
    yv = np.asarray([int(r["label"]) for r in validation], dtype=int)
    if set(y) != {0, 1}: raise RuntimeError("Both training labels required")
    chosen = None
    for C in (0.1, 1.0, 10.0):
        fit = LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=3101)
        fit.fit(X, y)
        pv = fit.predict_proba(Xv)[:, 1]
        loss = float(log_loss(yv, pv, labels=[0,1]))
        # Exact deterministic tie break: lower C wins.
        if chosen is None or loss < chosen[0] - 1e-12:
            chosen = (loss, C, fit)
    return chosen[2].predict_proba(Xt)[:, 1].tolist(), {
        "C": chosen[1], "validation_logloss": chosen[0],
        "features_seen_in_training": X.shape[1],
    }


def _format_cutoff(cutoff):
    return "full" if cutoff is None else str(cutoff)


def run(pairs, outdir, folds=(0, 1, 2), cutoffs=(0, 4, 8, 16, None), max_history=256):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    _check_pair_integrity(pairs)
    config = {"folds":list(folds), "cutoffs":[_format_cutoff(x) for x in cutoffs],
              "max_history":max_history, "baselines":list(BASELINES),
              "description":"Nonneural classical baselines on the SAME HistChess counterfactual test pair IDs",
              "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps({"config":config, "pairs":pairs}, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    if (out / "comparison_fingerprint.txt").exists():
        if (out / "comparison_fingerprint.txt").read_text().strip() != digest:
            raise ValueError("Comparison fingerprint mismatch; choose a new results directory")
    (out / "comparison_fingerprint.txt").write_text(digest + "\n", encoding="utf8")
    (out / "comparison_config.json").write_text(json.dumps(config,indent=2),encoding="utf8")
    predictions = []; metrics = []; tuning=[]
    for fold in folds:
        split = rotate_splits(pairs, fold)
        rows = rows_from_pairs(split)
        sets = {s:[r for r in rows if r["split"]==s] for s in ("train","validation","test")}
        if any(not entries for entries in sets.values()):
            raise ValueError("Missing training, validation, or test split")
        # Test labels are used ONLY for the explicit privileged reference.
        for cutoff in cutoffs:
            for method in BASELINES:
                if method == "privileged_replay_oracle":
                    probs = [float(row["label"]) for row in sets["test"]]
                    chosen = {"C":"NA", "validation_logloss":0.0}
                else:
                    probs, chosen = _predict_probabilities(sets["train"], sets["validation"], sets["test"],
                                                            method, cutoff, max_history)
                guesses = [int(float(x)>=0.5) for x in probs]
                info={"baseline":method, "fold":fold, "history_plies":_format_cutoff(cutoff)}
                for item in _metrics(sets["test"], guesses, probs):
                    metrics.append({**info, **item})
                for row,pred,score in zip(sets["test"],guesses,probs):
                    predictions.append({**info, "pair_id":row["pair_id"], "variant":row["variant"],
                                        "task":row["task"],"family":row["generator_family"],
                                        "label":row["label"], "prediction":pred,"score":score})
                tuning.append({**info, **chosen})
    _write_csv(out / "comparison_predictions.csv",predictions)
    _write_csv(out / "comparison_metrics.csv",metrics)
    _write_csv(out / "comparison_validation_selection.csv",tuning)
    by_group=collections.defaultdict(list)
    for m in metrics:
        if m["task"]=="all":by_group[(m["baseline"],m["history_plies"])].append(m)
    summary=[]
    for (method,context),entries in sorted(by_group.items()):
        summary.append({"baseline":method,"history_plies":context,"folds":len(entries),
            "mean_record_accuracy":sum(x["record_accuracy"] for x in entries)/len(entries),
            "mean_joint_pair_accuracy":sum(x["joint_pair_accuracy"] for x in entries)/len(entries),
            "mean_pair_ranking_accuracy":sum(x["pair_ranking_accuracy"] for x in entries)/len(entries),
            "note":"Correlated rotations of three synthetic families; descriptive only."})
    _write_csv(out / "comparison_summary.csv",summary)
    (out / "comparison_status.json").write_text(json.dumps({"complete":True,"fingerprint":digest,
        "metric_rows":len(metrics),"prediction_rows":len(predictions)},indent=2),encoding="utf8")
    return metrics, predictions


def main(argv=None):
    ap=argparse.ArgumentParser(description="Nonneural HistChess comparison on exactly the same counterfactual pairs")
    ap.add_argument("--pairs-jsonl",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--folds",nargs="+",type=int,default=[0,1,2])
    ap.add_argument("--cutoffs",nargs="+",default=["0","4","8","16","full"])
    ap.add_argument("--max-history",type=int,default=256)
    args=ap.parse_args(argv)
    pairs=[json.loads(line) for line in Path(args.pairs_jsonl).read_text(encoding="utf8").splitlines() if line.strip()]
    from .core import validate
    report=validate(pairs)
    if not report["passed"]:raise RuntimeError("Invalid/replayed chess pairs: " + str(report.get("errors",[])[:3]))
    metrics, pred=run(pairs,args.out,tuple(args.folds),
                      tuple(None if c=="full" else int(c) for c in args.cutoffs),args.max_history)
    print("COMPARISON COMPLETE",len(metrics),"metric rows",len(pred),"prediction rows",flush=True)

if __name__ == "__main__":main()
