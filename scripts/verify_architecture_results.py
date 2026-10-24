"""Independent, deterministic audit of an HistChess architecture output directory.

Works with original v3 outputs and the v4 extension. Recomputes metrics directly
from raw predictions; does not trust a successful Kaggle exit or run counter.
Uses no torch/chess dependency. Never fabricates missing metrics.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path


def read_csv(path):
    with open(path,newline="",encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def canonical_cutoff(value):
    return str(value)


def audit(outdir):
    path=Path(outdir)
    problems=[]
    def check(test,reason):
        if not test:problems.append(reason)
    required=("architecture_status.json","architecture_config.json",
              "architecture_metrics.csv","architecture_predictions.csv",
              "data/pairs.jsonl")
    missing=[x for x in required if not (path/x).is_file()]
    if missing:
        return {"passed":False,"errors":[f"Missing required file: {name}" for name in missing]}
    status=read_json(path/"architecture_status.json")
    config=read_json(path/"architecture_config.json")
    metrics=read_csv(path/"architecture_metrics.csv")
    predictions=read_csv(path/"architecture_predictions.csv")
    pairs=[json.loads(line) for line in (path/"data/pairs.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    check(bool(status.get("complete")),"Status does not report completion")
    pair_ids=set()
    pairs_by_id={}
    families=collections.defaultdict(set)
    for p in pairs:
        pid=p["pair_id"]
        check(pid not in pair_ids,f"Duplicate dataset pair ID: {pid}")
        pair_ids.add(pid)
        pairs_by_id[pid]=p
        families[p["task"]].add(p["generator_family"])
        check(p["positive"]["label"]==1 and p["negative"]["label"]==0,
              f"Incorrect labels for {pid}")
        check(p["positive"]["board_observation"]==p["negative"]["board_observation"]==p["board_key"],
              f"Mismatched pair observation: {pid}")
    models=list(config["models"])
    folds=[int(i) for i in config["folds"]]
    seeds=[int(i) for i in config["seeds"]]
    histories=[canonical_cutoff(c) for c in config["cutoffs"]]
    expected={(a,f,s,h) for a in models for f in folds for s in seeds for h in histories}
    observed_runs=[(x["model"],int(x["fold"]),int(x["seed"]),str(x["history"])) for x in status.get("runs",[])]
    check(set(observed_runs)==expected and len(observed_runs)==len(expected),
          f"Run manifest does not match {len(expected)} expected configurations")
    pred_groups=collections.defaultdict(dict)
    for p in predictions:
        key=(p["architecture"],int(p["fold"]),int(p["seed"]),p["history_plies"])
        pairkey=(p["pair_id"],p["variant"])
        if pairkey in pred_groups[key]:problems.append(f"Duplicate prediction {key} {pairkey}")
        pred_groups[key][pairkey]=p
        check(p["pair_id"] in pair_ids,f"Prediction references unknown pair: {p['pair_id']}")
        score=float(p["score"])
        check(math.isfinite(score) and 0<=score<=1,f"Invalid score at {key} {pairkey}")
        check(int(p["prediction"])==int(score>=0.5),f"Prediction threshold mismatch at {key} {pairkey}")
    check(set(pred_groups)==expected,"Prediction configurations disagree with run manifest")
    reported={}
    for m in metrics:
        key=(m["architecture"],int(m["fold"]),int(m["seed"]),m["history_plies"],m["task"])
        if key in reported:problems.append(f"Duplicate metric group: {key}")
        reported[key]=m
    check(len(reported)==5*len(expected),"Expected 5 per-task/overall metric rows per run")
    # A dataset may be v3 (three families per task) or a future expanded
    # version. Reconstruct the original three-family fold rotation from the
    # generator family order, which is included in generated pair order.
    ordered=collections.defaultdict(lambda:[None,None,None])
    stage={"train":0,"validation":1,"test":2}
    for p in pairs:
        if p.get("split") not in stage:
            problems.append(f"Cannot reconstruct generator-family order for {p['pair_id']}")
            continue
        ix=stage[p["split"]]
        old=ordered[p["task"]][ix]
        if old not in (None,p["generator_family"]):
            problems.append(f"Inconsistent baseline split/family for {p['task']}")
        ordered[p["task"]][ix]=p["generator_family"]
    for task,fams in ordered.items():
        check(len(set(fams))==3 and None not in fams,
              f"Expected three source generator families for {task}; got {fams}")
    for key in expected:
        model,fold,seed,history=key
        data=pred_groups.get(key,{})
        test_ids=set()
        for p in pairs:
            if len(ordered[p["task"]])!=3:continue
            ix=ordered[p["task"]].index(p["generator_family"])
            if (ix-fold)%3==2:test_ids.add(p["pair_id"])
        expected_records={(pid,variant) for pid in test_ids for variant in ("positive","negative")}
        if set(data)!=expected_records:
            problems.append(f"Unmatched test records in {key}: expected={len(expected_records)}, found={len(data)}")
            continue
        for pid in test_ids:
            pair=pairs_by_id[pid]
            for variant in ("positive","negative"):
                item=data[(pid,variant)]
                check(int(item["label"])==int(pair[variant]["label"]),f"Label mismatch {key} {pid} {variant}")
                check(item["task"]==pair["task"],f"Task mismatch {key} {pid} {variant}")
            if history=="0":
                a=data[(pid,"positive")];b=data[(pid,"negative")]
                check(a["prediction"]==b["prediction"],f"Zero-history pair has different predictions: {key} {pid}")
                check(abs(float(a["score"])-float(b["score"]))<=1e-6,
                      f"Zero-history pair has different scores: {key} {pid}")
        for task in ("all","castling","en_passant","halfmove","repetition"):
            ids=[pid for pid in test_ids if task=="all" or
                 pairs_by_id[pid]["task"]==task]
            if not ids:
                problems.append(f"Empty task {task} in {key}");continue
            correct=0;joint=0;ranking=0.0;brier=0.0
            for pid in ids:
                p=data[(pid,"positive")];n=data[(pid,"negative")]
                pp=float(p["score"]);nn=float(n["score"])
                correct+=int(p["prediction"]=="1")+int(n["prediction"]=="0")
                joint+=int(p["prediction"]=="1" and n["prediction"]=="0")
                ranking+=(pp>nn)+0.5*(pp==nn)
                brier+=(pp-1)**2+nn**2
            actual={"record_accuracy":correct/(2*len(ids)),
                    "joint_pair_accuracy":joint/len(ids),
                    "pair_ranking_accuracy":ranking/len(ids),
                    "brier_score":brier/(2*len(ids))}
            group=reported.get((*key,task))
            if group is None:
                problems.append(f"Missing metric row {key} {task}");continue
            for field,value in actual.items():
                if field in group and group[field] not in ("",None):
                    check(abs(float(group[field])-value)<=1e-6,
                          f"Metric mismatch {field}: {key} {task}")
            check(int(group["test_pairs"])==len(ids),f"Incorrect pair count {key} {task}")
            if history=="0":
                check(actual["record_accuracy"]==0.5 and actual["joint_pair_accuracy"]==0,
                      f"Zero-history mathematical invariant violated {key} {task}")
    # An OOD file is optional, but partial files or mismatched metrics are not.
    ood_metrics_file=path/"architecture_length_ood_metrics.csv"
    ood_pred_file=path/"architecture_length_ood_predictions.csv"
    check(ood_metrics_file.exists()==ood_pred_file.exists(),
          "Length-shift metric and prediction outputs must both exist")
    if ood_metrics_file.exists() and ood_pred_file.exists():
        ood_metric=read_csv(ood_metrics_file)
        ood_pred=read_csv(ood_pred_file)
        by_key=collections.defaultdict(list)
        for r in ood_pred:
            key=(r["architecture"],int(r["fold"]),int(r["seed"]),
                 r["training_history_plies"],r["evaluation_history_plies"])
            by_key[key].append(r)
        check(len(ood_metric)==5*len(by_key),"Length-shift metrics must contain 5 rows per evaluation")
        ood_reported={}
        for r in ood_metric:
            key=(r["architecture"],int(r["fold"]),int(r["seed"]),
                 r["training_history_plies"],r["evaluation_history_plies"],r["task"])
            check(key not in ood_reported,f"Duplicated length-shift metric {key}")
            ood_reported[key]=r
        for key,rows in by_key.items():
            record_map={(r["pair_id"],r["variant"]):r for r in rows}
            check(len(record_map)==len(rows),f"Repeated length-shift prediction: {key}")
            check((key[0],key[1],key[2],key[3]) in expected,
                  f"Length-shift evaluation has no corresponding trained checkpoint: {key}")
            original=pred_groups.get((key[0],key[1],key[2],key[3]),{})
            check(set(record_map)==set(original),
                  f"Length-shift evaluation changed test-set membership: {key}")
            for pkey,r in record_map.items():
                if pkey not in original:continue
                check(r["task"]==original[pkey]["task"] and
                      r["label"]==original[pkey]["label"],
                      f"Length-shift evaluation changed task/label: {key} {pkey}")
                score=float(r["score"])
                check(math.isfinite(score) and 0<=score<=1 and
                      int(r["prediction"])==int(score>=0.5),
                      f"Invalid length-shift score/prediction: {key} {pkey}")
            if not rows:continue
            for task in ("all","castling","en_passant","halfmove","repetition"):
                ids={r["pair_id"] for r in rows if task=="all" or r["task"]==task}
                if not ids:continue
                if any((pid,v) not in record_map for pid in ids for v in ("positive","negative")):
                    problems.append(f"Length-shift pair incomplete: {key} {task}")
                    continue
                correct=joint=0;ranking=brier=0.0
                for pid in ids:
                    pp=record_map[(pid,"positive")]; nn=record_map[(pid,"negative")]
                    pscore=float(pp["score"]); nscore=float(nn["score"])
                    correct+=int(pp["prediction"]=="1")+int(nn["prediction"]=="0")
                    joint+=int(pp["prediction"]=="1" and nn["prediction"]=="0")
                    ranking+=(pscore>nscore)+0.5*(pscore==nscore)
                    brier+=(1-pscore)**2+nscore**2
                actual={"record_accuracy":correct/(2*len(ids)),
                        "joint_pair_accuracy":joint/len(ids),
                        "pair_ranking_accuracy":ranking/len(ids),
                        "brier_score":brier/(2*len(ids))}
                row=ood_reported.get((*key,task))
                if row is None:
                    problems.append(f"Length-shift metric missing: {key} {task}")
                    continue
                for field,value in actual.items():
                    check(field in row and abs(float(row[field])-value)<1e-6,
                          f"Length-shift metric mismatch {field}: {key} {task}")
                check(int(row["test_pairs"])==len(ids),
                      f"Length-shift test pair count mismatch: {key} {task}")
                if key[4]=="0":
                    check(actual["record_accuracy"]==.5 and actual["joint_pair_accuracy"]==0,
                          f"Zero-history length-shift invariant failed {key} {task}")
    return {"passed":not problems,"errors":problems[:200],
            "checked_runs":len(pred_groups),"expected_runs":len(expected),
            "prediction_records":len(predictions),"metric_rows":len(metrics),
            "dataset_pairs":len(pairs),
            "note":"Pairs and folds are correlated; passing integrity checks is not publication-level external validity."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--write",default=None,help="Optional destination JSON audit report")
    args=ap.parse_args()
    result=audit(args.output_dir)
    if args.write:Path(args.write).write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))
    if not result["passed"]:sys.exit(1)


if __name__=="__main__":main()
