"""Paired uncertainty quantification and automatically generated paper artefacts."""
from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

from .core import TASKS
from .experiments import write_csv, save_json


def read_csv(path):
    path=Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def bootstrap_pair_ci(pairs, predictions, seed=3101, repeats=2000):
    """Resample whole pairs, not individual highly correlated examples."""
    if not pairs:
        return {"record_accuracy":None,"record_ci_low":None,"record_ci_high":None,
                "joint_pair_accuracy":None,"joint_ci_low":None,"joint_ci_high":None}
    values=[]
    for p in pairs:
        a=predictions.get((p["pair_id"],"positive"))
        b=predictions.get((p["pair_id"],"negative"))
        record=(int(a is True)+int(b is False))/2
        joint=int(a is True and b is False)
        values.append((record,joint))
    n=len(values)
    rng=random.Random(seed)
    acc_samples=[]; pair_samples=[]
    for _ in range(repeats):
        draw=[values[rng.randrange(n)] for __ in range(n)]
        acc_samples.append(sum(x[0] for x in draw)/n)
        pair_samples.append(sum(x[1] for x in draw)/n)
    def quantile(v,q):
        v.sort()
        return v[min(len(v)-1, max(0,round(q*(len(v)-1))))]
    return {"record_accuracy":round(sum(v[0] for v in values)/n,6),
            "record_ci_low":round(quantile(acc_samples,.025),6),
            "record_ci_high":round(quantile(acc_samples,.975),6),
            "joint_pair_accuracy":round(sum(v[1] for v in values)/n,6),
            "joint_ci_low":round(quantile(pair_samples,.025),6),
            "joint_ci_high":round(quantile(pair_samples,.975),6)}


def prediction_collections(pairs, outdir):
    """All evaluated methods, respecting unequal test sizes and repeated seeds."""
    outdir=Path(outdir)
    grouped=defaultdict(dict)
    # Symbolic methods evaluated on every test record.
    from .core import observed_label,replay
    for p in pairs:
        if p["split"]!="test":continue
        for variant in ("positive","negative"):
            r=p[variant]
            key=(p["pair_id"],variant)
            grouped[("rule_oracle", "all")][key]=observed_label(replay(r["history_uci"]),p["task"])
            grouped[("board_only_majority", "all")][key]=False
            parts=r["fen"].split()
            symbolic=("K" in parts[2] if p["task"]=="castling" else
                      parts[3]!="-" if p["task"]=="en_passant" else
                      int(parts[4])>=2 if p["task"]=="halfmove" else False)
            grouped[("full_FEN_rule_fields", "all")][key]=bool(symbolic)
    for r in read_csv(outdir/"gru_predictions.csv"):
        grouped[(f"GRU_history_{r['history_moves']}",r["seed"])][(r["pair_id"],r["variant"])]=int(r["prediction"])==1
    for r in read_csv(outdir/"llm_predictions.csv"):
        grouped[(r["model"],"llm")][(r["pair_id"],r["variant"])]=(None if not r["valid"].lower()=="true" else int(r["prediction"])==1)
    return grouped


def paper_statistics(pairs, outdir, *, bootstrap=2000, seed=3101):
    outdir=Path(outdir)
    grouped=prediction_collections(pairs,outdir)
    results=[]
    for (model,trial), preds in sorted(grouped.items()):
        for task in ("all",)+TASKS:
            eligible=[p for p in pairs if p["split"]=="test" and
                      (task=="all" or p["task"]==task) and
                      (p["pair_id"],"positive") in preds and
                      (p["pair_id"],"negative") in preds]
            ci=bootstrap_pair_ci(eligible,preds,seed=seed+TASKS.index(task) if task!="all" else seed,
                                 repeats=bootstrap)
            results.append({"model":model,"trial":trial,"task":task,
                            "n_pairs":len(eligible),"n_records":2*len(eligible),**ci})
    write_csv(outdir/"paired_confidence_intervals.csv",results)
    # Fair side-by-side comparison: use exactly the SAME held-out sample IDs
    # for the GRU, board-only control, privileged FEN reference, and each LLM.
    llm_sample_file=outdir/"llm_sample.json"
    matched=[]
    if llm_sample_file.exists():
        import json
        sample=set(json.loads(llm_sample_file.read_text())["pair_ids"])
        for (model,trial), preds in sorted(grouped.items()):
            if model.startswith("GRU_history_") and model not in ("GRU_history_0","GRU_history_full"):
                continue
            for task in ("all",)+TASKS:
                subset=[p for p in pairs if p["pair_id"] in sample and p["split"]=="test" and
                        (task=="all" or p["task"]==task)]
                if not subset or any((p["pair_id"],v) not in preds for p in subset for v in ("positive","negative")):
                    raise ValueError(f"Incomplete matched comparison {model} {trial} {task}")
                ci=bootstrap_pair_ci(subset,preds,seed=seed, repeats=bootstrap)
                matched.append({"model":model,"trial":trial,"task":task,
                                "n_pairs":len(subset),"n_records":len(subset)*2,**ci})
    write_csv(outdir/"matched_sample_comparison.csv",matched)
    # Paired sign-flip randomization on per-pair CORRECTNESS against a matched
    # board-only control. This p-value is descriptive, not multiplicity-adjusted.
    significance=[]
    for (model,trial), preds in grouped.items():
        if not model.startswith("GRU_history_") or model.endswith("_0"):
            continue
        for task in ("all",)+TASKS:
            subset=[p for p in pairs if p["split"]=="test" and (task=="all" or p["task"]==task)]
            delta=[]
            for p in subset:
                a=preds.get((p["pair_id"],"positive"))
                b=preds.get((p["pair_id"],"negative"))
                if a is None or b is None:
                    continue
                # board-only always predicts negative: 50% on matched balanced pairs
                delta.append((int(a is True)+int(b is False))/2 - 0.5)
            if not delta:continue
            obs=abs(sum(delta))
            rng=random.Random(f"{seed}|{model}|{trial}|{task}")
            draws=4096
            extreme=0
            for _ in range(draws):
                test=sum(x if rng.randrange(2) else -x for x in delta)
                extreme += abs(test)+1e-12 >= obs
            significance.append({"model":model,"trial":trial,"task":task,
                                 "n_pairs":len(delta),"accuracy_gain_vs_board_majority":round(sum(delta)/len(delta),6),
                                 "paired_sign_flip_p":round((extreme+1)/(draws+1),7),
                                 "permutation_draws":draws,
                                 "multiplicity_adjusted":False})
    write_csv(outdir/"paired_significance.csv",significance)
    return results


def make_figures(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir=Path(outdir)
    figs=outdir/"figures";figs.mkdir(exist_ok=True)
    records=read_csv(outdir/"paired_confidence_intervals.csv")
    seq=[r for r in records if r["model"].startswith("GRU_history_") and r["task"]=="all"]
    groups=defaultdict(list)
    for r in seq:
        length=r["model"].split("_")[-1]
        groups[length].append(float(r["record_accuracy"]))
    if groups:
        labels=sorted((x for x in groups if x!="full"),key=lambda x:int(x))
        if "full" in groups:labels.append("full")
        mean=[sum(groups[x])/len(groups[x]) for x in labels]
        spread=[(sum((v-m)**2 for v in groups[x])/len(groups[x]))**.5
                for x,m in zip(labels,mean)]
        plt.figure(figsize=(7.2,4.2))
        plt.errorbar(labels,mean,yerr=spread,marker="o",capsize=3)
        plt.ylim(0,1);plt.ylabel("Test record accuracy")
        plt.xlabel("Visible history (plies; full capped at configuration maximum)")
        plt.title("Retrained history-length ablation (mean ± SD across seeds)")
        plt.tight_layout();plt.savefig(figs/"history_length_ablation.png",dpi=180);plt.close()
    curve=read_csv(outdir/"gru_learning_curve.csv")
    full=[r for r in curve if r["history_moves"]=="full"]
    if full:
        plt.figure(figsize=(7.2,4.2))
        for trial in sorted({r["seed"] for r in full}):
            ss=[r for r in full if r["seed"]==trial]
            plt.plot([int(r["epoch"]) for r in ss],
                     [float(r["validation_loss"]) for r in ss],label=f"seed {trial}")
        plt.xlabel("Epoch");plt.ylabel("Validation BCE")
        plt.title("Full-history training (validation only)")
        plt.legend();plt.tight_layout();plt.savefig(figs/"validation_loss.png",dpi=180);plt.close()


def paper_tables(outdir, *, sample_note=""):
    outdir=Path(outdir)
    ci=read_csv(outdir/"paired_confidence_intervals.csv")
    rows=[r for r in ci if r["task"]=="all" and
          (r["model"].startswith("GRU_history_") or r["model"] in
           ("rule_oracle","board_only_majority","full_FEN_rule_fields"))]
    head="| Model | Seed/trial | Test pairs | Record accuracy [95% paired bootstrap CI] | Joint pair accuracy |"
    lines=["# HistChess automatically generated experimental tables", "",
           "All entries here derive from the CSV prediction files, not manually entered numbers.",
           "Bootstrap samples pairs as clusters; CIs describe dataset sampling variability, not template-family generalization.",
           "Full FEN is a privileged-input upper reference for three tasks, not a history-only baseline.",
           "The data are synthetic; no real-game generalization is asserted.","", head,
           "|---|---:|---:|---:|---:|"]
    for r in rows:
        label=r["model"].replace("_","\\_")
        lines.append(f"| {label} | {r['trial']} | {r['n_pairs']} | "
                     f"{float(r['record_accuracy']):.3f} [{float(r['record_ci_low']):.3f}, {float(r['record_ci_high']):.3f}] | "
                     f"{float(r['joint_pair_accuracy']):.3f} |")
    if sample_note:lines.extend(["",sample_note])
    matched=read_csv(outdir/"matched_sample_comparison.csv")
    if matched:
        lines.extend(["", "## Fair fixed-subset comparison (identical test pair IDs)", "",
                      "For these rows, the GRU, LLMs, oracle, and board-only comparator are evaluated on the **same test pairs**.",
                      "Full-FEN retains privileged label fields and must not be described as a history-only model.", "",
                      "| Model | Trial | Matched pairs | Accuracy [95% paired CI] | Joint accuracy |",
                      "|---|---|---:|---:|---:|"])
        for r in matched:
            if r["task"]!="all":continue
            lines.append(f"| {r['model']} | {r['trial']} | {r['n_pairs']} | "
                         f"{float(r['record_accuracy']):.3f} [{float(r['record_ci_low']):.3f}, {float(r['record_ci_high']):.3f}] | "
                         f"{float(r['joint_pair_accuracy']):.3f} |")
    if (outdir/"llm_metrics.csv").exists():
        lines.extend(["", "## Pretrained LLM fixed-subset evaluation", "",
                      "LLM rows below use the **fixed test subset**, not the full neural test set.", "",
                      "| Model | Task | Tested pairs | Accuracy | Joint accuracy | Invalid replies |",
                      "|---|---|---:|---:|---:|---:|"])
        for r in read_csv(outdir/"llm_metrics.csv"):
            lines.append(f"| {r['model']} | {r['task']} | {r['test_pairs']} | "
                         f"{float(r['accuracy_including_invalid']):.3f} | "
                         f"{float(r['paired_joint_accuracy']):.3f} | {r['invalid_or_missing']} |")
    (outdir/"paper_tables.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    # A small compilable input-only tabular fragment; no fabricated numbers.
    latex=["% Auto-generated; include with \\input{paper_tables.tex}",
           "\\begin{tabular}{llrrr}","\\hline",
           "Model & Trial & Pairs & Acc. & Joint acc. \\\\ ","\\hline"]
    for r in rows:
        name=r["model"].replace("_","\\_")
        latex.append(f"{name} & {r['trial']} & {r['n_pairs']} & {float(r['record_accuracy']):.3f} & {float(r['joint_pair_accuracy']):.3f} \\\\ ")
    latex.extend(["\\hline", "\\end{tabular}"])
    (outdir/"paper_tables.tex").write_text("\n".join(latex)+"\n",encoding="utf8")
