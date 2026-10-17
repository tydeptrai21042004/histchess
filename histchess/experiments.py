"""Auditable, CPU-friendly baseline experiments for HistChess."""
from __future__ import annotations

import collections
import csv
import gc
import json
import math
import os
import random
import re
from pathlib import Path

from .core import TASKS, rows_from_pairs, write_jsonl


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf8")


def metrics(predictions, pairs, model):
    """Compute record accuracy and strict joint-pair accuracy for one method."""
    out = []
    for task in ("all",) + TASKS:
        subset = [p for p in pairs if p["split"] == "test" and (task == "all" or p["task"] == task)]
        if not subset:
            continue
        pred_pairs = [(predictions.get((p["pair_id"], "positive")), predictions.get((p["pair_id"], "negative")), p) for p in subset]
        valid = sum(v is not None for pp in pred_pairs for v in pp[:2])
        right = sum(v == (variant == "positive") for a,b,_ in pred_pairs
                    for v,variant in [(a,"positive"),(b,"negative")] if v is not None)
        joint = sum(a is True and b is False for a,b,_ in pred_pairs)
        out.append({
            "model": model, "task": task, "test_pairs": len(subset),
            "attempted_records": 2 * len(subset), "valid_predictions": valid,
            "accuracy_including_invalid": round(right / (2*len(subset)), 6),
            "paired_joint_accuracy": round(joint/len(subset), 6),
            "invalid_or_missing": 2*len(subset)-valid,
        })
    return out


def baseline_experiments(pairs, outdir):
    outdir = Path(outdir)
    preds = {"full_history_rule_oracle": {}, "board_only_majority": {}, "full_FEN_rule_fields": {}}
    for p in pairs:
        if p["split"] != "test":
            continue
        for variant in ("positive", "negative"):
            r = p[variant]
            key = (p["pair_id"], variant)
            preds["full_history_rule_oracle"][key] = bool(r["label"])
            # Board-only Bayes baseline on a balanced matched-pair set: 0/1 tie.
            preds["board_only_majority"][key] = False
            fields = r["fen"].split()
            if p["task"] == "castling":
                val = "K" in fields[2]
            elif p["task"] == "en_passant":
                # python-chess canonical FEN emits en-passant only when legal.
                val = fields[3] != "-"
            elif p["task"] == "halfmove":
                val = int(fields[4]) >= 2
            else:
                # Repetition occurrence counts cannot be recovered from one FEN.
                val = False
            preds["full_FEN_rule_fields"][key] = bool(val)
    rows = [m for name,pred in preds.items() for m in metrics(pred, pairs, name)]
    write_csv(outdir / "baseline_metrics.csv", rows)
    return rows


def leakage_audit(pairs):
    groups = collections.defaultdict(set)
    hseq = collections.defaultdict(set)
    fen = collections.defaultdict(set)
    prefix = collections.defaultdict(set)
    for p in pairs:
        split = p["split"]
        groups[p["board_key"]].add(split)
        for variant in ("positive", "negative"):
            item = p[variant]
            hseq[" ".join(item["history_uci"])].add(split)
            fen[item["fen"]].add(split)
            prefix[" ".join(item["history_uci"][:4])].add(split)
    return {
        "board_group_leakage_count": sum(len(x)>1 for x in groups.values()),
        "full_history_leakage_count": sum(len(x)>1 for x in hseq.values()),
        "full_FEN_across_split_count": sum(len(x)>1 for x in fen.values()),
        "four_ply_prefix_across_split_count": sum(len(x)>1 for x in prefix.values()),
        "unique_board_groups": len(groups),
        "unique_full_histories": len(hseq),
        "notes": [
            "Splits are group-disjoint by observable board and hence canonical full FEN.",
            "The prefix overlap statistic warns about shared synthetic templates; it is not proof of memorization.",
            "All pairs are generated from a few rule mechanisms, so this is not a game-independent split.",
        ],
    }


def _torch_import():
    import torch
    from torch import nn
    return torch, nn


def sequence_experiment(pairs, outdir, epochs=3, batch_size=32, seed=3101):
    torch, nn = _torch_import()
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    rows = rows_from_pairs(pairs)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise RuntimeError("Missing train/test examples")
    PAD, UNK = "<PAD>", "<UNK>"
    vocab = {PAD:0, UNK:1}
    for task in TASKS:
        vocab["TASK_"+task] = len(vocab)
    def board_tokens(rec):
        fen_board, side = rec["board_key"].split()
        expanded = "".join("-"*int(ch) if ch.isdigit() else ch for ch in fen_board if ch != "/")
        if len(expanded) != 64:
            raise ValueError("Bad 64-square board encoding")
        return [f"SQ{i}_{piece}" for i, piece in enumerate(expanded)] + ["TURN_"+side]
    for r in train:
        for tok in board_tokens(r) + r["history_uci"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    max_len = 128

    def convert(batch, keep=None):
        xs, ys, lengths = [], [], []
        for r in batch:
            moves = r["history_uci"] if keep is None else r["history_uci"][-keep:] if keep else []
            tokens = board_tokens(r) + moves + ["TASK_"+r["task"]]
            seq = [vocab.get(tok, 1) for tok in tokens][-max_len:]
            lengths.append(len(seq))
            xs.append(seq + [0]*(max_len-len(seq)))
            ys.append(r["label"])
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.float32), torch.tensor(lengths, dtype=torch.long)

    class TinyGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), 48, padding_idx=0)
            self.rnn = nn.GRU(48, 64, batch_first=True)
            self.head = nn.Linear(64, 1)
        def forward(self, x, lengths):
            emb = self.emb(x)
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(packed)
            return self.head(hidden[-1]).squeeze(1)

    model = TinyGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    trace = []
    for ep in range(epochs):
        model.train()
        indices = rng.permutation(len(train))
        loss_sum = 0.0
        for start in range(0, len(train), batch_size):
            batch = [train[int(i)] for i in indices[start:start+batch_size]]
            x,y,lens = convert(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x,lens), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())*len(batch)
        model.eval()
        valid_acc = None
        if val:
            with torch.inference_mode():
                vx,vy,vl = convert(val)
                score = torch.sigmoid(model(vx,vl))
                valid_acc = float(((score >= 0.5).float() == vy).float().mean().item())
        trace.append({"epoch":ep+1,"train_loss":round(loss_sum/len(train),6), "validation_accuracy":None if valid_acc is None else round(valid_acc,6)})

    def evaluate(subset, keep):
        pred = {}
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(subset), batch_size):
                batch=subset[i:i+batch_size]
                x,_,lens=convert(batch, keep=keep)
                pp=(torch.sigmoid(model(x,lens)) >= 0.5).tolist()
                for rec,value in zip(batch,pp):
                    pred[(rec["pair_id"],rec["variant"])]=bool(value)
        return pred

    full_prediction=evaluate(test,None)
    full_metrics=metrics(full_prediction,pairs,"small_sequence_GRU_full_history")
    ablation = []
    test_pairs = [p for p in pairs if p["split"]=="test"]
    for keep in (0,1,2,4,8,16,32,None):
        prediction=evaluate(test,keep)
        metric=metrics(prediction,pairs,"gru_history_ablation")
        for row in metric:
            task=row["task"]
            subset=[p for p in test_pairs if task=="all" or p["task"]==task]
            can_distinguish=sum((p["positive"]["history_uci"][-keep:] if keep else []) !=
                                (p["negative"]["history_uci"][-keep:] if keep else []) for p in subset) if keep is not None else len(subset)
            ablation.append({"history_moves": "full" if keep is None else keep, "task":task,
                             "test_pairs":row["test_pairs"],
                             "gru_accuracy":row["accuracy_including_invalid"],
                             "gru_joint_pair_accuracy":row["paired_joint_accuracy"],
                             "pair_suffix_distinguishability_fraction":round(can_distinguish/len(subset),6),
                             "explanation":"Upper bound for pair discrimination using board+suffix; not a trained-model score"})
    write_csv(Path(outdir)/"gru_learning_curve.csv",trace)
    write_csv(Path(outdir)/"history_ablation.csv",ablation)
    save_json(Path(outdir)/"gru_model_config.json", {"epochs":epochs,"seed":seed,"vocab_size":len(vocab),"parameters":sum(x.numel() for x in model.parameters()),"max_history_tokens":max_len-1,"train_records":len(train),"validation_records":len(val),"test_records":len(test)})
    torch.save({"state_dict":model.state_dict(),"vocab":vocab,"seed":seed},Path(outdir)/"tiny_gru_model.pt")
    return full_metrics


def parse_yes_no(output):
    # A model that generates ambiguous text is counted as invalid, NOT corrected.
    s = output.strip().lower()
    m = re.match(r"^(?:answer\s*:\s*)?(yes|no)\b",s)
    return None if not m else (m.group(1)=="yes")


def llm_experiment(pairs, outdir, models, pairs_per_task=2, seed=3101):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    # One shared balanced fixed sample per model. Includes complete paired examples.
    selected=[]
    for task in TASKS:
        available=sorted([p for p in pairs if p["split"]=="test" and p["task"]==task],key=lambda p:p["pair_id"])
        if not available:
            raise RuntimeError(f"No test pairs for task {task}")
        rng=random.Random(seed+TASKS.index(task))
        selected.extend(rng.sample(available,min(pairs_per_task,len(available))))
    save_json(Path(outdir)/"llm_sample.json", {"pair_ids":[p["pair_id"] for p in selected],"pairs_per_task":pairs_per_task, "seed":seed})
    qs={
        "castling":"Does White still retain kingside castling rights (regardless of whether the move can be played immediately)?",
        "en_passant":"Is there a LEGAL en-passant capture for the side to move right now?",
        "halfmove":"Is the halfmove clock at least 2?",
        "repetition":"Has this exact rule-equivalent position already occurred at least three times INCLUDING the current position? Do not count a possible future repetition.",
    }
    final=[]
    records=[]
    for mid in models:
        print("Loading LLM (one at a time):",mid,flush=True)
        tokenizer=AutoTokenizer.from_pretrained(mid)
        lm=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32).to("cpu")
        lm.eval()
        model_preds={}
        for p in selected:
            for variant in ("positive","negative"):
                item=p[variant]
                query=("You are evaluating chess rules. All moves start from the standard initial chess position. "
                       "Read the WHOLE history; no board-only guesses. Reply with only YES or NO.\n"
                       f"Question: {qs[p['task']]}\n"
                       f"Final visible board and side: {item['board_observation']}\n"
                       f"Chronological moves (UCI): {' '.join(item['history_uci'])}\n"
                       "Answer:")
                messages=[{"role":"user","content":query}]
                if tokenizer.chat_template:
                    inp=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt",tokenize=True)
                else:
                    inp=tokenizer(query,return_tensors="pt")["input_ids"]
                with torch.inference_mode():
                    generated=lm.generate(inp,max_new_tokens=8,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                response=tokenizer.decode(generated[0][inp.shape[-1]:],skip_special_tokens=True)
                pred=parse_yes_no(response)
                model_preds[(p["pair_id"],variant)]=pred
                records.append({"model":mid,"pair_id":p["pair_id"],"task":p["task"],"variant":variant,
                                "expected":item["label"],"prediction":None if pred is None else int(pred),
                                "valid":pred is not None,"raw_response":response.strip()})
        for row in metrics(model_preds,[dict(p,split="test") for p in selected],mid):
            row["evaluation_scope"]="small_diagnostic_sample"
            final.append(row)
        del lm,tokenizer
        gc.collect()
    write_csv(Path(outdir)/"llm_predictions.csv",records)
    write_csv(Path(outdir)/"llm_metrics.csv",final)
    return final
