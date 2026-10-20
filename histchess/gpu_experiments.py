"""GPU-accelerated history/board controls, with matched-pair predictions.

No FEN clock, rights, or en-passant fields are passed to these neural models.
Each history ablation is independently trained using the SAME cutoff at
training, model selection, and test time, avoiding a truncation-only shift.
"""
from __future__ import annotations

import gc
import random
from pathlib import Path

from .core import TASKS, rows_from_pairs
from .experiments import metrics, save_json, write_csv

PIECES = "-PNBRQKpnbrqk"
SQUARES = "abcdefgh"


def move_coords(uci):
    """Encode source/target with fixed square vocabulary (no test-derived vocab)."""
    if len(uci) not in (4, 5) or uci[0] not in SQUARES or uci[2] not in SQUARES:
        raise ValueError(f"Invalid UCI move token {uci!r}")
    if uci[1] not in "12345678" or uci[3] not in "12345678":
        raise ValueError(f"Invalid UCI move rank {uci!r}")
    if len(uci) == 5 and uci[4] not in "qrbn":
        raise ValueError(f"Invalid promotion code {uci!r}")
    a = SQUARES.index(uci[0]) + 8 * (int(uci[1]) - 1)
    b = SQUARES.index(uci[2]) + 8 * (int(uci[3]) - 1)
    return a + 1, b + 1


def board_features(key):
    board, turn = key.split()
    squares = []
    for char in board.replace("/", ""):
        if char.isdigit():
            squares.extend([0] * int(char))
        else:
            squares.append(PIECES.index(char))
    if len(squares) != 64 or turn not in ("w", "b"):
        raise ValueError(f"Invalid board observation: {key}")
    return squares, 0 if turn == "w" else 1


def history_features(history, keep, max_history=128):
    visible = history if keep is None else (history[-keep:] if keep else [])
    visible = visible[-max_history:]
    moves = [move_coords(m) for m in visible]
    return moves or [(0, 0)], len(moves)


def make_data(rows, keep, device, max_history=128):
    import torch
    boards, sides, tasks, moves, lengths, targets = [], [], [], [], [], []
    for r in rows:
        sq, turn = board_features(r["board_key"])
        m, n = history_features(r["history_uci"], keep, max_history)
        boards.append(sq); sides.append(turn); tasks.append(TASKS.index(r["task"]))
        moves.append(m); lengths.append(n); targets.append(float(r["label"]))
    mx = max(map(len, moves))
    for m in moves:
        m.extend([(0,0)] * (mx - len(m)))
    return (
        torch.tensor(boards, dtype=torch.long, device=device),
        torch.tensor(sides, dtype=torch.long, device=device),
        torch.tensor(tasks, dtype=torch.long, device=device),
        torch.tensor(moves, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def history_experiment(pairs, outdir, *, seeds=(3101, 3102),
                       cutoffs=(0, 4, 8, 16, 32, None), epochs=12,
                       batch_size=128, patience=4, max_history=128,
                       require_gpu=True):
    import numpy as np
    import torch
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_gpu and device.type != "cuda":
        raise RuntimeError("CUDA GPU unavailable; select Kaggle accelerator GPU (T4 or P100).")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be >=1")
    if max_history < 1:
        raise ValueError("max_history must be >=1")

    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    rows = rows_from_pairs(pairs)
    split = {k: [r for r in rows if r["split"] == k]
             for k in ("train", "validation", "test")}
    if any(not v for v in split.values()):
        raise RuntimeError("Train, validation, and test must all be nonempty")
    if any(not any(r["task"] == task for r in split["test"]) for task in TASKS):
        raise RuntimeError("Every task needs test pairs")

    class HistoryBoardGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.piece_embed = nn.Embedding(len(PIECES), 12)
            self.board_net = nn.Sequential(nn.Linear(64 * 12, 96), nn.ReLU(), nn.Dropout(0.10))
            self.side_embed = nn.Embedding(2, 8)
            self.task_embed = nn.Embedding(len(TASKS), 16)
            self.square_embed = nn.Embedding(65, 24, padding_idx=0)
            self.history_net = nn.GRU(48, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(96+8+16+128, 128), nn.ReLU(),
                                      nn.Dropout(0.15), nn.Linear(128, 1))
        def forward(self, board, side, task, history, lengths):
            b = self.board_net(self.piece_embed(board).flatten(1))
            if lengths.max().item() == 0:
                h = torch.zeros((board.shape[0], 128), device=board.device, dtype=b.dtype)
            else:
                h_emb = self.square_embed(history).flatten(2)
                packed = nn.utils.rnn.pack_padded_sequence(
                    h_emb, lengths.clamp(min=1).cpu(), batch_first=True,
                    enforce_sorted=False)
                _, hidden = self.history_net(packed)
                h = torch.cat((hidden[-2], hidden[-1]), dim=1)
                h = h * (lengths > 0).to(h.dtype).unsqueeze(1)
            return self.head(torch.cat((b, self.side_embed(side),
                                         self.task_embed(task), h), dim=1)).squeeze(1)

    outdir = Path(outdir)
    metric_rows, curve, predictions, ablation = [], [], [], []
    best_cfg = {}
    test_pair_ids = {(r["pair_id"], r["variant"]) for r in split["test"]}
    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for keep in cutoffs:
            # Same initial weights per seed across cutoffs for a cleaner ablation.
            torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed_all(seed)
            tag = "full" if keep is None else str(keep)
            train = make_data(split["train"], keep, device, max_history)
            val = make_data(split["validation"], keep, device, max_history)
            test = make_data(split["test"], keep, device, max_history)
            model = HistoryBoardGRU().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)
            criterion = nn.BCEWithLogitsLoss()
            best_val, best_state, best_epoch, bad = float("inf"), None, None, 0
            print(f"GPU neural training seed={seed} visible_plies={tag}", flush=True)
            for epoch in range(1, epochs + 1):
                model.train()
                permutation = torch.randperm(len(split["train"]), device=device)
                total_loss = 0.
                for ids in permutation.split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(*(tensor[ids] for tensor in train[:5]))
                    loss = criterion(logits, train[5][ids]); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    total_loss += float(loss.detach()) * ids.numel()
                model.eval()
                with torch.inference_mode():
                    vloss = 0.
                    for ids in torch.arange(len(split["validation"]), device=device).split(batch_size):
                        logits = model(*(tensor[ids] for tensor in val[:5]))
                        vloss += float(criterion(logits, val[5][ids])) * ids.numel()
                vloss /= len(split["validation"])
                curve.append({"seed":seed,"history_moves":tag,"epoch":epoch,
                              "train_loss":round(total_loss / len(split["train"]), 7),
                              "validation_loss":round(vloss, 7),
                              "device":str(device)})
                if vloss < best_val - 0.0001:
                    best_val, bad, best_epoch = vloss, 0, epoch
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if best_state is None:
                raise RuntimeError("Training did not yield any checkpoint")
            model.load_state_dict(best_state)
            model.eval()
            probabilities = []
            with torch.inference_mode():
                for ids in torch.arange(len(split["test"]), device=device).split(batch_size):
                    probabilities.extend(torch.sigmoid(model(
                        *(tensor[ids] for tensor in test[:5]))).cpu().tolist())
            pred = {}
            for r,p in zip(split["test"],probabilities):
                key=(r["pair_id"], r["variant"])
                assert key in test_pair_ids
                pred[key] = p >= 0.5
                predictions.append({"model":"history_board_GRU", "seed":seed,
                                    "history_moves":tag,"pair_id":r["pair_id"],
                                    "task":r["task"],"variant":r["variant"],
                                    "label":r["label"],"score":float(p),
                                    "prediction":int(p>=.5)})
            for m in metrics(pred,pairs,"history_board_GRU"):
                m.update({"seed":seed,"history_moves":tag,"best_epoch":best_epoch})
                metric_rows.append(m)
                test_subset=[p for p in pairs if p["split"]=="test" and
                             (m["task"]=="all" or p["task"]==m["task"])]
                distinguishable=sum((p["positive"]["history_uci"][-keep:] if keep else []) !=
                                    (p["negative"]["history_uci"][-keep:] if keep else [])
                                    for p in test_subset) if keep is not None else sum(
                                    p["positive"]["history_uci"] != p["negative"]["history_uci"]
                                    for p in test_subset)
                ablation.append({"seed":seed,"history_moves":tag,"task":m["task"],
                                 "test_pairs":m["test_pairs"],
                                 "accuracy":m["accuracy_including_invalid"],
                                 "paired_joint_accuracy":m["paired_joint_accuracy"],
                                 "pair_input_distinguishability_ceiling":round(
                                     distinguishable/len(test_subset),6)})
            if seed == seeds[0] and keep is None:
                torch.save({"state_dict":best_state,"architecture":"HistoryBoardGRU",
                            "seed":seed,"max_history":max_history,"cutoff":"full",
                            "best_epoch":best_epoch,"pieces":PIECES,"tasks":TASKS},
                           outdir / "history_gru_best.pt")
            best_cfg[f"{seed}:{tag}"] = {"best_epoch":best_epoch,
                                        "validation_loss":round(best_val,7),
                                        "parameters":sum(p.numel() for p in model.parameters())}
            del model, optimizer, best_state, train, val, test
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(outdir / "gru_metrics.csv",metric_rows)
    write_csv(outdir / "gru_learning_curve.csv",curve)
    write_csv(outdir / "history_ablation.csv",ablation)
    write_csv(outdir / "gru_predictions.csv",predictions)
    save_json(outdir / "gru_model_config.json",{
        "method":"joint board+history multitask neural classifier",
        "board_features":"64 piece identities, side to move",
        "history_features":"UCI source and target squares, recent history only",
        "forbidden_features":"castling rights, EP target, halfmove clock, FEN counters, repetition count",
        "cutoffs": ["full" if k is None else k for k in cutoffs],
        "full_max_history_plies":max_history,
        "note":"Every cutoff trained from scratch with own validation checkpoint; full means at most max_history plies",
        "seeds":list(seeds),"device":str(device),"runs":best_cfg})
    return metric_rows
