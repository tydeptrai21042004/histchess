"""Research-grounded, causal architecture comparison for HistChess.

Architectures and references are recorded in ARCHITECTURES.  The Transformer is
an encoder stack with an *explicit causal mask*, not a pretrained GPT model.
Linear attention is a causal ELU+1 feature-map implementation, not a checkpoint
from Katharopoulos et al. The real diagonal SSM is an S4D-inspired ablation,
NOT the authors' S4D (which uses additional kernel/initialization machinery).

Models see only: board piece placement, side to move, task ID, UCI moves.
They never see castling/FEN rights, EP field, halfmove, repetitions or labels.
All model training, validation, cutoff selection and testing use the same input
cutoff; no test-only truncation. Families are rotated between splits without
reusing a pair within a fold.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TASKS = ("castling", "en_passant", "halfmove", "repetition")
PIECES = "-PNBRQKpnbrqk"
FILES = "abcdefgh"
FAMILIES = {
    "castling": ("rook_g1_return", "king_e2_return", "rook_h2_return"),
    "en_passant": ("quiet_vs_double", "two_black_doubles", "black_captures_white_double"),
    "halfmove": ("both_pawns_last", "white_pawn_penultimate", "black_pawn_last"),
    "repetition": ("legacy_two_cycles", "interleaved_reverse", "interleaved_forward"),
}
ARCHITECTURES = {
    "elman_rnn": {"reference": "Elman (1990), Finding Structure in Time", "doi": "10.1207/s15516709cog1402_1", "mechanism": "ungated recurrent state"},
    "lstm": {"reference": "Hochreiter and Schmidhuber (1997), Long Short-Term Memory", "doi": "10.1162/neco.1997.9.8.1735", "mechanism": "gated recurrent cell state"},
    "gru": {"reference": "Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder", "doi": "10.3115/v1/D14-1179", "mechanism": "gated recurrent hidden state"},
    "causal_transformer": {"reference": "Vaswani et al. (2017), Attention Is All You Need", "doi": "10.48550/arXiv.1706.03762", "mechanism": "causal softmax self-attention; trained from scratch"},
    "linear_attention": {"reference": "Katharopoulos et al. (2020), Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", "doi": "10.48550/arXiv.2006.16236", "mechanism": "causal feature-map linear attention (ELU+1)"},
    "diagonal_ssm": {"reference": "Gu, Gupta, Goel and Re (2022), On the Parameterization and Initialization of Diagonal State Space Models", "doi": "10.48550/arXiv.2206.11893", "mechanism": "real-valued diagonal recurrent linear state-space ablation; NOT full S4D"},
}


def move_id(uci):
    if not isinstance(uci, str) or len(uci) not in (4, 5):
        raise ValueError(f"Invalid UCI move {uci!r}")
    if uci[0] not in FILES or uci[2] not in FILES or uci[1] not in "12345678" or uci[3] not in "12345678":
        raise ValueError(f"Invalid UCI coordinate {uci!r}")
    if len(uci) == 5 and uci[4] not in "qrbn":
        raise ValueError(f"Invalid promotion code {uci!r}")
    return (1 + FILES.index(uci[0]) + 8*(int(uci[1])-1),
            1 + FILES.index(uci[2]) + 8*(int(uci[3])-1),
            0 if len(uci)==4 else 1+"qrbn".index(uci[4]))


def boards_to_ids(board_key):
    layout, turn = board_key.split()
    board = []
    for char in layout.replace("/", ""):
        if char.isdigit(): board.extend([0]*int(char))
        else: board.append(PIECES.index(char))
    if len(board) != 64 or turn not in ("w", "b"):
        raise ValueError("Invalid observable board")
    return board, 0 if turn == "w" else 1


def rows_from_pairs(pairs):
    rows = []
    for pair in pairs:
        for variant in ("positive", "negative"):
            item = pair[variant]
            rows.append({"pair_id": pair["pair_id"], "variant":variant,
                         "task": pair["task"], "board_key":pair["board_key"],
                         "generator_family":pair["generator_family"],
                         "history_uci":item["history_uci"], "label":int(item["label"]),
                         "split":pair.get("split", "unassigned")})
    return rows


def rotate_splits(pairs, fold):
    """Rotate whole template mechanisms. All three test mechanisms per task get evaluated."""
    if fold not in (0,1,2): raise ValueError("fold in {0,1,2} required")
    updated = []
    for pair in pairs:
        q = dict(pair)
        idx = FAMILIES[q["task"]].index(q["generator_family"])
        q["split"] = ("train","validation","test")[(idx-fold)%3]
        updated.append(q)
    for task in TASKS:
        groups = {split: {p["generator_family"] for p in updated
                           if p["task"]==task and p["split"]==split}
                  for split in ("train","validation","test")}
        if any(len(s)!=1 for s in groups.values()) or len(set.union(*groups.values()))!=3:
            raise ValueError(f"Invalid held-out family allocation for {task}")
    return updated


def make_tensors(rows, cutoff, max_history, device="cpu"):
    """Right padding, explicit lengths. Full history is capped, and audit reports truncation."""
    if cutoff is not None and cutoff < 0: raise ValueError("Negative cutoff")
    boards, sides, tasks, moves, lengths, labels = [], [], [], [], [], []
    for r in rows:
        board, side = boards_to_ids(r["board_key"])
        seq = r["history_uci"]
        seq = seq[-max_history:]
        if cutoff is not None: seq = seq[-cutoff:] if cutoff else []
        m = [move_id(x) for x in seq]
        boards.append(board); sides.append(side); tasks.append(TASKS.index(r["task"]))
        moves.append(m); lengths.append(len(m));labels.append(float(r["label"]))
    length = max(1,max(lengths,default=0))
    padded = [m+[(0,0,0)]*(length-len(m)) for m in moves]
    return (torch.tensor(boards,dtype=torch.long,device=device),
            torch.tensor(sides,dtype=torch.long,device=device),
            torch.tensor(tasks,dtype=torch.long,device=device),
            torch.tensor(padded,dtype=torch.long,device=device),
            torch.tensor(lengths,dtype=torch.long,device=device),
            torch.tensor(labels,dtype=torch.float32,device=device))


class CausalLinearBlock(nn.Module):
    """Causal phi(q)^T cumulative(phi(k) v^T) / phi(q)^T cumulative(phi(k))."""
    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(width); self.to_qkv=nn.Linear(width,3*width)
        self.project=nn.Linear(width,width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.drop=nn.Dropout(dropout)
    def forward(self, x, valid):
        z=self.norm1(x)
        q,k,v=self.to_qkv(z).chunk(3,dim=-1)
        q=F.elu(q)+1; k=(F.elu(k)+1)*valid.unsqueeze(-1)
        v=v*valid.unsqueeze(-1)
        # Prefix sums imply STRICT causal behavior; no access to future moves.
        kv=(k.unsqueeze(-1)*v.unsqueeze(-2)).cumsum(dim=1)
        ks=k.cumsum(dim=1)
        numerator=torch.einsum("btd,btdh->bth",q,kv)
        denominator=(q*ks).sum(-1,keepdim=True).clamp_min(1e-6)
        context=numerator/denominator
        x=x+self.drop(self.project(context))*valid.unsqueeze(-1)
        return x+self.drop(self.ff(self.norm2(x)))*valid.unsqueeze(-1)


class RealDiagonalSSMBlock(nn.Module):
    """Interpretable real diagonal state update; simplified research-inspired SSM.

    h_t = exp(-softplus(a)) h_{t-1} + b * u_t; y_t = c * h_t + d * u_t.
    This is NOT a faithful reproduction of S4D's complex kernel or HiPPO initialization.
    """
    def __init__(self,width,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(width)
        self.a=nn.Parameter(torch.full((width,),-4.0))
        self.b=nn.Parameter(torch.ones(width));self.c=nn.Parameter(torch.ones(width))
        self.d=nn.Parameter(torch.zeros(width));self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.drop=nn.Dropout(dropout)
    def forward(self,x,valid):
        u=self.norm1(x)
        decay=torch.exp(-F.softplus(self.a))
        state=torch.zeros_like(x[:,0,:]);outputs=[]
        for t in range(x.shape[1]):
            candidate=decay*state+self.b*u[:,t,:]
            state=torch.where(valid[:,t,None],candidate,state)
            outputs.append((self.c*state+self.d*u[:,t,:])*valid[:,t,None])
        y=torch.stack(outputs,dim=1)
        x=x+self.drop(y)
        return x+self.drop(self.ff(self.norm2(x)))*valid.unsqueeze(-1)


class HistoryArchitecture(nn.Module):
    def __init__(self,kind,width=64,max_history=256,dropout=0.1):
        super().__init__()
        if kind not in ARCHITECTURES: raise ValueError(f"Unknown architecture: {kind}")
        if width%4:raise ValueError("width must be divisible by four")
        self.kind=kind;self.width=width;self.max_history=max_history
        self.piece_embed=nn.Embedding(len(PIECES),12)
        self.board_net=nn.Sequential(nn.Linear(64*12,96),nn.ReLU(),nn.Dropout(dropout))
        self.side_embed=nn.Embedding(2,8); self.task_embed=nn.Embedding(len(TASKS),16)
        self.square_embed=nn.Embedding(65,width//2,padding_idx=0)
        self.promotion_embed=nn.Embedding(5,width)
        self.position_embed=nn.Embedding(max_history+1,width)
        self.norm_input=nn.LayerNorm(width)
        if kind in ("elman_rnn","gru","lstm"):
            module={"elman_rnn":nn.RNN,"gru":nn.GRU,"lstm":nn.LSTM}[kind]
            self.sequence=module(width,width,num_layers=2,batch_first=True,dropout=dropout,
                                 nonlinearity="tanh" if kind=="elman_rnn" else None) if kind=="elman_rnn" else module(width,width,num_layers=2,batch_first=True,dropout=dropout)
        elif kind=="causal_transformer":
            layer=nn.TransformerEncoderLayer(d_model=width,nhead=4,dim_feedforward=2*width,
                                             dropout=dropout,batch_first=True,activation="gelu",norm_first=True)
            self.sequence=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        elif kind=="linear_attention":
            self.sequence=nn.ModuleList([CausalLinearBlock(width,dropout) for _ in range(2)])
        else:
            self.sequence=nn.ModuleList([RealDiagonalSSMBlock(width,dropout) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(96+8+16+width,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,1))
    def forward(self,board,side,task,moves,lengths):
        board_repr=self.board_net(self.piece_embed(board).flatten(1))
        b,t,ch=moves.shape
        if ch!=3:raise ValueError("History must encode source, target and promotion")
        if t>self.max_history:raise ValueError("Sequence exceeds max_history")
        x=self.square_embed(moves[:,:,:2]).flatten(2)+self.promotion_embed(moves[:,:,2])
        pos=torch.arange(t,device=moves.device).unsqueeze(0)
        x=self.norm_input(x+self.position_embed(pos))
        valid=torch.arange(t,device=moves.device)[None,:]<lengths.clamp(min=1)[:,None]
        if self.kind in ("elman_rnn","gru","lstm"):
            packed=nn.utils.rnn.pack_padded_sequence(x,lengths.clamp(min=1).cpu(),
                                                       batch_first=True,enforce_sorted=False)
            result=self.sequence(packed)
            h=result[1][0] if self.kind=="lstm" else result[1]
            state=h[-1]
        else:
            x=x*valid.unsqueeze(-1)
            if self.kind=="causal_transformer":
                causal=torch.triu(torch.ones((t,t),device=moves.device,dtype=torch.bool),diagonal=1)
                x=self.sequence(x,mask=causal,src_key_padding_mask=~valid)
            else:
                for block in self.sequence: x=block(x,valid)
            idx=lengths.clamp(min=1).sub(1)
            state=x[torch.arange(b,device=moves.device),idx]
        state=state*(lengths>0).to(state.dtype).unsqueeze(-1)
        return self.head(torch.cat([board_repr,self.side_embed(side),self.task_embed(task),state],dim=-1)).squeeze(-1)


def num_parameters(model):return sum(x.numel() for x in model.parameters())


def make_budgeted_model(kind,target=150000,max_history=256,dropout=0.10,
                        widths=(32,40,48,56,64,72,80,96,112,128)):
    # Select width using a fixed public grid, never validation/test performance.
    options=[HistoryArchitecture(kind,w,max_history,dropout) for w in widths]
    best=min(options,key=lambda m:abs(num_parameters(m)-target))
    return best


def _metrics(rows,preds):
    items={(r["pair_id"],r["variant"]):(r["task"],int(r["label"]),int(g)) for r,g in zip(rows,preds)}
    result=[]
    for task in ("all",)+TASKS:
        sub={key:v for key,v in items.items() if task=="all" or v[0]==task}
        n=len(sub)
        if n==0: raise RuntimeError(f"Empty test subset {task}")
        correct=sum(int(v[1]==v[2]) for v in sub.values())
        ids={key[0] for key in sub}
        joint=sum(int(sub[(pid,"positive")][2]==1 and sub[(pid,"negative")][2]==0) for pid in ids)
        result.append({"task":task,"test_pairs":len(ids),"test_records":n,
                       "record_accuracy":correct/n,"joint_pair_accuracy":joint/len(ids)})
    return result


def _write_csv(path,records):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        path.write_text("",encoding="utf8");return
    fields=list(dict.fromkeys(k for record in records for k in record))
    with path.open("w",newline="",encoding="utf8") as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader();writer.writerows(records)


def _check_pair_integrity(pairs):
    keys=collections.Counter(p["pair_id"] for p in pairs)
    if not pairs or any(n!=1 for n in keys.values()):raise ValueError("Duplicate/empty pair ID set")
    for p in pairs:
        if p["task"] not in TASKS or p.get("generator_family") not in FAMILIES[p["task"]]:
            raise ValueError("Unsupported task/generator family")
        if p["positive"]["label"]!=1 or p["negative"]["label"]!=0:
            raise ValueError("Incorrect balanced counterfactual labels")
        if p["positive"]["board_observation"]!=p["negative"]["board_observation"] or p["positive"]["board_observation"]!=p["board_key"]:
            raise ValueError("Counterfactuals do not share their board")
    seen={}
    for p in pairs:
        old=seen.setdefault(p["board_key"],p["pair_id"])
        if old!=p["pair_id"]:raise ValueError("Duplicate boards across pairs")


def observability_bounds(pairs, folds=(0,1,2), cutoffs=(0,4,8,16,None), max_history=256):
    """Information-theoretic bounds induced by identical board+visible suffix.

    A pair with identical observations and opposite labels cannot have more
    than 50% record accuracy or zero joint-pair success for any deterministic
    model restricted to those observations (randomization cannot raise expected
    record accuracy). This is a bound, not a learned-model result.
    """
    results=[]
    for fold in folds:
        test=[p for p in rotate_splits(pairs,fold) if p["split"]=="test"]
        for cutoff in cutoffs:
            for task in ("all",)+TASKS:
                group=[p for p in test if task=="all" or p["task"]==task]
                distinguishable=0
                for p in group:
                    a=p["positive"]["history_uci"][-max_history:]
                    b=p["negative"]["history_uci"][-max_history:]
                    if cutoff is not None:
                        a=a[-cutoff:] if cutoff else []
                        b=b[-cutoff:] if cutoff else []
                    distinguishable+=int(a!=b)
                n=len(group)
                if not n:raise ValueError("Empty held-out task")
                frac=distinguishable/n
                results.append({"fold":fold,"history_plies":"full" if cutoff is None else cutoff,
                    "task":task,"test_pairs":n,"distinguishable_pairs":distinguishable,
                    "indistinguishable_pairs":n-distinguishable,
                    "record_accuracy_upper_bound":0.5+frac/2,
                    "joint_pair_accuracy_upper_bound":frac})
    return results


def _one_run(pairs,kind,fold,cutoff,seed,options,output):
    torch.manual_seed(seed);random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    train_pairs=rotate_splits(pairs,fold)
    rows=rows_from_pairs(train_pairs)
    sets={s:[r for r in rows if r["split"]==s] for s in ("train","validation","test")}
    if any(not x for x in sets.values()): raise RuntimeError("Missing data in a fold")
    device=torch.device(options.device)
    tensors={s:make_tensors(sets[s],cutoff,options.max_history,device)
             for s in sets}
    model=make_budgeted_model(kind,options.target_parameters,options.max_history).to(device)
    params=num_parameters(model)
    optim=torch.optim.AdamW(model.parameters(),lr=options.lr,weight_decay=options.weight_decay)
    loss_fn=nn.BCEWithLogitsLoss()
    best=float("inf");state=None;best_epoch=0;bad=0;curves=[]
    training_start=time.monotonic()
    for epoch in range(1,options.epochs+1):
        model.train();n=0;total=0.0
        perm=torch.randperm(len(sets["train"]),device=device)
        for ids in perm.split(options.batch_size):
            optim.zero_grad(set_to_none=True)
            logits=model(*(v[ids] for v in tensors["train"][:5]))
            loss=loss_fn(logits,tensors["train"][5][ids]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step();n+=ids.numel();total+=float(loss.detach())*ids.numel()
        model.eval();vloss=0.0
        with torch.inference_mode():
            for ids in torch.arange(len(sets["validation"]),device=device).split(options.batch_size):
                logits=model(*(v[ids] for v in tensors["validation"][:5]))
                vloss+=float(loss_fn(logits,tensors["validation"][5][ids]))*len(ids)
        vloss/=len(sets["validation"])
        curves.append({"architecture":kind,"fold":fold,"seed":seed,"history":str(cutoff),
                       "epoch":epoch,"train_loss":total/n,"validation_loss":vloss})
        if vloss<best-1e-4:
            best=vloss;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch;bad=0
        else:
            bad+=1
            if bad>=options.patience: break
    train_seconds=time.monotonic()-training_start
    if state is None:raise RuntimeError("Failed to establish a checkpoint")
    model.load_state_dict(state);model.eval()
    probabilities=[]
    start=time.monotonic()
    with torch.inference_mode():
        for ids in torch.arange(len(sets["test"]),device=device).split(options.batch_size):
            probabilities.extend(torch.sigmoid(model(*(v[ids] for v in tensors["test"][:5]))).cpu().tolist())
    inference_seconds=time.monotonic()-start
    preds=[int(v>=.5) for v in probabilities]
    met=[];pre=[]
    for row in _metrics(sets["test"],preds):
        met.append({"architecture":kind,"fold":fold,"seed":seed,
                    "history_plies":"full" if cutoff is None else str(cutoff),
                    "parameters":params,"target_parameters":options.target_parameters,
                    "best_epoch":best_epoch,"best_validation_bce":best,
                    "train_seconds":train_seconds,"test_seconds":inference_seconds,**row})
    for row,p,score in zip(sets["test"],preds,probabilities):
        pre.append({"architecture":kind,"fold":fold,"seed":seed,
                    "history_plies":"full" if cutoff is None else str(cutoff),
                    "pair_id":row["pair_id"],"variant":row["variant"],
                    "task":row["task"],"family":row["generator_family"],
                    "label":row["label"],"prediction":p,"score":score})
    return met,pre,curves


def _write_architecture_analysis(out, metrics, predictions, seed=3101):
    """Descriptive aggregates and genuinely matched per-pair model comparisons.

    Only exploratory pair-bootstrap intervals. Neither generator families nor
    stochastic seeds are independent observations in these intervals.
    """
    groups=collections.defaultdict(list)
    for row in metrics:
        if row["task"]=="all":
            groups[(row["architecture"],row["history_plies"])].append(row)
    summary=[]
    for (arch,history),rows in sorted(groups.items()):
        record=[r["record_accuracy"] for r in rows]
        joint=[r["joint_pair_accuracy"] for r in rows]
        summary.append({"architecture":arch,"history_plies":history,
            "fold_seed_runs":len(rows),"unique_folds":len({r["fold"] for r in rows}),
            "unique_training_seeds":len({r["seed"] for r in rows}),
            "mean_record_accuracy":sum(record)/len(record),
            "sd_record_accuracy_across_runs":(sum((x-sum(record)/len(record))**2 for x in record)/len(record))**.5,
            "min_record_accuracy_across_runs":min(record),
            "max_record_accuracy_across_runs":max(record),
            "mean_joint_pair_accuracy":sum(joint)/len(joint),
            "min_joint_pair_accuracy_across_runs":min(joint),
            "mean_parameters":sum(r["parameters"] for r in rows)/len(rows),
            "note":"Fold-seed rows correlated; SD is descriptive, not uncertainty over new generators."})
    _write_csv(Path(out)/"architecture_summary.csv",summary)
    indexed=collections.defaultdict(dict)
    for p in predictions:
        key=(p["fold"],p["seed"],p["history_plies"])
        indexed[key][(p["architecture"],p["pair_id"],p["variant"])]=p
    comparisons=[]
    for (fold,trial,history),d in sorted(indexed.items()):
        present={key[0] for key in d}
        for reference in ("gru","causal_transformer"):
            if reference not in present:continue
            for tested in sorted(present-{reference}):
                ids={key[1] for key in d if key[0]==reference}
                paired=[]
                for pid in ids:
                    a=[];b=[]
                    for v in ("positive","negative"):
                        ka=(tested,pid,v);kb=(reference,pid,v)
                        if ka not in d or kb not in d:
                            raise RuntimeError(f"Unmatched paired predictions: {tested} vs {reference}")
                        ra,rb=d[ka],d[kb]
                        if ra["label"]!=rb["label"]:raise RuntimeError("Target mismatch")
                        a.append(int(ra["prediction"]==ra["label"]))
                        b.append(int(rb["prediction"]==rb["label"]))
                    paired.append((sum(a)-sum(b))/2)
                mean=sum(paired)/len(paired)
                rng=random.Random(f"{seed}|{fold}|{trial}|{history}|{tested}|{reference}")
                boots=[]
                for _ in range(500):
                    boots.append(sum(paired[rng.randrange(len(paired))] for __ in paired)/len(paired))
                boots.sort()
                comparisons.append({"architecture":tested,"reference":reference,
                    "fold":fold,"seed":trial,"history_plies":history,
                    "matched_pairs":len(paired),"record_accuracy_difference":mean,
                    "paired_bootstrap_low_95_descriptive":boots[int(.025*(len(boots)-1))],
                    "paired_bootstrap_high_95_descriptive":boots[int(.975*(len(boots)-1))],
                    "multiplicity_adjusted":False,
                    "interpretation":"Pair-bootstrap within ONE held-out generator family per task; not independent-family inference."})
    _write_csv(Path(out)/"architecture_matched_comparisons.csv",comparisons)
    # Avoid accidental claims of full-history and long-history independence.
    (Path(out)/"analysis_limitations.txt").write_text(
        "Dataset is synthetic with only 3 generator families per task; rotations reuse the same pairs.\n"
        "Do not treat fold-seed rows as independent samples.\n"
        "No proof of a general Transformer or LLM impossibility follows from these measured scores.\n"
        "None of these models are pretrained LLMs; the causal Transformer is trained from scratch.\n"
        "Synthetic history lengths may be short; read architecture_config.json before drawing memory conclusions.\n"
        "Diagonal state-space implementation is S4D-inspired, NOT a canonical reproduction.\n"
        "No multiple-comparison adjustments have been made.\n",encoding="utf8")


def run_architecture_benchmark(pairs,outdir,*, architectures=None,folds=(0,1,2),seeds=(3101,3102,3103),
                                cutoffs=(0,8,None),epochs=12,patience=4,batch_size=64,
                                max_history=256,target_parameters=150000,
                                lr=0.001,weight_decay=0.001,device="cpu"):
    """Runs architecture x fold x seed x history, saving partial progress on each run.

    No results are claimed until each stage completes. This routine never invokes
    pretrained LLMs; the architectural conclusions pertain only to studied models.
    """
    _check_pair_integrity(pairs)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    architectures=tuple(architectures or ARCHITECTURES)
    if not set(architectures)<=set(ARCHITECTURES):raise ValueError("Unrecognized architecture")
    if not folds or not seeds or not cutoffs or epochs<1 or patience<1 or batch_size<1 or max_history<1:raise ValueError("Bad run configuration")
    if len(set(folds))!=len(folds) or len(set(seeds))!=len(seeds) or len(set(cutoffs))!=len(cutoffs):raise ValueError("Repeated fold, seed or cutoff")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    options=argparse.Namespace(epochs=epochs,patience=patience,batch_size=batch_size,
        max_history=max_history,target_parameters=target_parameters,lr=lr,
        weight_decay=weight_decay,device=device)
    # Explicitly fail if full histories will silently be truncated.
    longest=max(len(p[v]["history_uci"]) for p in pairs for v in ("positive","negative"))
    if longest>max_history:raise ValueError(f"max_history={max_history} truncates history length {longest}")
    if any(c is not None and c>max_history for c in cutoffs):raise ValueError("Cutoff above max_history")
    raw_path=out/"architecture_predictions.csv"
    metrics_path=out/"architecture_metrics.csv"
    curves_path=out/"architecture_learning_curves.csv"
    config={"models":{m: ARCHITECTURES[m] for m in architectures},"folds":list(folds),
            "seeds":list(seeds),"cutoffs":["full" if c is None else c for c in cutoffs],
            "history_max_in_dataset":longest,"max_history":max_history,
            "train_protocol":vars(options),"source":"synthetic diversified generator families",
            "move_encoding":"UCI from/to squares and promotion type (no unknown move vocabulary)",
            "notes":["No pretrained LLM tested. These models are trained from scratch.",
                     "This is a controlled architecture-family study, not exact replication of published experiments.",
                     "Diagonal SSM is a simplified real diagonal recurrence, not a faithful S4D reproduction.",
                     "Confidence intervals across four task-specific families require further independent generators."]}
    (out/"architecture_config.json").write_text(json.dumps(config,indent=2),encoding="utf8")
    # Persist an individual, content-bound shard immediately after each training
    # run. Kaggle notebook interruptions can be resumed without fabricating output.
    fingerprints={"model_config":config,"pair_ids":sorted(p["pair_id"] for p in pairs),
                  "pair_histories":[[p[v]["history_uci"] for v in ("positive","negative")] for p in pairs]}
    fingerprint=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    runroot=out/"architecture_run_shards";runroot.mkdir(exist_ok=True)
    ident=out/"architecture_run_fingerprint.txt"
    if ident.is_file() and ident.read_text().strip()!=fingerprint:
        raise ValueError("Run parameters/dataset changed: choose a new output directory to avoid mixing experiments")
    ident.write_text(fingerprint+"\n")
    _write_csv(out/"observability_bounds.csv",
               observability_bounds(pairs,folds,cutoffs,max_history))
    all_metrics=[];all_predictions=[];all_curves=[];completed=[]
    try:
        for kind in architectures:
            for fold in folds:
                for seed in seeds:
                    for cutoff in cutoffs:
                        tag="full" if cutoff is None else str(cutoff)
                        filename=f"{kind}_fold{fold}_seed{seed}_history{tag}.json"
                        shard=runroot/filename
                        if shard.is_file():
                            record=json.loads(shard.read_text())
                            if record.get("fingerprint")!=fingerprint:
                                raise ValueError("Cached run fingerprint mismatch: "+str(shard))
                            met,pre,curve=record["metrics"],record["predictions"],record["curve"]
                            print(f"[ARCH RESUME] {filename}",flush=True)
                        else:
                            print(f"[ARCH TRAIN] {filename}",flush=True)
                            met,pre,curve=_one_run(pairs,kind,fold,cutoff,seed,options,out)
                            tmp=shard.with_suffix(".tmp")
                            tmp.write_text(json.dumps({"fingerprint":fingerprint,
                                "metrics":met,"predictions":pre,"curve":curve}),encoding="utf8")
                            tmp.replace(shard)
                        all_metrics.extend(met);all_predictions.extend(pre);all_curves.extend(curve)
                        completed.append({"model":kind,"fold":fold,"seed":seed,"history":tag})
                        _write_csv(out/"architecture_metrics.csv",all_metrics)
                        _write_csv(out/"architecture_predictions.csv",all_predictions)
                        _write_csv(out/"architecture_learning_curves.csv",all_curves)
                        (out/"architecture_status.json").write_text(json.dumps({
                            "complete":False,"runs":completed,"fingerprint":fingerprint},indent=2))
        _write_architecture_analysis(out,all_metrics,all_predictions)
        (out/"architecture_status.json").write_text(json.dumps({
            "complete":True,"runs":completed,"fingerprint":fingerprint},indent=2))
    except BaseException as exc:
        (out/"architecture_status.json").write_text(json.dumps({
            "complete":False,"runs":completed,"fingerprint":fingerprint,"error":repr(exc)},indent=2))
        raise
    return all_metrics


def main():
    ap=argparse.ArgumentParser(description="Paper-grounded HistChess architecture benchmark")
    ap.add_argument("--pairs-jsonl",help="Use an independently validated v3 pairs.jsonl, not a result prediction CSV")
    ap.add_argument("--out",default="histchess_architecture_results")
    ap.add_argument("--pairs-per-task",type=int,default=300)
    ap.add_argument("--generator-seed",type=int,default=3101)
    ap.add_argument("--architectures",nargs="+",choices=tuple(ARCHITECTURES),default=list(ARCHITECTURES))
    ap.add_argument("--folds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--seeds",type=int,nargs="+",default=[3101,3102,3103])
    ap.add_argument("--cutoffs",nargs="+",default=["0","8","full"])
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--patience",type=int,default=4)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--max-history",type=int,default=256)
    ap.add_argument("--target-parameters",type=int,default=150000)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=ap.parse_args()
    if args.pairs_jsonl:
        data=Path(args.pairs_jsonl).read_bytes()
        pairs=[json.loads(s) for s in data.decode("utf8").splitlines() if s.strip()]
        sha=hashlib.sha256(data).hexdigest()
    else:
        from .core import generate,validate
        pairs,_=generate(args.pairs_per_task,args.generator_seed,generator="diverse",split_strategy="family")
        validation=validate(pairs)
        if not validation["passed"]:raise RuntimeError(f"Failed rule replay: {validation['errors']}")
        sha=hashlib.sha256(json.dumps(pairs,sort_keys=True).encode()).hexdigest()
        p=Path(args.out)/"data"/"pairs.jsonl";p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w",encoding="utf8") as f:
            for item in pairs:f.write(json.dumps(item,ensure_ascii=False)+"\n")
    Path(args.out).mkdir(parents=True,exist_ok=True)
    (Path(args.out)/"dataset_sha256.txt").write_text(sha+"\n")
    _check_pair_integrity(pairs)
    # Replay-validate when python-chess is present even for external datasets.
    from .core import validate
    report=validate(pairs)
    (Path(args.out)/"dataset_validation.json").write_text(json.dumps(report,indent=2))
    if not report["passed"]:raise RuntimeError("Chess replay validation failed")
    run_architecture_benchmark(pairs,args.out,architectures=args.architectures,folds=args.folds,
        seeds=args.seeds,cutoffs=tuple(None if x=="full" else int(x) for x in args.cutoffs),
        epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,
        max_history=args.max_history,target_parameters=args.target_parameters,device=args.device)


if __name__=="__main__":main()
