"""CPU checks for publication-facing HistChess architectural controls."""
import pytest

torch=pytest.importorskip("torch")
from histchess.architecture_benchmark import (
    ARCHITECTURES, TASKS, HistoryArchitecture, CausalLinearBlock,
    make_budgeted_model, num_parameters, make_tensors, observability_bounds,
    rotate_splits, _check_pair_integrity, move_id, boards_to_ids,
)


def _fake_pairs():
    board="8/8/8/8/8/8/8/4K2k w"
    entries=[]
    families={
        "castling": ("rook_g1_return", "king_e2_return", "rook_h2_return"),
        "en_passant": ("quiet_vs_double", "two_black_doubles", "black_captures_white_double"),
        "halfmove": ("both_pawns_last", "white_pawn_penultimate", "black_pawn_last"),
        "repetition": ("legacy_two_cycles", "interleaved_reverse", "interleaved_forward"),
    }
    for task,seq in families.items():
        for ix,family in enumerate(seq):
            # Make each board unique, but identical across the two variants.
            bd=board.replace("4K2k",("3K3k" if ix==1 else "K6k" if ix==2 else "4K2k"))
            entries.append(dict(pair_id=f"{task}_{ix}",task=task,
                generator_family=family,board_key=bd,
                positive={"board_observation":bd,"history_uci":["a2a4","b7b5","g1f3"],"label":1},
                negative={"board_observation":bd,"history_uci":["a2a4","b7b5","b1c3"],"label":0}))
    return entries


def test_every_architecture_finite_and_gradient():
    data=(torch.zeros(3,64,dtype=torch.long),torch.zeros(3,dtype=torch.long),
          torch.tensor([0,1,2]),torch.randint(0,5,(3,5,3)),torch.tensor([5,3,0]))
    for architecture in ARCHITECTURES:
        model=HistoryArchitecture(architecture,width=32,max_history=16,dropout=0.0)
        result=model(*data)
        assert result.shape==(3,) and torch.isfinite(result).all(),architecture
        result.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_future_information_in_transformer_or_linear_attention():
    torch.manual_seed(10)
    x=torch.randn(2,6,32)
    changed=x.clone();changed[:,4:]=torch.randn_like(changed[:,4:])*100
    valid=torch.ones(2,6,dtype=torch.bool)
    linear=CausalLinearBlock(32,dropout=0).eval()
    a=linear(x,valid);b=linear(changed,valid)
    assert torch.allclose(a[:,:4],b[:,:4],atol=1e-5)
    layer=torch.nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,
                           dropout=0,batch_first=True,norm_first=True)
    t=torch.nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False).eval()
    mask=torch.triu(torch.ones(6,6,dtype=torch.bool),diagonal=1)
    assert torch.allclose(t(x,mask=mask)[:,:4],t(changed,mask=mask)[:,:4],atol=1e-5)


def test_published_architecture_records_and_parameter_budget():
    for architecture,ref in ARCHITECTURES.items():
        assert ref["doi"] and ref["reference"] and ref["mechanism"]
        m=make_budgeted_model(architecture,target=150000,max_history=256)
        assert abs(num_parameters(m)/150000-1)<0.10,(architecture,num_parameters(m))


def test_exact_observability_bound_for_indistinguishable_pairs():
    pairs=_fake_pairs()
    # _check_pair_integrity is intentionally not called; these tiny fixtures
    # reuse board positions. The test is solely about visible suffix matching.
    folds=(0,1,2)
    bound=observability_bounds(pairs,folds,cutoffs=(0,1,None))
    z=[r for r in bound if r["fold"]==0 and r["history_plies"]==0 and r["task"]=="all"]
    assert len(z)==1 and z[0]["record_accuracy_upper_bound"]==0.5
    assert z[0]["joint_pair_accuracy_upper_bound"]==0
    h=[r for r in bound if r["fold"]==0 and r["history_plies"]=="full" and r["task"]=="all"]
    assert h[0]["record_accuracy_upper_bound"]==1 and h[0]["joint_pair_accuracy_upper_bound"]==1
    for fold in folds:
        split=rotate_splits(pairs,fold)
        assert {p["generator_family"] for p in split if p["split"]=="test" and p["task"]=="castling"}


def test_cutoff_and_no_latent_fen_features():
    rows=[dict(board_key="8/8/8/8/8/8/8/4K2k w",task="castling",label=1,
               history_uci=["a2a4","b7b5","g1f3"],pair_id="0",variant="positive")]
    board,side,task,moves,length,label=make_tensors(rows,1,32)
    assert moves.shape==(1,1,3)
    assert moves[0,0].tolist()==list(move_id("g1f3"))
    assert length.tolist()==[1]
    assert move_id("a7a8q")[-1]==1 and move_id("a7a8n")[-1]==4
    assert board.shape==(1,64)
    assert boards_to_ids(rows[0]["board_key"])[1]==0
    empty=make_tensors(rows,0,32)
    assert empty[4].item()==0


def test_no_unsafe_model_or_cutoff():
    with pytest.raises(ValueError): HistoryArchitecture("some_new_llm",32)
    with pytest.raises(ValueError): make_tensors([],cutoff=-1,max_history=32)


def test_sinusoidal_positions_defined_beyond_seen_training_length():
    from histchess.architecture_benchmark import sinusoidal_positions
    signal=sinusoidal_positions(512,32)
    assert signal.shape==(512,32)
    assert torch.isfinite(signal).all()
    assert torch.allclose(signal[0,0::2],torch.zeros(16))
    for kind in ARCHITECTURES:
        net=HistoryArchitecture(kind,width=32,max_history=128,dropout=0,
                                position_encoding="sinusoidal").eval()
        # Input lengths exceed typical 16-ply training windows, with defined
        # position values and without the learned out-of-domain lookup issue.
        boards=torch.zeros(2,64,dtype=torch.long)
        sides=torch.zeros(2,dtype=torch.long)
        tasks=torch.zeros(2,dtype=torch.long)
        moves=torch.ones(2,33,3,dtype=torch.long)
        lengths=torch.tensor([33,19])
        with torch.inference_mode():
            values=net(boards,sides,tasks,moves,lengths)
        assert values.shape==(2,) and torch.isfinite(values).all()


def test_rank_brier_and_exact_observation_ceiling():
    from histchess.architecture_benchmark import _metrics
    rows=[dict(pair_id="p",variant="positive",task="castling",label=1),
          dict(pair_id="p",variant="negative",task="castling",label=0),
          dict(pair_id="q",variant="positive",task="en_passant",label=1),
          dict(pair_id="q",variant="negative",task="en_passant",label=0),
          dict(pair_id="r",variant="positive",task="halfmove",label=1),
          dict(pair_id="r",variant="negative",task="halfmove",label=0),
          dict(pair_id="s",variant="positive",task="repetition",label=1),
          dict(pair_id="s",variant="negative",task="repetition",label=0)]
    m=_metrics(rows,[1,0,0,0,1,0,1,0],[.9,.1,.2,.8,.9,.1,.9,.1])[0]
    assert m["record_accuracy"]==.875
    assert m["joint_pair_accuracy"]==.75
    assert m["pair_ranking_accuracy"]==.75
    assert m["brier_score"]>0
    bounds=observability_bounds(_fake_pairs(),folds=(0,),cutoffs=(0,))
    for row in bounds:
        assert row["empirical_observation_ceiling"]<=row["record_accuracy_upper_bound"]
        assert row["empirical_observation_ceiling"]>=.5


def test_resume_dataset_metadata_protection_and_length_shift(tmp_path):
    from histchess.architecture_benchmark import run_architecture_benchmark
    # Synthetic input is sufficient for a model-run smoke test; unlike real
    # experiments it is not represented as legal chess data.
    pairs=[]
    for task_ix,task in enumerate(TASKS):
        for ix,family in enumerate({
            "castling":("rook_g1_return","king_e2_return","rook_h2_return"),
            "en_passant":("quiet_vs_double","two_black_doubles","black_captures_white_double"),
            "halfmove":("both_pawns_last","white_pawn_penultimate","black_pawn_last"),
            "repetition":("legacy_two_cycles","interleaved_reverse","interleaved_forward")
        }[task]):
            # Unique board: one extra white piece at a unique square.
            board=["-"]*64
            board[task_ix*3+ix]="P"
            # Canonical board notation generated manually for this isolated test.
            ranks=[]
            for row in range(8):
                current=""; blanks=0
                for ch in board[row*8:row*8+8]:
                    if ch=="-":blanks+=1
                    else:
                        if blanks:current+=str(blanks);blanks=0
                        current+=ch
                if blanks:current+=str(blanks)
                ranks.append(current)
            observation="/".join(ranks)+" w"
            pairs.append({"pair_id":f"{task_ix}_{ix}","task":task,"generator_family":family,
                          "board_key":observation,"split":("train","validation","test")[ix],
                          "positive":{"label":1,"board_observation":observation,
                                      "history_uci":["g1f3","g8f6","f3g1"]},
                          "negative":{"label":0,"board_observation":observation,
                                      "history_uci":["g1f3","g8f6","b1c3"]}})
    run_architecture_benchmark(pairs,tmp_path,architectures=("gru",),folds=(0,),seeds=(47,),
        cutoffs=(0,2),epochs=1,batch_size=8,target_parameters=16000,
        max_history=32,position_encoding="sinusoidal",length_eval_cutoffs=(4,None))
    assert (tmp_path/"architecture_length_ood_predictions.csv").exists()
    import json
    pairs_file=tmp_path/"data"/"pairs.jsonl"
    pairs_file.parent.mkdir(exist_ok=True)
    pairs_file.write_text("".join(json.dumps(p)+"\n" for p in pairs),encoding="utf8")
    import importlib.util
    from pathlib import Path
    verify_path=Path(__file__).resolve().parents[1]/"scripts"/"verify_architecture_results.py"
    spec=importlib.util.spec_from_file_location("histchess_offline_auditor",verify_path)
    auditor=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    verification=auditor.audit(tmp_path)
    assert verification["passed"], verification["errors"][:6]
    status=json.loads((tmp_path/"architecture_status.json").read_text())
    assert status["complete"] and len(status["runs"])==2
    # Metadata not previously hashed (board representation) must now prevent
    # old-shard reuse even if the pair IDs and histories happen to match.
    modified=[dict(x) for x in pairs]
    modified[0]=dict(modified[0]);modified[0]["generator_family"]="king_e2_return"
    with pytest.raises(ValueError,match="Run parameters/dataset changed"):
        run_architecture_benchmark(modified,tmp_path,architectures=("gru",),folds=(0,),seeds=(47,),
            cutoffs=(0,2),epochs=1,batch_size=8,target_parameters=16000,
            max_history=32,position_encoding="sinusoidal",length_eval_cutoffs=(4,None))
