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
