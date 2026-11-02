"""Numerical reference tests; no external chess data or downloads required."""
import math
import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F
from histchess.foundation_models import (
    OriginalGRUCell, OriginalGRU, MultiheadLinearAttentionBlock,
    S4DLinKernel, S4DLinBlock,
)
from histchess.architecture_benchmark import (
    HistoryArchitecture, ARCHITECTURES, cutoff_coverage, TASKS, FAMILIES,
)


def test_original_gru_cell_reference_equations_and_gradients():
    torch.manual_seed(7)
    cell=OriginalGRUCell(6, 5).double()
    x=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    h=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
    ig=F.linear(x,cell.input_gates.weight,cell.input_gates.bias)
    hg=F.linear(h,cell.hidden_gates.weight,cell.hidden_gates.bias)
    r,z=torch.sigmoid(ig+hg).chunk(2,-1)
    candidate=torch.tanh(F.linear(x,cell.input_candidate.weight,cell.input_candidate.bias)
              +F.linear(r*h,cell.hidden_candidate.weight,cell.hidden_candidate.bias))
    expected=z*h+(1-z)*candidate
    result=cell(x,h)
    torch.testing.assert_close(result,expected,rtol=1e-12,atol=1e-12)
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
    assert torch.isfinite(x.grad).all() and torch.isfinite(h.grad).all()


def test_original_gru_length_masking():
    torch.manual_seed(8)
    model=OriginalGRU(8,num_layers=2,dropout=0).eval()
    inputs=torch.randn(3,9,8)
    lengths=torch.tensor([9,3,0])
    y=model(inputs,lengths)
    assert y.shape==(3,8)
    assert torch.allclose(y[-1],torch.zeros_like(y[-1]))
    altered=inputs.clone();altered[1,3:]=torch.randn_like(altered[1,3:])*1e5
    torch.testing.assert_close(y[1],model(altered,lengths)[1])


def test_multihead_linear_attention_recurrent_parallel_reference():
    torch.manual_seed(9)
    model=MultiheadLinearAttentionBlock(16,n_heads=4,dropout=0).double().eval()
    x=torch.randn(2,8,16,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[1,1,1,1,1,1,1,1],[1,1,1,0,0,0,0,0]],dtype=torch.bool)
    parallel=model.attention(x,valid,recurrent=False)
    recurrent=model.attention(x,valid,recurrent=True)
    torch.testing.assert_close(parallel,recurrent,rtol=1e-12,atol=1e-12)
    gp=torch.autograd.grad(parallel.square().sum(),x,retain_graph=True)[0]
    gr=torch.autograd.grad(recurrent.square().sum(),x)[0]
    torch.testing.assert_close(gp,gr,rtol=1e-11,atol=1e-11)


def test_linear_attention_direct_kernel_ratio_matches_prefix():
    torch.manual_seed(10)
    model=MultiheadLinearAttentionBlock(16,n_heads=4,dropout=0).double().eval()
    x=torch.randn(2,6,16,dtype=torch.float64)
    valid=torch.ones(2,6,dtype=torch.bool)
    z=model.norm1(x)
    q,k,v=[a.reshape(2,6,4,4) for a in model.to_qkv(z).chunk(3,-1)]
    q,k=F.elu(q)+1,F.elu(k)+1
    scores=torch.einsum('bthd,bshd->bhts',q,k)
    causal=torch.tril(torch.ones(6,6,dtype=torch.bool))
    scores=scores*causal.view(1,1,6,6)
    expected=torch.einsum('bhts,bshd->bthd',scores,v)/scores.sum(-1).permute(0,2,1).unsqueeze(-1)
    actual=model.attention(x,valid).reshape(2,6,4,4)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)


def test_multihead_linear_attention_future_cannot_leak():
    torch.manual_seed(11)
    model=MultiheadLinearAttentionBlock(32,dropout=0).eval()
    x=torch.randn(2,10,32)
    altered=x.clone();altered[:,5:]=1000*torch.randn_like(altered[:,5:])
    valid=torch.ones(2,10,dtype=torch.bool)
    torch.testing.assert_close(model(x,valid)[:,:5],model(altered,valid)[:,:5],rtol=1e-5,atol=1e-5)


def test_s4d_lin_zoh_discretization_is_stable():
    torch.manual_seed(12)
    kernel=S4DLinKernel(4,state_size=16)
    Abar,Bbar=kernel.discretize()
    assert Abar.shape==(4,8) and Bbar.shape==(4,8)
    assert torch.all(torch.abs(Abar)<1.0)
    assert torch.isfinite(Abar).all() and torch.isfinite(Bbar).all()
    assert (kernel.log_dt.grad is None)


@pytest.mark.parametrize('length',[1,2,8,16,32])
def test_s4d_lin_kernel_convolution_matches_recurrent_steps(length):
    torch.manual_seed(13)
    kernel=S4DLinKernel(3,state_size=16).double()
    u=torch.randn(2,3,length,dtype=torch.float64,requires_grad=True)
    k=kernel(length)
    conv=torch.fft.irfft(torch.fft.rfft(u,n=2*length)*
                         torch.fft.rfft(k,n=2*length)[None],n=2*length)[...,:length]
    rec=kernel.recurrent(u)
    torch.testing.assert_close(conv,rec,rtol=1e-9,atol=1e-9)
    gc=torch.autograd.grad(conv.square().sum(),u,retain_graph=True)[0]
    gr=torch.autograd.grad(rec.square().sum(),u)[0]
    torch.testing.assert_close(gc,gr,rtol=1e-8,atol=1e-8)


def test_s4d_lin_block_causality_and_backprop():
    torch.manual_seed(14)
    block=S4DLinBlock(16,dropout=0,state_size=16).eval()
    x=torch.randn(2,12,16,requires_grad=True)
    valid=torch.tensor([[1]*12,[1]*8+[0]*4],dtype=torch.bool)
    altered=x.detach().clone();altered[:,7:]=torch.randn_like(altered[:,7:])*100
    torch.testing.assert_close(block(x,valid)[:,:7],block(altered,valid)[:,:7],rtol=1e-4,atol=1e-4)
    out=block(x,valid)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_all_added_model_variants_backward_and_param_budget():
    from histchess.architecture_benchmark import make_budgeted_model,num_parameters
    variants=['elman_original','gru_original','causal_transformer_postnorm',
              'linear_attention_multihead','s4d_lin']
    for name in variants:
        m=make_budgeted_model(name, target=150000,max_history=64,dropout=0,
                              position_encoding='sinusoidal')
        assert abs(num_parameters(m)/150000-1)<.10,(name,num_parameters(m))
        x=(torch.zeros(2,64,dtype=torch.long),torch.tensor([0,1]),torch.tensor([0,1]),
           torch.ones(2,8,3,dtype=torch.long),torch.tensor([8,0]))
        out=m(*x)
        assert out.shape==(2,) and torch.isfinite(out).all()
        out.sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_cutoff_coverage_identifies_duplicate_short_contexts():
    pairs=[]
    for i,task in enumerate(TASKS):
        for j,family in enumerate(FAMILIES[task]):
            board=f'8/8/8/8/8/8/8/{j+1}K{6-j} w' if j in (0,1,2) else '8/8/8/8/8/8/8/4K3 w'
            pairs.append(dict(pair_id=f'{i}-{j}', task=task, generator_family=family,
                board_key=board, positive=dict(history_uci=['e2e4','e7e5'],label=1),
                negative=dict(history_uci=['d2d4','d7d5'],label=0)))
    coverage=cutoff_coverage(pairs,folds=(0,),cutoffs=(0,4,8,16,32,None))
    rows={str(r['history_plies']):r for r in coverage if r['task']=='all'}
    assert rows['8']['identical_to_cutoff']=='4'
    assert rows['16']['identical_to_cutoff']=='4'
    assert rows['32']['identical_to_cutoff']=='4'
    assert rows['full']['identical_to_cutoff']=='4'
    assert rows['0']['genuinely_distinct_view']==1
