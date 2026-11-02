"""Auditable architecture variants tied to foundational model equations.

The blocks are tailored to HistChess's input and binary classification head;
these are NOT replications of the original papers' full tasks/optimizers.

S4D-Lin implements the minimal ZOH complex diagonal kernel described in
Gu et al. 2022 and official state-spaces/s4/models/s4/s4d.py, with N=32
complex-conjugate modes (16 stored) by default to control benchmark compute.
OriginalGRU implements the reset-before-hidden-matrix recurrence from Cho 2014.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class OriginalGRUCell(nn.Module):
    """Cho-style GRU: r acts on h before the hidden candidate projection.

    Follows Cho et al. EMNLP 2014 equations (5)-(8):
    h_t = z*h_old + (1-z)*h_candidate. This differs in gate convention
    from some later implementations, besides reset-gate placement.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_gates = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 2 * hidden_size)
        self.input_candidate = nn.Linear(input_size, hidden_size)
        self.hidden_candidate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ig = self.input_gates(x)
        hg = self.hidden_gates(h)
        reset, update = torch.sigmoid(ig + hg).chunk(2, dim=-1)
        candidate = torch.tanh(self.input_candidate(x)
                               + self.hidden_candidate(reset * h))
        return update * h + (1.0 - update) * candidate


class OriginalGRU(nn.Module):
    """Two-layer or one-layer original-equation GRU with explicit length masks."""
    def __init__(self, width: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList([OriginalGRUCell(width, width)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.width = width

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return last-layer hidden state, (batch, width)."""
        B, T, _ = x.shape
        current = x
        for layer_idx, cell in enumerate(self.layers):
            hidden = current.new_zeros(B, self.width)
            out = []
            for i in range(T):
                candidate = cell(current[:, i], hidden)
                hidden = torch.where((i < lengths).unsqueeze(-1), candidate, hidden)
                out.append(hidden)
            current = torch.stack(out, dim=1)
            if layer_idx < len(self.layers) - 1:
                current = self.dropout(current)
        return hidden * (lengths > 0).to(hidden.dtype).unsqueeze(-1)


class MultiheadLinearAttentionBlock(nn.Module):
    """Causal multi-head phi=ELU+1 attention with parallel/recurrent equivalence.

    This uses the causal *inclusive* prefix (a token can attend to itself).
    No positional encoding is included here; the HistChess wrapper adds it.
    """
    def __init__(self, width: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if width % n_heads:
            raise ValueError("width must be divisible by n_heads")
        self.width, self.n_heads, self.head_dim = width, n_heads, width // n_heads
        self.norm1 = nn.LayerNorm(width)
        self.to_qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.drop = nn.Dropout(dropout)

    def attention(self, x: torch.Tensor, valid: torch.Tensor,
                  *, recurrent: bool = False) -> torch.Tensor:
        B, T, W = x.shape
        z = self.norm1(x)
        q, k, v = self.to_qkv(z).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, T, self.n_heads, self.head_dim)
                   for t in (q, k, v)]
        q, k = F.elu(q) + 1, F.elu(k) + 1
        mask = valid[:, :, None, None].to(x.dtype)
        k, v = k * mask, v * mask
        if recurrent:
            state = x.new_zeros(B, self.n_heads, self.head_dim, self.head_dim)
            key_state = x.new_zeros(B, self.n_heads, self.head_dim)
            contexts = []
            for i in range(T):
                state = state + k[:, i].unsqueeze(-1) * v[:, i].unsqueeze(-2)
                key_state = key_state + k[:, i]
                numerator = torch.einsum('bhd,bhde->bhe', q[:, i], state)
                denominator = (q[:, i] * key_state).sum(-1, keepdim=True).clamp_min(1e-6)
                contexts.append(numerator / denominator)
            context = torch.stack(contexts, dim=1)
        else:
            kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=1)
            ks = k.cumsum(dim=1)
            numerator = torch.einsum('bthd,bthde->bthe', q, kv)
            denominator = (q * ks).sum(-1, keepdim=True).clamp_min(1e-6)
            context = numerator / denominator
        return context.reshape(B, T, W) * mask.squeeze(-1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                *, recurrent: bool = False) -> torch.Tensor:
        context = self.attention(x, valid, recurrent=recurrent)
        valid_f = valid.unsqueeze(-1).to(x.dtype)
        x = x + self.drop(self.project(context)) * valid_f
        return x + self.drop(self.ff(self.norm2(x))) * valid_f


class S4DLinKernel(nn.Module):
    """ZOH-discretized complex diagonal SSM kernel, S4D-Lin initialization.

    This matches the *minimal* S4D-Lin diagonal mechanism (not HiPPO-LegS).
    State dimension is even: N/2 conjugate complex modes are stored.
    """
    def __init__(self, channels: int, state_size: int = 32,
                 dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        if state_size < 2 or state_size % 2:
            raise ValueError('S4D state_size must be an even positive integer')
        n = state_size // 2
        self.channels, self.state_size = channels, state_size
        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(math.log(dt_min), math.log(dt_max)))
        self.C = nn.Parameter(torch.view_as_real(
            torch.randn(channels, n, dtype=torch.cfloat) / math.sqrt(n)).contiguous())
        self.log_A_real = nn.Parameter(torch.full((channels, n), math.log(0.5)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n, dtype=torch.float).expand(channels, -1).clone())
        # Official minimal S4D uses zero weight decay for state-space parameters.
        for parameter in (self.log_dt, self.log_A_real, self.A_imag):
            parameter._optim = {"weight_decay": 0.0}

    def discretize(self):
        A = torch.complex(-self.log_A_real.exp(), self.A_imag)
        dt = self.log_dt.exp().unsqueeze(-1)
        # Stable ZOH for the continuous-time diagonal system B=1:
        # Abar=exp(dt*A), Bbar=expm1(dt*A)/A.
        abar = torch.exp(dt * A)
        bbar = torch.expm1(dt * A) / A
        return abar, bbar

    def forward(self, length: int):
        if length < 1: raise ValueError("length must be positive")
        abar, bbar = self.discretize()
        C = torch.view_as_complex(self.C)
        powers = torch.arange(length, device=abar.device, dtype=abar.real.dtype)
        # 2*Re handles conjugate pairs, as in the reference S4D-Lin kernel.
        return 2 * torch.einsum('cn,cnl->cl', C*bbar,
                                abar.unsqueeze(-1) ** powers).real

    def recurrent(self, u: torch.Tensor) -> torch.Tensor:
        """Reference time-stepping with the same kernel parameters; (B,C,T)."""
        abar, bbar = self.discretize()
        C = torch.view_as_complex(self.C)
        B, _, T = u.shape
        state = abar.new_zeros((B, self.channels, self.state_size//2))
        outputs = []
        for i in range(T):
            state = abar.unsqueeze(0) * state + bbar.unsqueeze(0) * u[...,i].unsqueeze(-1)
            outputs.append(2 * (state * C.unsqueeze(0)).sum(-1).real)
        return torch.stack(outputs, dim=-1)


class S4DLinBlock(nn.Module):
    """HistChess residual block with minimal reference-style S4D-Lin kernel.

    Kernel is the original S4D-Lin ZOH construction; residual/FFN and masking
    are disclosed adaptations for the HistChess architecture comparison.
    """
    def __init__(self, width: int, dropout: float = 0.1, state_size: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.kernel = S4DLinKernel(width, state_size=state_size)
        self.D = nn.Parameter(torch.randn(width))
        self.act = nn.GELU()
        self.output_linear = nn.Conv1d(width, 2*width, kernel_size=1)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*width, width))
        self.drop = nn.Dropout(dropout)

    def sequence(self, u: torch.Tensor, recurrent: bool = False) -> torch.Tensor:
        """Apply diagonal kernel plus D skip, activation and GLU on (B,C,T)."""
        if recurrent:
            y = self.kernel.recurrent(u)
        else:
            L = u.shape[-1]
            k = self.kernel(L)
            y = torch.fft.irfft(torch.fft.rfft(u, n=2*L) *
                                torch.fft.rfft(k, n=2*L).unsqueeze(0), n=2*L)[...,:L]
        y = self.act(y + u * self.D.view(1,-1,1))
        return F.glu(self.output_linear(y), dim=1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1).to(x.dtype)
        u = (self.norm1(x) * mask).transpose(1,2)
        y = self.sequence(u).transpose(1,2) * mask
        x = x + self.drop(y) * mask
        return x + self.drop(self.ff(self.norm2(x))) * mask
