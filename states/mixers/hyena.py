
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import validate_call

from einops import rearrange


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x



class Filter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        num_heads: int = 1,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model 
        self.emb_dim = emb_dim 
        self.seq_len = seq_len 
        self.modulate = modulate
        self.num_heads = num_heads
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            self.implicit_filter = [
                nn.Linear(emb_dim, order),
                act,
            ]
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            # final linear layer
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            self.implicit_filter = nn.Sequential(*self.implicit_filter)
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, d_model, bias=False),
            )

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            # [MP] Currently does not work if k is None as the filter
            # comes in L, D instead of D, L
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias


        y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y.to(dtype=x.dtype)


class Hyena(nn.Module):
    NUM_PROJECTIONS = 3 
    
    @validate_call
    def __init__(
        self,
        d_model: int,
        l_max: int,
        filter_order: int=64,
        num_heads: int=1,
        num_blocks: int=1,
        outer_mixing: bool=False,
        dropout: float=0.0,
        filter_dropout: float=0.0,
        short_filter_order: int=3,
        return_state: bool=False,
        bidirectional: bool=False,
        layer_idx: int=None,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            return_state: (bool): whether to return a state
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"Model dimension {d_model} must be divisible by num heads {num_heads}"
        assert (
            l_max % num_blocks == 0
        ), f"Maximum signal length {l_max} must be divisible by block dimension {num_blocks}"
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        self.d_model=d_model
        self.l_max=l_max
        self.num_heads=num_heads
        self.block_dim=block_dim
        self.head_dim=head_dim
        self.filter_order=filter_order
        self.short_filter_order=short_filter_order
        self.num_blocks=num_blocks
        self.filter_dropout=filter_dropout
        self.outer_mixing=outer_mixing
        self.return_state=return_state
        
        self.dropout = nn.Dropout(dropout)

        # setup projections 
        self.in_proj = nn.Linear(self.d_model, self.NUM_PROJECTIONS * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.bidirectional = bidirectional

        total_width = self.d_model * self.NUM_PROJECTIONS

        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1,
        )

        if "channels" not in filter_args:
            filter_args["channels"] = 1
        self.filter_fn = Filter(
            self.head_dim,
            order=self.filter_order,
            seq_len=self.l_max,
            dropout=self.filter_dropout,
            bidirectional=self.bidirectional,
            l_max=self.l_max,
            **filter_args,
        )


    def forward(self, u, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        l = u.size(1)
        assert l <= self.l_max, f"Input length {l} exceeds maximum length {self.max_l}"

        # in projection
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        # short filter
        uc = self.short_filter(u)[..., :l]

        uc = rearrange(
            uc,
            "b (ho v) (z l) -> b ho v z l",
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * self.NUM_PROJECTIONS,
        )

        x1, x2, v = uc.split(self.d_model, dim=2)

        # pre-gating
        v = v * x1
        v = self.dropout(v) 

        # long convolution
        if self.bidirectional:
            # print(f"self.bidirectional: {self.bidirectional}")
            k_rev = self.filter_fn.filter_rev(l, device=u.device)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
        else:
            k_rev = None
        k = self.filter_fn.filter(l, device=u.device)
        k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default
        v = self.filter_fn(v, l, k=k, k_rev=k_rev, bias=self.filter_fn.bias[None, :, None])
        
        # post-gating
        v = v * x2

        y = rearrange(
            v,
            "b h v z l -> b (z l) (h v)",
            z=self.num_blocks,
            h=self.num_heads,
        )
        y = self.out_proj(y)

        if self.return_state:
            return y, None
        return y

    def state_size(self, sequence_length: int=2048) -> int:
        return self.d_model * sequence_length
