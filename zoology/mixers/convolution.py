import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
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

    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)
    

class ShortConvolution(nn.Module):

    def __init__(
        self, 
        d_model: int,
        kernel_size: int
    ): 
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        l = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)
        return y 



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


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
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
        self.z = nn.Parameter(z, requires_grad=False)
        self.t = nn.Parameter(t, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class LongConvolution(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
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
        self.filter = nn.Parameter(torch.randn(self.d_model, l_max), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)



class ExponentialModulation(nn.Module):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        self.deltas = nn.Parameter(torch.linspace(min_decay, max_decay, d_model)[None, None], requires_grad=False)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x

class ImplicitLongConvolution(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        emb_dim: int=3,  # dim of input to MLP, augments with positional encoding
        order: int=16,  # width of the implicit MLP
        dropout=0.0,
        frequency: int=1,  # frequency of periodic activations
        bias: bool=True,
        num_inner_mlps: int=2,
        normalized: bool=False,
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
        self.use_bias = bias

        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=frequency)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, l_max)

        # uses a variable number of inner linear layers
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

        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized

    def filter(self, l: int, *args, **kwargs):
        z, t = self.pos_emb(l)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)

