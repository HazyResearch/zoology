import torch
from torch import nn
import math

def fft_conv(u: torch.Tensor, k: torch.Tensor):
    """
    Args:
        u (torch.Tensor): Float tensor of shape (batch_size, d_model, seq_len)
        k (torch.Tensor): Float tensor of shape (d_model, l_max)
    Return: 
        y (torch.Tensor): Float tensor of shape (batch_size, d_model, seq_len)
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return y

class BaseConv(torch.nn.Module):

    def __init__(self, d_model: int, l_max: int, **kwargs):
        super().__init__()
        self.d_model, l_max = d_model, l_max, 
        self.projection = torch.nn.Linear(self.d_model,  self.d_model)
        self.filter = torch.nn.Parameter(torch.randn(self.d_model, l_max), requires_grad=True)      

    def forward(self, u: torch.Tensor):
        """
        Args:
            u (torch.Tensor): Float tensor of shape (batch_size, seq_len, d_model)
        Return:
            y (torch.Tensor): Float tensor of shape (batch_size, seq_len, d_model)
        """
        u_conv = fft_conv(u.transpose(1, 2), self.filter).transpose(1, 2)
        u_proj = self.projection(u)
        y = u_conv * u_proj
        return y + u



class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()
        t = torch.linspace(0, 1, seq_len)[None, :, None]  # 1, L, 1
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]

class BaseImplicitConv(nn.Module):
    """
    BaseConv with implicit filter parameterized by an MLP.

    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
        d_emb (int, optional): The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine). Defaults to 3.
        d_hidden (int, optional): The number of features in the hidden layer of the MLP. Defaults to 16.
    """

    def __init__(self, d_model: int, l_max: int, d_emb: int=3, d_hidden: int = 16,):
        """
        Long convolution with implicit filter parameterized by an MLP.
        """
        super().__init__()
        self.pos_emb = PositionalEmbedding(d_emb, l_max)
        self.filter_mlp = nn.Sequential(nn.Linear(d_emb, d_hidden), torch.nn.ReLU(), nn.Linear(d_hidden, d_model))
        self.projection = torch.nn.Linear(d_model, d_model)


    def forward(self, u: torch.Tensor, *args, **kwargs):
        """
        Args:
            u (torch.Tensor): Float tensor of shape (batch_size, seq_len, d_model)
        Return:
            y (torch.Tensor): Float tensor of shape (batch_size, seq_len, d_model)
        """
        filter = self.filter_mlp(self.pos_emb(u.shape[1])).transpose(1, 2)
        u_conv = fft_conv(u.transpose(1, 2), filter).transpose(1, 2).to(dtype=u.dtype)
        u_proj = self.projection(u)
        y = u_conv * u_proj
        return y + u