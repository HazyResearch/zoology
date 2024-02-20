import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
import wandb


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        seqlen = qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class SelectiveLookups(nn.Module):
    """"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.selecting = nn.Linear(d_model, 1)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv)
        attn_output = self.out_proj(rearrange(context, "... h d -> ... (h d)"))

        selection = torch.softmax(self.selecting(x), dim=1)

        # selection = self.selecting(x)
        # selection = (selection / selection.sum(dim = 1, keepdim=True)) * math.sqrt(x.shape[-1])

        if wandb.run.step % 100 == 0:
            wandb.log(
                {
                    "selection/mean": selection.mean(),
                    "selection/std": selection.std(),
                    "selection/max": selection.max(),
                    "selection/significant": (selection > 0.01).float().mean(),
                    "selection/hist": wandb.Histogram(
                        np_histogram=np.histogram(
                            selection[0].squeeze().detach().cpu().numpy(), bins=20
                        )
                    ),
                },
                commit=False,
            )

        if self.training:
            y = attn_output * selection + x
        else:
            _, l, d = x.shape
            k = math.ceil(math.sqrt(l))
            out = torch.topk(selection, k=k, dim=1, sorted=False)
            src = torch.gather(
                attn_output * selection, dim=1, index=out.indices.repeat(1, 1, d)
            )
            y = x.scatter_add_(dim=1, index=out.indices.repeat(1, 1, d), src=src)

        return y


class SigmoidLookups(nn.Module):
    """"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        selection_penalty: float = 0.001,
        n_lookups: int = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.selecting = nn.Linear(d_model, 1, bias=False)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.selection_penalty = selection_penalty
        self.n_lookups = n_lookups

        self._auxiliary_loss = 0.0

    def forward(self, x: torch.Tensor):
        """"""
        _, l, d = x.shape
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv)
        attn_output = self.out_proj(rearrange(context, "... h d -> ... (h d)"))

        selection = torch.sigmoid(self.selecting(x))

        # selection = self.selecting(x)
        # selection = (selection / selection.sum(dim = 1, keepdim=True)) * math.sqrt(x.shape[-1])

        if wandb.run.step % 100 == 0:
            wandb.log(
                {
                    "selection/mean": selection.mean(),
                    "selection/std": selection.std(),
                    "selection/max": selection.max(),
                    "selection/significant": (selection > 0.01).float().mean(),
                    "selection/hist": wandb.Histogram(
                        np_histogram=np.histogram(
                            selection[0].squeeze().detach().cpu().numpy(), bins=20
                        )
                    ),
                    "selection/weight": self.selecting.weight.data.squeeze(),
                },
                commit=False,
            )

        if self.training:
            n_lookups = math.sqrt(l) if self.n_lookups is None else self.n_lookups
            y = attn_output * selection + x
            self._auxiliary_loss = (
                torch.relu(selection.mean() - (n_lookups / l))
                * self.selection_penalty
            )

            y = y  # + torch.randn_like(x) * (torch.rand(1).to(device=x.device) > 0.8).float() * 1 *  selection
        else:
            k = math.ceil(math.sqrt(l)) if self.n_lookups is None else self.n_lookups
            out = torch.topk(selection, k=k, dim=1, sorted=False)
            src = torch.gather(
                attn_output * selection, dim=1, index=out.indices.repeat(1, 1, d)
            )
            y = x.scatter_add_(dim=1, index=out.indices.repeat(1, 1, d), src=src)

        return y

    def get_auxiliary_loss(self):
        loss = self._auxiliary_loss
        self._auxiliary_loss = 0.0
        return loss


class SMA(nn.Module):
    """"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        alpha: float = 0.3,
        layer_idx: int = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.selection_proj = nn.Linear(d_model, 2)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(d_model) * alpha)

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv)
        attn_output = self.out_proj(rearrange(context, "... h d -> ... (h d)"))

        selection = torch.softmax(self.selection_proj(x) / self.temperature, dim=-1)[
            ..., 1:
        ]

        # selection = self.selecting(x)
        # selection = (selection / selection.sum(dim = 1, keepdim=True)) * math.sqrt(x.shape[-1])

        if wandb.run.step % 100 == 0:
            wandb.log(
                {
                    "selection/mean": selection.mean(),
                    "selection/std": selection.std(),
                    "selection/max": selection.max(),
                    "selection/temperature": self.temperature.item(),
                    "selection/significant": (selection > 0.01).float().mean(),
                    "selection/hist": wandb.Histogram(
                        np_histogram=np.histogram(
                            selection[0].squeeze().detach().cpu().numpy(), bins=20
                        )
                    ),
                },
                commit=False,
            )

        if self.training:
            y = attn_output * selection + x
        else:
            _, l, d = x.shape
            k = math.ceil(math.sqrt(l))
            out = torch.topk(selection, k=k, dim=1, sorted=False)
            src = torch.gather(
                attn_output * selection, dim=1, index=out.indices.repeat(1, 1, d)
            )
            y = x.scatter_add_(dim=1, index=out.indices.repeat(1, 1, d), src=src)

        return y
