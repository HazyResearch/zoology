# Copyright (c) 2023, Tri Dao.

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

import sys
from zoology.mixers.scatterbrain.feature_maps_sb import SBPerformerFeatures

import sys
sys.path.append('/var/cr01_data/sabri/code/based/train')
from csrc import causal_dot_product 

class GlobalPerformer(nn.Module):
    def __init__(self, dim_heads=0, nb_features=0, softmax_scale=1.0, window_size=0):
        super().__init__()
        self.feature_map = SBPerformerFeatures(dim_heads, nb_features, ortho_scaling=0,
                                               softmax_temp=softmax_scale, eps=1e-4)
        self.softmax_scale = softmax_scale
        self.local_context = window_size

    def forward(self, query, key, value, key_padding_mask=None):

        mask_val = -10000.0
        self.feature_map.new_feature_map(query.device)
        B, H, L, D = query.shape
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query) 
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        global_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        
        global_out_cumsum = torch.einsum('...nm,...nm->...n', q_prime, k_prime.cumsum(dim=-2))
        global_out = causal_dot_product(
            q_prime.to(torch.float32), 
            k_prime.to(torch.float32), 
            value.to(torch.float32), 
        ).to(query.dtype)

        # compute SWA output on featurized QK
        scores = torch.einsum("b h m d, b h n d -> b h m n", q_prime, k_prime) * self.softmax_scale    
        mask_positions = torch.zeros(L, L)
        for i in range(L):
            mask_positions[i, 0 : max(0, i - self.local_context)] = mask_val
            mask_positions[i, i + 1 :] = mask_val
        mask_positions = mask_positions.to(scores.device)
        print(f"{scores.shape=}, {mask_positions.shape=}")
        global_qk = scores + mask_positions[None, :, :]

        # local_dot_product fills in -1e24 for invalid locations. We want to set them to zero.
        global_qk = global_qk.masked_fill(global_qk <= 0, 0.0)
        assert torch.all(global_qk >= 0)

        return global_out, global_out_cumsum, global_scale, global_qk


class SelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.1, window_size=None, dim_heads=None, nb_features=24):
        super().__init__()
        self.causal = True
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

        # scatterbrain
        self.get_global = GlobalPerformer(dim_heads=dim_heads, nb_features=nb_features, softmax_scale=softmax_scale, window_size=window_size)
        self.local_context = window_size
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, qkv, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        query, key, value = qkv.unbind(dim=2)
        mask_val = -10000.0

        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        Bk, Tk, Hk, Ek = key.shape
        Bv, S, Hv, D = value.shape
        assert Hv == Hk, "The number of heads in the key and value tensor must be the same."
        assert S == Tk, "The sequence length of the key and value tensor must be the same."
        softmax_temp = 1 / math.sqrt(E)

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        # Global attn
        global_out, global_out_cumsum, global_scale, global_qk = self.get_global(query, key, value)

        # Local attn (SWA)
        scores = torch.einsum("b h m d, b h n d -> b h m n", query, key) * self.softmax_scale    
        mask_positions = torch.zeros(T, T)
        for i in range(T):
            mask_positions[i, 0 : max(0, i - self.local_context)] = mask_val
            mask_positions[i, i + 1 :] = mask_val
        mask_positions = mask_positions.to(scores.device)
        local_qk = scores + mask_positions[None, :, :]
        
        # Compute the weightings
        QK_lse = torch.logsumexp(local_qk, dim=-1, keepdim=True)
        global_qk_sum = global_qk.sum(dim=-1, keepdim=True)
        global_log_normalization = torch.log(
            (rearrange(global_out_cumsum, 'b h s -> b h s 1') - global_qk_sum).clamp_min_(1e-24)
        ) + global_scale
        log_normalization = torch.logaddexp(QK_lse, global_log_normalization)
        global_prime_scale = torch.exp(global_scale - global_log_normalization)
        
        # Compute local output, after removing double counting effect
        local_score = self.dropout(torch.softmax(local_qk - log_normalization, dim=-1)) - global_qk * global_prime_scale
        # out_local = local_weighted_average(   # breaks causality from default scatterbrain impl.
        #     local_score.to(torch.float32), 
        #     value.to(torch.float32)
        # ).to(query.dtype)
        out_local = torch.matmul(local_score, value)

        # combine
        out = out_local + global_out * global_prime_scale
        return rearrange(out, 'b h t d -> b t h d')


class SBLocalAttention(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        d_model,
        num_heads,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.1,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        device=None,
        dtype=None,
        feature_dim=24, 
        **kwargs,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        causal = True
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)

        softmax_scale = 1 / math.sqrt(self.head_dim)
        self.feature_dim = feature_dim
        self.inner_attn = SelfAttention(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
            window_size=window_size,
            dim_heads=self.head_dim,
            nb_features=feature_dim,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)
    

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None

        seqlen_offset = 0
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if self.num_heads_kv == self.num_heads:
            qkv = self.Wqkv(x)
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            print(f"qkv shape: {qkv.shape}")
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(
                    qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            context = self.inner_attn(qkv)
        else:
            assert 0, print("Wrong codepath.")
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out



    def state_size(self, sequence_length: int=2048):
        return (
            self.inner_attn.local_context * self.d_model * 2 +  # sliding window attention 
            self.num_heads_kv * self.head_dim * self.feature_dim +  # numerator
            self.num_heads_kv * self.feature_dim  # denominator
        )
