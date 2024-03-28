import math
from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange


class BigBIRDSelfAttention(nn.Module):
    def __init__(self, block_size=128, attention_dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        batch_size, l, d_model = qkv.shape[0], qkv.shape[1], qkv.shape[2]
        self.window_size = 3 * self.block_size
        self.random_toks = 3 * self.block_size
        self.global_toks = 2 * self.block_size
        q, k, v = qkv.split(d_model // 3, dim=-1)
        
        # compute attention for each chunk
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, softmax_scale * k.transpose(-1, -2))

        # mask out the sliding window
        mask_positions = torch.zeros(l, l)
        for i in range(l):
            mask_positions[i, 0 : max(0, i - self.window_size)] =-10000.0
            mask_positions[i, i + 1 :] = -10000.0
        mask_positions = mask_positions.to(scores.device)

        # add back the strip of attention for the first strip
        mask_positions[:, :self.global_toks] = 0.0

        # add back random positions in the l x l matrix
        positions = torch.randint(l-1, (self.random_toks,)), torch.randint(l-1, (self.random_toks,))
        positions = positions[0][positions[0] > positions[1]], positions[1][positions[0] > positions[1]] # take causal only
        off_positions_1 = (positions[0] + 1, positions[1] + 1)
        off_positions_2 = (positions[0] + 1, positions[1])
        off_positions_3 = (positions[0], positions[1] + 1)
        mask_positions[off_positions_1] = 0.0
        mask_positions[off_positions_2] = 0.0
        mask_positions[off_positions_3] = 0.0
        mask_positions[positions] = 0.0

        # causal masking
        causal_mask = torch.triu(torch.full((l, l), -20000.0), diagonal=1).to(scores.device)
        mask_positions = mask_positions + causal_mask

        # apply the mask and compute the output
        scores = scores + mask_positions[None, :, :]
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.matmul(attention_drop, v)
        return output


class BigBIRDAttention(nn.Module):
    def __init__(
            self, d_model, 
            block_size=128,
            bias=True, 
            dropout=0.0,
            **kwargs
        ) -> None:
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = BigBIRDSelfAttention(block_size=block_size, 
                                         attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, key_padding_mask=None, bigram_vector=None,  **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        """
        qkv = self.Wqkv(x)
        context = self.inner_attn(qkv)
        out = self.out_proj(context)
        return out

    def state_size(self, sequence_length: int=2048):
        return 2 * 7 * self.block_size * self.d_model