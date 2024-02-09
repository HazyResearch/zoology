import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingSelfAttention(nn.Module):
    def __init__(self, block_size=256, attention_dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        batch_size, seqlen, d_model = qkv.shape[0], qkv.shape[1], qkv.shape[2]
        q, k, v = qkv.split(d_model // 3, dim=-1)
        
        # compute attention for each chunk
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, softmax_scale * k.transpose(-1, -2))

        # only keep the previous block size elements - also add causal mask
        mask_positions = torch.zeros(seqlen, seqlen)
        for i in range(seqlen):
            mask_positions[i, 0 : max(0, i - self.block_size)] =-10000.0
            mask_positions[i, i + 1 :] = -10000.0
        mask_positions = mask_positions.to(scores.device)

        scores = scores + mask_positions[None, :, :]
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.matmul(attention_drop, v)
        return output


class SlidingAttn(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        block_size=256,
                 bias=True, 
                 dropout=0.0,
                    **kwargs
                 ) -> None:
        super().__init__()
        self.d_model = d_model
        print(f"SlidingAttn: d_model={d_model}, block_size={block_size}")
        import time
        time.sleep(10)
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = SlidingSelfAttention(block_size=block_size, 
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

    def state_size(self, sequence_length: int):
        return self.inner_attn.block_size * self.d_model * 2
    
