
import torch
import torch.nn as nn
import math
import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_path)

from .attention import mra2_attention

class MRAAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        max_position_embeddings: int,
        num_heads: int=1,
        num_block_per_row: int=8,
        attention_approx_mode: str="full",
        initial_prior_first_n_blocks: int=0,
        attention_initial_prior_diagonal_n_blocks: int=0,
        attention_input_shape: tuple=None,
        layer_idx: int =None
    ):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(d_model / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.num_block = (max_position_embeddings // 32) * num_block_per_row

        self.num_block = min(self.num_block, int((max_position_embeddings // 32) ** 2))
        self.approx_mode = attention_approx_mode
        self.initial_prior_first_n_blocks = initial_prior_first_n_blocks 
        self.initial_prior_diagonal_n_blocks = attention_initial_prior_diagonal_n_blocks 
        self.input_shape = attention_input_shape

    def extra_repr(self):
        rep = [
            f'num_block = {self.num_block}',
            f'approx_mode = {self.approx_mode}',
            f'initial_prior: first_n_blocks = {self.initial_prior_first_n_blocks}',
            f'initial_prior: diagonal_n_blocks = {self.initial_prior_diagonal_n_blocks}',
            f'input_shape = {self.input_shape}',
        ]
        return "\n".join(rep)

    def forward(self, X):

        batch_size, seq_len, dim = X.shape

        mask = None
        #torch.zeros((batch_size, seq_len, seq_len), device=X.device, dtype=X.dtype)
        # mask = -1000.0 * (1.0 - mask.float())  # https://github.com/mlpen/mra-attention/blob/7e08afeb6f34dcb15a7812f45d66e42d9a21bbc5/src/roberta/models/postnorm.py#L84C26-L84C84

        Q = self.split_heads(self.query(X))
        K = self.split_heads(self.key(X))
        V = self.split_heads(self.value(X))

        attn_out = mra2_attention(
            Q.float(), K.float(), V.float(), mask.float(), self.num_block,
            approx_mode = self.approx_mode,
            initial_prior_first_n_blocks = self.initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks = self.initial_prior_diagonal_n_blocks
        ).to(X.dtype)

        attn_out = self.combine_heads(attn_out)

        if self.input_shape is not None:
            attn_out = attn_out.reshape(batch_size, H // 4, W // 8, 4, 8, dim)
            attn_out = attn_out.permute(0, 1, 3, 2, 4, 5)
            attn_out = attn_out.reshape(batch_size, seq_len, dim)

        return attn_out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.all_head_size)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_attention_heads, self.attention_head_size)
        X = X.transpose(1, 2)
        return X

    def state_size(self, sequence_length: int=2048):
        return 0