
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, curr_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from autograd_function import SampledDenseMM, SparseDenseMM, ReduceSum
from kernel import sparse_max, sparse_mask_B

def get_low_resolution_logit(Q, K, block_size, mask = None, V = None):
    batch_size, seq_len, head_dim = Q.size()

    num_block_per_row = seq_len // block_size

    V_hat = None
    if mask is not None:
        token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim = -1)
        Q_hat = Q.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim = -2) / (token_count[:, :, None] + 1e-6)
        K_hat = K.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim = -2) / (token_count[:, :, None] + 1e-6)
        if V is not None:
            V_hat = V.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim = -2) / (token_count[:, :, None] + 1e-6)
    else:
        token_count = block_size * torch.ones(batch_size, num_block_per_row, dtype = torch.float, device = Q.device)
        Q_hat = Q.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim = -2)
        K_hat = K.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim = -2)
        if V is not None:
            V_hat = V.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim = -2)

    low_resolution_logit = torch.matmul(Q_hat, K_hat.transpose(-1, -2)) / math.sqrt(head_dim)

    low_resolution_logit_row_max = low_resolution_logit.max(dim = -1, keepdims = True).values

    if mask is not None:
        low_resolution_logit = low_resolution_logit - 1e4 * ((token_count[:, None, :] * token_count[:, :, None]) < 0.5).float()

    return low_resolution_logit, token_count, low_resolution_logit_row_max, V_hat

def get_block_idxes(low_resolution_logit, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks):
    batch_size, total_blocks_per_row, _ = low_resolution_logit.shape

    if initial_prior_diagonal_n_blocks > 0:
        offset = initial_prior_diagonal_n_blocks // 2
        temp_mask = torch.ones(total_blocks_per_row, total_blocks_per_row, device = low_resolution_logit.device)
        diagonal_mask = torch.tril(torch.triu(temp_mask, diagonal = -offset), diagonal = offset)
        low_resolution_logit = low_resolution_logit + diagonal_mask[None, :, :] * 5e3

    if initial_prior_first_n_blocks > 0:
        low_resolution_logit[:, :initial_prior_first_n_blocks, :] = low_resolution_logit[:, :initial_prior_first_n_blocks, :] + 5e3
        low_resolution_logit[:, :, :initial_prior_first_n_blocks] = low_resolution_logit[:, :, :initial_prior_first_n_blocks] + 5e3

    top_k_vals = torch.topk(low_resolution_logit.reshape(batch_size, -1), num_blocks, dim = -1, largest = True, sorted = False)
    indices = top_k_vals.indices
    
    if approx_mode == "full":
        threshold = top_k_vals.values.min(dim = -1).values
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    elif approx_mode == "sparse":
        high_resolution_mask = None
    else:
        raise Exception()

    return indices, high_resolution_mask

def mra2_attention(
    Q, K, V, mask, num_blocks,
    approx_mode,
    block_size = 32,
    initial_prior_first_n_blocks = 0,
    initial_prior_diagonal_n_blocks = 0
):
    batch_size, num_head, seq_len, head_dim = Q.size()
    meta_batch = batch_size * num_head

    assert seq_len % block_size == 0
    num_block_per_row = seq_len // block_size

    Q = Q.reshape(meta_batch, seq_len, head_dim)
    K = K.reshape(meta_batch, seq_len, head_dim)
    V = V.reshape(meta_batch, seq_len, head_dim)

    # SE (03/27): Force no mask
    # mask = None if torch.all(mask == 1).item() else mask[:, None, :].repeat(1, num_head, 1).reshape(meta_batch, seq_len)
    mask = None

    if mask is not None:
        Q = Q * mask[:, :, None]
        K = K * mask[:, :, None]
        V = V * mask[:, :, None]

    if approx_mode == "full":
        low_resolution_logit, token_count, low_resolution_logit_row_max, V_hat = get_low_resolution_logit(Q, K, block_size, mask, V)
    elif approx_mode == "sparse":
        with torch.no_grad():
            low_resolution_logit, token_count, low_resolution_logit_row_max, _ = get_low_resolution_logit(Q, K, block_size, mask)
    else:
        raise Exception()

    with torch.no_grad():
        low_resolution_logit_normalized = low_resolution_logit - low_resolution_logit_row_max
        indices, high_resolution_mask = get_block_idxes(low_resolution_logit_normalized, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks)

    high_resolution_logit = SampledDenseMM.operator_call(Q, K, indices, block_size = block_size) / math.sqrt(head_dim)
    max_vals, max_vals_scatter = sparse_max(high_resolution_logit, indices, num_block_per_row, num_block_per_row)
    high_resolution_logit = high_resolution_logit - max_vals_scatter
    if mask is not None:
        high_resolution_logit = high_resolution_logit - 1e4 * (1 - sparse_mask_B(mask, indices)[:, :, :, None])
    high_resolution_attn = torch.exp(high_resolution_logit)
    high_resolution_attn_out = SparseDenseMM.operator_call(high_resolution_attn, indices, V, num_block_per_row)
    high_resolution_normalizer = ReduceSum.operator_call(high_resolution_attn, indices, num_block_per_row, num_block_per_row)

    if approx_mode == "full":
        low_resolution_attn = torch.exp(low_resolution_logit - low_resolution_logit_row_max - 1e4 * high_resolution_mask) * token_count[:, None, :]

        low_resolution_attn_out = torch.matmul(low_resolution_attn, V_hat)[:, :, None, :].repeat(1, 1, block_size, 1).reshape(meta_batch, seq_len, head_dim)
        low_resolution_normalizer = low_resolution_attn.sum(dim = -1)[:, :, None].repeat(1, 1, block_size).reshape(meta_batch, seq_len)

        log_correction = low_resolution_logit_row_max.repeat(1, 1, block_size).reshape(meta_batch, seq_len) - max_vals
        if mask is not None:
            log_correction = log_correction * mask

        low_resolution_corr = torch.exp(log_correction * (log_correction <= 0).float())
        low_resolution_attn_out = low_resolution_attn_out * low_resolution_corr[:, :, None]
        low_resolution_normalizer = low_resolution_normalizer * low_resolution_corr

        high_resolution_corr = torch.exp(- log_correction * (log_correction > 0).float())
        high_resolution_attn_out = high_resolution_attn_out * high_resolution_corr[:, :, None]
        high_resolution_normalizer = high_resolution_normalizer * high_resolution_corr

        attn = (high_resolution_attn_out + low_resolution_attn_out) / (high_resolution_normalizer[:, :, None] + low_resolution_normalizer[:, :, None] + 1e-6)

    elif approx_mode == "sparse":
        attn = high_resolution_attn_out / (high_resolution_normalizer[:, :, None] + 1e-6)
    else:
        raise Exception()

    if mask is not None:
        attn = attn * mask[:, :, None]

    attn = attn.reshape(batch_size, num_head, seq_len, head_dim)

    return attn
