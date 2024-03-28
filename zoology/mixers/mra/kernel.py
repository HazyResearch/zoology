
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import time
import math
import numpy as np

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, "cuda", file) for file in src_files]
cuda_kernel = load('cuda_kernel', src_files, verbose = True)

import cuda_kernel

def sparse_max_cuda(sparse_C, indices, A_num_block, B_num_block):
    assert len(sparse_C.size()) == 4
    assert len(indices.size()) == 2
    assert sparse_C.size(2) == 32
    assert sparse_C.size(3) == 32
    
    index_vals = sparse_C.max(dim = -2).values.transpose(-1, -2)
    if not index_vals.is_contiguous():
        index_vals = index_vals.contiguous()
        
    indices = indices.int()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    max_vals, max_vals_scatter = cuda_kernel.index_max(index_vals, indices, A_num_block, B_num_block)
    max_vals_scatter = max_vals_scatter.transpose(-1, -2)[:, :, None, :]
    
    return max_vals, max_vals_scatter

def sparse_max(sparse_C, indices, A_num_block, B_num_block):
    return sparse_max_cuda(sparse_C, indices, A_num_block, B_num_block)

def sparse_mask_B(mask, indices, block_size = 32):
    # mask = [batch_size, num_block * block_size]
    # indices = [batch_size, num_block]
    
    assert len(mask.size()) == 2
    assert len(indices.size()) == 2
    assert mask.shape[0] == indices.shape[0]
    
    batch_size, seq_len = mask.shape
    num_block = seq_len // block_size
    
    batch_idx = torch.arange(indices.size(0), dtype = torch.long, device = indices.device)
    mask = mask.reshape(batch_size, num_block, block_size)
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]
    
    return mask

def sparse_to_dense(sparse_A, indices, A_num_block, block_size = 32):
    batch_size, _, _, _ = sparse_A.shape
    eyes = torch.eye(A_num_block * block_size, device = sparse_A.device).repeat(batch_size, 1, 1)
    eyes = eyes.reshape(batch_size, A_num_block, block_size, A_num_block * block_size).transpose(-1, -2)
    dense_A = sparse_dense_mm_torch(sparse_A, indices, eyes, A_num_block)
    dense_A = dense_A.transpose(-1, -2).reshape(batch_size, A_num_block * block_size, A_num_block * block_size)
    return dense_A

def mm_to_sparse_cuda(dense_A, dense_B, indices):

    # dense_A = [batch_size, A_num_block, dim, 32]
    # dense_B = [batch_size, B_num_block, dim, 32]
    # indices = [batch_size, num_block]

    assert len(dense_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2
    assert dense_A.size(3) == 32
    assert dense_B.size(3) == 32

    if not dense_A.is_contiguous():
        dense_A = dense_A.contiguous()

    if not dense_B.is_contiguous():
        dense_B = dense_B.contiguous()

    indices = indices.int()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    assert dense_A.is_contiguous()
    assert dense_B.is_contiguous()
    assert indices.is_contiguous()

    return cuda_kernel.mm_to_sparse(dense_A, dense_B, indices.int())

def mm_to_sparse_torch(dense_A, dense_B, indices, block_size = 32):

    # dense_A = [batch_size, A_num_block, dim, block_size]
    # dense_B = [batch_size, B_num_block, dim, block_size]
    # indices = [batch_size, num_block]

    assert len(dense_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2

    _, B_num_block, _, _ = dense_B.size()

    batch_idx = torch.arange(indices.size(0), dtype = torch.long, device = indices.device)
    dense_A = dense_A[batch_idx[:, None], torch.div(indices, B_num_block, rounding_mode = 'floor').long(), :, :]
    dense_B = dense_B[batch_idx[:, None], (indices % B_num_block).long(), :, :]

    sparse_C = torch.matmul(dense_B.transpose(-1, -2), dense_A)

    return sparse_C

def mm_to_sparse(dense_A, dense_B, indices, block_size = 32, cuda = True):
    batch_size, A_size, dim = dense_A.size()
    _, B_size, dim = dense_B.size()
    assert A_size % block_size == 0
    assert B_size % block_size == 0

    dense_A = dense_A.reshape(batch_size, A_size // block_size, block_size, dim).transpose(-1, -2)
    dense_B = dense_B.reshape(batch_size, B_size // block_size, block_size, dim).transpose(-1, -2)
    if block_size == 32 and dim % 8 == 0 and cuda:
        return mm_to_sparse_cuda(dense_A, dense_B, indices)
    else:
        return mm_to_sparse_torch(dense_A, dense_B, indices)

def sparse_dense_mm_cuda(sparse_A, indices, dense_B, A_num_block):

    # sparse_A = [batch_size, num_block, 32, 32]
    # indices = [batch_size, num_block]
    # dense_B = [batch_size, B_num_block, dim, 32]

    assert len(sparse_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2
    assert sparse_A.size(2) == 32
    assert sparse_A.size(3) == 32
    assert dense_B.size(3) == 32

    if not sparse_A.is_contiguous():
        sparse_A = sparse_A.contiguous()

    indices = indices.int()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    if not dense_B.is_contiguous():
        dense_B = dense_B.contiguous()

    assert sparse_A.is_contiguous()
    assert indices.is_contiguous()
    assert dense_B.is_contiguous()

    return cuda_kernel.sparse_dense_mm(sparse_A, indices, dense_B, A_num_block)

def sparse_dense_mm_torch(sparse_A, indices, dense_B, A_num_block):

    # sparse_A = [batch_size, num_block, block_size, block_size]
    # indices = [batch_size, num_block]
    # dense_B = [batch_size, B_num_block, dim, block_size]

    assert len(sparse_A.size()) == 4
    assert len(dense_B.size()) == 4
    assert len(indices.size()) == 2

    batch_size, num_block = indices.size()
    _, B_num_block, dim, block_size = dense_B.size()

    batch_idx = torch.arange(indices.size(0), dtype = torch.long, device = indices.device)
    dense_B = dense_B[batch_idx[:, None], (indices % B_num_block).long(), :, :]

    dense_C = torch.matmul(dense_B, sparse_A)
    dense_C = dense_C.reshape(batch_size * num_block, dim * block_size)
    global_idxes = (torch.div(indices, B_num_block, rounding_mode = 'floor').long() + batch_idx[:, None] * A_num_block).reshape(batch_size * num_block)
    temp = torch.zeros((batch_size * A_num_block, dim * block_size), dtype = dense_C.dtype, device = dense_C.device)
    dense_C = temp.index_add(0, global_idxes, dense_C).reshape(batch_size, A_num_block, dim, block_size)

    return dense_C

def sparse_dense_mm(sparse_A, indices, dense_B, A_num_block, block_size = 32, cuda = True):
    batch_size, B_size, dim = dense_B.size()

    assert B_size % block_size == 0
    assert sparse_A.size(2) == block_size
    assert sparse_A.size(3) == block_size

    dense_B = dense_B.reshape(batch_size, B_size // block_size, block_size, dim).transpose(-1, -2)
    if block_size == 32 and dim % 64 == 0 and cuda:
        dense_C = sparse_dense_mm_cuda(sparse_A, indices, dense_B, A_num_block)
    else:
        dense_C = sparse_dense_mm_torch(sparse_A, indices, dense_B, A_num_block)
    dense_C = dense_C.transpose(-1, -2).reshape(batch_size, A_num_block * block_size, dim)
    return dense_C

def reduce_sum_cuda(sparse_A, indices, A_num_block, B_num_block):

    # sparse_A = [batch_size, num_block, 32, 32]
    # indices = [batch_size, num_block]

    assert len(sparse_A.size()) == 4
    assert len(indices.size()) == 2
    assert sparse_A.size(2) == 32
    assert sparse_A.size(3) == 32

    if not sparse_A.is_contiguous():
        sparse_A = sparse_A.contiguous()

    indices = indices.int()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    assert sparse_A.is_contiguous()
    assert indices.is_contiguous()

    return cuda_kernel.reduce_sum(sparse_A, indices, A_num_block, B_num_block)

def reduce_sum_torch(sparse_A, indices, A_num_block, B_num_block):

    # sparse_A = [batch_size, num_block, block_size, block_size]
    # indices = [batch_size, num_block]

    assert len(sparse_A.size()) == 4
    assert len(indices.size()) == 2

    _, _, block_size, _ = sparse_A.size()
    batch_size, num_block = indices.size()

    sparse_A = sparse_A.sum(dim = 2).reshape(batch_size * num_block, block_size)

    batch_idx = torch.arange(indices.size(0), dtype = torch.long, device = indices.device)
    global_idxes = (torch.div(indices, B_num_block, rounding_mode = 'floor').long() + batch_idx[:, None] * A_num_block).reshape(batch_size * num_block)
    temp = torch.zeros((batch_size * A_num_block, block_size), dtype = sparse_A.dtype, device = sparse_A.device)
    output = temp.index_add(0, global_idxes, sparse_A).reshape(batch_size, A_num_block, block_size)

    return output

def reduce_sum(sparse_A, indices, A_num_block, B_num_block, cuda = False):
    batch_size, num_block, block_size, _ = sparse_A.size()

    # if block_size == 32 and cuda:
    #     output = reduce_sum_cuda(sparse_A, indices, A_num_block, B_num_block)
    # else:
    #     output = reduce_sum_torch(sparse_A, indices, A_num_block, B_num_block)
    output = reduce_sum_torch(sparse_A, indices, A_num_block, B_num_block)
    output = output.reshape(batch_size, A_num_block * block_size)
    return output

def transpose_indices(indices, dim_1_block, dim_2_block):
    return ((indices % dim_2_block) * dim_1_block + torch.div(indices, dim_2_block, rounding_mode='floor')).long()
