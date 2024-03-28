
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import time
import math
import numpy as np

from kernel import mm_to_sparse, sparse_dense_mm, reduce_sum, transpose_indices

class SampledDenseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_A, dense_B, indices, block_size):
        sparse_AB = mm_to_sparse(dense_A, dense_B, indices, block_size)
        ctx.save_for_backward(dense_A, dense_B, indices)
        ctx.block_size = block_size
        return sparse_AB

    @staticmethod
    def backward(ctx, grad):
        dense_A, dense_B, indices = ctx.saved_tensors
        block_size = ctx.block_size
        A_num_block = dense_A.size(1) // block_size
        B_num_block = dense_B.size(1) // block_size
        indices_T = transpose_indices(indices, A_num_block, B_num_block)
        grad_B = sparse_dense_mm(grad.transpose(-1, -2), indices_T, dense_A, B_num_block)
        grad_A = sparse_dense_mm(grad, indices, dense_B, A_num_block)
        return grad_A, grad_B, None, None

    @staticmethod
    def operator_call(dense_A, dense_B, indices, block_size = 32):
        return SampledDenseMM.apply(dense_A, dense_B, indices, block_size)

class SparseDenseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_A, indices, dense_B, A_num_block):
        sparse_AB = sparse_dense_mm(sparse_A, indices, dense_B, A_num_block)
        ctx.save_for_backward(sparse_A, indices, dense_B)
        ctx.A_num_block = A_num_block
        return sparse_AB

    @staticmethod
    def backward(ctx, grad):
        sparse_A, indices, dense_B = ctx.saved_tensors
        A_num_block = ctx.A_num_block
        B_num_block = dense_B.size(1) // sparse_A.size(-1)
        indices_T = transpose_indices(indices, A_num_block, B_num_block)
        grad_B = sparse_dense_mm(sparse_A.transpose(-1, -2), indices_T, grad, B_num_block)
        grad_A = mm_to_sparse(grad, dense_B, indices)
        return grad_A, None, grad_B, None

    @staticmethod
    def operator_call(sparse_A, indices, dense_B, A_num_block):
        return SparseDenseMM.apply(sparse_A, indices, dense_B, A_num_block)

class ReduceSum():
    @staticmethod
    def operator_call(sparse_A, indices, A_num_block, B_num_block):
        return reduce_sum(sparse_A, indices, A_num_block, B_num_block)
