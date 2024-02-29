"""
Feature map for CosFormer from cosFormer: Rethinking Softmax in Attention
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


from .base import FeatureMap


class CosFormerFeatureMap(FeatureMap):
    """
    Code from https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
    """
    def __init__(self, act_fun = 'relu', 
                 *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
        self.n_heads = None
        self.head_dim = None
        self.act_fun = self.get_act_fun(act_fun=act_fun)
        
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)
    
    def get_act_fun(self, act_fun = 'relu'):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        
    def forward(self, x: torch.Tensor):
        b, h, l, d = x.shape
        if self.head_dim is None:
            self.head_dim = d
        if self.n_heads is None:
            self.n_heads = h
        
        x = self.act_fun(x)
        
        x = rearrange(x, 'b h l d -> (b h) l d')

        # cos transform
        m = x.shape[1]
        tgt_len = m
        
        weight_index = self.get_index(m).to(x)
        x = torch.cat([x * torch.sin(weight_index[:, :tgt_len, :] / m), 
                       x * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        return rearrange(x, '(b h) l d -> b h l d', b=b)

    def expanded_size(self):
        return 2 * self.input_dim