"""
Spiky feature maps based on applying exp() element- or dimension-wise
"""
from typing import Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureMap


class FullSpaceMap(nn.Module):
    """
    Project positive features to upper and lower "halfspaces"
    """
    def __init__(self, 
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12,
                 **kwargs: any):
        super().__init__()
        self.head_dim_idx = head_dim_idx
        self.eps = eps
        
    def forward(self, x: torch.Tensor, fmap: Callable = None):
        return torch.cat([x, -x], dim=self.head_dim_idx).clamp(min=self.eps)


class ExpDim(FeatureMap):
    """
    Feature maps based on applying exp() element- or dimension-wise
    """
    def __init__(self, 
                 fullspace: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        if fullspace:
            self.fs_map = FullSpaceMap(**kwargs)
        else:
            self.fs_map = nn.Identity()

    def forward(self, x: torch.Tensor):
        return torch.exp(self.fs_map(x * self.temp))


class ExpMaxNorm(ExpDim):
    def forward(self, x: torch.Tensor):
        x = x * self.temp
        x = x - x.amax(self.head_dim_idx, keepdim=True)
        return torch.exp(self.fs_map(x))

    
class SoftmaxDim(ExpDim):
    """
    Compute softmax across fullspace
    """
    def forward(self, x: torch.Tensor):
        x = x * self.temp
        return torch.cat([
            torch.softmax( x, dim=self.head_dim_idx),
            torch.softmax(-x, dim=self.head_dim_idx)
        ], dim=self.head_dim_idx).clamp(min=self.eps)


class SoftmaxDimHalfspace(ExpDim):
    def forward(self, x: torch.Tensor):
        x = x * self.temp
        return torch.softmax(x, dim=self.head_dim_idx).clamp(min=self.eps)


class ExpSumNorm(ExpDim):
    """
    Like softmax except we normalize by the denominator in torch.abs(x).softmax()
    and ignore interactions between terms with different signs

    assumes self.head_dim_idx = -1
    """
    def __init__(self, head_dim_idx: int = -1, *args, **kwargs):
        assert head_dim_idx == -1
        super().__init__(head_dim_idx=head_dim_idx, *args, **kwargs)
        
    def forward(self, x: torch.Tensor):
        b, h, l = x.shape[:3]
        x = x * self.temp  # b h l d 2
        x = (torch.stack([x >= 0, x < 0], dim=self.head_dim_idx) * 
             x.abs().softmax(dim=self.head_dim_idx).unsqueeze(-1))
        return x.view(b, h, l, -1).clamp(min=self.eps)
        # return x.flatten(start_dim=-2).clamp(min=self.eps)


class ExpSumNormHalfspace(ExpDim):
    """
    Same as above but only positive values
    """
    def __init__(self, head_dim_idx: int = -1, *args, **kwargs):
        assert head_dim_idx == -1
        super().__init__(head_dim_idx=head_dim_idx, *args, **kwargs)
        
    def forward(self, x: torch.Tensor):
        # x = x * self.temp  # b h l d 2
        return ((x >= 0) *
                x.abs().softmax(dim=self.head_dim_idx)).clamp(min=self.eps)