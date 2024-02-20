import math
import numpy as np
import torch

from .base import FeatureMap
import opt_einsum as oe


class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    # Running these in parallel
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([torch.ones(x[..., :1].shape).to(x.device), 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)
        
    def forward_mem_save(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x) s.t. f(x)^T f(x') = 1 + x^Tx' + (x^Tx')^2 / 2
        -> Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        # Slow but memory-saving way to compute 2nd-order terms; how do w/o outer-product first?
        x2  = oe.contract('...m,...n->...mn', x, x) / self.rd
        x2d = torch.diagonal(x2, dim1=-2, dim2=-1) / self.r2
        x2  = x2[..., self.tril_indices[0], self.tril_indices[1]]
        x   = torch.cat([torch.ones(x[..., :1].shape).to(x.device), 
                         x / self.rrd, x2d, x2], dim=-1)
        return x 

    def expanded_size(self):
        return (self.input_dim) * (self.input_dim + 1) / 2 + self.input_dim + 1