import math
import numpy as np
import torch
from torch import nn

from .base import FeatureMap
import opt_einsum as oe


class AllPolyMap(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int=None,
        **kwargs: any
    ):
        super().__init__(input_dim, **kwargs)
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.proj_1 = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.proj_2 = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        x1 = self.proj_1(x)
        x2 = self.proj_2(x * x)
        return (x1 * x1 + x2) / math.sqrt(2)

    def expanded_size(self):
        return self.output_dim