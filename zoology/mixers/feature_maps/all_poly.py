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
        init: str="randn",
        learnable: bool=True,
        **kwargs: any
    ):
        super().__init__(input_dim, **kwargs)
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.proj_1 = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.proj_2 = nn.Linear(self.input_dim, self.output_dim, bias=False)

        if not learnable:
            for p in self.parameters():
                p.requires_grad = False

        if init == "randn":
            nn.init.normal_(self.proj_1.weight, std=1 / np.sqrt(self.input_dim))
            nn.init.normal_(self.proj_2.weight, std=1 / np.sqrt(self.input_dim))
        elif init == "kaiming":
            pass
        elif init == "plusminus":
            self.proj_1.weight.copy_((torch.rand_like(self.proj_1.weight) > 0.5) * 2 - 1)
            self.proj_2.weight.copy_((torch.rand_like(self.proj_2.weight) > 0.5) * 2 - 1)
        else:
            raise ValueError(f"Invalid init method: {init}")
            
    def forward(self, x: torch.Tensor):
        x1 = self.proj_1(x) ** 2
        x2 = self.proj_2(x ** 2)
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)  # compute squared norm
        return (x1 + x2) / math.sqrt(2)

    def expanded_size(self):
        return self.output_dim


class SquaredMap(FeatureMap):

    def __init__(
        self, 
        input_dim: int, 
        **kwargs: any
    ):
        super().__init__(input_dim, **kwargs)
        

    def forward(self, x: torch.Tensor):
        return x * x / math.sqrt(2)

    def expanded_size(self):
        return self.input_dim

