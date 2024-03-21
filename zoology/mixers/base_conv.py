
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import validate_call
from einops import rearrange

from .convolution import ImplicitLongConvolution, ShortConvolution, LongConvolution

class BaseConv(nn.Module):

    @validate_call
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_size: Union[int, List[int]]=3,
        layer_idx: int=None,
        implicit_long_conv: bool=True,
        use_act=False,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx=layer_idx

        self.projection = nn.Linear(self.d_model,  self.d_model)
        self.use_act = use_act
        if self.use_act:
            self.act = nn.SiLU() 
        
        # support for different kernel sizes per layer
        if isinstance(kernel_size, List):
            if layer_idx is  None or layer_idx >= len(kernel_size):
                raise ValueError("kernel_size must be an int or a list of ints with length equal to the number of layers")
            kernel_size = kernel_size[layer_idx]

        # prepare convolution
        if kernel_size == -1:
            conv = ImplicitLongConvolution if implicit_long_conv else LongConvolution
            self.conv = conv(d_model, l_max=l_max)
        else:
            self.conv = ShortConvolution(d_model, kernel_size=kernel_size)

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u_conv = self.conv(u)
        u_proj = self.projection(u)
        if self.use_act:
            y = self.act(u_conv) * self.act(u_proj)
        else:
            y = u_conv * u_proj
        return y + u

    def state_size(self, sequence_length: int):
        return self.conv.state_size(sequence_length=sequence_length)