import torch
import torch.nn as nn
from typing import List


class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last. 
    """

    def __init__(
        self, 
        d_model: int,
        kernel_size: int,
        **kwargs,
    ): 
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )
    
    def forward(self, x: torch.Tensor, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        l = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)
        return y 


class BaseConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_sizes: List[int]=[3, 128],
        layer_idx: int=None,
        implicit_long_conv: bool=True,
        use_proj_act: bool=True,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx=layer_idx

        self.projection = nn.Linear(self.d_model,  self.d_model)
        self.use_proj_act = use_proj_act
        
        assert len(kernel_sizes) == 2, "kernel_sizes must be a list of length 2"

        # prepare convolution
        convs = []
        for kernel_size in kernel_sizes:
            convs.append(ShortConvolution(d_model, kernel_size=kernel_size))
        self.conv1, self.conv2 = convs
        
    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """

        u_conv1 = self.conv1(u)
        u_conv1 = nn.functional.silu(u_conv1)
        u_conv2 = self.conv2(u_conv1)

        u_proj = self.projection(u)
        if self.use_proj_act:
            u_proj = nn.functional.silu(u_proj)
        
        y = u_conv2.to(torch.float32) * u_proj.to(torch.float32)
        return y.to(u.dtype) 
    