
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .hyena import Filter
from .convolution import ImplicitLongConvolution, ShortConvolution



class BaseConvOld(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        filter_order=64,
        outer_mixing=False,
        dropout=0.0,
        filter_dropout=0.0,
        short_filter_order=3,
        activation="id",
        return_state=False,
        num_heads=1,

        layer_idx=None,

        **filter_args,
    ):
        super().__init__()
        
        # hard-code the heads to be 1
        # num_heads=1
        num_blocks=1
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads
        head_dim = ( d_model // 2 ) // num_heads
        self.num_heads = num_heads

        self.d_model = d_model
        self.l_max = l_max
        self.num_heads = num_heads
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.num_blocks = num_blocks
        self.filter_dropout = filter_dropout
        self.outer_mixing = outer_mixing
        self.activation = activation
        self.return_state = return_state
        
        self.dropout = nn.Dropout(dropout)

        self.layer_idx=layer_idx
        self.in_proj = nn.Linear(self.d_model,  self.d_model)

        self.do_long1 = False
        self.do_long2 = False
        self.do_short1 = False
        self.do_short2 = False

        if layer_idx < 2: # == 0:
            # self.do_long1 = True
            self.do_long2 = True
            self.do_short1 = True
            # self.do_short2 = True
        else:
            # self.do_long1 = True
            self.do_long2 = True 
            self.do_short1 = True
            # self.do_short2 = True

        self.setup_filters(filter_args)


    def setup_filters(self, filter_args):
        "Initializes the explicit and implicit filters"
        # assert self.order >= 2, f"Order must be at least 2, (got {self.order})"

        if self.do_short1:
            self.short_filter = nn.Conv1d(
                in_channels=self.d_model // 2,
                out_channels=self.d_model // 2,
                kernel_size=self.short_filter_order,
                groups=self.d_model // 2,
                padding=self.short_filter_order - 1,
            )

        if self.do_short2:
            self.short_filter2 = nn.Conv1d(
                in_channels=self.d_model // 2,
                out_channels=self.d_model // 2,
                kernel_size=self.short_filter_order,
                groups=self.d_model // 2,
                padding=self.short_filter_order - 1,
            )


        if self.do_long1:
            self.filter_fn1 = Filter(
                self.head_dim,
                order=self.filter_order,
                seq_len=self.l_max,
                dropout=self.filter_dropout,
                channels=1,
                num_heads=self.num_heads,
                **filter_args,
            )

        if self.do_long2:
            self.filter_fn = Filter(
                self.head_dim,
                order=self.filter_order,
                seq_len=self.l_max,
                dropout=self.filter_dropout,
                channels=1,
                num_heads=self.num_heads,
                **filter_args,
            )

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u = u.transpose(1, 2)
        l = u.size(-1)
        u_orig = u

        # split the d_model in two sections, one for the short one for the long
        u_short, u_long = u.split(self.d_model // 2, dim=1)

        # long filter 1
        if self.do_long1:
            k1 = self.filter_fn1.filter(l, device=u.device)
            k1 = rearrange(k1, "c l d -> c d l")[0] # `c` is always 1 by default
            u_short = rearrange(
                u_short,
                "b (ho v) (z l) -> b ho v z l",
                z=self.num_blocks,
                ho=self.num_heads,
                v=self.head_dim,
            )
            v_long1 = self.filter_fn1(u_short, l, k=k1, bias= self.filter_fn1.bias[None, :, None])
            v_short = rearrange(
                v_long1,
                "b h v z l -> b (z l) (h v)",
                z=self.num_blocks,
                h=self.num_heads,
            )

        # short filter 1 
        if self.do_short1: 
            # u_short = self.in_proj(u_short.transpose(1, 2)).transpose(1, 2)
            # u_short, x1_short = u_short.split(self.d_model // 2, dim=1)
            v_short = self.short_filter(u_short)[..., :l]
            v_short = v_short.squeeze()

            # gating
            # v_short = v_short * x1_short

            if len(v_short.shape) < 3:
                v_short = torch.unsqueeze(v_short, 0).transpose(1, 2)
            else:
                v_short = v_short.transpose(1, 2)

        # long filter 2
        if self.do_long2:
            k = self.filter_fn.filter(l, device=u.device)
            k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default
            u_long = rearrange(
                u_long,
                "b (ho v) (z l) -> b ho v z l",
                z=self.num_blocks,
                ho=self.num_heads,
                v=self.head_dim,
            )
            v_long = self.filter_fn(u_long, l, k=k, bias= self.filter_fn.bias[None, :, None])
            v_long = rearrange(
                v_long,
                "b h v z l -> b (z l) (h v)",
                z=self.num_blocks,
                h=self.num_heads,
            )

        # short filter 2 
        if self.do_short2:
            v_long = self.short_filter2(u_long)[..., :l]
            v_long = v_long.squeeze()

            if len(v_long.shape) < 3:
                v_long = torch.unsqueeze(v_long, 0).transpose(1, 2)
            else:
                v_long = v_long.transpose(1, 2)

        # concat long and short, alternate order depending on the layer
        if self.layer_idx % 2 == 0:
            y = torch.concat([v_long, v_short], dim=-1)
        else:
            y = torch.concat([v_short, v_long], dim=-1)

        y = rearrange(y, "b l d -> b d l") 

        # final gating
        y = y * self.in_proj(u_orig.transpose(1, 2)).transpose(1, 2)

        if self.return_state:
            return y, u_orig
        return y.transpose(1, 2)





class BaseConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_size: Union[int, List[int]]=3,
        layer_idx: int=None,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx=layer_idx

        self.projection = nn.Linear(self.d_model,  self.d_model)
        
        # support for different kernel sizes per layer
        if isinstance(kernel_size, List):
            if layer_idx is  None or layer_idx >= len(kernel_size):
                raise ValueError("kernel_size must be an int or a list of ints with length equal to the number of layers")
            kernel_size = kernel_size[layer_idx]

        # prepare convolution
        if kernel_size != -1:
            self.conv = ImplicitLongConvolution(d_model, l_max=l_max, channels=1)
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
        y = u_conv * u_proj
        return y
