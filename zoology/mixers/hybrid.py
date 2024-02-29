
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import ModuleConfig


class Hybrid(nn.Module):
    def __init__(
        self,
        d_model: int,
        configs: List[ModuleConfig],
        layer_idx: int=None,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model

        self.mixer = ModuleConfig(
            **configs[layer_idx % len(configs)]
        ).instantiate(d_model=d_model, layer_idx=layer_idx)

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        return  self.mixer(u, *args, **kwargs)

    def state_size(self, **kwargs):
        return self.mixer.state_size(**kwargs)
    