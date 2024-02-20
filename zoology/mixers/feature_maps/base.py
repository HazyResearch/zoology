"""
Parent feature map class along with baseline versions (Pos ELU, ReLU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, 
                 input_dim: int,                 
                 temp: int = None,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.temp = 1. if temp is None else temp
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x

    def expanded_size(self):
        return self.input_dim 



class Identity(FeatureMap):
    def forward(self, x: torch.Tensor):
        return x
    
class PosELU(FeatureMap):
    def forward(self, x: torch.Tensor):
        return F.elu(x * self.temp) + 1
    

class ReLU(FeatureMap):
    def forward(self, x: torch.Tensor):
        return F.relu(x * self.temp).clamp(min=self.eps)

class Square(FeatureMap):
    def forward(self, x: torch.Tensor):
        return x ** 2