"""
Feature map for Performer (FAVOR+) from Rethinking Attention with Performers
"""
import math
import numpy as np
import torch

from .base import FeatureMap

# class PerformerFeatureMap(FeatureMap):
#     """
#     Code from https://github.com/teddykoker/performer/blob/main/performer.py
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.random_feats = None
        
#     def forward(self, x: torch.Tensor):
#         """Assume x.shape is (b, h, l, d)"""
#         d = x.shape[-1]
        
#         # Build + cache random features
#         if self.random_feats is None:
#             m = d  # self.output_dim   # * d
#             self.random_feats = self.orthogonal_gaussian(m, d).type(x.type())
#             # print(self.random_feats.shape, self.random_feats)
            
#         fmap = self.feature_map(self.h, [torch.exp], self.random_feats)
#         return fmap(x / (d ** 0.25))
    
#     def feature_map(self, h, fs, random_feats):
#         """Random feature map"""
#         m, d = random_feats.shape
#         # MZ 1/30/24 -> add back scaling factor to get SM from RBF
#         return lambda x: (self.h(x) / math.sqrt(m) * torch.cat(
#             [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
#             dim=-1))
#         # return lambda x: (1 / math.sqrt(m) * torch.cat(
#         #     [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
#         #     dim=-1))
    
#     def h(self, x):
#         """Adjust to get softmax (from RBF)"""
#         x = torch.exp(-torch.pow(x, 2).sum(dim=-1, keepdims=True) / 2)
#         return x
    
#     def iid_gaussian(self, m, d):
#         """Generate IID Gaussian random features"""
#         return torch.randn(size=(m, d))

#     def orthogonal_gaussian(self, m, d):
#         """Generate orthogonal Gaussian random features"""
#         def orthogonal_square():
#             # create orthogonal square matrix using Gram-Schmidt
#             q, _ = np.linalg.qr(self.iid_gaussian(d, d))
#             return q.T

#         num_squares = int(m / d)
#         blocks = [orthogonal_square() for _ in range(num_squares)]

#         remainder = m - d * num_squares
#         if remainder:
#             blocks.append(orthogonal_square()[:remainder])

#         matrix = np.vstack(blocks)
#         matrix /= np.sqrt(num_squares + remainder / d)
#         # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

#         return torch.from_numpy(matrix) 
    



class PerformerFeatureMap(FeatureMap):
    """
    Code from https://github.com/teddykoker/performer/blob/main/performer.py
    """
    def __init__(
        self, 
        input_dim: int = 16, 
        expanded_dim: int = 256,
        *args, 
        **kwargs
    ):
        super().__init__(input_dim=input_dim, *args, **kwargs)
        self.expanded_dim = expanded_dim
        print(f"using expanded dim {expanded_dim}!!")
        
        random_feats = self.orthogonal_gaussian(
            self.expanded_dim, 
            self.input_dim
        )
        self.register_buffer("random_feats", random_feats)
        
    def forward(self, x: torch.Tensor):
        """Assume x.shape is (b, h, l, d)"""
        d = x.shape[-1]
    
            
        fmap = self.feature_map(self.h, [torch.exp], self.random_feats)
        return fmap(x / (d ** 0.25))
    
    def feature_map(self, h, fs, random_feats):
        """Random feature map"""
        m, d = random_feats.shape
        # MZ 1/30/24 -> add back scaling factor to get SM from RBF
        return lambda x: (self.h(x) / math.sqrt(m) * torch.cat(
            [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
            dim=-1))
        # return lambda x: (1 / math.sqrt(m) * torch.cat(
        #     [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
        #     dim=-1))
    
    def h(self, x):
        """Adjust to get softmax (from RBF)"""
        x = torch.exp(-torch.pow(x, 2).sum(dim=-1, keepdims=True) / 2)
        return x
    
    def iid_gaussian(self, m, d):
        """Generate IID Gaussian random features"""
        return torch.randn(size=(m, d))

    def orthogonal_gaussian(self, m, d):
        """Generate orthogonal Gaussian random features"""
        def orthogonal_square():
            # create orthogonal square matrix using Gram-Schmidt
            q, _ = np.linalg.qr(self.iid_gaussian(d, d))
            return q.T

        num_squares = int(m / d)
        blocks = [orthogonal_square() for _ in range(num_squares)]

        remainder = m - d * num_squares
        if remainder:
            blocks.append(orthogonal_square()[:remainder])

        matrix = np.vstack(blocks)
        matrix /= np.sqrt(num_squares + remainder / d)
        # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

        return torch.from_numpy(matrix) 