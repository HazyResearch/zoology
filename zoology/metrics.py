import torch
import torch.nn.functional as F
from typing import Tuple


def compute_mse(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    loss = F.mse_loss(outputs, targets)
    return loss, loss.item()


def compute_ce_with_embeddings(
    outputs: torch.Tensor,
    true_indices: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = outputs @ value_embeddings.T
    loss = F.cross_entropy(logits, true_indices)
    return loss, logits


    