"""Loss functions (paper Sections IV-D, IV-E).

- ce:      plain cross-entropy over 5 classes (baseline)
- ordinal: per-threshold BCE on cumulative targets (Chen et al. style ordinal baseline)
- koa:     class-weighted (inverse frequency) + focal ordinal BCE (proposed)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_GRADES


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    """Grade y -> 4 binary targets [y>0, y>1, y>2, y>3]."""
    ks = torch.arange(NUM_GRADES - 1, device=y.device)
    return (y.unsqueeze(1) > ks).float()


def inverse_freq_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.maximum(counts, 1).astype(np.float64)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class OrdinalBCE(nn.Module):
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, ordinal_targets(y))


class FocalOrdinalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("cw", class_weights)
        self.gamma = gamma

    def forward(self, logits, y):
        t = ordinal_targets(y)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        focal = (1 - p_t) ** self.gamma * bce          # (B, 4)
        w = self.cw[y].unsqueeze(1)                    # per-sample class weight
        return (w * focal).sum(dim=1).mean()


def build_loss(name: str, counts: np.ndarray, gamma: float = 2.0):
    if name == "ce":
        return nn.CrossEntropyLoss(), "softmax"
    if name == "ordinal":
        return OrdinalBCE(), "ordinal"
    if name == "koa":
        return FocalOrdinalLoss(inverse_freq_weights(counts), gamma), "ordinal"
    raise ValueError(f"Unknown loss {name}")
