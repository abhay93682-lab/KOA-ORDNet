"""ResNet-50 backbone with ordinal or softmax head (paper Sections IV-C, IV-D)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_GRADES


class KOAOrdNet(nn.Module):
    """ResNet-50 feature extractor + head.

    head="ordinal": K-1 = 4 sigmoid units, unit k estimates P(y > k) (cumulative link).
    head="softmax": 5-way logits (cross-entropy baseline).
    """

    def __init__(self, head: str = "ordinal", pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_type = head
        out = NUM_GRADES - 1 if head == "ordinal" else NUM_GRADES
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, out)

    @property
    def target_layer(self):
        return self.backbone.layer4  # final conv block, used by Grad-CAM

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


def logits_to_grade(logits: torch.Tensor, head: str) -> torch.Tensor:
    if head == "ordinal":
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return logits.argmax(dim=1)


def logits_to_probs(logits: torch.Tensor, head: str) -> torch.Tensor:
    """Per-grade probabilities (for display). Ordinal: P(y=k) = P(y>k-1) - P(y>k)."""
    if head != "ordinal":
        return torch.softmax(logits, dim=1)
    p_gt = torch.sigmoid(logits)                       # (B, 4): P(y>0..3)
    p_gt = torch.cummin(p_gt, dim=1).values            # enforce monotonic
    ones = torch.ones_like(p_gt[:, :1])
    zeros = torch.zeros_like(p_gt[:, :1])
    upper = torch.cat([ones, p_gt], dim=1)
    lower = torch.cat([p_gt, zeros], dim=1)
    return (upper - lower).clamp_min(0)
