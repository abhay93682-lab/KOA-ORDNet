"""Grad-CAM with quantitative IoU scoring (paper Section IV-F).

The Chen (2018) images are crops centred on the knee joint, so the tibiofemoral
joint margin sits in a horizontal band around the image centre. JOINT_BOX gives that
fixed region as fractions (x0, y0, x1, y1) of the image size; change it if your
dataset is cropped differently.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

JOINT_BOX = (0.15, 0.35, 0.85, 0.65)


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self._h = [layer.register_forward_hook(self._fwd),
                   layer.register_full_backward_hook(self._bwd)]

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def remove(self):
        for h in self._h:
            h.remove()

    def __call__(self, x: torch.Tensor, grade: int | None = None) -> tuple[np.ndarray, int]:
        """x: (1,3,H,W). Returns (cam in [0,1] at HxW, predicted grade).

        For the ordinal head the score explained is the sum of the threshold logits
        that are 'on' for the predicted grade (or the first threshold when grade 0),
        i.e. the evidence that pushed the knee up to its grade.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if self.model.head_type == "ordinal":
            pred = int((torch.sigmoid(logits) > 0.5).sum())
            g = pred if grade is None else grade
            score = logits[0, :g].sum() if g > 0 else -logits[0, 0]
        else:
            pred = int(logits.argmax())
            g = pred if grade is None else grade
            score = logits[0, g]
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(dim=1))[0].cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred


def joint_mask(h: int, w: int, box=JOINT_BOX) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    return m


def cam_iou(cam: np.ndarray, threshold: float = 0.5, box=JOINT_BOX) -> float:
    act = cam >= threshold
    ref = joint_mask(*cam.shape, box=box)
    union = np.logical_or(act, ref).sum()
    return float(np.logical_and(act, ref).sum() / union) if union else 0.0


def overlay(gray_or_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4, box=JOINT_BOX) -> np.ndarray:
    img = gray_or_rgb
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (cam.shape[1], cam.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    h, w = cam.shape
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 255), 1)
    return out
