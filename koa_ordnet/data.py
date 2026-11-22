"""Dataset loading and preprocessing (paper Sections IV-A, IV-B).

Expected layout (Chen 2018 / Kaggle "Knee Osteoarthritis Dataset with Severity Grading"):
    <root>/train/{0,1,2,3,4}/*.png
    <root>/val/{0,1,2,3,4}/*.png
    <root>/test/{0,1,2,3,4}/*.png
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NUM_GRADES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def list_split(root: str | Path, split: str | None, limit_per_class: int | None = None):
    base = Path(root) / split if split else Path(root)
    items = []
    for g in range(NUM_GRADES):
        files = sorted(p for p in (base / str(g)).glob("*") if p.suffix.lower() in IMG_EXT)
        if limit_per_class:
            files = files[:limit_per_class]
        items += [(p, g) for p in files]
    if not items:
        raise FileNotFoundError(f"No images found under {base}/<0..4>/")
    return items


def preprocess(gray: np.ndarray, size: int = 224) -> np.ndarray:
    """Grayscale uint8 -> CLAHE -> resize -> 3ch float normalized (H, W, 3)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = _clahe.apply(gray)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def augment(gray: np.ndarray) -> np.ndarray:
    """Small rotation (+-10 deg), translation and zoom. No horizontal flip (keeps L/R orientation)."""
    h, w = gray.shape[:2]
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.95, 1.10)
    tx, ty = random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx, ty)
    return cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_REFLECT)


class KneeDataset(Dataset):
    def __init__(self, items, img_size: int = 224, train: bool = False):
        self.items, self.img_size, self.train = items, img_size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, grade = self.items[i]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read {path}")
        if self.train:
            gray = augment(gray)
        x = preprocess(gray, self.img_size).transpose(2, 0, 1)
        return torch.from_numpy(x), grade


def class_counts(items) -> np.ndarray:
    return np.bincount([g for _, g in items], minlength=NUM_GRADES)
