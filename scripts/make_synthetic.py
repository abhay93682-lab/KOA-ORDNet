"""Create a tiny synthetic dataset with the real folder layout, for smoke tests only.

    python scripts/make_synthetic.py data/synthetic --per-class 4

Images are random noise with a darker 'joint gap' whose width shrinks with grade.
They are NOT medical data and results on them mean nothing.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main(root: Path, per_class: int, size: int = 128):
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for g in range(5):
            d = root / split / str(g)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img = rng.normal(150, 25, (size, size)).clip(0, 255).astype(np.uint8)
                gap = max(2, 14 - 3 * g)
                c = size // 2
                img[c - gap // 2:c + gap // 2, 15:size - 15] = 40
                cv2.imwrite(str(d / f"{i}.png"), img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-class", type=int, default=4)
    a = ap.parse_args()
    main(a.root, a.per_class)
