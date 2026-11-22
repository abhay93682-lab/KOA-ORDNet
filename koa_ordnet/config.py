"""Training configuration (paper Section V-A)."""
from dataclasses import dataclass, field

NUM_GRADES = 5  # KL grades 0..4


@dataclass
class Config:
    data_dir: str = "data/kneeKL224"     # folder with train/ val/ test/ subfolders, each with 0..4
    external_dir: str | None = None      # optional external dataset (same 0..4 layout), test only
    out_dir: str = "runs/koa_ordnet"
    loss: str = "koa"                    # "ce" (baseline) | "ordinal" (Chen-style) | "koa" (proposed)
    img_size: int = 224
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 60
    patience: int = 8                    # early stopping on val macro-F1
    lr_factor: float = 0.5
    lr_patience: int = 3
    focal_gamma: float = 2.0
    num_workers: int = 2
    pretrained: bool = True
    seed: int = 42
    limit_per_class: int | None = None   # for quick smoke runs
    extra: dict = field(default_factory=dict)
