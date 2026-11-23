"""Train KOA-ORDNet or a baseline.

    python train.py --data data/kneeKL224 --loss koa      # proposed
    python train.py --data data/kneeKL224 --loss ordinal  # Chen-style ordinal baseline
    python train.py --data data/kneeKL224 --loss ce       # cross-entropy baseline
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from koa_ordnet.config import Config
from koa_ordnet.data import KneeDataset, class_counts, list_split
from koa_ordnet.losses import build_loss
from koa_ordnet.metrics import compute_metrics
from koa_ordnet.model import KOAOrdNet, logits_to_grade


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def predict(model, loader, head, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(logits_to_grade(model(x.to(device)), head).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main(cfg: Config):
    seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir) / (cfg.extra.get("name") or cfg.loss)
    out.mkdir(parents=True, exist_ok=True)

    tr_items = list_split(cfg.data_dir, "train", cfg.limit_per_class)
    va_items = list_split(cfg.data_dir, "val", cfg.limit_per_class)
    counts = class_counts(tr_items)
    print(f"device={device} train={len(tr_items)} val={len(va_items)} class counts={counts.tolist()}")

    dl = lambda items, train: DataLoader(KneeDataset(items, cfg.img_size, train), batch_size=cfg.batch_size,
                                         shuffle=train, num_workers=cfg.num_workers, pin_memory=device == "cuda")
    tr_dl, va_dl = dl(tr_items, True), dl(va_items, False)

    criterion, head = build_loss(cfg.loss, counts, cfg.focal_gamma)
    model = KOAOrdNet(head=head, pretrained=cfg.pretrained).to(device)
    criterion = criterion.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience)

    best, bad, history = -1.0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(y); n += len(y)
        yv, pv = predict(model, va_dl, head, device)
        m = compute_metrics(yv, pv)
        sched.step(m["macro_f1"])
        history.append({"epoch": epoch, "train_loss": total / n, **{k: m[k] for k in ("accuracy", "macro_f1", "qwk", "mae")}})
        print(f"epoch {epoch:02d} loss={total/n:.4f} val_acc={m['accuracy']:.4f} "
              f"val_f1={m['macro_f1']:.4f} val_qwk={m['qwk']:.4f} ({time.time()-t0:.0f}s)")
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            torch.save({"state_dict": model.state_dict(), "head": head, "config": asdict(cfg)}, out / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {epoch}")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val macro-F1 = {best:.4f}; checkpoint: {out/'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Config.data_dir)
    ap.add_argument("--out", default=Config.out_dir)
    ap.add_argument("--loss", choices=["ce", "ordinal", "koa"], default="koa")
    ap.add_argument("--epochs", type=int, default=Config.max_epochs)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--workers", type=int, default=Config.num_workers)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--gamma", type=float, default=Config.focal_gamma, help="focal gamma (koa loss)")
    ap.add_argument("--name", default=None, help="run folder name (default: the loss name)")
    a = ap.parse_args()
    main(Config(data_dir=a.data, out_dir=a.out, loss=a.loss, max_epochs=a.epochs, batch_size=a.batch_size,
                num_workers=a.workers, limit_per_class=a.limit_per_class, pretrained=not a.no_pretrained,
                focal_gamma=a.gamma, extra={"name": a.name}))
