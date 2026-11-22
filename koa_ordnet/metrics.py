"""Evaluation metrics (paper Section V-B)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error, recall_score)

from .config import NUM_GRADES


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(NUM_GRADES))
    per_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    severe = y_true >= 3
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "per_grade_recall": {str(k): float(r) for k, r in zip(labels, per_recall)},
        # Grade 3-4 recall: fraction of severe knees predicted as grade 3 or 4
        "grade34_recall": float((y_pred[severe] >= 3).mean()) if severe.any() else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
