"""Depth evaluation metrics: AbsRel, RMSE, delta < 1.25, coverage."""
from __future__ import annotations
import numpy as np


def per_image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compare a predicted depth map against ground truth.

    Only pixels where BOTH maps are valid (> 0 and finite) are evaluated.
    No scale alignment is applied — this is a metric-depth comparison.

    Coverage = n_valid_pred / n_valid_gt, so 1.0 means the prediction
    covers every GT pixel; values < 1 indicate missing regions.
    """
    valid_gt = np.isfinite(gt) & (gt > 0)
    valid = (pred > 0) & valid_gt
    n_valid_gt = int(valid_gt.sum())
    n_pred = int(valid.sum())

    if n_pred == 0:
        return {
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "delta_1_25": float("nan"),
            "coverage": 0.0,
            "n_valid_pred": 0,
            "n_valid_gt": n_valid_gt,
        }

    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    abs_rel = float(np.mean(np.abs(g - p) / g))
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))
    # delta: fraction of pixels where max(p/g, g/p) < threshold
    ratio = np.maximum(p / g, g / p)
    delta = float(np.mean(ratio < 1.25))
    coverage = n_pred / max(n_valid_gt, 1)

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta_1_25": delta,
        "coverage": coverage,
        "n_valid_pred": n_pred,
        "n_valid_gt": n_valid_gt,
    }
