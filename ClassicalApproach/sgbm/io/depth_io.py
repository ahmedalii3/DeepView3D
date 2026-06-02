"""Save and load depth maps in .npy format.

Convention: float32 (H, W), value 0.0 = invalid pixel.
A companion _mask.npy (bool) is written alongside for convenience.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def save_depth(path_npy: Path, depth: np.ndarray) -> None:
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    path_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_npy, depth)
    # Side-car mask: identical info as depth > 0, but handy for quick loading
    mask_path = path_npy.with_name(path_npy.stem + "_mask.npy")
    np.save(mask_path, depth > 0)


def load_depth(path_npy: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth float32, valid_mask bool)."""
    depth = np.load(path_npy)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    return depth, depth > 0
