"""Write point clouds to ASCII PLY files."""
from __future__ import annotations
from pathlib import Path

import numpy as np


def _header(n: int, with_rgb: bool) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if with_rgb:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_ply_xyz(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(_header(xyz.shape[0], with_rgb=False))
        np.savetxt(f, xyz, fmt="%.6f %.6f %.6f")


def write_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb count mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Stack into a single matrix so np.savetxt writes one row per point (fast, no Python loop)
    combined = np.hstack([xyz, rgb.astype(np.float32)])
    with open(path, "w") as f:
        f.write(_header(xyz.shape[0], with_rgb=True))
        np.savetxt(f, combined, fmt="%.6f %.6f %.6f %d %d %d")
