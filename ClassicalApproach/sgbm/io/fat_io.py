"""Load FAT stereo images, camera intrinsics, and ground-truth depth."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Physical distance between left and right camera optical centers
BASELINE_M: float = 0.06


@dataclass
class FatCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsics matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @classmethod
    def from_camera_settings(cls, path: Path, eye: str = "left") -> "FatCamera":
        """Parse intrinsics from FAT's _camera_settings.json for one eye."""
        data = json.loads(Path(path).read_text())
        for entry in data["camera_settings"]:
            if entry["name"] == eye:
                K = entry["intrinsic_settings"]
                S = entry["captured_image_size"]
                return cls(
                    fx=float(K["fx"]),
                    fy=float(K["fy"]),
                    cx=float(K["cx"]),
                    cy=float(K["cy"]),
                    width=int(S["width"]),
                    height=int(S["height"]),
                )
        raise KeyError(f"Eye {eye!r} not found in {path}")


def load_rgb(path: Path) -> np.ndarray:
    """Read image and return uint8 RGB (not BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt_depth(path: Path) -> np.ndarray:
    """Read 16-bit depth PNG and convert to meters float32.

    FAT encodes depth as 0.1 mm per count, so dividing by 10 000 gives meters.
    Pixels with raw value 0 are marked invalid (depth = 0.0).
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read GT depth: {path}")
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel depth PNG, got shape {raw.shape}")
    depth = raw.astype(np.float32) / 10000.0  # 0.1 mm → meters
    depth[raw == 0] = 0.0
    depth[depth < 0] = 0.0
    return depth
