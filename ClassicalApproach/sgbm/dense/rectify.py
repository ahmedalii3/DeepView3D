"""Stereo rectification for the FAT rig (identical K, zero distortion, parallel axes)."""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectifyMaps:
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    R_left_rect: np.ndarray    # rotation applied to left camera during rectification
    R_right_rect: np.ndarray
    P_left: np.ndarray
    P_right: np.ndarray
    Q: np.ndarray              # disparity-to-depth reprojection matrix
    K_rect: np.ndarray         # intrinsics of the rectified left image
    image_size: tuple[int, int]


def rectify_stereo(
    K: np.ndarray, image_size: tuple[int, int], baseline_m: float
) -> RectifyMaps:
    """Compute rectification maps for a parallel stereo rig.

    FAT cameras are already aligned (R = I between eyes), so rectification
    is essentially a no-op rotation-wise, but we still run it to get a
    consistent K_rect and Q matrix for the disparity→depth step.

    T = (-baseline, 0, 0) because the right camera is to the left of the
    left camera in the standard OpenCV convention (negative X direction).

    alpha=0 crops to only valid pixels — no black borders in the output.
    """
    dist = np.zeros(5, dtype=np.float64)  # FAT has no lens distortion
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline_m, 0.0, 0.0], dtype=np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1l, map2l = cv2.initUndistortRectifyMap(K, dist, R1, P1, image_size, cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(K, dist, R2, P2, image_size, cv2.CV_32FC1)
    # P_left is [K_rect | 0]; extract the top-left 3x3 for back-projection
    K_rect = P1[:3, :3]
    return RectifyMaps(
        map1_left=map1l, map2_left=map2l,
        map1_right=map1r, map2_right=map2r,
        R_left_rect=R1, R_right_rect=R2,
        P_left=P1, P_right=P2,
        Q=Q, K_rect=K_rect, image_size=image_size,
    )


def remap_pair(
    left: np.ndarray, right: np.ndarray, maps: RectifyMaps
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rectification maps to both images."""
    lr = cv2.remap(left, maps.map1_left, maps.map2_left, cv2.INTER_LINEAR)
    rr = cv2.remap(right, maps.map1_right, maps.map2_right, cv2.INTER_LINEAR)
    return lr, rr


def overlay_horizontal_lines(
    left_rect: np.ndarray, right_rect: np.ndarray, n_lines: int = 20
) -> np.ndarray:
    """Diagnostic: side-by-side view with green epipolar lines.
    After correct rectification all lines should pass through the same
    row in both images, confirming horizontal alignment.
    """
    if left_rect.ndim == 2:
        left_rect = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
    if right_rect.ndim == 2:
        right_rect = cv2.cvtColor(right_rect, cv2.COLOR_GRAY2BGR)
    sbs = np.hstack([left_rect, right_rect])
    H = sbs.shape[0]
    for i in range(1, n_lines):
        y = int(H * i / n_lines)
        cv2.line(sbs, (0, y), (sbs.shape[1], y), (0, 255, 0), 1)
    return sbs
