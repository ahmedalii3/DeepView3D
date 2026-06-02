"""Stereo SGBM disparity matching and disparity-to-depth conversion."""
from __future__ import annotations

import cv2
import numpy as np

from ..config import DEFAULT_PARAMS, SgbmParams


def make_sgbm(params: SgbmParams = DEFAULT_PARAMS) -> cv2.StereoSGBM:
    block = params.sgbm_block_size
    # P1/P2 are smoothness penalties; the 8*3 and 32*3 multipliers are the
    # standard OpenCV recommendation for 3-channel images
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=params.sgbm_num_disparities,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=params.sgbm_disp12_max_diff,
        uniquenessRatio=params.sgbm_uniqueness_ratio,
        speckleWindowSize=params.sgbm_speckle_window,
        speckleRange=params.sgbm_speckle_range,
        preFilterCap=params.sgbm_pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(
    left_gray: np.ndarray, right_gray: np.ndarray, params: SgbmParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Left-referenced disparity map (each pixel gives shift to the right image)."""
    sgbm = make_sgbm(params)
    # SGBM stores disparity as fixed-point with 4 fractional bits → divide by 16
    raw = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    raw[raw <= 0] = 0.0  # treat negative / invalid markers as missing
    return raw


def compute_disparity_right(
    left_gray: np.ndarray, right_gray: np.ndarray, params: SgbmParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Right-referenced disparity: flip both images, run SGBM, flip result back.

    A right-image pixel at (u_r, v) with disparity d_r corresponds to
    left-image pixel (u_r + d_r, v).  We use this to fill the dead band
    on the left edge of the left disparity map.
    """
    sgbm = make_sgbm(params)
    # Mirroring makes the right image act as the reference for SGBM
    lf = cv2.flip(left_gray, 1)
    rf = cv2.flip(right_gray, 1)
    raw = sgbm.compute(rf, lf).astype(np.float32) / 16.0
    raw[raw <= 0] = 0.0
    return cv2.flip(raw, 1)  # flip back to original orientation


def fuse_lr_disparity(
    disp_left: np.ndarray, disp_right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the left-edge dead band by reprojecting right-reference disparities.

    SGBM can only match a left pixel if the corresponding right pixel is at
    most numDisparities columns away — the leftmost numDisparities columns
    have no valid left disparity at all.  We fix this by projecting each
    valid right-reference pixel back into left coordinates:
        x_left = x_right + d_right
    and filling any still-zero slot with that disparity.

    Returns (fused_disparity, source_mask) where source_mask is:
        0 = invalid, 1 = original left disparity, 2 = filled from right
    """
    H, W = disp_left.shape
    fused = disp_left.copy()
    src = np.where(disp_left > 0, 1, 0).astype(np.uint8)

    ys, xs_r = np.nonzero(disp_right > 0)
    if ys.size == 0:
        return fused, src

    d = disp_right[ys, xs_r]
    xs_l = np.round(xs_r + d).astype(np.int32)       # project into left image
    in_b = (xs_l >= 0) & (xs_l < W)
    ys, xs_l, d = ys[in_b], xs_l[in_b], d[in_b]

    need_fill = fused[ys, xs_l] <= 0                  # only touch empty pixels
    ys_f, xs_f, d_f = ys[need_fill], xs_l[need_fill], d[need_fill]
    fused[ys_f, xs_f] = d_f
    src[ys_f, xs_f] = 2
    return fused, src


def disparity_to_depth(
    disparity: np.ndarray, fx: float, baseline_m: float, max_depth_m: float
) -> np.ndarray:
    """Z = fx * baseline / disparity  (stereo triangulation formula)."""
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (fx * baseline_m) / disparity[valid]
    depth[depth > max_depth_m] = 0.0   # discard implausibly far points
    depth[depth < 0] = 0.0
    return depth


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Map disparity to a Turbo colormap image for visualization."""
    d = disparity.copy()
    valid = d > 0
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = float(d[valid].min()), float(d[valid].max())
    norm = np.zeros_like(d)
    if hi > lo:
        norm[valid] = (d[valid] - lo) / (hi - lo)
    norm = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
