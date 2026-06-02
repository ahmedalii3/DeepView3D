"""Dense stereo pipeline: rectify → SGBM → LR-fuse → depth → point cloud."""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import DEFAULT_PARAMS, SgbmParams, output_dir
from ..io.depth_io import save_depth
from ..io.fat_io import BASELINE_M, FatCamera, load_rgb
from ..io.ply_io import write_ply_xyz_rgb
from .rectify import overlay_horizontal_lines, rectify_stereo, remap_pair
from .sgbm import (
    colorize_disparity,
    compute_disparity,
    compute_disparity_right,
    disparity_to_depth,
    fuse_lr_disparity,
)

log = logging.getLogger("sgbm.dense")


@dataclass
class DenseOutputs:
    n_valid_disparity: int
    n_valid_depth: int
    timings: dict
    out_dir: Path


def _backproject(
    depth: np.ndarray, K: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map into a 3-D point cloud using the pinhole model.

    We use K_rect (from the rectified image), NOT the original camera K,
    because disparity was computed in rectified image coordinates.
    """
    valid = depth > 0
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    ys, xs = np.nonzero(valid)
    Z = depth[ys, xs]
    X = (xs.astype(np.float32) - K[0, 2]) * Z / K[0, 0]
    Y = (ys.astype(np.float32) - K[1, 2]) * Z / K[1, 1]
    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    colors = rgb[ys, xs]
    return xyz, colors


def run_dense(
    scene: str,
    frame: str,
    left_jpg: Path,
    right_jpg: Path,
    cam: FatCamera,
    params: SgbmParams = DEFAULT_PARAMS,
) -> DenseOutputs:
    timings: dict = {}
    out = output_dir(scene, frame, "dense")
    diag = output_dir(scene, frame) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    left_rgb = load_rgb(left_jpg)
    right_rgb = load_rgb(right_jpg)
    timings["load_s"] = time.perf_counter() - t0

    K = cam.K
    image_size = (cam.width, cam.height)

    # ── Rectify ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    maps = rectify_stereo(K, image_size, BASELINE_M)

    # Sanity check: for a pre-aligned rig, R_left_rect should be near identity
    rot_dev = float(np.degrees(np.arccos(np.clip(
        (np.trace(maps.R_left_rect) - 1.0) / 2.0, -1.0, 1.0
    ))))
    if rot_dev > 0.1:
        log.warning(
            "[%s/%s] R_left_rect deviates from identity by %.3f deg",
            scene, frame, rot_dev,
        )

    left_rect_rgb, right_rect_rgb = remap_pair(left_rgb, right_rgb, maps)
    left_rect_gray = cv2.cvtColor(left_rect_rgb, cv2.COLOR_RGB2GRAY)
    right_rect_gray = cv2.cvtColor(right_rect_rgb, cv2.COLOR_RGB2GRAY)
    timings["rectify_s"] = time.perf_counter() - t0

    # Save epipolar-line overlay as a quick visual sanity check
    overlay = overlay_horizontal_lines(left_rect_rgb, right_rect_rgb)
    cv2.imwrite(
        str(diag / "rectification_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    # ── Disparity (left + right, then fuse) ───────────────────────────────
    t0 = time.perf_counter()
    disparity_l = compute_disparity(left_rect_gray, right_rect_gray, params)
    disparity_r = compute_disparity_right(left_rect_gray, right_rect_gray, params)
    # LR fusion fills the ~128-pixel dead band on the left edge
    disparity, disp_src = fuse_lr_disparity(disparity_l, disparity_r)
    timings["sgbm_s"] = time.perf_counter() - t0

    n_left_only = int((disp_src == 1).sum())
    n_right_filled = int((disp_src == 2).sum())
    log.info(
        "[%s/%s] disparity: left=%d filled-from-right=%d",
        scene, frame, n_left_only, n_right_filled,
    )

    np.save(out / "disparity.npy", disparity)
    np.save(out / "disparity_source.npy", disp_src)
    cv2.imwrite(str(diag / "disparity_colormap.png"), colorize_disparity(disparity))

    # ── Depth ─────────────────────────────────────────────────────────────
    # Use K_rect's focal length — it may differ from the original K after rectification
    fx_rect = float(maps.K_rect[0, 0])
    depth = disparity_to_depth(disparity, fx_rect, BASELINE_M, params.depth_max_m)
    save_depth(out / "depth.npy", depth)

    # ── Back-project to 3-D point cloud ───────────────────────────────────
    t0 = time.perf_counter()
    xyz, colors = _backproject(depth, maps.K_rect, left_rect_rgb)
    write_ply_xyz_rgb(out / "points.ply", xyz, colors)
    timings["write_s"] = time.perf_counter() - t0

    n_disp = int((disparity > 0).sum())
    n_depth = int((depth > 0).sum())

    recon = {
        "scene": scene,
        "frame": frame,
        "K_input": K.tolist(),
        "K_rect": maps.K_rect.tolist(),
        "Q": maps.Q.tolist(),
        "R_left_rect": maps.R_left_rect.tolist(),
        "R_right_rect": maps.R_right_rect.tolist(),
        "baseline_m": BASELINE_M,
        "n_valid_disparity": n_disp,
        "n_valid_disparity_left_only": n_left_only,
        "n_valid_disparity_right_filled": n_right_filled,
        "n_valid_depth": n_depth,
        "rotation_deviation_deg": rot_dev,
    }
    (out / "reconstruction.json").write_text(json.dumps(recon, indent=2))
    (out / "timings.json").write_text(json.dumps(timings, indent=2))

    log.info(
        "[%s/%s] dense: disp_px=%d depth_px=%d (%.1f%% coverage)",
        scene, frame, n_disp, n_depth, 100.0 * n_depth / (cam.width * cam.height),
    )

    return DenseOutputs(
        n_valid_disparity=n_disp,
        n_valid_depth=n_depth,
        timings=timings,
        out_dir=out,
    )
