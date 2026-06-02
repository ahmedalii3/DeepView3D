"""Pipeline: predicted depth map → point cloud + metrics + visuals.

Discovers every subdirectory under depthMap-comparisons/ that contains
    predicted_depth.npy   (float32, any consistent depth unit)
    ground_truth_depth.npy (uint8 or float32, same unit as predicted)

Optional per-instance files:
    rgb.png / rgb.jpg / left.jpg   — colours the point cloud and compare panel
    camera.json                    — {"fx":…,"fy":…,"cx":…,"cy":…}
                                     falls back to a synthetic pinhole if absent

Outputs go to outputs/depth_comparisons/<name>/:
    depth.npy               predicted depth as float32
    points.ply              back-projected point cloud (xyz + rgb if available)
    compare.png             pred depth | gt depth | point-cloud overlay  (+ rgb column)
    metrics.json            abs_rel, rmse, delta_1.25, coverage
_summary/
    metrics_table.csv
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import DEPTH_COMP_ROOT, DEPTH_COMP_OUTPUT_ROOT
from ..eval.metrics import per_image_metrics
from ..io.depth_io import save_depth
from ..io.ply_io import write_ply_xyz, write_ply_xyz_rgb

log = logging.getLogger("sgbm.from_depth")

_SUMMARY_DIR = DEPTH_COMP_OUTPUT_ROOT / "_summary"
_METRIC_KEYS = ["abs_rel", "rmse", "delta_1_25", "coverage", "n_valid_pred", "n_valid_gt"]

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DepthInstance:
    name: str
    pred_path: Path
    gt_path: Path
    rgb_path: Path | None = field(default=None)
    camera_json: Path | None = field(default=None)


def _find_rgb(d: Path) -> Path | None:
    for name in ("rgb.png", "rgb.jpg", "left.jpg", "image.png", "image.jpg"):
        p = d / name
        if p.exists():
            return p
    return None


def discover_depth_instances(root: Path = DEPTH_COMP_ROOT) -> list[DepthInstance]:
    instances: list[DepthInstance] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pred = d / "predicted_depth.npy"
        gt = d / "ground_truth_depth.npy"
        if not pred.exists() or not gt.exists():
            log.warning("Skipping %s: missing predicted_depth.npy or ground_truth_depth.npy", d.name)
            continue
        cam_json = d / "camera.json"
        instances.append(DepthInstance(
            name=d.name,
            pred_path=pred,
            gt_path=gt,
            rgb_path=_find_rgb(d),
            camera_json=cam_json if cam_json.exists() else None,
        ))
    return instances


# ── Camera helpers ────────────────────────────────────────────────────────────

def _load_K(inst: DepthInstance, h: int, w: int) -> np.ndarray:
    """Return a 3×3 intrinsics matrix.  Uses camera.json when present;
    otherwise synthesises a centred pinhole with fx=fy=w (≈53° horizontal FOV)."""
    if inst.camera_json is not None:
        d = json.loads(inst.camera_json.read_text())
        fx = float(d["fx"]); fy = float(d["fy"])
        cx = float(d["cx"]); cy = float(d["cy"])
        log.info("[%s] Camera from camera.json: fx=%.2f fy=%.2f cx=%.1f cy=%.1f", inst.name, fx, fy, cx, cy)
    else:
        fx = fy = float(w)
        cx = w / 2.0; cy = h / 2.0
        log.info("[%s] No camera.json — using synthetic K: fx=fy=%.1f cx=%.1f cy=%.1f", inst.name, fx, cx, cy)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ── Core steps ────────────────────────────────────────────────────────────────

def _load_depth_npy(path: Path) -> np.ndarray:
    """Load a .npy depth map and return float32.  uint8/uint16 are cast directly."""
    arr = np.load(path)
    return arr.astype(np.float32)


def _backproject(depth: np.ndarray, K: np.ndarray,
                 rgb: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    valid = depth > 0
    if not valid.any():
        return np.zeros((0, 3), np.float32), None
    ys, xs = np.nonzero(valid)
    Z = depth[ys, xs]
    X = (xs.astype(np.float32) - K[0, 2]) * Z / K[0, 0]
    Y = (ys.astype(np.float32) - K[1, 2]) * Z / K[1, 1]
    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    colors = rgb[ys, xs] if rgb is not None else None
    return xyz, colors


def _depth_to_color(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(depth[valid], 2))
    hi = float(np.percentile(depth[valid], 98))
    norm = np.zeros_like(depth, dtype=np.float32)
    if hi > lo:
        norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _make_pc_overlay(xyz: np.ndarray, colors_rgb: np.ndarray | None,
                     K: np.ndarray, H: int, W: int,
                     bg_bgr: np.ndarray | None = None) -> np.ndarray:
    """Re-project xyz back onto a 2-D image for the compare panel."""
    if bg_bgr is None:
        bg_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    dim = (bg_bgr.astype(np.float32) * 0.4).astype(np.uint8)
    if xyz.shape[0] == 0:
        return dim
    Z = xyz[:, 2]; valid = Z > 0
    u = (K[0, 0] * xyz[:, 0] / np.where(valid, Z, 1) + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * xyz[:, 1] / np.where(valid, Z, 1) + K[1, 2]).astype(np.int32)
    in_b = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pu, pv = u[in_b], v[in_b]
    if colors_rgb is not None:
        pcol = colors_rgb[in_b][:, ::-1]  # RGB → BGR
    else:
        # Colour by depth using turbo
        col_img = _depth_to_color(xyz[:, 2:3].reshape(1, -1))  # (1, N, 3)
        # just use a flat colour pass
        d_norm = np.zeros(xyz.shape[0], np.float32)
        d_valid = xyz[in_b, 2]
        lo, hi = d_valid.min(), d_valid.max()
        if hi > lo:
            d_norm_in = ((d_valid - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            d_norm_in = np.zeros(in_b.sum(), np.uint8)
        tmp = cv2.applyColorMap(d_norm_in.reshape(1, -1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        pcol = tmp  # BGR already

    order = np.argsort(-xyz[in_b, 2])  # far → near
    pu, pv, pcol = pu[order], pv[order], pcol[order]
    overlay = dim.copy()
    overlay[pv, pu] = pcol
    return overlay


def run_one_depth(inst: DepthInstance) -> dict:
    out_dir = DEPTH_COMP_OUTPUT_ROOT / inst.name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = _load_depth_npy(inst.pred_path)
    gt = _load_depth_npy(inst.gt_path)
    H, W = pred.shape

    rgb: np.ndarray | None = None
    if inst.rgb_path is not None:
        bgr = cv2.imread(str(inst.rgb_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))

    K = _load_K(inst, H, W)

    # ── Save depth ────────────────────────────────────────────────────────────
    save_depth(out_dir / "depth.npy", pred)

    # ── Back-project → PLY (predicted + GT) ──────────────────────────────────
    xyz, colors = _backproject(pred, K, rgb)
    if colors is not None:
        write_ply_xyz_rgb(out_dir / "predicted_points.ply", xyz, colors)
    else:
        write_ply_xyz(out_dir / "predicted_points.ply", xyz)
    log.info("[%s] predicted PLY: %d points", inst.name, xyz.shape[0])

    xyz_gt, colors_gt = _backproject(gt, K, rgb)
    if colors_gt is not None:
        write_ply_xyz_rgb(out_dir / "gt_points.ply", xyz_gt, colors_gt)
    else:
        write_ply_xyz(out_dir / "gt_points.ply", xyz_gt)
    log.info("[%s] GT PLY: %d points", inst.name, xyz_gt.shape[0])

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = per_image_metrics(pred, gt)
    metrics_out = {"name": inst.name, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    log.info("[%s] metrics: %s", inst.name, json.dumps(metrics_out))

    # ── Compare panel ─────────────────────────────────────────────────────────
    bg_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None else None
    pc_overlay = _make_pc_overlay(xyz, colors, K, H, W, bg_bgr)

    panels: list[np.ndarray] = []
    labels: list[str] = []
    if rgb is not None:
        panels.append(bg_bgr)
        labels.append("RGB")
    panels += [_depth_to_color(pred), _depth_to_color(gt), pc_overlay]
    labels += ["predicted depth", "GT depth", "point cloud"]

    labelled = []
    for p, lbl in zip(panels, labels):
        out_p = p.copy()
        cv2.rectangle(out_p, (0, 0), (out_p.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out_p, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        labelled.append(out_p)

    compare = np.hstack(labelled)
    cv2.imwrite(str(out_dir / "compare.png"), compare)
    log.info("[%s] Wrote compare.png", inst.name)

    return metrics_out


# ── Summary grid ──────────────────────────────────────────────────────────────

def _build_summary_grid() -> None:
    instances = discover_depth_instances()
    rows = []
    for inst in instances:
        p = DEPTH_COMP_OUTPUT_ROOT / inst.name / "compare.png"
        img = cv2.imread(str(p))
        if img is None:
            continue
        label = inst.name
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(img)
    if not rows:
        return
    max_w = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for r in rows
    ]
    _SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_SUMMARY_DIR / "grid.png"), np.vstack(rows))
    print(f"Wrote {_SUMMARY_DIR / 'grid.png'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    instances = discover_depth_instances()
    if not instances:
        print(f"No depth-map instances found under {DEPTH_COMP_ROOT}")
        return 1
    log.info("Discovered %d depth-map instance(s)", len(instances))

    _SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for inst in instances:
        try:
            m = run_one_depth(inst)
        except Exception as e:
            log.exception("[%s] failed: %s", inst.name, e)
            continue
        row = {"name": inst.name}
        for k in _METRIC_KEYS:
            row[k] = m.get(k)
        rows.append(row)

    metrics_csv = _SUMMARY_DIR / "metrics_table.csv"
    if rows:
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {metrics_csv}")

    _build_summary_grid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
