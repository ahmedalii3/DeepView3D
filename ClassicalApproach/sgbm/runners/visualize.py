"""Visualization commands for FAT dense-stereo outputs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import OUTPUT_ROOT
from ..io.depth_io import load_depth
from ..io.fat_io import FatCamera, load_gt_depth, load_rgb
from .discover import find_instance


def _depth_to_color(depth: np.ndarray) -> np.ndarray:
    """Map depth to a Turbo colormap; invalid (0) pixels stay black.
    Uses percentile 2–98 normalization to avoid outliers washing out the range.
    """
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


_COL_LABELS = ("RGB", "dense", "GT", "point cloud")


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _build_compare_panels(scene: str, frame: str) -> list[np.ndarray]:
    inst = find_instance(scene, frame)
    base = OUTPUT_ROOT / scene / frame
    rgb = load_rgb(inst.left_jpg)
    dense, _ = load_depth(base / "dense" / "depth.npy")
    gt = load_gt_depth(inst.left_depth_png)
    pc_overlay = _make_pointcloud_overlay(scene, frame)
    return [
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        _depth_to_color(dense),
        _depth_to_color(gt),
        pc_overlay,
    ]


def cmd_compare(scene: str, frame: str) -> None:
    """4-panel comparison: RGB | dense depth | GT depth | point-cloud overlay."""
    base = OUTPUT_ROOT / scene / frame
    panels = _build_compare_panels(scene, frame)
    labelled = [_label(p, lbl) for p, lbl in zip(panels, _COL_LABELS)]
    out = np.hstack(labelled)
    out_path = base / "compare.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")


def cmd_grid(out_path: Path | None = None) -> None:
    """Stack every instance's compare panels into one summary grid."""
    from .discover import discover_instances
    insts = discover_instances()
    if not insts:
        print("No instances found.")
        return
    rows = []
    for inst in insts:
        panels = _build_compare_panels(inst.scene, inst.frame)
        row = np.hstack(panels)
        row = _label(row, f"{inst.scene} / {inst.frame}")
        rows.append(row)
    big = np.vstack(rows)
    # Add a column-label header row above the grid
    header = np.zeros((36, big.shape[1], 3), dtype=np.uint8)
    panel_w = big.shape[1] // len(_COL_LABELS)
    for i, lbl in enumerate(_COL_LABELS):
        cv2.putText(header, lbl, (i * panel_w + 12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    big = np.vstack([header, big])
    final = out_path or (OUTPUT_ROOT / "_summary" / "grid.png")
    final.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(final), big)
    print(f"Wrote {final}")


def _load_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ASCII PLY file and return (xyz float32, rgb uint8)."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f)
    if data.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    if data.ndim == 1:
        data = data[None, :]
    xyz = data[:, :3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else np.full((data.shape[0], 3), 200, np.uint8)
    return xyz, rgb


def _project_points_to_image(
    xyz: np.ndarray, K: np.ndarray, image_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D points onto the image plane; return (pixel coords, in-bounds mask)."""
    H, W = image_hw
    Z = xyz[:, 2]
    valid = Z > 0
    proj = np.zeros((xyz.shape[0], 2), dtype=np.int32)
    if valid.any():
        u = K[0, 0] * xyz[:, 0] / Z + K[0, 2]
        v = K[1, 1] * xyz[:, 1] / Z + K[1, 2]
        proj[:, 0] = np.round(u).astype(np.int32)
        proj[:, 1] = np.round(v).astype(np.int32)
    in_bounds = valid & (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < H)
    return proj, in_bounds


def _sample_cloud(
    scene: str, frame: str, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the PLY, optionally subsample, and return (xyz, rgb, K_rect, rgb_left).
    n_samples <= 0 keeps every point.
    """
    inst = find_instance(scene, frame)
    base = OUTPUT_ROOT / scene / frame
    xyz, rgb = _load_ply_xyz_rgb(base / "dense" / "points.ply")
    if n_samples > 0 and xyz.shape[0] > n_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(xyz.shape[0], n_samples, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    recon = json.loads((base / "dense" / "reconstruction.json").read_text())
    K_rect = np.asarray(recon["K_rect"], dtype=np.float64)
    rgb_left = load_rgb(inst.left_jpg)
    return xyz, rgb, K_rect, rgb_left


def _make_pointcloud_overlay(
    scene: str, frame: str, n_samples: int = 0
) -> np.ndarray:
    """Re-project the point cloud back onto the left image.

    Uses a Z-buffer pass (paint far→near) so near points always win.
    Dense clouds (> 80k points) use single-pixel dots to avoid smearing;
    sparse ones use a 3×3 footprint to stay visible.
    """
    xyz_s, rgb_s, K_rect, rgb_left = _sample_cloud(scene, frame, n_samples)
    H, W = rgb_left.shape[:2]
    # Dim the background so reprojected points stand out
    dim = (cv2.cvtColor(rgb_left, cv2.COLOR_RGB2BGR).astype(np.float32) * 0.4).astype(np.uint8)
    if xyz_s.shape[0] == 0:
        return dim

    proj, in_b = _project_points_to_image(xyz_s, K_rect, (H, W))
    pu, pv = proj[in_b, 0], proj[in_b, 1]
    pcol = rgb_s[in_b][:, ::-1]   # RGB → BGR for OpenCV

    # Z-buffer: sort far-to-near so the nearest point wins at each pixel
    z = xyz_s[in_b, 2]
    order = np.argsort(-z)         # descending Z = paint far first
    pu, pv, pcol = pu[order], pv[order], pcol[order]

    overlay = dim.copy()
    use_dilate = xyz_s.shape[0] < 80_000  # sparse cloud needs a larger dot to be visible
    if use_dilate:
        for r in range(-1, 2):
            for c in range(-1, 2):
                yy = np.clip(pv + r, 0, H - 1)
                xx = np.clip(pu + c, 0, W - 1)
                overlay[yy, xx] = pcol
    else:
        overlay[pv, pu] = pcol
    return overlay


def cmd_pointcloud(
    scene: str,
    frame: str,
    n_samples_overlay: int = 0,
    n_samples_3d: int = 0,
) -> None:
    """Generate a 2-D overlay PNG and a 4-panel matplotlib 3-D figure.
    n_samples <= 0 uses every point in the cloud (can be 400k+, allow time).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = OUTPUT_ROOT / scene / frame

    overlay = _make_pointcloud_overlay(scene, frame, n_samples_overlay)
    cv2.imwrite(str(base / "pointcloud_overlay.png"), overlay)
    n_overlay_used = _last_overlay_count(scene, frame, n_samples_overlay)
    print(f"Wrote {base / 'pointcloud_overlay.png'} ({n_overlay_used:,} pts)")

    xyz_s, rgb_s, _K_rect, rgb_left = _sample_cloud(scene, frame, n_samples_3d)
    if xyz_s.shape[0] == 0:
        print(f"Empty point cloud: {base / 'dense' / 'points.ply'}")
        return

    rgb_f = rgb_s.astype(np.float32) / 255.0
    n_3d = xyz_s.shape[0]
    # Scale marker size down proportionally — very dense clouds look better as tiny dots
    marker_s = max(0.02, min(0.5, 8000.0 / n_3d))

    # Flip Y so the scene appears right-side-up in the 3-D plot (OpenCV Y is down)
    Xd = xyz_s[:, 0]
    Yd = -xyz_s[:, 1]
    Zd = xyz_s[:, 2]

    fig = plt.figure(figsize=(24, 8))
    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(rgb_left); ax0.set_title("Left RGB"); ax0.axis("off")

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Cloud overlay ({n_overlay_used:,} pts)"); ax1.axis("off")

    def _scatter(ax, elev, azim, title):
        ax.scatter(Xd, Zd, Yd, c=rgb_f, s=marker_s, marker=".", linewidths=0)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)

    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    _scatter(ax2, elev=15, azim=-65, title=f"3D rotated ({n_3d:,} pts)")
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    _scatter(ax3, elev=80, azim=-90, title="Top-down")

    fig.suptitle(f"{scene} / {frame} — dense point cloud", fontsize=14)
    fig.tight_layout()
    out_3d = base / "pointcloud_views.png"
    fig.savefig(out_3d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_3d}")


def _last_overlay_count(scene: str, frame: str, n_samples: int) -> int:
    base = OUTPUT_ROOT / scene / frame
    xyz, _ = _load_ply_xyz_rgb(base / "dense" / "points.ply")
    return xyz.shape[0] if n_samples <= 0 else min(n_samples, xyz.shape[0])


def cmd_pointcloud_grid() -> None:
    """Stack all instances' overlay + 3-D view images into one summary PNG."""
    from .discover import discover_instances
    insts = discover_instances()
    if not insts:
        print("No instances found.")
        return
    rows = []
    for inst in insts:
        base = OUTPUT_ROOT / inst.scene / inst.frame
        a = cv2.imread(str(base / "pointcloud_overlay.png"))
        b = cv2.imread(str(base / "pointcloud_views.png"))
        if a is None or b is None:
            print(f"Skipping {inst.scene}/{inst.frame}: missing pointcloud images. "
                  f"Run `visualize pointcloud {inst.scene} {inst.frame}` first.")
            continue
        # Scale views image to match overlay height
        target_h = a.shape[0]
        bw = int(b.shape[1] * (target_h / b.shape[0]))
        b_rs = cv2.resize(b, (bw, target_h))
        row = np.hstack([a, b_rs])
        row = _label(row, f"{inst.scene} / {inst.frame}")
        rows.append(row)
    if not rows:
        return
    max_w = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for r in rows
    ]
    big = np.vstack(rows)
    out_path = OUTPUT_ROOT / "_summary" / "pointcloud_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), big)
    print(f"Wrote {out_path}")


def cmd_depth(scene: str, frame: str) -> None:
    """Blend dense depth colormap over the RGB image (50/50)."""
    inst = find_instance(scene, frame)
    base = OUTPUT_ROOT / scene / frame
    rgb = load_rgb(inst.left_jpg)
    depth, _ = load_depth(base / "dense" / "depth.npy")
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    color = _depth_to_color(depth)
    mask = (depth > 0)[..., None]
    blended = np.where(mask, (0.5 * overlay + 0.5 * color).astype(np.uint8), overlay)
    out_path = base / "overlay_dense.png"
    cv2.imwrite(str(out_path), blended)
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize FAT dense-stereo outputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("depth", help="Depth-RGB blend overlay")
    p_d.add_argument("scene"); p_d.add_argument("frame")

    p_c = sub.add_parser("compare", help="4-panel comparison image")
    p_c.add_argument("scene"); p_c.add_argument("frame")

    sub.add_parser("grid", help="Compare grid for all instances")

    p_pc = sub.add_parser("pointcloud", help="2-D overlay + 3-D scatter")
    p_pc.add_argument("scene"); p_pc.add_argument("frame")
    p_pc.add_argument("--samples-overlay", type=int, default=0,
                      help="2D overlay point cap; 0 = all (default)")
    p_pc.add_argument("--samples-3d", type=int, default=0,
                      help="3D scatter point cap; 0 = all (default)")

    sub.add_parser("pointcloud-grid", help="Summary grid of all point clouds")

    args = ap.parse_args()
    if args.cmd == "depth":
        cmd_depth(args.scene, args.frame)
    elif args.cmd == "compare":
        cmd_compare(args.scene, args.frame)
    elif args.cmd == "grid":
        cmd_grid()
    elif args.cmd == "pointcloud":
        cmd_pointcloud(args.scene, args.frame, args.samples_overlay, args.samples_3d)
    elif args.cmd == "pointcloud-grid":
        cmd_pointcloud_grid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
