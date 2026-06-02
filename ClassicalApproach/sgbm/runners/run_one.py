"""Run the dense stereo pipeline for a single FAT instance."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from ..config import DEFAULT_PARAMS, output_dir
from ..dense.pipeline import run_dense
from ..eval.metrics import per_image_metrics
from ..io.depth_io import load_depth
from ..io.fat_io import FatCamera, load_gt_depth
from .discover import Instance, find_instance

log = logging.getLogger("sgbm.run_one")


def _setup_logging(out_dir: Path) -> None:
    """Reset root logger handlers so re-running doesn't produce duplicate lines."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    # Remove any handlers added by previous calls or imports
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "log.txt", mode="w")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


def run_one(instance: Instance, params=DEFAULT_PARAMS) -> dict:
    base_out = output_dir(instance.scene, instance.frame)
    _setup_logging(base_out)
    log.info("== %s/%s ==", instance.scene, instance.frame)

    cam = FatCamera.from_camera_settings(instance.camera_settings, "left")
    log.info(
        "Camera: fx=%.2f fy=%.2f cx=%.1f cy=%.1f size=%dx%d",
        cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
    )

    dense = run_dense(
        instance.scene, instance.frame, instance.left_jpg, instance.right_jpg, cam, params,
    )

    gt_depth = load_gt_depth(instance.left_depth_png)
    dense_depth, _ = load_depth(dense.out_dir / "depth.npy")

    metrics = {
        "scene": instance.scene,
        "frame": instance.frame,
        "dense": per_image_metrics(dense_depth, gt_depth),
    }
    (base_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("metrics: %s", json.dumps(metrics, indent=None))

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Run dense stereo for one FAT instance.")
    ap.add_argument("scene", help="Scene name, e.g. Kitchen")
    ap.add_argument("frame", help="Zero-padded frame ID, e.g. 000000")
    args = ap.parse_args()
    inst = find_instance(args.scene, args.frame)
    run_one(inst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
