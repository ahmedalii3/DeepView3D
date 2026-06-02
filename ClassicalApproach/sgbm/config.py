"""Global configuration: paths, SGBM parameters, and defaults."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Repo root is two levels above this file (FinalProject/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Put your FAT stereo pairs under datasets/fat/<scene>/
COMPARISON_ROOT = PROJECT_ROOT / "datasets" / "fat"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sgbm_fat"
SUMMARY_DIR = OUTPUT_ROOT / "_summary"

# External depth-map comparisons (predicted_depth.npy + ground_truth_depth.npy pairs)
DEPTH_COMP_ROOT = PROJECT_ROOT / "depthMap-comparisons"
DEPTH_COMP_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "depth_comparisons"


def output_dir(scene: str, frame: str, branch: str | None = None) -> Path:
    """Return (and create) the per-frame output directory."""
    p = OUTPUT_ROOT / scene / frame
    if branch is not None:
        p = p / branch
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class SgbmParams:
    # ── SGBM disparity search ──────────────────────────────────────────────
    sgbm_num_disparities: int = 128   # must be divisible by 16; also equals left dead-band width
    sgbm_block_size: int = 5          # odd number; larger = smoother but loses fine detail
    sgbm_uniqueness_ratio: int = 10   # reject ambiguous matches (higher = stricter)
    sgbm_speckle_window: int = 100    # max isolated-blob area to discard as noise
    sgbm_speckle_range: int = 32      # disparity jump allowed inside a speckle blob
    sgbm_pre_filter_cap: int = 63
    sgbm_disp12_max_diff: int = 1     # left-right consistency check tolerance (pixels)

    # ── Depth clamping ────────────────────────────────────────────────────
    depth_max_m: float = 20.0         # discard very far points (sky, reflections)


DEFAULT_PARAMS = SgbmParams()
