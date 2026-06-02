"""Smoke tests for FAT IO against the live ComparisonFolder data."""
from __future__ import annotations
import numpy as np
import pytest

from ..config import COMPARISON_ROOT
from ..io.fat_io import FatCamera, load_gt_depth, load_rgb
from ..runners.discover import discover_instances


@pytest.fixture(scope="module")
def instances():
    insts = discover_instances(COMPARISON_ROOT)
    if not insts:
        pytest.skip("No FAT instances found under ComparisonFolder/")
    return insts


def test_camera_settings_are_consistent(instances):
    for inst in instances:
        cam_l = FatCamera.from_camera_settings(inst.camera_settings, "left")
        cam_r = FatCamera.from_camera_settings(inst.camera_settings, "right")
        assert abs(cam_l.fx - 768.16) < 1e-1
        assert abs(cam_l.fy - 768.16) < 1e-1
        assert cam_l.cx == 480 and cam_l.cy == 270
        assert cam_l.width == 960 and cam_l.height == 540
        assert (cam_l.K.shape == (3, 3))
        assert np.allclose(cam_l.K, cam_r.K)


def test_load_rgb(instances):
    inst = instances[0]
    img = load_rgb(inst.left_jpg)
    assert img.dtype == np.uint8
    assert img.shape == (540, 960, 3)


def test_load_gt_depth(instances):
    for inst in instances:
        d = load_gt_depth(inst.left_depth_png)
        assert d.dtype == np.float32
        assert d.shape == (540, 960)
        valid = (d > 0.5) & (d < 10.0) & np.isfinite(d)
        ratio = float(valid.sum()) / d.size
        assert ratio > 0.5, f"{inst.scene}/{inst.frame}: only {ratio:.2%} of depth pixels in [0.5, 10] m"
