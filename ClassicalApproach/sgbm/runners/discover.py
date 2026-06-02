"""Auto-discover FAT stereo-pair instances under datasets/fat/."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import COMPARISON_ROOT

# Matches filenames like "000000.left.jpg" and captures the zero-padded frame ID
_FRAME_RE = re.compile(r"^(\d{6})\.left\.jpg$")


@dataclass
class Instance:
    scene: str
    frame: str
    scene_dir: Path
    left_jpg: Path
    right_jpg: Path
    left_depth_png: Path
    right_depth_png: Path
    left_json: Path
    right_json: Path
    camera_settings: Path

    @property
    def all_required(self) -> list[Path]:
        """All files that must exist for an instance to be considered valid."""
        return [
            self.left_jpg, self.right_jpg,
            self.left_depth_png, self.right_depth_png,
            self.left_json, self.right_json,
            self.camera_settings,
        ]


def _build(scene_dir: Path, frame: str) -> Instance:
    scene = scene_dir.name
    return Instance(
        scene=scene,
        frame=frame,
        scene_dir=scene_dir,
        left_jpg=scene_dir / f"{frame}.left.jpg",
        right_jpg=scene_dir / f"{frame}.right.jpg",
        left_depth_png=scene_dir / f"{frame}.left.depth.png",
        right_depth_png=scene_dir / f"{frame}.right.depth.png",
        left_json=scene_dir / f"{frame}.left.json",
        right_json=scene_dir / f"{frame}.right.json",
        camera_settings=scene_dir / "_camera_settings.json",
    )


def discover_instances(root: Path = COMPARISON_ROOT) -> list[Instance]:
    """Walk every scene directory and collect all valid stereo pairs."""
    instances: list[Instance] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(scene_dir.iterdir()):
            m = _FRAME_RE.match(f.name)
            if not m:
                continue
            frame = m.group(1)
            inst = _build(scene_dir, frame)
            missing = [p for p in inst.all_required if not p.exists()]
            if missing:
                print(f"WARN: skipping {inst.scene}/{frame}: missing {[m.name for m in missing]}")
                continue
            instances.append(inst)
    return instances


def find_instance(scene: str, frame: str, root: Path = COMPARISON_ROOT) -> Instance:
    """Look up a single instance; raise FileNotFoundError if any required file is missing."""
    inst = _build(root / scene, frame)
    missing = [p for p in inst.all_required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Instance {scene}/{frame} missing: {[m.name for m in missing]}"
        )
    return inst
