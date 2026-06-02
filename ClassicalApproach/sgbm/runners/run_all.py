"""Run the dense pipeline for all discovered FAT instances and write summary CSVs."""
from __future__ import annotations
import csv
import json
import logging
import sys

from ..config import SUMMARY_DIR
from .discover import discover_instances
from .run_one import run_one
from .visualize import cmd_compare, cmd_depth, cmd_pointcloud, cmd_grid, cmd_pointcloud_grid

log = logging.getLogger("sgbm.run_all")

_METRIC_KEYS = ["abs_rel", "rmse", "delta_1_25", "coverage", "n_valid_pred", "n_valid_gt"]


def main() -> int:
    instances = discover_instances()
    log.warning("Discovered %d instances", len(instances))
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    timing_rows = []
    for inst in instances:
        try:
            metrics = run_one(inst)
        except Exception as e:
            log.exception("[%s/%s] failed: %s", inst.scene, inst.frame, e)
            continue

        row = {"scene": inst.scene, "frame": inst.frame, "method": "dense"}
        for k in _METRIC_KEYS:
            row[k] = metrics["dense"].get(k)
        rows.append(row)

        # Attach timing data written by the pipeline
        tpath = SUMMARY_DIR.parent / inst.scene / inst.frame / "dense" / "timings.json"
        try:
            t = json.loads(tpath.read_text())
        except FileNotFoundError:
            t = {}
        timing_rows.append(
            {"scene": inst.scene, "frame": inst.frame, "method": "dense", **t}
        )

        # Generate per-instance visual comparisons
        try:
            cmd_depth(inst.scene, inst.frame)
            cmd_compare(inst.scene, inst.frame)
            cmd_pointcloud(inst.scene, inst.frame)
        except Exception as e:
            log.warning("[%s/%s] visualize failed: %s", inst.scene, inst.frame, e)

    # Write metrics table
    metrics_csv = SUMMARY_DIR / "metrics_table.csv"
    if rows:
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Write timings table (columns vary by instance, so union all keys)
    timings_csv = SUMMARY_DIR / "timings_table.csv"
    if timing_rows:
        keys = sorted({k for r in timing_rows for k in r.keys()})
        keys = ["scene", "frame", "method"] + [k for k in keys if k not in {"scene", "frame", "method"}]
        with open(timings_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(timing_rows)

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {timings_csv}")

    # Regenerate summary grids with all instances
    try:
        cmd_grid()
        cmd_pointcloud_grid()
    except Exception as e:
        log.warning("Summary grid failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
