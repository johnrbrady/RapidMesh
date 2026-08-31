"""
Peak-memory measurement harness — PLAN.md §5 item 10.

Runs each workload in its own process and reports the operating system's peak
RSS for it, because a peak is a high-water mark and two workloads in one process
share one. Every figure says which instrument produced it.

    python tools/measure_peak_memory.py --rows 500 --cols 2000

The fixture is generated once into a `.npz` and every measured run loads it, so
no pipeline figure carries the cost of `synthetic.generate`. That matters more
than it sounds: the generator ray-casts every lattice cell, and its transients
dominate a small station. A production run reads an E57; it does not ray-cast a
room, so charging generation to the pipeline would overstate every row.

Workload functions are module level and take JSON-serialisable arguments, so
`rapidmesh.memory.measure_in_child` can name one in a spec and a reader can
reproduce a row from that spec alone. `tests/test_peak_memory.py` calls the same
functions, so the numbers in the report and the numbers the gate asserts come
from one definition.

**Scope.** Synthetic fixtures only. Gate 1's memory row is written for the
14.5 M-point reference station and that station is **not** measured here: no
`H:\\Sample` access is made and none is claimed. What this establishes is the
instrument, the method, the budget behaviour at stated synthetic sizes, and the
scaling those sizes imply.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rapidmesh.memory import (  # noqa: E402
    PEAK_RSS_BUDGET_BYTES,
    WORKING_MEMORY_BUDGET_BYTES,
    MemoryMeasurement,
    measure_in_child,
)

TOOLS_DIR = str(pathlib.Path(__file__).resolve().parent)


# ---------------------------------------------------------------------------
# fixture, prepared once and loaded by every measured run
# ---------------------------------------------------------------------------


def prepare_fixture(path: str, rows: int, cols: int, seed: int = 7) -> dict[str, Any]:
    """Generate a scan once and save its arrays. Not a pipeline figure."""
    import numpy as np

    from rapidmesh import synthetic

    scan = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=rows, cols=cols,
        dropout=0.01, range_noise=0.002, seed=seed, station_id="mem",
    ).scan
    lattice = scan.lattice
    np.savez(
        path,
        row=scan.row, col=scan.col, xyz=scan.xyz, rng=scan.rng,
        rgb=np.zeros((0, 3), np.uint8) if scan.rgb is None else scan.rgb,
        sample_id=np.zeros(0, np.int64) if scan.sample_id is None else scan.sample_id,
        translation=scan.pose.translation, rotation=scan.pose.rotation,
        lattice=np.array(
            [lattice.rows, lattice.cols, lattice.az_step, lattice.el_step,
             lattice.az0, lattice.el0], np.float64,
        ),
        counts=np.array(
            [scan.source_sample_count or 0, scan.dropped_no_return, scan.dropped_other],
            np.int64,
        ),
    )
    return {
        "rows": rows, "cols": cols, "samples": len(scan),
        "bytes_on_disk": pathlib.Path(path).stat().st_size,
    }


def load_fixture(fixture: str) -> Any:
    """Rebuild the scan from `prepare_fixture`'s arrays and nothing else."""
    import numpy as np

    from rapidmesh.types import LatticeInfo, LatticeSource, ScanPose, StructuredScan

    data = np.load(fixture)
    lat, counts = data["lattice"], data["counts"]
    rgb, sample_id = data["rgb"], data["sample_id"]
    return StructuredScan(
        row=data["row"], col=data["col"], xyz=data["xyz"], rng=data["rng"],
        pose=ScanPose(translation=data["translation"], rotation=data["rotation"]),
        lattice=LatticeInfo(
            rows=int(lat[0]), cols=int(lat[1]), az_step=float(lat[2]),
            el_step=float(lat[3]), az0=float(lat[4]), el0=float(lat[5]),
            source=LatticeSource.SYNTHETIC,
        ),
        rgb=None if rgb.shape[0] == 0 else rgb,
        station_id="mem",
        sample_id=None if sample_id.shape[0] == 0 else sample_id,
        source_sample_count=int(counts[0]) or None,
        dropped_no_return=int(counts[1]),
        dropped_other=int(counts[2]),
    )


# ---------------------------------------------------------------------------
# workloads — each runs in its own child process
# ---------------------------------------------------------------------------


def load_only(fixture: str) -> dict[str, Any]:
    """The input scan alone. The floor every measured run starts from."""
    scan = load_fixture(fixture)
    return {"samples": len(scan), "lattice_cells": scan.lattice.cells}


def in_memory_station(fixture: str, measure: bool = False) -> dict[str, Any]:
    """`mesh_station` — the reference path, including the global `cull_islands`."""
    from rapidmesh.pipeline import mesh_station

    scan = load_fixture(fixture)
    result = mesh_station(scan, measure=measure)
    return {
        "samples": len(scan),
        "vertices": result.mesh.vertex_count,
        "triangles": result.mesh.triangle_count,
    }


def streamed_station(
    fixture: str, band_rows: int = 64, measure: bool = False
) -> dict[str, Any]:
    """`mesh_station_streamed` — Pass A then Pass B, segments via a temp dir."""
    from rapidmesh.pipeline import mesh_station_streamed

    scan = load_fixture(fixture)
    result = mesh_station_streamed(
        scan, band_rows=band_rows, chunk_points=250_000, halo=3, measure=measure
    )
    diagnostics = result.diagnostics
    return {
        "samples": len(scan),
        "vertices": result.mesh.vertex_count,
        "triangles": result.mesh.triangle_count,
        "bands": 0 if diagnostics is None else diagnostics.band_count,
        "segment_bytes": 0 if diagnostics is None else diagnostics.segment_bytes,
    }


def streamed_pass_a(fixture: str, band_rows: int = 64) -> dict[str, Any]:
    """Pass A alone: filter, triangulate, label, retire, write segments.

    Measured on its own so Pass B's share can be derived exactly. A peak only
    rises, so `peak(whole run) = max(peak(A), peak(B))`; where the whole run
    exceeds Pass A, the excess is Pass B's own high-water mark.
    """
    import shutil
    import tempfile

    from rapidmesh.islands import pass_a_sweep

    scan = load_fixture(fixture)
    work = pathlib.Path(tempfile.mkdtemp(prefix="rapidmesh-passa-"))
    try:
        result = pass_a_sweep(scan, work_dir=work, band_rows=band_rows, halo=3)
        return {
            "samples": len(scan),
            "bands": result.band_count,
            "max_live_components": result.max_live_components,
            "max_frontier_occupied": result.max_frontier_occupied,
            "segment_bytes": sum(
                p.stat().st_size for p in work.rglob("*") if p.is_file()
            ),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def global_cull_only(fixture: str) -> dict[str, Any]:
    """`cull_islands` alone — the whole-station intermediate item 8 removed."""
    from rapidmesh.filters import clean
    from rapidmesh.grid import ScanGrid
    from rapidmesh.triangulate import cull_islands, triangulate

    scan = load_fixture(fixture)
    cleaned, _ = clean(scan)
    grid = ScanGrid.build(cleaned)
    tris = triangulate(grid)
    kept = cull_islands(grid.scan.xyz, tris)
    return {"triangles": int(tris.shape[0]), "kept": int(kept.shape[0])}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def run_table(fixture: str, band_rows: int) -> list[MemoryMeasurement]:
    common: dict[str, Any] = {"fixture": fixture}
    plan = [
        ("input scan only", "load_only", common),
        ("global cull_islands", "global_cull_only", common),
        ("mesh_station", "in_memory_station", common),
        ("streamed Pass A only", "streamed_pass_a", {**common, "band_rows": band_rows}),
        ("mesh_station_streamed", "streamed_station", {**common, "band_rows": band_rows}),
    ]
    out: list[MemoryMeasurement] = []
    for label, attr, kwargs in plan:
        out.append(
            measure_in_child(
                "measure_peak_memory", attr, kwargs,
                label=label, sys_path=[TOOLS_DIR],
            )
        )
        print(out[-1].describe(), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    import tempfile

    parser = argparse.ArgumentParser(description="Peak-memory table (PLAN §5 item 10)")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--cols", type=int, default=1200)
    parser.add_argument("--band-rows", type=int, default=64)
    args = parser.parse_args(argv)

    print(f"budgets: working <= {WORKING_MEMORY_BUDGET_BYTES:,} B   "
          f"peak RSS <= {PEAK_RSS_BUDGET_BYTES:,} B")
    with tempfile.TemporaryDirectory(prefix="rapidmesh-fixture-") as tmp:
        fixture = str(pathlib.Path(tmp) / "scan.npz")
        made = measure_in_child(
            "measure_peak_memory", "prepare_fixture",
            {"path": fixture, "rows": args.rows, "cols": args.cols},
            label="prepare fixture (not a pipeline row)", sys_path=[TOOLS_DIR],
        )
        print(f"fixture: {args.rows} x {args.cols} synthetic, "
              f"{made.detail.get('samples', 0):,} samples, band_rows={args.band_rows}")
        print("each row is one child process; peak RSS is the OS high-water mark.")
        print(made.describe())
        print()
        results = run_table(fixture, args.band_rows)

    by_label = {m.label: m for m in results}
    pass_a, whole = by_label.get("streamed Pass A only"), by_label.get("mesh_station_streamed")
    if pass_a and whole:
        excess = whole.peak_rss_bytes - pass_a.peak_rss_bytes
        share = 100.0 * excess / max(whole.peak_rss_bytes, 1)
        print(
            f"\nPass B share: peak(whole) - peak(Pass A) = {excess:,} B "
            f"({share:.1f}% of the streamed peak). A peak only rises, so where "
            f"this is positive it is Pass B's own high-water mark."
        )
    scan_only = by_label.get("input scan only")
    if scan_only and whole:
        samples = max(int(whole.detail.get("samples", 0)), 1)
        print(
            f"per retained sample: input scan {scan_only.working_set_delta_bytes / samples:.1f} B, "
            f"streamed run {whole.working_set_delta_bytes / samples:.1f} B "
            f"({samples:,} samples)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
