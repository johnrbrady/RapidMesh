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


def prepare_fixture(
    path: str, rows: int, cols: int, seed: int = 7, extent_scale: float = 1.0
) -> dict[str, Any]:
    """Generate a scan once and save its arrays. Not a pipeline figure.

    `extent_scale` grows the *room* while the lattice stays put — WP-1.G0's
    extent ladder. It is a uniform scale on the scene, scanner included, so the
    result is a geometric similarity: the same rays, the same lattice cells, the
    same sample count, at scaled ranges. That is the whole point. WP-3.2's
    density ladder varied the other axis and could not separate the tile term
    from the sample-store floor, because it never moved the tile count.

    `range_noise` stays at 2 mm at every scale. It is an instrument parameter,
    not a scene one; see `RoomScene.scaled`.
    """
    import numpy as np

    from rapidmesh import synthetic

    scene = synthetic.RoomScene(mover=True)
    if extent_scale != 1.0:
        scene = scene.scaled(extent_scale)
    scan = synthetic.generate(
        scene, rows=rows, cols=cols,
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
        "extent_scale": float(extent_scale),
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


def prepare_e57(path: str, rows: int, cols: int, seed: int = 7) -> dict[str, Any]:
    """Write the same station as a structured E57. Not a pipeline figure.

    The streamed rows below read from this file, so the input genuinely lives
    on disk rather than being handed over as an array. No reference-corpus data
    is touched.
    """
    import numpy as np
    import pye57

    from rapidmesh import synthetic

    scan = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=rows, cols=cols,
        dropout=0.01, range_noise=0.002, seed=seed, station_id="mem",
    ).scan
    writer = pye57.E57(path, mode="w")
    writer.write_scan_raw(
        {
            "cartesianX": scan.xyz[:, 0].astype(np.float64),
            "cartesianY": scan.xyz[:, 1].astype(np.float64),
            "cartesianZ": scan.xyz[:, 2].astype(np.float64),
            "rowIndex": scan.row.astype(np.int64),
            "columnIndex": scan.col.astype(np.int64),
            "cartesianInvalidState": np.zeros(len(scan), np.int64),
        },
        name="mem",
    )
    del writer
    return {
        "rows": rows, "cols": cols, "samples": len(scan),
        "bytes_on_disk": pathlib.Path(path).stat().st_size,
    }


def e57_metadata_only(e57: str, chunk_points: int = 250_000) -> dict[str, Any]:
    """The bounded metadata passes alone: lattice, pose, frame and colour."""
    from rapidmesh.streaming_e57 import e57_station_metadata

    metadata, policy = e57_station_metadata(e57, chunk_points=chunk_points)
    return {
        "rows": metadata.lattice.rows, "cols": metadata.lattice.cols,
        "rewrite_to_local": policy.rewrite_to_local,
    }


def e57_pass_a(
    e57: str, band_rows: int = 64, chunk_points: int = 250_000
) -> dict[str, Any]:
    """Pass A driven from the file — the input is never a whole-station array."""
    import shutil
    import tempfile

    from rapidmesh.islands import pass_a_sweep_bands
    from rapidmesh.streaming_e57 import e57_band_filter_results, e57_station_metadata

    metadata, policy = e57_station_metadata(e57, chunk_points=chunk_points)
    work = pathlib.Path(tempfile.mkdtemp(prefix="rapidmesh-e57a-"))
    try:
        result = pass_a_sweep_bands(
            metadata,
            e57_band_filter_results(
                e57, metadata, policy, chunk_points=chunk_points,
                band_rows=band_rows, halo=3,
            ),
            work_dir=work,
        )
        return {
            "bands": result.band_count,
            "max_live_components": result.max_live_components,
            "segment_bytes": sum(
                p.stat().st_size for p in work.rglob("*") if p.is_file()
            ),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def e57_streamed_station(
    e57: str, band_rows: int = 64, chunk_points: int = 250_000,
    measure: bool = False,
) -> dict[str, Any]:
    """The whole streamed pipeline, input never resident."""
    from rapidmesh.pipeline import mesh_station_from_chunks
    from rapidmesh.streaming_e57 import e57_band_to_scan, e57_chunks, e57_station_metadata

    metadata, policy = e57_station_metadata(e57, chunk_points=chunk_points)
    result = mesh_station_from_chunks(
        e57_chunks(e57, chunk_points=chunk_points), metadata,
        band_rows=band_rows, chunk_points=chunk_points, halo=3, measure=measure,
        converter=lambda band, meta: e57_band_to_scan(band, meta, policy),
    )
    return {
        "vertices": result.mesh.vertex_count,
        "triangles": result.mesh.triangle_count,
        "bands": 0 if result.diagnostics is None else result.diagnostics.band_count,
    }


def e57_tiled_station(
    e57: str,
    band_rows: int = 64,
    chunk_points: int = 250_000,
    tile_size: float = 4.0,
    measure: bool = False,
    out: str = "",
) -> dict[str, Any]:
    """The whole tiled pipeline driven from an E57, input never resident.

    The Round 4 reference row. `e57_streamed_station` runs the same sweep but
    without an output directory, so it takes Pass B's *resident* branch and
    assembles a whole-station `MeshData` — exactly what ADR-006 Decision 2
    forbids at reference scale. This one passes `out_dir`, so Pass B files
    triangles into per-tile spools and never holds the station's mesh.

    Returns counts only. No path, no station name, no coordinate reaches the
    caller: the measurement is authorised read-only and nothing identifying may
    leave it.
    """
    import shutil
    import tempfile

    from rapidmesh.pipeline import mesh_station_from_chunks
    from rapidmesh.streaming_e57 import e57_band_to_scan, e57_chunks, e57_station_metadata

    owned = not out
    root = pathlib.Path(out or tempfile.mkdtemp(prefix="rapidmesh-e57tiles-"))
    try:
        metadata, policy = e57_station_metadata(e57, chunk_points=chunk_points)
        result = mesh_station_from_chunks(
            e57_chunks(e57, chunk_points=chunk_points), metadata,
            band_rows=band_rows, chunk_points=chunk_points, halo=3,
            measure=measure, out_dir=str(root), tile_size=tile_size,
            converter=lambda band, meta: e57_band_to_scan(band, meta, policy),
        )
        diagnostics = result.diagnostics
        store = result.tiles
        return {
            "rows": metadata.lattice.rows,
            "cols": metadata.lattice.cols,
            "vertices": 0 if store is None else store.vertex_count,
            "triangles": 0 if store is None else store.triangle_count,
            "tiles": 0 if store is None else len(store.tile_ids),
            "tile_size": tile_size,
            "tile_bytes": 0 if diagnostics is None else diagnostics.tile_bytes,
            "segment_bytes": 0 if diagnostics is None else diagnostics.segment_bytes,
            "bands": 0 if diagnostics is None else diagnostics.band_count,
            "resident_mesh": result.mesh is not None,
            "retained": result.stats.retained,
            "input_points": result.stats.input_points,
            "balanced": result.stats.balanced,
        }
    finally:
        if owned:
            shutil.rmtree(root, ignore_errors=True)


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


def tiled_station(
    fixture: str,
    band_rows: int = 64,
    tile_size: float = 4.0,
    measure: bool = False,
    out: str = "",
) -> dict[str, Any]:
    """`mesh_station_streamed` writing an incremental spatial generation.

    The WP-3.2 memory row. ADR-006 Decision 1 excludes bytes already written to
    disk from the working-memory budget, so `tile_bytes` is returned beside the
    peak: the exclusion is only defensible if the amount excluded is stated, and
    a run that quietly kept the output in memory would show a large peak and a
    tile byte count of zero.
    """
    import shutil
    import tempfile

    from rapidmesh.pipeline import mesh_station_streamed

    scan = load_fixture(fixture)
    owned = not out
    root = pathlib.Path(out or tempfile.mkdtemp(prefix="rapidmesh-tiles-"))
    try:
        result = mesh_station_streamed(
            scan, band_rows=band_rows, chunk_points=250_000, halo=3,
            measure=measure, out_dir=str(root), tile_size=tile_size,
        )
        diagnostics = result.diagnostics
        store = result.tiles
        return {
            "samples": len(scan),
            "vertices": 0 if store is None else store.vertex_count,
            "triangles": 0 if store is None else store.triangle_count,
            "tiles": 0 if store is None else len(store.tile_ids),
            "tile_size": tile_size,
            "tile_bytes": 0 if diagnostics is None else diagnostics.tile_bytes,
            "segment_bytes": 0 if diagnostics is None else diagnostics.segment_bytes,
            "resident_mesh": result.mesh is not None,
        }
    finally:
        if owned:
            shutil.rmtree(root, ignore_errors=True)


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


def merge_only(
    records: int,
    block_records: int = 200_000,
    merge_bytes: int = 4_000_000,
    output_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """The external merge alone, over synthetic records.

    Blocks are generated lazily at a fixed size, so nothing about the *input*
    scales with `records` — only the number of runs does. The peak this reports
    is therefore the merge layer's own, and it is what ITEM-009 claims is
    bounded by its caps rather than by the station.

    The caps are arguments so a caller can put both sizes in the *same* regime.
    Comparing a run small enough to fit under the cap against one that reaches
    it measures the cap being approached, not the bound holding.
    """
    import shutil
    import tempfile

    import numpy as np

    from rapidmesh.pass_b_merge import merge_runs, write_runs

    dtype = np.dtype([("a", "<i8"), ("b", "<i8")])

    def blocks() -> Any:
        rng = np.random.default_rng(0)
        for start in range(0, records, block_records):
            count = min(block_records, records - start)
            out = np.empty(count, dtype)
            out["a"] = rng.integers(0, 1_000_000, count)
            out["b"] = rng.integers(0, 1_000_000, count)
            yield out

    work = pathlib.Path(tempfile.mkdtemp(prefix="rapidmesh-merge-"))
    try:
        runs = write_runs(blocks(), work, "bench", ("a", "b"))
        seen, largest = 0, 0
        for block in merge_runs(runs, merge_bytes=merge_bytes, output_bytes=output_bytes):
            seen += int(block.shape[0])
            largest = max(largest, int(block.nbytes))
        return {"records": seen, "runs": len(runs.paths), "largest_block_bytes": largest}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# QA stage attribution — WP-3.1
#
# Forward and reverse QA are measured against a mesh **loaded from disk**, in a
# child that has built nothing. WP-1.10 learned this the hard way: with the mesh
# built in the same process the pipeline's own high-water mark sits above the QA
# transient and hides it, and a floor read that way is not a floor. The marginal
# reported here is therefore `peak(mesh + QA) - peak(mesh alone)`, both rows in
# their own fresh interpreter.
# ---------------------------------------------------------------------------


def prepare_mesh(fixture: str, path: str) -> dict[str, Any]:
    """Mesh the fixture once and save the mesh plus its QA inputs.

    Not a pipeline row. `mesh_station`'s retained offsets are bitwise the mesh's
    own vertices — both are `rotate_local(xyz[used_final]).astype(f32)` — so the
    forward-QA query set is stored once and reused rather than re-derived.
    """
    import numpy as np

    from rapidmesh.filters import clean
    from rapidmesh.grid import ScanGrid
    from rapidmesh.triangulate import build_mesh, cull_islands, triangulate, used_vertices

    scan = load_fixture(fixture)
    cleaned, _ = clean(scan)
    grid = ScanGrid.build(cleaned)
    tris = cull_islands(grid.scan.xyz, triangulate(grid), min_area=0.005)
    mesh = build_mesh(grid.scan, tris)
    final = used_vertices(len(grid.scan), tris)
    np.savez(
        path,
        origin=mesh.origin,
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        source_sample_id=mesh.source_sample_id,
        rows=grid.scan.row[final].astype(np.int32),
        translation=grid.scan.pose.translation,
        rotation=grid.scan.pose.rotation,
    )
    return {
        "samples": len(scan),
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "bytes_on_disk": pathlib.Path(path).stat().st_size,
    }


def load_mesh(path: str) -> Any:
    """Rebuild the `MeshData` and its QA inputs from `prepare_mesh`'s arrays."""
    import numpy as np

    from rapidmesh.types import MeshData, ScanPose

    data = np.load(path)
    pose = ScanPose(translation=data["translation"], rotation=data["rotation"])
    mesh = MeshData(
        origin=data["origin"],
        vertices=data["vertices"],
        triangles=data["triangles"],
        source_pose=pose,
        source_sample_id=data["source_sample_id"],
    )
    return mesh, np.asarray(data["rows"])


def _warm_scipy() -> None:
    """Load scipy's KD-tree machinery before the baseline row is taken.

    Charging `scipy.spatial`'s import to forward QA would put tens of megabytes
    of interpreter state in a figure that is supposed to be the QA working set.
    The production prefix has it loaded already — `filters.clean` builds trees —
    so every row here loads it too, and the marginal is the computation.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(np.zeros((8, 3), np.float64))
    tree.query(np.zeros((8, 3), np.float64), k=2, workers=1)


def mesh_only(path: str) -> dict[str, Any]:
    """The mesh prefix: the finished mesh resident, and nothing measured."""
    _warm_scipy()
    mesh, rows = load_mesh(path)
    return {
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "rows": int(rows.shape[0]),
    }


def mesh_forward_qa(
    path: str, samples: int = 300_000, block: int = 0
) -> dict[str, Any]:
    """The mesh prefix plus one forward `deviation_report`."""
    from rapidmesh.qa import QA_QUERY_BLOCK, deviation_report

    _warm_scipy()
    mesh, rows = load_mesh(path)
    report = deviation_report(
        mesh, mesh.vertices, max_samples=samples,
        block=block or QA_QUERY_BLOCK,
    )
    return {
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "rows": int(rows.shape[0]),
        "sampled": report.sampled_points,
        "rms": report.rms,
    }


def mesh_forward_qa_caps(
    path: str,
    samples: int = 300_000,
    vertex_bytes: int = 0,
    triangle_bytes: int = 0,
    pair_block: int = 0,
    seed_cap: int = 0,
    stage: str = "all",
) -> dict[str, Any]:
    """Forward QA with each cap set explicitly, so a peak can be attributed.

    `stage` stops after a named stage — `seed`, `knn`, or `all` — because the
    only way to know which cap sets the high-water mark is to measure the peak
    with the later stages absent.
    """
    import numpy as np

    from rapidmesh.qa import QA_QUERY_BLOCK
    from rapidmesh.qa_neighbours import QA_SEED_VERTICES, _seed_radius, nearest_vertices
    from rapidmesh.qa_stream import (
        QA_PAIR_BLOCK,
        QA_TRIANGLE_BLOCK_BYTES,
        QA_VERTEX_BLOCK_BYTES,
        ResidentMesh,
        incident_distances,
    )

    _warm_scipy()
    mesh, _ = load_mesh(path)
    source = ResidentMesh(
        mesh,
        vertex_block_bytes=vertex_bytes or QA_VERTEX_BLOCK_BYTES,
        triangle_block_bytes=triangle_bytes or QA_TRIANGLE_BLOCK_BYTES,
    )
    queries = mesh.vertices
    if queries.shape[0] > samples:
        rs = np.random.default_rng(0)
        queries = queries[rs.choice(queries.shape[0], samples, replace=False)]
    q64 = queries.astype(np.float64)
    cap = seed_cap or QA_SEED_VERTICES

    if stage == "queries":
        # The floor every later stage stands on: 300,000 float64 query points
        # and the subsample that chose them. Neither is a geometry structure and
        # neither can be bounded away, so a marginal is only interpretable
        # beside this row.
        return {"queries": int(q64.shape[0]), "bytes": int(q64.nbytes)}

    if stage == "seed":
        radius = _seed_radius(
            source, q64, k=2, workers=1, block=QA_QUERY_BLOCK, cap=cap
        )
        return {"queries": int(q64.shape[0]), "median_radius": float(np.median(radius))}

    found, ids = nearest_vertices(
        source, q64, k=2, workers=1, block=QA_QUERY_BLOCK, seed_cap=cap
    )
    if stage == "knn":
        return {"queries": int(q64.shape[0]), "median_d1": float(np.median(found[:, 1]))}

    d = incident_distances(
        source, q64, ids, found[:, 0], pair_block=pair_block or QA_PAIR_BLOCK
    )
    return {
        "queries": int(d.size),
        "triangles": mesh.triangle_count,
        "rms": float((d * d).mean() ** 0.5),
    }


def mesh_forward_qa_resident(
    path: str, samples: int = 300_000, block: int = 0
) -> dict[str, Any]:
    """The same row through `qa_reference.resident_distances`.

    The sensitivity break for the WP-3.1 gate: a bound that the implementation it
    replaced also satisfies is not a bound, it is the fixture being small.
    """
    from rapidmesh.qa import QA_QUERY_BLOCK
    from rapidmesh.qa_reference import resident_distances

    _warm_scipy()
    mesh, rows = load_mesh(path)
    d = resident_distances(
        mesh, mesh.vertices, max_samples=samples, block=block or QA_QUERY_BLOCK
    )
    return {
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "sampled": int(d.size),
        "rms": float((d * d).mean() ** 0.5),
    }


def mesh_reverse_qa(
    path: str, samples: int = 300_000, window: int = 8
) -> dict[str, Any]:
    """The mesh prefix plus one `reverse-qa-v2` report."""
    from rapidmesh.reverse_qa import mesh_to_source_report_v2

    _warm_scipy()
    mesh, rows = load_mesh(path)
    report, evidence = mesh_to_source_report_v2(
        mesh, mesh.vertices, source_rows=rows,
        qa_window_rows=window, max_samples=samples,
    )
    return {
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "sampled": report.sampled_points,
        "rms": report.rms,
        "selected": evidence.samples_selected,
    }


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
