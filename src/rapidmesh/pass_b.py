"""
Pass B — bounded finalisation. ITEM-009, DEC-009 step 1.

`PHASE1-ISLANDS-FINALISATION.md` §4 splits the station's finalisation from its
sweep: Pass A decides component *membership* while streaming, Pass B evaluates
`component-area-v1` over completed components, culls, finalises the ledger,
meshes and measures. WP-1.5 then measured what Pass B actually cost — 45% of
the streamed pipeline's peak at 158k samples, rising to 76% at 990k — because
it held the whole station's triangles several times over at once.

The area reduction that makes the streaming safe — §4.1's exactness condition,
and why it licenses a block-wise accumulation — now lives in `pass_b_area.py`,
which this module re-exports from. That split is DEC-010 only; nothing about the
computation changed.

Two merged passes replace it. The first reduces area per component; the second
emits the survivors into an array preallocated from the first pass's counts, so
the station's triangles are never held twice. What lives between them is one
entry per *component*, not one per triangle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .pass_b_area import (
    BOUNDARY_SAFETY,
    EXACTNESS_RATIO_LIMIT,
    StreamedAreaAccumulator,
    _block_indices,
    exact_component_areas,
    stream_component_areas,
)
from .pass_b_merge import RunSet, merge_runs, write_runs
from .tiles import DEFAULT_TILE_SIZE_M
from .types import FilterStats, StructuredScan

# Re-exported for callers that predate the DEC-010 split: `tile_build` imports
# `_block_indices` from here, and `test_pass_b_bound` the area names. The split
# is a file boundary, not an interface change.
__all__ = [
    "BOUNDARY_SAFETY",
    "EXACTNESS_RATIO_LIMIT",
    "StreamedAreaAccumulator",
    "TRIANGLE_RUN_FIELDS",
    "TRIANGLE_RUN_KEY",
    "build_triangle_runs",
    "exact_component_areas",
    "pass_b_finalise",
    "stream_component_areas",
    "stream_kept_triangles",
]

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .islands import PassAResult
    from .pipeline import MeshResult

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


TRIANGLE_RUN_FIELDS = [
    ("root", "<i8"), ("c0", "<i8"), ("c1", "<i8"), ("c2", "<i8"), ("rot", "i1"),
]
TRIANGLE_RUN_KEY = ("root", "c0", "c1", "c2")


def build_triangle_runs(
    segments: Sequence[Any], alias: I64, work_dir: Path
) -> tuple[RunSet, I64]:
    """One sorted run per band segment, keyed `(root, canonical cell triple)`.

    The key is the stable lattice cell id, never a band-local array index —
    `PHASE1-HALO-CALCULUS.md` §7 — and the triple is canonicalised by rotation
    only, never reversal, so a reversed winding stays a different triangle.
    A band's segment is already bounded, so a run is too.

    `rot` records how far the triple was rotated to reach canonical form, so
    the **original winding can be restored on emission**. That matters more
    than it looks: cyclic rotation preserves orientation and is therefore
    invisible to the T2 triangle comparison, but `reverse-qa-v2` builds its
    barycentric interior point from the stored corner order, so emitting a
    rotated triple would silently move every reverse-QA sample. Measured, not
    reasoned about — see the WP-1.7 report.

    Also returns the sorted unique lattice cells named by any triangle. Pass B
    loads only those samples from `pos` (Round 4c): the rest of `pos` is
    retained-but-unmeshed and is counted, not materialised.
    """
    import numpy as np

    from .segments_io import read_tri_segment

    unique_chunks: list[Any] = []

    def blocks() -> Iterator[Any]:
        for segment in segments:
            cells, provisional = read_tri_segment(segment.tri_path)
            if cells.shape[0] == 0:
                continue
            unique_chunks.append(np.unique(cells))
            pick = np.argmin(cells, axis=1)
            rows = np.arange(cells.shape[0])
            record = np.empty(cells.shape[0], dtype=np.dtype(TRIANGLE_RUN_FIELDS))
            record["root"] = alias[provisional]
            record["rot"] = pick.astype(np.int8)
            for offset, name in enumerate(("c0", "c1", "c2")):
                record[name] = cells[rows, (pick + offset) % 3]
            yield record

    runs = write_runs(blocks(), work_dir, "tri", TRIANGLE_RUN_KEY)
    if not unique_chunks:
        return runs, np.empty(0, np.int64)
    # Round 4e. `np.unique(np.concatenate(chunks))` held three copies at once —
    # the per-band chunks, their concatenation, and the sorted copy `np.unique`
    # makes internally — and that transient, not any resident structure, set
    # Pass B's high-water mark. On the most fragmented station measured
    # (175,274 components) this was the *only* call in the pass that raised the
    # process peak at all, by 110,825,472 B, while every later call reported no
    # rise. The chunks are released before the sort, the sort is in place, and
    # an adjacent-difference mask replaces the second copy, so one copy of the
    # concatenation survives where three did.
    needed = np.concatenate(unique_chunks)
    unique_chunks.clear()
    needed.sort()
    if needed.size > 1:
        keep = np.empty(needed.size, dtype=bool)
        keep[0] = True
        np.not_equal(needed[1:], needed[:-1], out=keep[1:])
        needed = needed[keep]
    return runs, needed.astype(np.int64, copy=False)


def stream_kept_triangles(
    runs: RunSet, cells: I64, keep_root: Any, kept_total: int, vertices: int
) -> tuple[I64, Any, Any]:
    """Phase two: the surviving triangles, and both vertex-membership masks.

    `keep_root` is a bool array indexed by component id, not a mapping: the
    lookup happens once per block rather than once per triangle.

    The kept array is preallocated from the phase-one counts, so the station's
    triangles are never held twice — no list of blocks, no concatenate. The
    pre-cull mask is accumulated here too, because `dropped_island` needs it
    and materialising the pre-cull triangle array to get it is exactly the
    allocation this pass removes.
    """
    import numpy as np

    kept = np.empty((kept_total, 3), np.int64)
    before = np.zeros(vertices, dtype=bool)
    final = np.zeros(vertices, dtype=bool)
    at = 0
    for block in merge_runs(runs):
        indices = _block_indices(block, cells)
        before[indices.ravel()] = True
        survives = keep_root[block["root"]]
        if not bool(survives.any()):
            continue
        chosen = indices[survives]
        kept[at : at + chosen.shape[0]] = chosen
        final[chosen.ravel()] = True
        at += chosen.shape[0]
    if at != kept_total:
        raise ValueError(f"kept triangle count disagrees: {at} != {kept_total}")
    return kept, before, final


def pass_b_finalise(
    scan: StructuredScan, pass_a: PassAResult, *, work_dir: Path,
    min_component_area: float, measure: bool, measure_samples: int,
    qa_window_rows: int, timings: dict[str, float], streaming: tuple[int, int, int],
    settings: dict[str, float | int | bool | str],
    out_dir: Path | None = None, tile_size: float = DEFAULT_TILE_SIZE_M,
    qa_workers: int = 1, qa_seed: int = 0,
) -> MeshResult:
    """Re-read the segments, cull on completed components, mesh and measure.

    The ordering is not incidental. `dropped_island` needs the completed
    component table, and retained-but-unmeshed needs certainty that *no* band
    produced a triangle containing the sample — one in band *k*'s halo may be a
    vertex of a triangle band *k+1* owns, so deciding either at band *k* would
    mis-attribute it (§4.3). Both are written once, here. QA follows, against
    final dispositions only: measuring provisional geometry is the defect
    `FINDING-002` records.

    **Round 4c — the mesh vertex store, not the whole `pos` set.** Pass A writes
    every filter-retained sample to `pos`. Only samples named by a triangle are
    ever indexed for area, cull, mesh or tiles. Loading the rest was the Class F
    Gate 1 binding term (~1.1 GB of `pos` for ~0.4% final retention). This path
    therefore builds triangle runs first, takes the unique cells they name, and
    loads only those samples. Retained-but-unmeshed is counted from the segment
    totals minus that set, never materialised.

    **Two output shapes, and the difference is the WP-3.2 memory row.** With an
    `out_dir`, the surviving triangles are filed into an immutable spatial
    generation one tile at a time and no whole-station `MeshData` is ever
    constructed (ADR-006 Decision 2). Without one there is nowhere for the
    generation to live that outlives this call, so the old shape runs instead:
    `build_mesh` over the station, exactly as before. That path is kept
    deliberately — it is the reference the tiled one is compared against, the way
    `qa_reference` is for forward QA — but it is **not** the shape the memory gate
    is measured on, and a caller that wants the gate has to say where the output
    goes.
    """
    import numpy as np

    from .obs_store import write_store
    from .pipeline import MeshResult, StreamedDiagnostics
    from .qa import deviation_report
    from .retained_io import retained_scan_for_cells
    from .reverse_qa import mesh_to_source_report_v2
    from .segments_io import read_component_alias, write_finalised_component_table
    from .tile_build import build_tiles
    from .triangulate import MIN_COMPONENT_TRIANGLES, build_mesh

    # ITEM-009. The station's triangles are never resident: they are sorted
    # per band into bounded runs, then merged twice — once to reduce area per
    # component, once to emit the survivors. What lives between the passes is
    # one entry per *component*, not one per triangle.
    #
    # Round 4e (ITEM-014). Nothing in this pass reads `pass_a.table`: Pass A
    # wrote the completed table to disk before returning, and the file is what
    # is read here. `mesh_station_from_chunks` releases that in-memory table at
    # the handoff, because it is 30.8 MB on a 175,274-component station and is
    # otherwise still live at the moment this pass sets its peak.
    alias = read_component_alias(pass_a.component_table_path)
    runs, mesh_cells = build_triangle_runs(pass_a.segments, alias, work_dir)

    # Round 4c: load only triangle-named pos samples (the mesh vertex store).
    pos_samples_total = sum(seg.sample_count for seg in pass_a.segments)
    retained = retained_scan_for_cells(scan, pass_a.segments, mesh_cells)
    retained_unmeshed = pos_samples_total - len(retained)
    cells = retained.row.astype(np.int64) * scan.lattice.cols + retained.col
    if cells.size and not bool(np.all(np.diff(cells) > 0)):
        raise ValueError("pos segments are not in strictly increasing cell order")
    if cells.size != mesh_cells.size or (
        cells.size and not bool(np.array_equal(cells, mesh_cells))
    ):
        raise ValueError("mesh vertex store does not match the triangle cell set")

    accumulator = stream_component_areas(runs, cells, retained.xyz)
    flagged = accumulator.needs_exact()
    if flagged:
        accumulator.totals.update(
            exact_component_areas(runs, cells, retained.xyz, flagged)
        )
    area = accumulator.as_area_result(flagged)

    survives = (
        np.ones(area.root_ids.size, dtype=bool)
        if min_component_area <= 0
        else (area.areas >= min_component_area)
        & (area.counts >= MIN_COMPONENT_TRIANGLES)
    )
    keep_root = np.zeros(alias.size, dtype=bool)
    keep_root[area.root_ids] = survives
    kept_total = int(area.counts[survives].sum())

    triangles_before_cull = runs.record_count
    write_finalised_component_table(
        work_dir, pass_a.component_table_path, area,
        min_component_area=min_component_area, min_triangles=MIN_COMPONENT_TRIANGLES,
    )

    t0 = time.perf_counter()
    mesh: Any = None
    tiles = None
    built: Any = None
    qa_source: Any = None
    observation_bytes = 0
    observation_count = 0
    if out_dir is None:
        kept_tris, before, final = stream_kept_triangles(
            runs, cells, keep_root, kept_total, len(retained)
        )
        # Released before the mesh is assembled: the cell index and the run
        # buffers have no reader left, and `build_mesh` is the largest single
        # allocation in this pass. Holding them across it is how the old shape
        # reached its peak.
        del cells
        mesh = build_mesh(retained, kept_tris)
        triangles_after_cull = int(kept_tris.shape[0])
        del kept_tris
        qa_source = mesh
    else:
        built = build_tiles(
            retained, runs, cells, keep_root,
            work_dir=out_dir, tile_size=tile_size,
        )
        del cells
        before, final = built.before, built.final
        triangles_after_cull = built.triangles_written
        tiles = built.store
        # WP-1.G1b B2. Written here because this is the one moment the pipeline
        # holds the retained set and knows which of it survived the cull, and
        # because the generation directory is already open. Costs one sequential
        # write and no second read of the source.
        observation_bytes = write_store(
            out_dir / "generations" / "00000000" / "obs" / "observations.rmobs",
            retained, final, retained.pose,
        )
        observation_count = int(final.sum())
        # WP-3.3: both QA directions read the written generation. `TileStore` is
        # a `qa_stream.GeometrySource`, so neither direction has to know whether
        # it was handed a resident mesh or a tile store.
        qa_source = tiles
    timings["assemble"] = time.perf_counter() - t0

    deviation = None
    reverse = None
    reverse_evidence = None
    triangle_count = (
        triangles_after_cull if mesh is None else mesh.triangle_count
    )
    if measure and triangle_count and qa_source is not None:
        t0 = time.perf_counter()
        offsets = retained.pose.rotate_local(retained.xyz[final]).astype(np.float32)
        deviation = deviation_report(
            qa_source, offsets, max_samples=measure_samples,
            seed=qa_seed, workers=qa_workers,
        )
        # §4.4: both directions run here, after culling and the ledger, against
        # final dispositions only.
        reverse, reverse_evidence = mesh_to_source_report_v2(
            qa_source, offsets, source_rows=retained.row[final],
            qa_window_rows=qa_window_rows, max_samples=measure_samples,
            seed=qa_seed, work_dir=work_dir,
        )
        timings["measure"] = time.perf_counter() - t0

    source_count = scan.source_sample_count
    stats = FilterStats(
        input_points=(
            source_count
            if source_count is not None
            else len(scan) + scan.dropped_no_return + scan.dropped_other
        ),
        retained=int(final.sum()),
        dropped_no_return=scan.dropped_no_return,
        dropped_despeckle=pass_a.dropped_despeckle,
        dropped_mover_carve=pass_a.dropped_mover_carve,
        dropped_island=int((before & ~final).sum()),
        dropped_other=scan.dropped_other + retained_unmeshed,
        restored_from_carve=pass_a.restored_from_carve,
    )
    stats.require_balanced()

    band_rows, chunk_points, halo = streaming
    diagnostics = StreamedDiagnostics(
        band_rows=band_rows, chunk_points=chunk_points, halo=halo,
        band_count=pass_a.band_count,
        max_live_components=pass_a.max_live_components,
        max_frontier_occupied=pass_a.max_frontier_occupied,
        component_count=int(area.root_ids.size),
        triangles_before_cull=triangles_before_cull,
        triangles_after_cull=triangles_after_cull,
        retained_unmeshed=retained_unmeshed,
        component_area_version=area.version,
        max_area_ratio=area.max_ratio,
        area_fallback_components=area.fallback_components,
        segment_bytes=sum(
            p.stat().st_size
            for seg in pass_a.segments
            for p in (seg.tri_path, seg.pos_path, seg.disp_path)
        ),
        tiles=None if built is None else built.describe(),
        tile_bytes=0 if built is None else built.tile_bytes,
        observation_bytes=observation_bytes,
        observation_count=observation_count,
    )
    return MeshResult(
        mesh=mesh, stats=stats, lattice=scan.lattice, deviation=deviation,
        mesh_to_source=reverse, timings=timings, settings=settings,
        station_id=scan.station_id, diagnostics=diagnostics,
        reverse_qa_evidence=reverse_evidence, tiles=tiles,
    )
