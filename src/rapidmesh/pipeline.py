"""
Pipeline — the stages in order, with timings and honest numbers attached.

    read -> clean -> triangulate -> cull -> assemble -> measure

Decimation is not in this list yet. It goes between cull and assemble, and the
order is deliberate: decimating before the deviation measurement would leave
nothing to measure against, and decimating before island culling would waste
effort simplifying fragments about to be deleted.

Nothing here catches its own exceptions. Cairn's `run_conversion` wraps the
mesh step in a broad `except` because a scan that converts but fails to mesh is
still a successful scan, and that is the right call *at the integration
boundary*. Burying it in the library would hide real bugs during development,
so failures propagate here and the caller decides.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .carvegrid import (
    DEFAULT_CHUNK_POINTS,
    GridStore,
    StationRef,
    build_or_load,
    select_neighbours,
)
from .evidence import (
    DEFAULT_QA_WORKERS,
    FRAME_PATH_UNRECORDED,
    resolve_qa_workers,
)
from .filters import clean
from .grid import CARVE_MAX_CELLS, CoarseRangeGrid, ScanGrid
from .qa import QA_QUERY_BLOCK
from .qa_neighbours import QA_SEED_VERTICES
from .qa_stream import (
    QA_PAIR_BLOCK,
    QA_TRIANGLE_BLOCK_BYTES,
    QA_VERTEX_BLOCK_BYTES,
)
from .result import MeshResult as MeshResult
from .reverse_qa import DEFAULT_QA_WINDOW_ROWS as REVERSE_QA_WINDOW_ROWS
from .reverse_qa import REVERSE_QA_VERSION
from .streaming import StreamedDiagnostics as StreamedDiagnostics
from .streaming import StreamedMeshingNotImplemented as StreamedMeshingNotImplemented
from .tiles import DEFAULT_TILE_SIZE_M
from .triangulate import build_mesh, cull_islands, triangulate, used_vertices
from .types import (
    FilterStats,
    StructuredScan,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .e57_reader import RawChunk
    from .streaming import (
        StationMetadata,
    )


def mesh_station_streamed(
    scan: StructuredScan,
    *,
    band_rows: int,
    chunk_points: int,
    halo: int,
    others: list[CoarseRangeGrid] | None = None,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_component_area: float = 0.005,
    despeckle: bool = True,
    measure: bool = True,
    measure_samples: int = 300_000,
    qa_window_rows: int = REVERSE_QA_WINDOW_ROWS,
    qa_workers: int = DEFAULT_QA_WORKERS,
    qa_seed: int = 0,
    frame_path: str = FRAME_PATH_UNRECORDED,
    work_dir: str | None = None,
    out_dir: str | None = None,
    tile_size: float = DEFAULT_TILE_SIZE_M,
    partition: str = "lattice",
    window: tuple[int, int] | None = None,
) -> MeshResult:
    """Two-pass streamed pipeline for one station (PLAN.md §5 items 6–8).

    Pass A sweeps the lattice once — filtering each band with the composed
    halo, triangulating the quads that band owns, labelling components against
    a one-row frontier, and writing provisional segments to `work_dir` before
    moving on. Pass B re-reads those segments, evaluates `component-area-v1`
    over completed components, culls, finalises the ledger and measures.

    It does not call ``mesh_station`` and it does not reassemble the input scan
    and delegate. What it *does* reconstruct, in Pass B, is the **retained**
    set, from the `pos` segments Pass A wrote — that is the mesh's own vertex
    store, which any implementation has to materialise to emit a mesh, and it
    is what the two-pass structure exists to make affordable. The thing being
    avoided is the global triangle edge list and whole-station
    `connected_components` of `cull_islands`, which is over 1 GB of
    intermediate on the reference station before the mesh is counted.

    ``chunk_points`` is **live** since DEC-009 step 2: the scan is re-emitted as
    a chunk stream at that capacity, `iter_row_bands` reassembles row bands from
    it, and Pass A never sees the whole station. Moving chunk boundaries
    therefore genuinely moves where reads split, and the harness's demonstration
    that the output does not move is now a statement about the code rather than
    about an unused parameter.

    Handing a resident `StructuredScan` in does not itself save memory — the
    caller already has one, and the equivalence harness needs the same station
    down both paths. `mesh_station_from_chunks` is the form that never
    materialises it.
    """
    from .streaming import StationMetadata, scan_to_chunks

    return mesh_station_from_chunks(
        scan_to_chunks(scan, chunk_points),
        StationMetadata.from_scan(scan, frame_path=frame_path),
        band_rows=band_rows, chunk_points=chunk_points, halo=halo, others=others,
        max_incidence_deg=max_incidence_deg, noise_floor=noise_floor,
        min_component_area=min_component_area, despeckle=despeckle,
        measure=measure, measure_samples=measure_samples,
        qa_window_rows=qa_window_rows, qa_workers=qa_workers, qa_seed=qa_seed,
        work_dir=work_dir, out_dir=out_dir, tile_size=tile_size,
        partition=partition, window=window,
    )


def mesh_station_from_chunks(
    chunks: Iterable[RawChunk],
    metadata: StationMetadata,
    *,
    band_rows: int,
    chunk_points: int,
    halo: int,
    others: list[CoarseRangeGrid] | None = None,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_component_area: float = 0.005,
    despeckle: bool = True,
    measure: bool = True,
    measure_samples: int = 300_000,
    qa_window_rows: int = REVERSE_QA_WINDOW_ROWS,
    qa_workers: int = DEFAULT_QA_WORKERS,
    qa_seed: int = 0,
    work_dir: str | None = None,
    out_dir: str | None = None,
    tile_size: float = DEFAULT_TILE_SIZE_M,
    partition: str = "lattice",
    window: tuple[int, int] | None = None,
    converter: Any | None = None,
    chunk_factory: Any | None = None,
) -> MeshResult:
    """The streamed pipeline with **no resident input scan** — DEC-009 step 2.

    Points enter as a chunk stream and are never assembled into a whole-station
    array. What is held across the sweep is `metadata`: a pose, a lattice and
    four counts, a fixed size whatever the station's.

    Pass B rebuilds a *mesh vertex store* from the `pos` segments — only the
    lattice cells named by a triangle (Round 4c), not every filter-retained
    sample Pass A wrote. Retained-but-unmeshed stays a ledger count.

    `converter` turns one raw band into a band-local scan. The default reads
    the wire vocabulary `streaming.scan_to_chunks` emits; an E57 producer
    passes `streaming.e57_band_to_scan` bound to its station-level frame and
    colour policy, which must be decided once and not per band.
    """
    import tempfile
    from pathlib import Path

    from .islands import ComponentTable, pass_a_sweep_bands
    from .pass_b import pass_b_finalise
    from .streaming import iter_band_filter_results

    # WP-1.9 refused any count but the default here, because `pass_b.py` owned
    # this path's QA call sites and was read-only in that package: silently
    # running at 1 while recording the requested value would have put a false
    # reproduction condition in the envelope, which is the one thing the
    # envelope exists to prevent. WP-3.3 edits `pass_b.py`, so the count and the
    # seed are threaded through to the call sites and the refusal is lifted. The
    # recorded value is now the value that ran.
    workers = resolve_qa_workers(qa_workers)
    t: dict[str, float] = {}
    owned = work_dir is None
    root = (
        Path(tempfile.mkdtemp(prefix="rapidmesh-stream-"))
        if owned
        else Path(str(work_dir))
    )
    try:
        t0 = time.perf_counter()
        pass_a = pass_a_sweep_bands(
            metadata,
            iter_band_filter_results(
                chunks, metadata, others=others, despeckle=despeckle,
                band_rows=band_rows, halo=halo,
                **({} if converter is None else {"converter": converter}),
                **({} if chunk_factory is None else {"chunk_factory": chunk_factory}),
            ),
            work_dir=root,
            max_incidence_deg=max_incidence_deg,
            noise_floor=noise_floor,
        )
        t["pass_a"] = time.perf_counter() - t0

        # Round 4e (ITEM-014). Pass A's working `ComponentTable` — the union-
        # find parent list, one int object per component id, the triangle-count
        # dict and the retired set — is dead the moment Pass A writes the
        # completed table to disk, but it is reachable from `pass_a` and so
        # stayed resident through the whole of Pass B. Measured at 30.8 MB on a
        # 175,274-component station, and live at the exact moment Pass B set
        # its high-water mark. Released here, at the handoff that owns both
        # passes, rather than inside `pass_b_finalise`, which does not own its
        # argument. `pass_a_sweep` callers that read `.table` never run Pass B.
        pass_a.table = ComponentTable()

        t0 = time.perf_counter()
        result = pass_b_finalise(
            metadata.handle_scan(),
            pass_a,
            work_dir=root,
            min_component_area=min_component_area,
            measure=measure,
            measure_samples=measure_samples,
            qa_window_rows=qa_window_rows,
            timings=t,
            settings={
                "max_incidence_deg": max_incidence_deg,
                "noise_floor": noise_floor,
                "min_component_area": min_component_area,
                "despeckle": despeckle,
                "measure": measure,
                "measure_samples": measure_samples,
                "qa_window_rows": qa_window_rows,
                "qa_workers": workers,
                "qa_query_block": QA_QUERY_BLOCK,
                "qa_vertex_block_bytes": QA_VERTEX_BLOCK_BYTES,
                "qa_triangle_block_bytes": QA_TRIANGLE_BLOCK_BYTES,
                "qa_pair_block": QA_PAIR_BLOCK,
                "qa_seed_vertices": QA_SEED_VERTICES,
                "reverse_qa_version": REVERSE_QA_VERSION,
            },
            streaming=(band_rows, chunk_points, halo),
            out_dir=None if out_dir is None else Path(out_dir),
            tile_size=tile_size,
            tile_partition=partition, tile_window=window,
            qa_workers=workers,
            qa_seed=qa_seed,
        )
        t["pass_b"] = time.perf_counter() - t0
        # `pass_b_finalise` owns the MeshResult and is read-only in this
        # package, so the envelope's three run-identity fields are attached
        # here, where the values are known, rather than threaded through it.
        result.qa_workers = workers
        result.seeds = {"qa_seed": qa_seed}
        result.frame_path = metadata.frame_path
        return result
    finally:
        if owned:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


def mesh_station(
    scan: StructuredScan,
    others: list[CoarseRangeGrid] | None = None,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_component_area: float = 0.005,
    despeckle: bool = True,
    measure: bool = True,
    measure_samples: int = 300_000,
    qa_window_rows: int = REVERSE_QA_WINDOW_ROWS,
    qa_workers: int = DEFAULT_QA_WORKERS,
    qa_seed: int = 0,
    frame_path: str = FRAME_PATH_UNRECORDED,
) -> MeshResult:
    """Full pipeline for one station.

    `others` are coarse range grids from neighbouring stations, used only to
    carve out transients — never merged into the geometry. Per-station
    isolation is a design commitment, not an implementation shortcut
    (`00-PRODUCT-DEFINITION.md` §5).

    `noise_floor` should be set from the instrument: roughly six times the
    range sigma. 12 mm suits a 2 mm-sigma scanner. Too low perforates flat
    surfaces at close range; too high starts bridging genuine thin gaps.
    """
    from .qa import deviation_report
    from .reverse_qa import mesh_to_source_report_v2

    workers = resolve_qa_workers(qa_workers)
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    cleaned, stats = clean(scan, others=others, despeckle=despeckle)
    t["clean"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = ScanGrid.build(cleaned)
    tris_before_cull = triangulate(
        grid, max_incidence_deg=max_incidence_deg, noise_floor=noise_floor
    )
    t["triangulate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if min_component_area > 0:
        tris = cull_islands(
            grid.scan.xyz, tris_before_cull, min_area=min_component_area
        )
    else:
        tris = tris_before_cull
    t["cull"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = build_mesh(grid.scan, tris)
    t["assemble"] = time.perf_counter() - t0

    used_before_cull = used_vertices(len(grid.scan), tris_before_cull)
    used_final = used_vertices(len(grid.scan), tris)

    dev = None
    reverse = None
    reverse_evidence = None
    if measure and mesh.triangle_count:
        import numpy as np

        t0 = time.perf_counter()
        retained_offsets = grid.scan.pose.rotate_local(
            grid.scan.xyz[used_final]
        ).astype(np.float32)
        dev = deviation_report(
            mesh, retained_offsets, max_samples=measure_samples, seed=qa_seed,
            workers=workers,
        )
        # Reverse QA runs against final dispositions only, on the same retained
        # set the forward direction uses (`PHASE1-ISLANDS-FINALISATION.md` §4.4).
        reverse, reverse_evidence = mesh_to_source_report_v2(
            mesh, retained_offsets,
            source_rows=grid.scan.row[used_final],
            qa_window_rows=qa_window_rows, max_samples=measure_samples,
            seed=qa_seed,
        )
        t["measure"] = time.perf_counter() - t0

    # Samples that survived filtering but ended up in no triangle disappear in
    # `build_mesh`'s compaction — either their component was culled as an
    # island, or every quad they belonged to failed the discontinuity test.
    # Both are "removed by the meshing stage" and are attributed here rather
    # than left unaccounted for.
    stats = FilterStats(
        input_points=stats.input_points,
        retained=int(used_final.sum()),
        dropped_no_return=stats.dropped_no_return,
        dropped_despeckle=stats.dropped_despeckle,
        dropped_mover_carve=stats.dropped_mover_carve,
        dropped_island=int((used_before_cull & ~used_final).sum()),
        dropped_other=stats.dropped_other + int((~used_before_cull).sum()),
        restored_from_carve=stats.restored_from_carve,
    )
    stats.require_balanced()

    return MeshResult(
        mesh=mesh,
        stats=stats,
        lattice=cleaned.lattice,
        deviation=dev,
        mesh_to_source=reverse,
        timings=t,
        settings={
            "max_incidence_deg": max_incidence_deg,
            "noise_floor": noise_floor,
            "min_component_area": min_component_area,
            "despeckle": despeckle,
            "measure": measure,
            "measure_samples": measure_samples,
            "qa_window_rows": qa_window_rows,
            "qa_workers": workers,
            "qa_query_block": QA_QUERY_BLOCK,
            "qa_vertex_block_bytes": QA_VERTEX_BLOCK_BYTES,
            "qa_triangle_block_bytes": QA_TRIANGLE_BLOCK_BYTES,
            "qa_pair_block": QA_PAIR_BLOCK,
            "qa_seed_vertices": QA_SEED_VERTICES,
            "reverse_qa_version": REVERSE_QA_VERSION,
        },
        station_id=scan.station_id,
        reverse_qa_evidence=reverse_evidence,
        qa_workers=workers,
        seeds={"qa_seed": qa_seed},
        frame_path=frame_path,
    )


def carve_grids(
    scans: list[StructuredScan],
    exclude: str,
    nearest: int = 2,
    *,
    store: GridStore | None = None,
    max_cells: int = CARVE_MAX_CELLS,
    fill_holes: bool = True,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
) -> list[CoarseRangeGrid]:
    """Coarse range grids from the `nearest` stations to `exclude`.

    Two neighbours is Cairn's choice and a reasonable default: carving recall
    rises quickly with the first two and slowly after, while cost is linear.
    In open areas with widely-spaced setups, three or four is worth trying —
    `qa.score_mover_filter` on a synthetic fixture will show whether it helps.

    Taking a `list[StructuredScan]` means every neighbour's points are already
    resident before this is called, so **this signature cannot bound memory**;
    `PHASE1-TILE-CONTRACT-V0.md` §6.2 calls it evidence of behaviour rather
    than a production interface. It is kept because the in-memory callers are
    real, and it now delegates to `carve_grids_streamed`, which builds one grid
    at a time through the chunked path. `carve_grids_streamed` is the form to
    use when the neighbours are not already in memory.
    """
    refs = [StationRef.from_scan(scan) for scan in scans]
    return carve_grids_streamed(
        refs,
        exclude,
        nearest,
        store=store,
        max_cells=max_cells,
        fill_holes=fill_holes,
        chunk_points=chunk_points,
    )


def carve_grids_streamed(
    refs: Sequence[StationRef],
    exclude: str,
    nearest: int = 2,
    *,
    store: GridStore | None = None,
    max_cells: int = CARVE_MAX_CELLS,
    fill_holes: bool = True,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
) -> list[CoarseRangeGrid]:
    """The memory-safe form: metadata for every station, points for one.

    Neighbour selection reads station origins only, so the caller holds poses,
    lattices and digests for the whole project without holding anyone's point
    arrays. Each selected neighbour is then loaded, binned and released before
    the next is touched, and a cached grid skips the load entirely — the
    schedule of `PHASE1-TILE-CONTRACT-V0.md` §6.2, steps 1 to 4.

    Peak is therefore one neighbour's points plus the grids built so far, not
    every neighbour's points at once.
    """
    return [
        build_or_load(
            ref,
            store=store,
            max_cells=max_cells,
            fill_holes=fill_holes,
            chunk_points=chunk_points,
        )
        for ref in select_neighbours(refs, exclude, nearest)
    ]
