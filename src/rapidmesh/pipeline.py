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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .carvegrid import (
    DEFAULT_CHUNK_POINTS,
    GridStore,
    StationRef,
    build_or_load,
    select_neighbours,
)
from .filters import clean
from .grid import CARVE_MAX_CELLS, CoarseRangeGrid, ScanGrid
from .triangulate import build_mesh, cull_islands, triangulate, used_vertices
from .types import (
    DeviationReport,
    FilterStats,
    LatticeInfo,
    MeshData,
    QAReportMetadata,
    StationQAReport,
    StructuredScan,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class MeshResult:
    """Everything produced for one station, including the parts that are
    inconvenient. A result with no deviation figure is an incomplete result."""

    mesh: MeshData
    stats: FilterStats
    lattice: LatticeInfo
    deviation: DeviationReport | None = None
    mesh_to_source: DeviationReport | None = None
    timings: dict[str, float] = field(default_factory=dict)
    settings: dict[str, float | int | bool] = field(default_factory=dict)
    station_id: str = ""
    diagnostics: StreamedDiagnostics | None = None
    """Internal streamed-run evidence. Never reaches `evidence_report`."""

    def summary(self) -> str:
        lines = [
            f"station    {self.station_id or '(unnamed)'}",
            f"lattice    {self.lattice.describe()}",
            f"filters    {self.stats.summary()}",
            f"mesh       {self.mesh.vertex_count:,} verts  {self.mesh.triangle_count:,} tris",
        ]
        if self.deviation:
            lines.append(f"retained   {self.deviation.summary()}")
        if self.mesh_to_source:
            lines.append(f"reverse    {self.mesh_to_source.summary()}")
        if self.timings:
            total = sum(self.timings.values())
            parts = "  ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
            lines.append(f"time       {total:.2f}s  ({parts})")
        return "\n".join(lines)

    def evidence_report(
        self, source_sha256: str, peak_rss_bytes: int | None = None
    ) -> StationQAReport:
        """Package all QA outputs with reproducibility metadata.

        Station identity and coordinates are intentionally absent. The source
        digest is the join to an authorised evidence register.
        """
        from . import __version__

        metadata = QAReportMetadata(
            source_sha256=source_sha256,
            rapidmesh_version=__version__,
            settings=tuple(
                (key, str(value)) for key, value in sorted(self.settings.items())
            ),
            exclusions=(
                "no-return",
                "despeckled",
                "carved",
                "island-culled",
                "otherwise-excluded",
            ),
            processing_seconds=sum(self.timings.values()),
            peak_rss_bytes=peak_rss_bytes,
        )
        return StationQAReport(
            retained_surface=self.deviation,
            filtering_ledger=self.stats,
            mesh_to_source=self.mesh_to_source,
            metadata=metadata,
            lattice_source=self.lattice.source.value,
        )


class StreamedMeshingNotImplemented(NotImplementedError):
    """Retained for the harness's named-failure contract.

    ``mesh_station_streamed`` raised this while PLAN.md §5 items 6–8 were
    outstanding. All three now exist, so the streamed entry point runs and this
    exception is no longer raised on the implemented path. It stays defined —
    and stays exported — because it is the harness's way of distinguishing "not
    built yet" from a crash, and a configuration this pipeline genuinely cannot
    stream must still say so by name rather than by traceback.
    """


@dataclass(frozen=True)
class StreamedDiagnostics:
    """Internal, non-exported evidence about one streamed run.

    Deliberately not part of ``StationQAReport``: the equivalence harness
    compares metadata field for field (SPEC §7), so the streamed run's settings
    must be identical to the in-memory run's. The streaming axes and the
    `PHASE1-ISLANDS-FINALISATION.md` §7 claim-A instrumentation live here
    instead, where a test can read them and a report cannot accidentally claim
    them as product metadata.
    """

    band_rows: int
    chunk_points: int
    halo: int
    band_count: int
    max_live_components: int
    max_frontier_occupied: int
    component_count: int
    triangles_before_cull: int
    triangles_after_cull: int
    retained_unmeshed: int
    component_area_version: str
    max_area_ratio: float
    area_fallback_components: int
    segment_bytes: int


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
    work_dir: str | None = None,
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

    ``chunk_points`` is accepted and recorded but is not yet a live axis on
    this path: it belongs to `e57_reader.iter_row_bands`, and driving Pass A
    from a raw E57 stream is not part of item 8. It is in the signature and in
    the diagnostics so the harness matrix can vary it and demonstrate that the
    output does not move, which is a weaker statement than exercising it and is
    reported as such.
    """
    import tempfile
    from pathlib import Path

    from .islands import pass_a_sweep, pass_b_finalise

    t: dict[str, float] = {}
    owned = work_dir is None
    root = (
        Path(tempfile.mkdtemp(prefix="rapidmesh-stream-"))
        if owned
        else Path(str(work_dir))
    )
    try:
        t0 = time.perf_counter()
        pass_a = pass_a_sweep(
            scan,
            work_dir=root,
            others=others,
            despeckle=despeckle,
            band_rows=band_rows,
            halo=halo,
            max_incidence_deg=max_incidence_deg,
            noise_floor=noise_floor,
        )
        t["pass_a"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = pass_b_finalise(
            scan,
            pass_a,
            work_dir=root,
            min_component_area=min_component_area,
            measure=measure,
            measure_samples=measure_samples,
            timings=t,
            settings={
                "max_incidence_deg": max_incidence_deg,
                "noise_floor": noise_floor,
                "min_component_area": min_component_area,
                "despeckle": despeckle,
                "measure": measure,
                "measure_samples": measure_samples,
            },
            streaming=(band_rows, chunk_points, halo),
        )
        t["pass_b"] = time.perf_counter() - t0
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
    from .qa import deviation_report, mesh_to_source_report

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
    if measure and mesh.triangle_count:
        import numpy as np

        t0 = time.perf_counter()
        retained_offsets = grid.scan.pose.rotate_local(
            grid.scan.xyz[used_final]
        ).astype(np.float32)
        dev = deviation_report(mesh, retained_offsets, max_samples=measure_samples)
        reverse = mesh_to_source_report(
            mesh, retained_offsets, max_samples=measure_samples
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
        },
        station_id=scan.station_id,
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
