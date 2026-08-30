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

from .filters import clean
from .grid import CoarseRangeGrid, ScanGrid
from .triangulate import build_mesh, cull_islands, triangulate
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
    import numpy as np
    import numpy.typing as npt


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
    """Raised by ``mesh_station_streamed`` until band-local geometry exists.

    PLAN.md §5 item 6 (band-local despeckle/carve/restore with composed halos)
    is done and lives in ``filters.clean_bands`` / ``filters.iter_clean_bands``.
    Items 7 (streaming carve-grid build) and 8 (band-local triangulation and
    island finalisation) are the work still missing, and item 7 is why this
    entry point cannot yet supply ``others`` without materialising every
    neighbour scan. This stub must not call ``mesh_station`` and must not
    synthesise geometry by reassembling bands.
    """


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
) -> MeshResult:
    """Streamed entry point for the equivalence harness (PLAN.md §5 item 5).

    Signature accepts the streaming axes ``band_rows``, ``chunk_points`` and
    ``halo`` so the harness matrix can name configurations before band-local
    geometry is connected. Raises ``StreamedMeshingNotImplemented`` until
    PLAN.md §5 items 7 and 8 are implemented. Does not call ``mesh_station``.

    The filtering half is available now: ``filters.iter_clean_bands`` runs
    despeckle, carve and restore band by band and yields core-owned output,
    and ``filters.clean_bands`` is the drop-in equivalent of ``filters.clean``.
    Wiring them in here would still need whole-neighbour carve grids (item 7)
    and band-local triangulation with two-pass island finalisation (item 8),
    so this entry point stays a stub rather than becoming a partial pipeline
    that quietly skips them.
    """
    # Signature is the future contract; parameters are reserved until items 7-8.
    _ = (
        scan,
        others,
        max_incidence_deg,
        noise_floor,
        min_component_area,
        despeckle,
        measure,
        measure_samples,
    )
    raise StreamedMeshingNotImplemented(
        "mesh_station_streamed is not implemented: PLAN.md §5 item 6 "
        "(band-local despeckle/carve/restore with composed halos) is done in "
        "filters.clean_bands, but this path still requires items "
        "7 (streaming carve-grid build) and 8 (band-local triangulation "
        "and island finalisation). band_rows="
        f"{band_rows} chunk_points={chunk_points} halo={halo}"
    )


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

    used_before_cull = _used_vertices(len(grid.scan), tris_before_cull)
    used_final = _used_vertices(len(grid.scan), tris)

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


def _used_vertices(
    n: int, tris: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Boolean source-membership mask for a triangle array."""
    import numpy as np

    out = np.zeros(n, dtype=bool)
    array = np.asarray(tris)
    if array.size:
        out[array.ravel()] = True
    return out


def carve_grids(scans: list[StructuredScan], exclude: str, nearest: int = 2) -> list[CoarseRangeGrid]:
    """Coarse range grids from the `nearest` stations to `exclude`.

    Two neighbours is Cairn's choice and a reasonable default: carving recall
    rises quickly with the first two and slowly after, while cost is linear.
    In open areas with widely-spaced setups, three or four is worth trying —
    `qa.score_mover_filter` on a synthetic fixture will show whether it helps.
    """
    import numpy as np

    target = next((s for s in scans if s.station_id == exclude), None)
    if target is None:
        return []
    o = target.pose.translation
    ranked = sorted(
        (s for s in scans if s.station_id != exclude),
        key=lambda s: float(np.linalg.norm(s.pose.translation - o)),
    )
    return [CoarseRangeGrid.build(s) for s in ranked[:nearest]]
