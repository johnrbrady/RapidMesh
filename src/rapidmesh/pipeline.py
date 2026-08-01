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
from .types import DeviationReport, FilterStats, LatticeInfo, MeshData, StructuredScan

if TYPE_CHECKING:
    pass


@dataclass
class MeshResult:
    """Everything produced for one station, including the parts that are
    inconvenient. A result with no deviation figure is an incomplete result."""

    mesh: MeshData
    stats: FilterStats
    lattice: LatticeInfo
    deviation: DeviationReport | None = None
    timings: dict[str, float] = field(default_factory=dict)
    station_id: str = ""

    def summary(self) -> str:
        lines = [
            f"station    {self.station_id or '(unnamed)'}",
            f"lattice    {self.lattice.describe()}",
            f"filters    {self.stats.summary()}",
            f"mesh       {self.mesh.vertex_count:,} verts  {self.mesh.triangle_count:,} tris",
        ]
        if self.deviation:
            lines.append(f"deviation  {self.deviation.summary()}")
        if self.timings:
            total = sum(self.timings.values())
            parts = "  ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
            lines.append(f"time       {total:.2f}s  ({parts})")
        return "\n".join(lines)


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
    from .qa import deviation_report

    t: dict[str, float] = {}

    t0 = time.perf_counter()
    cleaned, stats = clean(scan, others=others, despeckle=despeckle)
    t["clean"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = ScanGrid.build(cleaned)
    tris = triangulate(
        grid, max_incidence_deg=max_incidence_deg, noise_floor=noise_floor
    )
    t["triangulate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if min_component_area > 0:
        tris = cull_islands(grid.scan.xyz, tris, min_area=min_component_area)
    t["cull"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = build_mesh(grid.scan, tris)
    t["assemble"] = time.perf_counter() - t0

    dev = None
    if measure and mesh.triangle_count:
        t0 = time.perf_counter()
        dev = deviation_report(mesh, grid.scan.xyz, max_samples=measure_samples)
        t["measure"] = time.perf_counter() - t0

    # Samples that survived filtering but ended up in no triangle disappear in
    # `build_mesh`'s compaction — either their component was culled as an
    # island, or every quad they belonged to failed the discontinuity test.
    # Both are "removed by the meshing stage" and are attributed here rather
    # than left unaccounted for.
    stats = FilterStats(
        input_points=stats.input_points,
        dropped_despeckle=stats.dropped_despeckle,
        dropped_mover_carve=stats.dropped_mover_carve,
        dropped_island=max(len(grid.scan) - mesh.vertex_count, 0),
    )

    return MeshResult(
        mesh=mesh,
        stats=stats,
        lattice=cleaned.lattice,
        deviation=dev,
        timings=t,
        station_id=scan.station_id,
    )


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
