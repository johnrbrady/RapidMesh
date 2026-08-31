"""
`MeshResult` — everything one station produced, including the inconvenient parts.

Split out of `pipeline.py` by WP-3.2, for the ordinary reason: the tiled path
added `tiles` and made `mesh` optional, and `pipeline.py` went past the 500-line
rule (DEC-010). Nothing else moved with it, and `pipeline` re-exports the name so
`from .pipeline import MeshResult` keeps working.

The one thing worth knowing before reading it: **`mesh` is `None` on the tiled
path.** ADR-006 Decision 2 forbids assembling a whole-station `MeshData`, so a
run that wrote an incremental spatial generation has `tiles` and no `mesh`. A
reader that wants the array anyway is the equivalence harness, and it goes
through `tile_equivalence.reconstitute_mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .evidence import DEFAULT_QA_WORKERS, FRAME_PATH_UNRECORDED

if TYPE_CHECKING:
    from .reverse_qa import ReverseQAEvidence
    from .streaming import StreamedDiagnostics
    from .tiles import TileStore
    from .types import (
        DeviationReport,
        FilterStats,
        LatticeInfo,
        MeshData,
        StationQAReport,
    )


@dataclass
class MeshResult:
    """Everything produced for one station, including the parts that are
    inconvenient. A result with no deviation figure is an incomplete result."""

    mesh: MeshData | None
    """The station's mesh, when a run kept one. `None` on the tiled path, where
    ADR-006 Decision 2 forbids assembling it; read `tiles` instead, or reassemble
    through `tile_equivalence.reconstitute_mesh` if you are the harness."""
    stats: FilterStats
    lattice: LatticeInfo
    deviation: DeviationReport | None = None
    mesh_to_source: DeviationReport | None = None
    timings: dict[str, float] = field(default_factory=dict)
    settings: dict[str, float | int | bool | str] = field(default_factory=dict)
    station_id: str = ""
    diagnostics: StreamedDiagnostics | None = None
    """Internal streamed-run evidence. Never reaches `evidence_report`."""
    tiles: TileStore | None = None
    """The written spatial generation, when the run produced one (WP-3.2)."""
    reverse_qa_evidence: ReverseQAEvidence | None = None
    """How the reverse figure was produced — window, sample counts, version.
    The window and version also reach `settings`, and therefore the exported
    metadata; the counts stay here because they are results, not settings."""
    qa_workers: int = DEFAULT_QA_WORKERS
    """KD-tree query threads this run used. Recorded, never `-1`."""
    seeds: dict[str, int] = field(default_factory=dict)
    """Every RNG seed that could move a figure (SPEC §2)."""
    frame_path: str = FRAME_PATH_UNRECORDED
    """Which `_resolve_frame` branch the source took, or `unrecorded`."""

    def _mesh_line(self) -> str:
        if self.mesh is not None:
            return (
                f"{self.mesh.vertex_count:,} verts  "
                f"{self.mesh.triangle_count:,} tris"
            )
        if self.tiles is not None:
            return (
                f"{self.tiles.vertex_count:,} verts  "
                f"{self.tiles.triangle_count:,} tris  "
                f"in {len(self.tiles.tile_ids)} tiles at "
                f"{self.tiles.tile_size:g} m"
            )
        return "(no geometry)"

    def summary(self) -> str:
        lines = [
            f"station    {self.station_id or '(unnamed)'}",
            f"lattice    {self.lattice.describe()}",
            f"filters    {self.stats.summary()}",
            f"mesh       {self._mesh_line()}",
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
        """Package all QA outputs with the full reproduction context.

        The envelope itself is assembled in `evidence.py`; this is the entry
        point callers already had. Station identity and coordinates are
        intentionally absent — the source digest is the join to an authorised
        evidence register.
        """
        from .evidence import build_station_report

        return build_station_report(self, source_sha256, peak_rss_bytes)
