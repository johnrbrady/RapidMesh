"""Decimate a generation under a chosen output partition — WP-3.6.

`decimate_tiles.decimate_generation` is unchanged and stays the reference: it
decimates one intermediate tile at a time, which is the DEC-021 512x512 window,
and Rounds 10 to 12 are all measured against it. This runs the same decimator
over patches chosen by `output_tiles.OutputPartition` instead, so that ADR-006's
Phase 3b question — what output tile size — can be asked by measurement.

**Nothing about the decimator changes.** A patch still arrives at
`decimate.decimate_patch` with the locked set `decimate_tiles.lock_boundary`
computes, under the settings the caller passes, and at
`OutputPartition(kind="lattice", rows=512, cols=512)` the patches are the
intermediate tiles themselves — so this path reproduces `decimate_generation`
byte for byte, and `tests/test_output_tiles.py` holds it to that. A benchmark
whose own baseline case did not reproduce the thing it is benchmarking against
would measure nothing.

What the ledger carries, and why each field is there
-----------------------------------------------------
ADR-006's Phase 3b list, in its own order, and where each comes from:

1. *peak working memory* — the caller's business, measured around this call in
   a child process (`rapidmesh.memory.rss_sample`), because a peak is a
   high-water mark and a function cannot measure its own.
2. *total output size* — `output_bytes` is what was written; `geometry_bytes` is
   positions and triangles alone. Both, because `.npz` is not the container and
   DEC-005 puts that choice at Round 15: a container-neutral number is the one
   that survives that decision.
3. *percentage of locked boundary vertices* — `locked_fraction`, over owned
   vertices, which is the surface the partition refused permission to simplify.
4. *seam-local maximum deviation* — `output_measure.seam_local_deviation`, run
   by the caller over what this wrote.
5. *cracks or normal discontinuities* — the crack half here, exactly, via
   `decimate_tiles.check_seams`; the kink half in `output_measure`.
6. *decimation ratio* — `triangles_in / triangles_out`.
7. *LOD0 tile size and decode time* — per-tile bytes here; decode time is timed
   separately by the caller, since timing a read inside the writer would time
   the page cache rather than a decode.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .decimate import DecimationSettings, decimate_patch
from .decimate_tiles import boundary_edges, check_seams, lock_boundary
from .output_tiles import DEFAULT_MAX_PATCH_VERTICES, OutputPartition, patches

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .tiles import TileStore

    F32 = npt.NDArray[np.float32]
    I64 = npt.NDArray[np.int64]

#: The on-disk contract for a partitioned tier. Separate from
#: `decimate_tiles.DECIMATED_CONTRACT` because the manifest carries a partition
#: block that one has no field for, and a harness reading the wrong one should
#: fail loudly rather than find a missing key.
OUTPUT_CONTRACT = "rapidmesh-output-tier-v0"


@dataclass
class OutputTile:
    """What one output tile did. Counts and bytes only — no identifier."""

    key: tuple[int, int]
    source_tiles: tuple[int, ...]
    vertices_in: int
    owned_in: int
    triangles_in: int
    locked: int
    vertices_out: int
    triangles_out: int
    collapses: int
    rejected_deviation: int
    plane_deviation_bound_m: float
    output_bytes: int
    geometry_bytes: int
    seconds: float


@dataclass
class OutputRun:
    """The tier-level ledger: ADR-006's quantities 2, 3, 5 and 6 in one object."""

    partition: dict[str, Any] = field(default_factory=dict)
    tiles: list[OutputTile] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    vertices_in: int = 0
    triangles_in: int = 0
    vertices_out: int = 0
    triangles_out: int = 0
    locked_total: int = 0
    owned_total: int = 0
    output_bytes: int = 0
    geometry_bytes: int = 0
    plane_deviation_bound_m: float = 0.0
    seam_boundary_edges_before: int = 0
    seam_boundary_edges_after: int = 0
    seam_edges_lost: int = 0
    seam_edges_gained: int = 0
    decimate_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def decimation_ratio(self) -> float | None:
        """ADR-006 quantity 6."""
        return self.triangles_in / self.triangles_out if self.triangles_out else None

    @property
    def locked_fraction(self) -> float | None:
        """ADR-006 quantity 3, over owned vertices rather than over stored ones.

        Stored vertices include each tile's halo, which is an artefact of how a
        patch is assembled rather than of the partition; owned vertices are the
        surface exactly once, so the ratio is comparable between a partition that
        carries a large halo and one that carries none.
        """
        return self.locked_total / self.owned_total if self.owned_total else None

    @property
    def seams_clean(self) -> bool:
        return self.seam_edges_lost == 0 and self.seam_edges_gained == 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tiles"] = [asdict(t) for t in self.tiles]
        out.update({
            "tile_count": self.tile_count,
            "decimation_ratio": self.decimation_ratio,
            "locked_fraction": self.locked_fraction,
            "seams_clean": self.seams_clean,
        })
        return out


def decimate_output(
    store: TileStore,
    out_dir: Path,
    settings: DecimationSettings,
    partition: OutputPartition,
    *,
    max_patch_vertices: int = DEFAULT_MAX_PATCH_VERTICES,
    kernel: str = "auto",
) -> OutputRun:
    """Decimate one generation into output tiles of the given partition.

    One patch is resident at a time, so the working set follows the **output**
    tile and not the station — which is the whole reason tile size is a memory
    parameter, and is what quantity 1 measures as tile size climbs.

    A patch too large for `max_patch_vertices` is recorded in `refused` and
    skipped rather than attempted. The size that could not be held is the
    measurement; an allocation failure would report nothing at all.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    run = OutputRun(partition=partition.describe())
    before: list[I64] = []
    after: list[I64] = []
    started = time.perf_counter()

    for item in patches(store, partition, max_patch_vertices=max_patch_vertices):
        if isinstance(item, tuple):
            key, needed = item
            run.refused.append({
                "key": list(key),
                "vertices_needed": int(needed),
                "ceiling": int(max_patch_vertices),
            })
            continue

        locked = lock_boundary(item.triangles, item.vertex_count, item.owned_count)
        before.append(boundary_edges(item.triangles, item.global_ids))

        t0 = time.perf_counter()
        patch = decimate_patch(
            item.positions, item.triangles, locked, settings, kernel=kernel
        )
        seconds = time.perf_counter() - t0

        out_gids = item.global_ids[patch.source_index]
        out_locked = locked[patch.source_index]
        after.append(boundary_edges(patch.triangles, out_gids))

        path = out_dir / f"{item.key[0]:06d}_{item.key[1]:06d}.npz"
        np.savez(
            path,
            positions=patch.positions,
            triangles=patch.triangles,
            global_vertex_index=out_gids,
            locked=out_locked,
            moved=patch.moved,
        )
        geometry = int(patch.positions.nbytes + patch.triangles.nbytes)
        run.tiles.append(OutputTile(
            key=item.key,
            source_tiles=item.source_tiles,
            vertices_in=item.vertex_count,
            owned_in=item.owned_count,
            triangles_in=item.triangle_count,
            locked=patch.locked_vertices,
            vertices_out=patch.vertex_count,
            triangles_out=patch.triangle_count,
            collapses=patch.collapses,
            rejected_deviation=patch.rejected_deviation,
            plane_deviation_bound_m=patch.plane_deviation_bound_m,
            output_bytes=int(path.stat().st_size),
            geometry_bytes=geometry,
            seconds=round(seconds, 4),
        ))
        run.vertices_in += item.owned_count
        run.triangles_in += item.triangle_count
        # Owned survivors only, so a vertex shared by two output tiles is counted
        # once and the tier's vertex total is comparable across partitions.
        run.vertices_out += int(np.count_nonzero(patch.source_index < item.owned_count))
        run.triangles_out += patch.triangle_count
        run.owned_total += item.owned_count
        run.locked_total += int(np.count_nonzero(out_locked))
        run.output_bytes += int(path.stat().st_size)
        run.geometry_bytes += geometry
        run.decimate_seconds += seconds
        run.plane_deviation_bound_m = max(
            run.plane_deviation_bound_m, patch.plane_deviation_bound_m
        )
        del item, patch, locked, out_gids, out_locked

    counts = check_seams(before, after)
    run.seam_boundary_edges_before = counts[0]
    run.seam_boundary_edges_after = counts[1]
    run.seam_edges_lost = counts[2]
    run.seam_edges_gained = counts[3]
    run.total_seconds = time.perf_counter() - started

    (out_dir / "manifest.json").write_text(
        json.dumps({
            "contract": OUTPUT_CONTRACT,
            "partition": partition.describe(),
            "settings": settings.describe(),
            "source_contract": store.manifest.get("contract"),
            "run": run.to_dict(),
        }, indent=2),
        encoding="utf-8",
    )
    return run


def decode_tiles(out_dir: Path) -> dict[str, Any]:
    """ADR-006 quantity 7: per-tile bytes, and how long one takes to decode.

    **This is a first-paint proxy and is not first paint.** Time to first paint
    needs the container and the browser, which are Rounds 15 and 16; ADR-006 says
    so itself and `CLAUDE.md` §4 rule 6 forbids claiming a deferred capability.
    What this measures is the CPU cost of turning one written tile into arrays.

    Every tile is decoded once and the samples are reported individually as well
    as summarised, because ITEM-026 leaves the repeat rule for a lower-is-better
    measurement unsettled and a package should not invent one.
    """
    import numpy as np

    paths = sorted(out_dir.glob("*.npz"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        size = int(path.stat().st_size)
        t0 = time.perf_counter()
        with np.load(path) as data:
            vertices = int(np.asarray(data["positions"]).shape[0])
            triangles = int(np.asarray(data["triangles"]).shape[0])
        rows.append({
            "bytes": size,
            "vertices": vertices,
            "triangles": triangles,
            "decode_seconds": time.perf_counter() - t0,
        })
    if not rows:
        return {"tiles": 0}
    times = sorted(r["decode_seconds"] for r in rows)
    sizes = sorted(r["bytes"] for r in rows)
    return {
        "tiles": len(rows),
        "bytes_total": sum(sizes),
        "bytes_min": sizes[0],
        "bytes_median": sizes[len(sizes) // 2],
        "bytes_max": sizes[-1],
        "decode_seconds_min": times[0],
        "decode_seconds_median": times[len(times) // 2],
        "decode_seconds_max": times[-1],
        "decode_seconds_total": sum(times),
        "per_tile": rows,
        "proxy_note": (
            "LOD0 tile bytes and decode time. A first-paint proxy, not first "
            "paint: that needs the container (Round 15) and the browser "
            "(Round 16)."
        ),
    }
