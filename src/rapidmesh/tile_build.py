"""
Pass B's incremental tile emission — ADR-006 Decision 2, DEC-009 step 3, WP-3.2.

ADR-006's arithmetic is the reason this file exists: the reference station's
finished mesh is **628.9 MB of output alone**, against a 512 MB working-memory
budget, so *"output is written incrementally as spatial tiles, not assembled and
then saved"*. Decision 1 excludes bytes already written from the working budget;
what it does not excuse is holding the mesh while writing it.

What this removes, precisely
----------------------------
Before: `stream_kept_triangles` returned the station's kept triangles as one
`(T,3)` int64 array — 48 bytes per retained sample at ADR-006's measured 1.97
triangles per vertex — and `build_mesh` then turned it into a whole-station
`MeshData` with float32 positions, uint32 faces, float32 normals and colour
resident at once. Neither exists now. The triangles are filed to per-tile spools
as they stream past, and each tile is assembled, written, digested and released
before the next is touched.

What Round 4c removed on top
----------------------------
Pass B no longer rebuilds the whole `pos` set. It loads only the lattice cells
named by a triangle (`retained_scan_for_cells`). On real structured stations
that is the difference between ~O(input) resident samples and ~O(mesh vertices).
Per-tile position spools remain a separate, later bound if the mesh vertex
store itself grows to the budget.

The lifecycle, since the gate asks for it in writing
-----------------------------------------------------
    Pass alpha  merge -> vertex-membership masks         2 bool arrays
    bounds      keep_idx blocks -> grid                  one block projected
    assign      keep_idx blocks -> vertex_tile           4 B per vertex
    Pass beta   merge -> per-tile triangle spools        one merge block
    finalise    one tile: read, remap, normals, write    one tile
    publish     manifest, then current.json last         nothing

At no point are a whole-station vertex array and a whole-station triangle array
live together, and no `MeshData` for the station is ever constructed.

Normals, and why a tile stores fewer of them than it has vertices
------------------------------------------------------------------
A triangle is spooled to the tile of **each of its corners**, so a tile holds
every triangle incident to a vertex it owns, in the order the merge produced
them. Repeating `build_mesh`'s three-column `np.add.at` over that subsequence
gives the identical float64 sum — which is why `tile_io` stores normals only for
owned vertices and refuses, structurally, to store a partial one. See
`tile_io.py`'s module docstring.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .memory import arm_top_allocations, record_phase
from .tiles import CURRENT_NAME, MANIFEST_NAME, TILE_CONTRACT, TileGrid, TileStore

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .pass_b_merge import RunSet
    from .types import ScanPose, StructuredScan

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

# Vertices projected per block while the grid bounds and the ownership map are
# built. 262,144 float32 positions is 3.1 MB, the same order as the QA caps.
PROJECT_BLOCK = 262_144

_SPOOL_FIELDS = [("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4"), ("owner", "<i4")]


@dataclass(frozen=True)
class TileBuildResult:
    """A written generation, plus the counts Pass B's ledger and report need."""

    store: TileStore
    before: BOOL
    final: BOOL
    grid: TileGrid
    triangles_written: int
    tile_count: int
    largest_tile_triangles: int
    largest_tile_vertices: int
    boundary_vertices: int
    tile_bytes: int

    def describe(self) -> str:
        return (
            f"{self.tile_count} tiles at {self.grid.size:g} m, "
            f"{self.triangles_written:,} triangles, {self.tile_bytes:,} B written, "
            f"largest tile {self.largest_tile_triangles:,} tris / "
            f"{self.largest_tile_vertices:,} verts, "
            f"{self.boundary_vertices:,} boundary duplicates"
        )


def build_tiles(
    retained: StructuredScan,
    runs: RunSet,
    cells: I64,
    keep_root: Any,
    *,
    work_dir: Path,
    tile_size: float,
    generation: str = "00000000",
) -> TileBuildResult:
    """Stream the surviving triangles into an immutable spatial generation."""
    import numpy as np

    from .pass_b_area import _block_indices
    from .pass_b_merge import merge_runs
    from .tile_spool import TileSpoolSet

    before, final = _membership(runs, cells, keep_root, len(retained))
    keep_idx = np.flatnonzero(final)
    remap = np.full(len(retained), -1, np.int32)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=np.int32)

    grid = _grid_for(retained.pose, retained.xyz, keep_idx, tile_size)
    vertex_tile = _assign_tiles(retained.pose, retained.xyz, keep_idx, grid)

    # WP-11m.s. Scalars only, and only when the environment asks — `nbytes`,
    # never the array, so the snapshot cannot keep alive what it is measuring.
    record_phase(
        "assign",
        retained_xyz_bytes=int(retained.xyz.nbytes),
        cells_bytes=int(cells.nbytes),
        before_bytes=int(before.nbytes), final_bytes=int(final.nbytes),
        keep_idx_bytes=int(keep_idx.nbytes), remap_bytes=int(remap.nbytes),
        vertex_tile_bytes=int(vertex_tile.nbytes),
    )

    dtype = np.dtype(_SPOOL_FIELDS)
    spool = TileSpoolSet(work_dir / "tilespool", "tri", dtype)
    written = 0
    root = work_dir / "generations" / generation
    try:
        for block in merge_runs(runs):
            survives = keep_root[block["root"]]
            if not bool(survives.any()):
                continue
            rows = _block_indices(block[survives], cells)
            corners = remap[rows]
            if corners.size and int(corners.min()) < 0:
                raise ValueError("a surviving triangle names an unmeshed sample")
            written += int(corners.shape[0])
            _spool_block(spool, corners, vertex_tile, dtype)
        spool.finish()
        # WP-11m.1 (ITEM-022). `remap` is the spool phase's own station-scale
        # array — one int32 per retained sample, 52.4 MB on ordinal 20 — and the
        # `_spool_block` call above is its last reader. It stayed bound through
        # finalise, which is where this function peaks. Round 6b' made the same
        # correction in `pass_b.py` for the same reason: being allocated before
        # the peak is not being freed before it.
        del remap
        record_phase("spooled", spooled_records=int(spool.total))
        entries, stats = _finalise_tiles(
            spool, retained, keep_idx, vertex_tile, grid, root
        )
        record_phase("finalised", tiles_written=len(entries))
    finally:
        spool.finish()
    # Every spool was removed as its tile was finalised; the directory itself is
    # scratch and has no place in a published generation.
    with contextlib.suppress(OSError):
        (work_dir / "tilespool").rmdir()

    manifest = {
        "contract": TILE_CONTRACT,
        "generation": generation,
        "grid": grid.describe(),
        # The station's float64 anchor belongs to the *generation*, not to a
        # tile: it is the same for every tile, and a generation with no tiles at
        # all — a station whose every component was culled — still has one. A
        # reader that took it from the first tile would get zeros for that case
        # and would be reassembling geometry against the wrong origin.
        "origin": [float(v) for v in retained.pose.translation],
        # Whether the *source* carried colour, which is not the same question as
        # whether any tile did. A generation with no tiles at all still has to
        # reproduce the resident path's empty colour array rather than its
        # absence, or a station that meshes to nothing compares unequal on a
        # field neither run has any data for.
        "has_rgb": retained.rgb is not None,
        "owned_vertex_count": int(keep_idx.size),
        "triangle_count": written,
        "tiles": entries,
    }
    _publish(work_dir, root, manifest, generation)
    return TileBuildResult(
        store=TileStore.open(work_dir),
        before=before,
        final=final,
        grid=grid,
        triangles_written=written,
        tile_count=len(entries),
        largest_tile_triangles=stats["largest_triangles"],
        largest_tile_vertices=stats["largest_vertices"],
        boundary_vertices=stats["boundary"],
        tile_bytes=stats["bytes"],
    )


def _membership(
    runs: RunSet, cells: I64, keep_root: Any, vertices: int
) -> tuple[BOOL, BOOL]:
    """Pre-cull and post-cull vertex-membership masks, no triangles retained.

    The same two masks `pass_b.stream_kept_triangles` produced, without its third
    return value — the station's kept triangle array, which is exactly what this
    package removes. `dropped_island` and retained-but-unmeshed are derived from
    these, so the ledger is unchanged.
    """
    import numpy as np

    from .pass_b_area import _block_indices
    from .pass_b_merge import merge_runs

    before = np.zeros(vertices, dtype=bool)
    final = np.zeros(vertices, dtype=bool)
    for block in merge_runs(runs):
        indices = _block_indices(block, cells)
        before[indices.ravel()] = True
        survives = keep_root[block["root"]]
        if not bool(survives.any()):
            continue
        final[indices[survives].ravel()] = True
    return before, final


def _grid_for(
    pose: ScanPose, xyz: F32, keep_idx: I64, tile_size: float
) -> TileGrid:
    """Bounds of the projected vertices, one block at a time."""
    import numpy as np

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for start in range(0, keep_idx.size, PROJECT_BLOCK):
        piece = _project(pose, xyz, keep_idx[start : start + PROJECT_BLOCK])
        low = np.minimum(low, piece.min(axis=0))
        high = np.maximum(high, piece.max(axis=0))
    if not np.all(np.isfinite(low)):
        low = np.zeros(3)
        high = np.zeros(3)
    return TileGrid.covering(low, high, tile_size)


def _assign_tiles(
    pose: ScanPose, xyz: F32, keep_idx: I64, grid: TileGrid
) -> Any:
    """Owning tile per mesh vertex, as int32. Four bytes a vertex, and the only
    station-wide array this package adds."""
    import numpy as np

    out = np.empty(keep_idx.size, np.int32)
    for start in range(0, keep_idx.size, PROJECT_BLOCK):
        stop = min(start + PROJECT_BLOCK, keep_idx.size)
        out[start:stop] = grid.index_of(
            _project(pose, xyz, keep_idx[start:stop])
        ).astype(np.int32)
    return out


def _project(pose: ScanPose, xyz: F32, index: I64) -> F32:
    """Scanner-local offsets to project-axis float32, exactly as `build_mesh`.

    The rotation is applied once and only then narrowed, per
    `SPATIAL-CONTRACT.md`; doing it on a subset is the same arithmetic row by
    row, so a tile's stored position is bitwise what the resident mesh held.
    """
    import numpy as np

    out: F32 = pose.rotate_local(xyz[index]).astype(np.float32)
    return out


def _spool_block(spool: Any, corners: Any, vertex_tile: Any, dtype: Any) -> None:
    """File one merge block's triangles under each distinct corner tile.

    Ordered by `(triangle, corner)` before being handed over, and the spool sorts
    stably by tile, so a tile receives its triangles in global merge order. That
    ordering is what makes the per-tile normal accumulation reproduce
    `build_mesh`'s; filing column by column instead would interleave them and
    move every boundary normal by a rounding step.
    """
    import numpy as np

    count = int(corners.shape[0])
    if count == 0:
        return
    tiles = vertex_tile[corners]
    record = np.empty(count, dtype)
    for column, name in enumerate(("v0", "v1", "v2")):
        record[name] = corners[:, column]
    record["owner"] = tiles.min(axis=1)

    fresh = np.ones((count, 3), bool)
    fresh[:, 1] = tiles[:, 1] != tiles[:, 0]
    fresh[:, 2] = (tiles[:, 2] != tiles[:, 0]) & (tiles[:, 2] != tiles[:, 1])
    rows = np.tile(np.arange(count, dtype=np.int64), 3)
    cols = np.repeat(np.arange(3, dtype=np.int64), count)
    take = fresh.T.ravel()
    rows, cols = rows[take], cols[take]
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    spool.append(tiles[rows, cols].astype(np.int64), record[rows])


def _finalise_tiles(
    spool: Any,
    retained: StructuredScan,
    keep_idx: I64,
    vertex_tile: Any,
    grid: TileGrid,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Assemble, write, digest and release one tile at a time."""
    from .tile_assemble import assemble_tile
    from .tile_io import tile_digest, write_tile, write_tile_join

    entries: list[dict[str, Any]] = []
    stats = {"largest_triangles": 0, "largest_vertices": 0, "boundary": 0, "bytes": 0}
    ids = spool.tile_ids()
    # WP-11m.s. The stage peaks inside the largest tile's assembly, and only
    # that one is worth a `tracemalloc` ranking — the counts are already known,
    # so which tile it is does not have to be discovered by watching.
    largest = max(ids, key=spool.count) if ids else None
    for tile_id in ids:
        records = _read_tile_records(spool, tile_id)
        record_phase(
            "tile_records", tile_id=int(tile_id), records_bytes=int(records.nbytes)
        )
        if tile_id == largest:
            arm_top_allocations("assemble_peak_largest_tile")
        payload, join = assemble_tile(
            tile_id, records, retained, keep_idx, vertex_tile, grid
        )
        del records
        if payload is None or join is None:
            spool.remove(tile_id)
            continue
        path = write_tile(root / "tile" / f"{tile_id:012d}.rmtile", payload)
        write_tile_join(root / "tile" / f"{tile_id:012d}.rmtjoin", join)
        entries.append({
            "tile_id": tile_id,
            "vertex_count": payload.vertex_count,
            "owned_count": payload.owned_count,
            "triangle_count": payload.triangle_count,
            "bounds": [float(v) for v in payload.bounds],
            "sha256": tile_digest(path),
        })
        stats["largest_triangles"] = max(
            stats["largest_triangles"], payload.triangle_count
        )
        stats["largest_vertices"] = max(stats["largest_vertices"], payload.vertex_count)
        stats["boundary"] += payload.vertex_count - payload.owned_count
        stats["bytes"] += path.stat().st_size
        del payload, join
        spool.remove(tile_id)
    return entries, stats


def _read_tile_records(spool: Any, tile_id: int) -> Any:
    """One tile's spool as a single array, without ever holding it twice.

    `read` already yields bounded blocks and `count` is exact, so the destination
    can be allocated once and filled in place. Collecting the blocks into a list
    and concatenating them held the whole tile's records a second time at the
    moment finalise peaks — 282.2 MB of duplicate on ordinal 20's largest tile,
    against a tile whose own mesh is 425.4 MB.

    The count check is not redundant with `read`'s. `read` returns without
    yielding when the spool file is missing, and its own check never runs; a tile
    that silently came back empty would be dropped from the generation with its
    geometry, and the manifest would still look complete.
    """
    import numpy as np

    records = np.empty(spool.count(tile_id), spool.dtype)
    at = 0
    for block in spool.read(tile_id):
        records[at : at + block.shape[0]] = block
        at += block.shape[0]
    if at != records.shape[0]:
        raise ValueError(
            f"tile {tile_id} yielded {at} records, {records.shape[0]} were appended"
        )
    return records


def _publish(
    work_dir: Path, generation_root: Path, manifest: dict[str, Any], generation: str
) -> None:
    """Manifest into the generation, then `current.json` last.

    `PHASE1-ISLANDS-FINALISATION.md` §4.5 and contract §3: publication is one
    temp-write / fsync / `os.replace` of the pointer after the generation is
    complete, so a reader sees the old complete generation or the new complete
    one and never a manifest naming bytes that are not there yet.
    """
    _write_json(generation_root / MANIFEST_NAME, manifest)
    _write_json(work_dir / CURRENT_NAME, {"generation": generation})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
