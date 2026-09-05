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
from .tiles import (
    CURRENT_NAME,
    DEFAULT_WINDOW_COLS,
    DEFAULT_WINDOW_ROWS,
    MANIFEST_NAME,
    TILE_CONTRACT,
    LatticeGrid,
    Partition,
    TileStore,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .pass_b_merge import RunSet
    from .types import ScanPose, StructuredScan

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

_SPOOL_FIELDS = [("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4"), ("owner", "<i4")]


@dataclass(frozen=True)
class TileBuildResult:
    """A written generation, plus the counts Pass B's ledger and report need."""

    store: TileStore
    #: Membership over **lattice cells**, not over a station-wide array: which
    #: cells any triangle named, and which survived the cull. WP-B replaced the
    #: two bool arrays over `retained` with these; the ledger reads their
    #: `total`, which is the same integer `mask.sum()` gave.
    meshed: Any
    surviving: Any
    grid: Partition
    triangles_written: int
    tile_count: int
    largest_tile_triangles: int
    largest_tile_vertices: int
    boundary_vertices: int
    tile_bytes: int

    def describe(self) -> str:
        cut = self.grid.describe()
        how = (
            f"{cut['window_rows']}x{cut['window_cols']} lattice windows"
            if cut.get("kind") == "lattice"
            else f"tiles at {cut.get('size_m', 0.0):g} m"
        )
        return (
            f"{self.tile_count} {how}, "
            f"{self.triangles_written:,} triangles, {self.tile_bytes:,} B written, "
            f"largest tile {self.largest_tile_triangles:,} tris / "
            f"{self.largest_tile_vertices:,} verts, "
            f"{self.boundary_vertices:,} boundary duplicates"
        )


def build_tiles(
    runs: RunSet,
    keep_root: Any,
    *,
    work_dir: Path,
    lattice: Any,
    pose: ScanPose,
    segments: Any,
    has_rgb: bool,
    has_sample_id: bool,
    wrap: bool,
    tile_size: float,
    partition: str = "lattice",
    window: tuple[int, int] | None = None,
    generation: str = "00000000",
    retained: StructuredScan | None = None,
) -> TileBuildResult:
    """Stream the surviving triangles into an immutable spatial generation.

    `partition` selects how the intermediate is cut. **`"lattice"` is the
    production partition** (DEC-021): windows of `window` = (rows, cols) lattice
    cells, which bound a window's owned vertices at `W x H` by construction.
    `"metric"` is the pre-WP-A behaviour, kept because `test_tiles.py` pins
    tile-independent outputs against it and because ADR-006's later decimated
    tiers are still metric — it is **not** a memory lever and DEC-021 says so.

    **WP-B (ITEM-022 T1).** The lattice partition takes no station-wide arrays
    at all. Membership is two bitsets over the lattice, the global vertex index
    is rank in the surviving one, ownership is an integer divide on the cell id,
    and positions arrive through a per-tile vertex spool filled by one pass over
    `pos`. `retained` is accepted **only** for the metric partition, which
    decides ownership by projecting positions and therefore cannot be filled
    from a lattice ring; passing it on the lattice path is a caller error rather
    than a fallback, because a fallback is how a station-wide array quietly
    comes back.

    `tile_size` is read only by the metric partition. It stays in the signature
    because it is a recorded setting of the generation either way, and dropping
    it would silently change what a caller's argument means.
    """
    import numpy as np

    from .cellrank import CellRank
    from .pass_b_area import _block_cells
    from .pass_b_merge import merge_runs
    from .tile_metric import _assign_tiles, _grid_for
    from .tile_spool import TileSpoolSet
    from .vertex_spool import (
        spool_vertices_from_segments,
        vertex_dtype,
        vertices_from_scan,
    )

    lattice_rows, lattice_cols = int(lattice.rows), int(lattice.cols)
    meshed_flags, final_flags = _membership(runs, keep_root, lattice_rows * lattice_cols)
    meshed = CellRank.of(meshed_flags)
    surviving = CellRank.of(final_flags)

    vertex_spool: Any = None
    lattice_grid: LatticeGrid | None = None
    keep_idx: Any = None
    vertex_tile: Any = None
    if partition == "lattice":
        if retained is not None:
            raise ValueError(
                "the lattice partition does not take a station-wide scan; "
                "passing one would defeat the change that removed it"
            )
        del meshed_flags, final_flags
        window_rows, window_cols = window or (DEFAULT_WINDOW_ROWS, DEFAULT_WINDOW_COLS)
        lattice_grid = LatticeGrid(
            lattice_rows=lattice_rows, lattice_cols=lattice_cols,
            window_rows=int(window_rows), window_cols=int(window_cols),
            wrap=bool(wrap),
        )
        grid: Partition = lattice_grid
        vertex_spool = TileSpoolSet(work_dir / "vertspool", "vert", vertex_dtype())
        spooled_vertices = spool_vertices_from_segments(
            vertex_spool, segments, grid=lattice_grid, final=surviving,
            meshed=meshed, lattice_rows=lattice_rows, lattice_cols=lattice_cols,
            has_sample_id=has_sample_id,
        )
        vertex_spool.finish()
        record_phase(
            "vertices_spooled",
            surviving=int(surviving.total), spooled=int(spooled_vertices),
            packed_bytes=int(surviving.packed.nbytes),
            cumulative_bytes=int(surviving.cumulative.nbytes),
        )
    elif partition == "metric":
        if retained is None:
            raise ValueError("the metric partition needs the retained scan")
        cells_sorted = np.flatnonzero(meshed_flags)
        keep_idx = np.flatnonzero(final_flags[cells_sorted])
        del meshed_flags, final_flags, cells_sorted
        metric_scan: StructuredScan = retained
        grid = _grid_for(pose, metric_scan.xyz, keep_idx, tile_size)
        vertex_tile = _assign_tiles(pose, metric_scan.xyz, keep_idx, grid)
    else:
        raise ValueError(
            f"unknown partition {partition!r}; expected 'lattice' or 'metric'"
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
            triple = _block_cells(block[survives])
            corners = surviving.rank(triple).astype(np.int32)
            # Ownership: arithmetic on the cell for the lattice, the projected
            # assignment for the metric grid. Either way it is per corner, and
            # `_spool_block` never learns which partition produced it.
            # Narrowed to the concrete grid on purpose: `index_of` takes cell
            # ids on the lattice and projected positions on the metric grid, and
            # `mypy --strict` will not let the `Partition` union hide that.
            tiles = (
                lattice_grid.index_of(triple).astype(np.int32)
                if lattice_grid is not None
                else vertex_tile[corners]
            )
            written += int(corners.shape[0])
            _spool_block(spool, corners, tiles, dtype)
        spool.finish()
        record_phase("spooled", spooled_records=int(spool.total))

        def vertices_for(tile_id: int, tile_records: Any) -> Any:
            """The vertices one tile needs, bounded by that tile either way.

            The lattice partition reads them from the spool the `pos` pass
            filled; the metric partition gathers them from the station for the
            indices this tile actually spooled. Bounded by the tile either way,
            so keeping the old partition for the tests does not reintroduce the
            station-scale array this package removes.
            """
            if vertex_spool is not None:
                return _read_tile_records(vertex_spool, tile_id)
            used = np.unique(
                np.stack(
                    [tile_records["v0"], tile_records["v1"], tile_records["v2"]],
                    axis=1,
                )
            ).astype(np.int64)
            return vertices_from_scan(metric_scan, keep_idx, vertex_tile, used)

        entries, stats = _finalise_tiles(
            spool, vertices_for, pose, grid, root, has_rgb=has_rgb
        )
        record_phase("finalised", tiles_written=len(entries))
    finally:
        spool.finish()
        if vertex_spool is not None:
            for spooled_tile in vertex_spool.tile_ids():
                vertex_spool.remove(spooled_tile)
    # Every spool was removed as its tile was finalised; the directories are
    # scratch and have no place in a published generation.
    for scratch in ("tilespool", "vertspool"):
        with contextlib.suppress(OSError):
            (work_dir / scratch).rmdir()

    manifest = {
        "contract": TILE_CONTRACT,
        "generation": generation,
        "grid": grid.describe(),
        # The station's float64 anchor belongs to the *generation*, not to a
        # tile: it is the same for every tile, and a generation with no tiles at
        # all — a station whose every component was culled — still has one. A
        # reader that took it from the first tile would get zeros for that case
        # and would be reassembling geometry against the wrong origin.
        "origin": [float(v) for v in pose.translation],
        # Whether the *source* carried colour, which is not the same question as
        # whether any tile did. A generation with no tiles at all still has to
        # reproduce the resident path's empty colour array rather than its
        # absence, or a station that meshes to nothing compares unequal on a
        # field neither run has any data for.
        "has_rgb": bool(has_rgb),
        "owned_vertex_count": int(surviving.total),
        "triangle_count": written,
        "tiles": entries,
    }
    _publish(work_dir, root, manifest, generation)
    return TileBuildResult(
        store=TileStore.open(work_dir),
        meshed=meshed,
        surviving=surviving,
        grid=grid,
        triangles_written=written,
        tile_count=len(entries),
        largest_tile_triangles=stats["largest_triangles"],
        largest_tile_vertices=stats["largest_vertices"],
        boundary_vertices=stats["boundary"],
        tile_bytes=stats["bytes"],
    )


def _membership(
    runs: RunSet, keep_root: Any, cell_count: int
) -> tuple[BOOL, BOOL]:
    """Pre-cull and post-cull membership, as flags over the **lattice**.

    The same two masks `pass_b.stream_kept_triangles` produced, keyed by lattice
    cell instead of by position in a station-wide array — so no such array has
    to exist for the ledger to be computed. `dropped_island` and
    retained-but-unmeshed are derived from their counts, unchanged.

    Canonical corner order is used deliberately: membership is a set question
    and the winding cannot change which cells a triangle names.
    """
    import numpy as np

    from .pass_b_merge import merge_runs

    meshed = np.zeros(cell_count, dtype=bool)
    final = np.zeros(cell_count, dtype=bool)
    for block in merge_runs(runs):
        triple = np.stack([block["c0"], block["c1"], block["c2"]], axis=1)
        meshed[triple.ravel()] = True
        survives = keep_root[block["root"]]
        if not bool(survives.any()):
            continue
        final[triple[survives].ravel()] = True
    return meshed, final


def _spool_block(spool: Any, corners: Any, tiles: Any, dtype: Any) -> None:
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
    vertices_for: Any,
    pose: ScanPose,
    grid: Partition,
    root: Path,
    *,
    has_rgb: bool,
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
        vertices = vertices_for(tile_id, records)
        payload, join = assemble_tile(
            tile_id, records, vertices, pose, grid, has_rgb=has_rgb
        )
        del records, vertices
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
