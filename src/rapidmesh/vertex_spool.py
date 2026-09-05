"""
Per-tile vertex spools — WP-B, ITEM-022 T1, advisory §3.2.

`assemble_tile` needs five things per vertex: position, colour, source sample
id, lattice row, and which tile owns it. Until WP-B it got them by indexing a
station-wide `retained` scan — 481 MB of arrays on ordinal 20, held from the
`retained` stage to the end of QA so that one tile at a time could read a
264,196-vertex slice of it.

This module fills that need from one sequential pass over the `pos` segments
instead. A surviving sample is written to the spool of the window that owns it
and of every window whose 1-cell halo ring reaches it — at most four, and only
for samples on a window boundary. What is resident is one band in and the
spool's write buffers out; what a tile later reads back is bounded by
`(W+2) x (H+2)` records, which is the bound DEC-021's partition exists to give.

Order is the contract
---------------------
`pos` segments are in ascending lattice-cell order across the whole station,
and rank is monotonic in cell, so appending in segment order gives each tile its
vertices **ascending by global index**. `assemble_tile` binary-searches that, and
`TileSpoolSet` preserves append order within a tile
(`test_tiles.py::test_the_spool_preserves_append_order_within_a_tile`). Filing
window by window instead of sample by sample would interleave them and break the
search — which is why the emission below mirrors `tile_build._spool_block`'s
lexsort rather than looping over the ring offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from .cellrank import CellRank
    from .tiles import LatticeGrid
    from .types import StructuredScan

    I64 = npt.NDArray[np.int64]

#: One spooled vertex. 40 bytes, and every field is one `assemble_tile` reads.
#: `gidx` is the global vertex index (rank among surviving cells) and is what
#: the triangle spool's `v0/v1/v2` name; `owner` is the window that owns the
#: vertex, so assembly never has to re-derive ownership from geometry.
_VERTEX_FIELDS = [
    ("gidx", "<i4"), ("owner", "<i4"), ("row", "<i4"), ("pad", "<i4"),
    ("sample_id", "<i8"),
    ("xyz", "<f4", (3,)),
    ("rgb", "u1", (3,)), ("pad2", "u1"),
]

#: The 3x3 neighbourhood a 1-cell ring spans. Distinct windows among these are
#: at most four for any window larger than one cell, because a window id is a
#: pair of integer divisions and each can take at most two values over +/-1.
_RING = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def vertex_dtype() -> Any:
    import numpy as np

    return np.dtype(_VERTEX_FIELDS)


def _window_ids(
    rows: I64, cols: I64, grid: LatticeGrid, lattice_rows: int, lattice_cols: int
) -> Any:
    """Window id per (sample, ring offset), shape `(n, 9)`.

    Rows clamp at the lattice edge — there is no ring above row 0 — while
    columns wrap when the station does, because the filters already joined
    across the azimuth seam and a window there really does own geometry on both
    sides of column 0.
    """
    import numpy as np

    out = np.empty((rows.shape[0], len(_RING)), np.int64)
    for slot, (dr, dc) in enumerate(_RING):
        near_row = np.clip(rows + dr, 0, lattice_rows - 1)
        near_col = (
            (cols + dc) % lattice_cols
            if grid.wrap
            else np.clip(cols + dc, 0, lattice_cols - 1)
        )
        out[:, slot] = grid.index_of(near_row * lattice_cols + near_col)
    return out


def spool_vertices(
    spool: Any,
    *,
    cell: I64,
    sample_id: I64,
    xyz: Any,
    rgb: Any,
    grid: LatticeGrid,
    final: CellRank,
    lattice_rows: int,
    lattice_cols: int,
) -> int:
    """File one band's surviving samples into their window and ring spools."""
    import numpy as np

    keep = final.contains(cell)
    count = int(keep.sum())
    if not count:
        return 0
    kept_cell = cell[keep]
    rows = kept_cell // lattice_cols
    cols = kept_cell % lattice_cols

    record = np.zeros(count, vertex_dtype())
    record["gidx"] = final.rank(kept_cell).astype(np.int32)
    record["row"] = rows.astype(np.int32)
    record["sample_id"] = sample_id[keep]
    record["xyz"] = xyz[keep]
    if rgb is not None:
        record["rgb"] = rgb[keep]

    wids = _window_ids(rows, cols, grid, lattice_rows, lattice_cols)
    record["owner"] = wids[:, _RING.index((0, 0))].astype(np.int32)

    # Distinct windows per sample, keeping the sample's own first appearance.
    ordered = np.sort(wids, axis=1)
    fresh = np.ones(ordered.shape, bool)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]

    n = count
    slots = len(_RING)
    row_index = np.tile(np.arange(n, dtype=np.int64), slots)
    slot_index = np.repeat(np.arange(slots, dtype=np.int64), n)
    take = fresh.T.ravel()
    row_index, slot_index = row_index[take], slot_index[take]
    # Sample-major: every window a sample belongs to is emitted before the next
    # sample, so each tile receives its vertices in ascending `gidx`.
    order = np.lexsort((slot_index, row_index))
    row_index, slot_index = row_index[order], slot_index[order]
    spool.append(ordered[row_index, slot_index], record[row_index])
    return count


def spool_vertices_from_segments(
    spool: Any,
    segments: Sequence[Any],
    *,
    grid: LatticeGrid,
    final: CellRank,
    meshed: CellRank,
    lattice_rows: int,
    lattice_cols: int,
    has_sample_id: bool,
) -> int:
    """One sequential pass over `pos`, filling every tile's vertex spool.

    `has_sample_id` decides what a vertex's `source_sample_id` is, and the two
    answers are the ones the station-backed path gave: the scan's own id where
    it has one, and otherwise the vertex's position in the **meshed** cell set,
    which is what indexing `retained` by row used to produce.
    """
    from .segments_io import read_pos_segment

    written = 0
    for segment in segments:
        block = read_pos_segment(segment.pos_path)
        cell = block["cell"]
        if cell.size == 0:
            continue
        ids = block["sample_id"] if has_sample_id else meshed.rank(cell)
        written += spool_vertices(
            spool, cell=cell, sample_id=ids, xyz=block["xyz"], rgb=block["rgb"],
            grid=grid, final=final,
            lattice_rows=lattice_rows, lattice_cols=lattice_cols,
        )
    return written


def vertices_from_scan(
    retained: StructuredScan, keep_idx: I64, vertex_tile: Any, used: I64
) -> Any:
    """One tile's vertices, gathered from the station, for the **metric** path.

    The metric grid decides ownership by projecting positions, so it cannot be
    filled by a lattice ring and it still needs the station. That is acceptable
    because the metric partition is not production — DEC-021 moved the
    intermediate to lattice windows, and `tile_build.build_tiles` keeps the
    metric branch for the tests that pin pre-WP-A behaviour. Building the same
    record layout here is what lets `assemble_tile` have one code path instead
    of a production one and a test one.

    `used` is the tile's own global vertex indices, ascending, so this is
    bounded by the tile exactly as the lattice spool is. Building the whole
    surviving set instead would put a 40-byte-per-vertex station-scale array
    back into the pass — which is the thing this package removes, and it showed
    up immediately as a changed figure on `test_extent_ladder.py`.
    """
    import numpy as np

    rows = keep_idx[used]
    out = np.zeros(int(used.size), vertex_dtype())
    out["gidx"] = np.asarray(used, np.int32)
    out["owner"] = np.asarray(vertex_tile, np.int32)[used]
    out["row"] = np.asarray(retained.row, np.int32)[rows]
    out["sample_id"] = (
        rows.astype(np.int64)
        if retained.sample_id is None
        else np.asarray(retained.sample_id, np.int64)[rows]
    )
    out["xyz"] = retained.xyz[rows]
    if retained.rgb is not None:
        out["rgb"] = retained.rgb[rows]
    return out
