"""
One tile, assembled from its spool — WP-3.2.

This is the step ADR-006 Decision 2 calls *"write tile -> release band"*: the
only place a finished piece of surface exists in memory, and the only thing whose
size the tile parameter actually controls. Everything else in `tile_build.py` is
bounded by a byte cap; this is bounded by the tile.

Three sets, and keeping them straight is the whole job
-------------------------------------------------------
**Spooled** — every triangle with a corner in this tile. Larger than what the
tile emits, deliberately: it is exactly the incidence needed to finish an owned
vertex's normal.

**Emitted** — triangles this tile owns, by ADR-006 2a's lowest-corner-tile rule.
Each triangle is emitted by exactly one tile, so the generation's triangle
multiset is the mesh's, with multiplicity, and no de-duplication pass exists to
hide an ownership bug.

**Stored vertices** — the tile's owned vertices that any spooled triangle uses,
plus the corners of its emitted triangles that another tile owns. The second
group is the locked boundary ring of ADR-006 2a: both neighbours hold the same
position bits for it, so they agree on the seam exactly rather than nearly.
Normals are written for the first group only; `tile_io` has no slot for the
second, because a partial sum written into a normal array is indistinguishable
from a finished one.

Bitwise, against `build_mesh`
------------------------------
Positions are `pose.rotate_local(local).astype(float32)` on a subset — row-wise
arithmetic, so subsetting is exact. Normals repeat `build_mesh`'s expression
term for term, including its float64 promotion of the first difference and its
three-column `np.add.at` in merge order over the complete incident set. Neither
is "equivalent"; both are the same operations on the same bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tile_io import TileJoin, TilePayload

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .tiles import TileGrid
    from .types import StructuredScan

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


def assemble_tile(
    tile_id: int,
    records: Any,
    retained: StructuredScan,
    keep_idx: I64,
    vertex_tile: Any,
    grid: TileGrid,
) -> tuple[TilePayload | None, TileJoin | None]:
    """Build one tile's payload and its server-only join, or `(None, None)`.

    `records` is the tile's spool in merge order: one row per spooled triangle,
    carrying the three global vertex indices and the owning tile id. `keep_idx`
    maps a global vertex index back to its row in `retained`.

    A tile is written when it owns **either** a triangle or a vertex, and both
    halves of that matter. A tile whose every triangle is owned by a
    lower-indexed neighbour can still be the only holder of some vertex's exact
    normal, and dropping it would lose that vertex from the generation entirely —
    which is how the first version of this failed: the reassembled global index
    range came back with holes in it. A tile that owns neither is genuinely
    nothing and is not written.
    """
    import numpy as np

    if records.shape[0] == 0:
        return None, None

    spooled = np.stack(
        [records["v0"], records["v1"], records["v2"]], axis=1
    ).astype(np.int64)
    used = np.unique(spooled)
    local = np.searchsorted(used, spooled)

    rows = keep_idx[used]
    local_verts = retained.xyz[rows]
    normals_all = _accumulate_normals(local_verts, local, retained.pose)

    emitted = np.asarray(records["owner"], np.int64) == tile_id
    owned = vertex_tile[used] == tile_id
    if not bool(emitted.any() or owned.any()):
        return None, None
    needed = np.zeros(used.shape[0], bool)
    needed[np.unique(local[emitted])] = True
    slots = np.flatnonzero(owned | needed)
    # Owned first, then the boundary ring, each ascending by global index —
    # `used` is already sorted, so taking the two masks in order is enough.
    order = np.concatenate((slots[owned[slots]], slots[~owned[slots]]))
    owned_count = int(owned[slots].sum())

    place = np.full(used.shape[0], -1, np.int64)
    place[order] = np.arange(order.size, dtype=np.int64)
    faces = place[local[emitted]]
    if faces.size and int(faces.min()) < 0:
        raise ValueError(f"tile {tile_id} emitted a triangle with an unstored corner")

    positions = retained.pose.rotate_local(local_verts[order]).astype(np.float32)
    colour = None if retained.rgb is None else retained.rgb[rows[order]]
    payload = TilePayload(
        tile_id=tile_id,
        origin=np.asarray(retained.pose.translation, np.float64),
        bounds=_bounds(positions, grid, tile_id),
        positions=positions,
        normals=normals_all[order[:owned_count]],
        triangles=faces.astype(np.uint32),
        rgb=colour,
    )
    sample_id = (
        rows[order].astype(np.int64)
        if retained.sample_id is None
        else retained.sample_id[rows[order]].astype(np.int64)
    )
    join = TileJoin(
        source_sample_id=sample_id,
        global_vertex_index=used[order],
        row=retained.row[rows[order]].astype(np.int32),
        owned=np.arange(order.size) < owned_count,
    )
    return payload, join


def _accumulate_normals(local_verts: F32, faces: I64, pose: Any) -> F32:
    """`build_mesh`'s normals, over one tile's incident set.

    Copied expression for expression from `triangulate.build_mesh`, including
    two details that look cosmetic and are not: the first difference promotes to
    float64 before subtracting, and the three `np.add.at` calls are per column
    over all faces rather than per face over all columns. Reordering either
    changes the last bits of a float64 sum, and normals are compared at
    `PHASE1-DETERMINISM-SPEC.md` §7's T3 tier where that would show up as a real
    angle rather than as noise.
    """
    import numpy as np

    count = int(local_verts.shape[0])
    acc = np.zeros((count, 3), np.float64)
    if faces.shape[0]:
        fn = np.cross(
            local_verts[faces[:, 1]].astype(np.float64) - local_verts[faces[:, 0]],
            local_verts[faces[:, 2]].astype(np.float64) - local_verts[faces[:, 0]],
        )
        for col in range(3):
            np.add.at(acc, faces[:, col], fn)
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc /= np.maximum(norm, 1e-12)
    facing = np.einsum("ij,ij->i", acc, -local_verts.astype(np.float64))
    acc[facing < 0] *= -1.0
    out: F32 = pose.rotate_local(acc).astype(np.float32)
    return out


def _bounds(positions: F32, grid: TileGrid, tile_id: int) -> F64:
    """The tile's actual extent, clamped into nothing.

    The grid cell is what *decides* ownership; the recorded bounds are what the
    tile's geometry occupies, which is what a viewer culls against. They are not
    the same box — a boundary ring reaches outside the cell — and recording the
    cell instead would make a frustum test drop geometry that is really there.
    """
    import numpy as np

    if positions.shape[0] == 0:
        return grid.bounds_of(tile_id)
    low = positions.min(axis=0).astype(np.float64)
    high = positions.max(axis=0).astype(np.float64)
    out: F64 = np.concatenate((low, high))
    return out
