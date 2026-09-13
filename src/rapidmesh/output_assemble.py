"""Building one output patch out of intermediate tiles — WP-3.6.

`output_tiles.py` decides **which** intermediate tiles feed an output tile and
which of their vertices and triangles it owns. This is the machinery that turns
that decision into a patch `decimate.decimate_patch` will accept: deduplicate by
global id, compact owned-first, re-index, and drop what no surviving triangle
names.

The two are separate because they are separately wrong-able, and both were wrong
once in this package. The partition rules are about geometry: one re-derived a
triangle ownership Pass B had already fixed and silently dropped 274 triangles;
one reused a rule valid over row bands on ground cells, where it is not a total
order, and over-counted by 6,668. The compaction here is index bookkeeping,
shared unchanged by both partition kinds, so a fault in one is not a fault in the
other and they are tested apart.

The two ownership rules, and why one is not enough
---------------------------------------------------
A triangle must land in exactly one output tile however the surface is cut, and
the rule has to reach that answer **without consulting which tile is currently
being assembled** — otherwise two tiles can each find a reason to claim it.

* `triangle_bucket` names a **row band**. Bands are a total order, so the minimum
  band over a triangle's owned corners names one band.
* `designated_corner` names a **vertex**, and therefore whatever cell that vertex
  falls in, under any partition at all. Ground cells are not ordered, so the band
  rule cannot be reused for them.

Both fall back to the lowest-id corner for a triangle whose payload owns none,
which keeps them total rather than leaving such a triangle unclaimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class OutputPatch:
    """One output tile's geometry before decimation, plus what it may not move.

    `owned_count` vertices come first, so `decimate_tiles.lock_boundary` applies
    unchanged: everything past that mark is a halo vertex another tile owns, and
    is locked for that reason alone.
    """

    key: tuple[int, int]
    positions: F32              # (V,3) project-axis offsets, as the tile store holds
    triangles: I64              # (T,3) local indices
    global_ids: I64             # (V,)
    owned_count: int
    source_tiles: tuple[int, ...]

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])



def triangle_bucket(local: Any, bucket: Any, owned: Any) -> Any:
    """Which row band a payload's triangle belongs to, exactly once.

    The band of its lowest **owned** corner, falling back to its lowest corner
    when it owns none. Well defined without reference to which band is being
    assembled, which is what makes it exclusive: the triangle sits in one payload
    and this picks one band of it, so summing the bands reproduces the payload's
    own triangle list.

    Row bands are a total order, so a minimum over them names a band. A ground
    cell is not — see `designated_corner`, which is why the metric partition
    needs a different rule rather than this one with different numbers. Using
    this one there, with "this cell" as 0 and every other cell as 1, claimed a
    triangle for *every* cell holding one of its owned corners and over-counted
    the fixture by 6,668 triangles.
    """
    import numpy as np

    if local.shape[0] == 0:
        return np.empty(0, np.int64)
    corners = bucket[local]
    guarded = np.where(owned[local], corners, np.iinfo(np.int64).max)
    lowest = guarded.min(axis=1)
    return np.where(lowest == np.iinfo(np.int64).max, corners.min(axis=1), lowest)


def designated_corner(local: Any, owned: Any, gids: Any) -> Any:
    """The one corner that decides which output tile a triangle belongs to.

    The **owned** corner with the lowest global id, falling back to the lowest-id
    corner when the payload owns none. It names a vertex and therefore a cell —
    any cell, in any partition — without consulting which cell is being
    assembled, so every triangle is claimed exactly once however the ground is
    cut.
    """
    import numpy as np

    if local.shape[0] == 0:
        return np.empty(0, np.int64)
    ids = gids[local]
    guarded = np.where(owned[local], ids, np.iinfo(np.int64).max)
    pick = guarded.argmin(axis=1)
    none_owned = guarded.min(axis=1) == np.iinfo(np.int64).max
    if bool(none_owned.any()):
        pick = np.where(none_owned, ids.argmin(axis=1), pick)
    out: Any = local[np.arange(local.shape[0]), pick]
    return out


def finish_patch(
    key: tuple[int, int], gid: Any, pos: Any, own: Any, tri_gid: Any,
    source_tiles: tuple[int, ...],
) -> OutputPatch:
    """Deduplicate by global id, compact owned-first, re-index the triangles.

    Shared by both partition kinds, because the only thing that differs between
    them is *which* vertices a patch owns and *which* triangles it was given —
    and once those are decided the compaction is the same work. `np.unique`
    returns ascending ids, which is the deterministic order ADR-006 asks for.
    """
    import numpy as np

    unique, first = np.unique(gid, return_index=True)
    position = pos[first]
    owned_any = np.zeros(unique.shape[0], bool)
    np.logical_or.at(owned_any, np.searchsorted(unique, gid), own)

    # Compact owned-first, each half ascending in global id, so that
    # `lock_boundary`'s "everything past owned_count is halo" holds verbatim.
    order = np.concatenate([np.flatnonzero(owned_any), np.flatnonzero(~owned_any)])
    slot = np.empty(unique.shape[0], np.int64)
    slot[order] = np.arange(unique.shape[0], dtype=np.int64)
    triangles = (
        slot[np.searchsorted(unique, tri_gid)]
        if tri_gid.shape[0] else np.empty((0, 3), np.int64)
    )
    owned_count = int(owned_any.sum())
    keep = _reachable(triangles, unique.shape[0], owned_count)
    return _compact(key, position[order], triangles, unique[order],
                    owned_count, source_tiles, keep)


def _reachable(triangles: Any, count: int, owned_count: int) -> Any:
    """Which vertex slots a surviving triangle still names.

    A grouped patch inherits every source tile's halo, and most of that halo is
    another *member* of the group — already present as an owned vertex — or
    belongs to a triangle this patch does not own. Dropping the unreferenced
    ones is not tidiness: an unreferenced vertex is locked by rule 1, would be
    counted in "percentage of locked boundary vertices", and would make a
    grouped tile look as seam-bound as the tiles it replaced.
    """
    import numpy as np

    used = np.zeros(count, bool)
    if triangles.shape[0]:
        used[np.unique(triangles)] = True
    # Owned vertices are kept whether or not a triangle names them, so the
    # generation's owned-vertex total is conserved across any partition.
    used[:owned_count] = True
    return used


def _compact(
    key: tuple[int, int], position: Any, triangles: Any, gid: Any,
    owned_count: int, source_tiles: tuple[int, ...], keep: Any,
) -> OutputPatch:
    import numpy as np

    if bool(keep.all()):
        return OutputPatch(key, position, triangles, gid, owned_count, source_tiles)
    remap = np.full(keep.shape[0], -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    return OutputPatch(
        key=key,
        positions=position[keep],
        triangles=remap[triangles] if triangles.shape[0] else triangles,
        global_ids=gid[keep],
        # Owned vertices are never dropped, so the owned-first ordering and the
        # count both survive compaction untouched.
        owned_count=owned_count,
        source_tiles=source_tiles,
    )


# ---------------------------------------------------------------------------
# the metric partition
# ---------------------------------------------------------------------------


