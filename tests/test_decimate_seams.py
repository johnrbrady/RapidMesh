"""
No cracks and no T-junctions at tile seams — WP-3.4 task 2, Round 10.

This is the package's sensitivity evidence, and it is built the way the
supervision protocol prefers: the check is shown **red against a deliberately
broken seam** before it is shown green against the real rule. A green run of a
check nobody has seen fail proves only that the check ran.

The fixture, and why it is the real structure
----------------------------------------------
A `rows x cols` grid, split by column into two tiles that mirror what
`tile_assemble` actually writes:

* tile 0 **owns** the left columns and emits every triangle with a corner in
  them — ADR-006 2a's lowest-owning-tile rule — so its vertex array is its owned
  columns plus a one-column **ring** it stores positions for but does not own;
* tile 1 owns the right columns and emits only the triangles whose three corners
  are all its own.

The shared column is therefore owned by tile 1, stored by both, and reached by
triangles from both. That is exactly a seam.

The three failures this can have, and how each shows
-----------------------------------------------------
**A crack.** One tile collapses a seam edge its neighbour keeps. The edge is
then used once across the union instead of twice, so it appears as a *gained*
boundary edge.

**A T-junction.** One tile collapses a mid-seam vertex `m`, so it holds `(a,b)`
where its neighbour still holds `(a,m)` and `(m,b)`. All three become once-used,
so all three appear as gained boundary edges.

**A near miss.** Both tiles keep the vertex but disagree about where it is.
`_position_conflicts` compares float32 bit patterns, so agreeing to a hair and
not to a bit is a conflict — which is the difference ADR-006 2a's locked
boundary exists to remove.

Honesty condition (standing rule 8)
------------------------------------
The green assertions are regression tests on new code and are expected to pass
on first run; they are not presented as proven-failing. The red assertions —
`lock_edges=False`, and the hand-built T-junction — are the proof that the check
can fail, and were confirmed red before the green ones were trusted.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import numpy as np

from rapidmesh.decimate import DecimationSettings, decimate_patch
from rapidmesh.decimate_tiles import (
    _position_conflicts,
    boundary_edges,
    check_seams,
    lock_boundary,
)

ROWS, COLS, SPLIT = 34, 40, 20


def two_tiles() -> list[dict]:
    """Two tile-shaped patches over one grid, in the layout `tile_io` writes.

    Each patch carries local positions, local triangles, the global vertex id of
    every local slot, and the owned count — which is all `lock_boundary` and
    `boundary_edges` ever need, and is why a tile can be decimated alone.
    """
    xs, ys = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
    z = 0.05 * np.sin(xs * 0.3) * np.cos(ys * 0.3)
    z = z + 0.001 * np.random.default_rng(7).standard_normal(z.shape)
    world = np.stack(
        [xs.ravel() * 0.01, ys.ravel() * 0.01, z.ravel() + 2.0], axis=1
    ).astype(np.float32)

    a = (np.arange(ROWS - 1)[:, None] * COLS + np.arange(COLS - 1)[None, :]).ravel()
    faces = np.concatenate(
        [np.stack([a, a + 1, a + COLS + 1], 1), np.stack([a, a + COLS + 1, a + COLS], 1)]
    ).astype(np.int64)

    column = np.arange(world.shape[0]) % COLS
    owner = (column >= SPLIT).astype(np.int64)          # 0 = left tile, 1 = right
    emitter = owner[faces].min(axis=1)

    tiles = []
    for tile_id in (0, 1):
        mine = faces[emitter == tile_id]
        owned = np.flatnonzero(owner == tile_id)
        used = np.unique(mine)
        ring = np.setdiff1d(used, owned, assume_unique=False)
        owned_used = np.intersect1d(owned, used, assume_unique=False)
        # Owned first, then the ring — `tile_io`'s layout, and the reason
        # `lock_boundary` can find the ring from a single integer.
        gids = np.concatenate((owned_used, ring))
        slot = np.full(world.shape[0], -1, np.int64)
        slot[gids] = np.arange(gids.size)
        tiles.append(
            {
                "tile_id": tile_id,
                "positions": world[gids],
                "triangles": slot[mine].astype(np.uint32),
                "gids": gids,
                "owned_count": int(owned_used.size),
            }
        )
    return tiles


def decimate_both(*, lock_edges: bool, factor: int = 6) -> dict:
    """Decimate each tile alone and collect the seam ledger both sides produce."""
    before, after, ids, xyz = [], [], [], []
    for tile in two_tiles():
        locked = lock_boundary(
            tile["triangles"], tile["positions"].shape[0], tile["owned_count"],
            lock_edges=lock_edges,
        )
        before.append(boundary_edges(tile["triangles"], tile["gids"]))
        patch = decimate_patch(
            tile["positions"], tile["triangles"], locked,
            DecimationSettings(target_triangles=tile["triangles"].shape[0] // factor),
        )
        out_gids = tile["gids"][patch.source_index]
        after.append(boundary_edges(patch.triangles, out_gids))
        rows = np.flatnonzero(locked[patch.source_index])
        ids.append(out_gids[rows])
        xyz.append(patch.positions[rows])
    counts = check_seams(before, after)
    return {
        "before": counts[0], "after": counts[1],
        "lost": counts[2], "gained": counts[3],
        "conflicts": _position_conflicts(ids, xyz),
        "boundary_before": before, "boundary_after": after,
    }


# -- the red cases, first ---------------------------------------------------


def test_without_the_boundary_lock_the_seam_breaks() -> None:
    """The deliberately broken seam. If this passed, the green test proves nothing."""
    broken = decimate_both(lock_edges=False)
    assert broken["lost"] + broken["gained"] > 0, (
        "with the patch boundary unlocked the two tiles must disagree at the seam"
    )


def test_a_hand_built_t_junction_is_reported() -> None:
    """A seam vertex removed on one side only, with no decimator involved.

    Tile 0 keeps `(a, m)` and `(m, b)`; tile 1 is given `(a, b)` instead. That is
    a T-junction by construction, and all three edges must show as gained.
    """
    good = decimate_both(lock_edges=True)
    a, m, b = 1_000, 1_001, 1_002
    left = np.array([[a, m], [m, b]], np.int64)
    right_intact = np.array([[a, m], [m, b]], np.int64)
    right_broken = np.array([[a, b]], np.int64)

    intact = check_seams([left, right_intact], [left, right_intact])
    assert intact[2] == 0 and intact[3] == 0

    broken = check_seams([left, right_intact], [left, right_broken])
    assert broken[3] == 3, "a, m and m, b and a, b should all become once-used"
    assert good["gained"] == 0


def test_a_seam_vertex_moved_by_one_bit_is_a_conflict() -> None:
    ids = [np.array([42], np.int64), np.array([42], np.int64)]
    same = np.array([[1.0, 2.0, 3.0]], np.float32)
    nudged = np.nextafter(same, np.float32(4.0))
    assert _position_conflicts(ids, [same, same.copy()]) == 0
    assert _position_conflicts(ids, [same, nudged]) == 1


# -- and then the green one -------------------------------------------------


def test_two_tiles_decimated_alone_still_agree_at_the_seam() -> None:
    clean = decimate_both(lock_edges=True)
    assert clean["lost"] == 0, "a boundary edge was collapsed: that is a crack"
    assert clean["gained"] == 0, "a boundary edge appeared: crack or T-junction"
    assert clean["conflicts"] == 0, "two tiles disagree about a shared position"
    assert clean["before"] == clean["after"] > 0


def test_the_locked_rule_actually_removes_work_rather_than_all_of_it() -> None:
    """The seam holds *and* the tiles are still decimated — not a null result."""
    tiles = two_tiles()
    total_in = sum(t["triangles"].shape[0] for t in tiles)
    total_out = 0
    for tile in tiles:
        locked = lock_boundary(
            tile["triangles"], tile["positions"].shape[0], tile["owned_count"]
        )
        assert 0 < int(locked.sum()) < locked.size, "everything or nothing is locked"
        patch = decimate_patch(
            tile["positions"], tile["triangles"], locked,
            DecimationSettings(target_triangles=tile["triangles"].shape[0] // 6),
        )
        total_out += patch.triangle_count
    assert total_out < total_in / 2.0
