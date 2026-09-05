"""
`CellRank` is `remap` in 1.13 bits a cell — WP-B, ITEM-022 T1.

The global vertex index has always been *rank among surviving cells in
ascending cell order*: `remap[keep_idx] = arange(keep_idx.size)` over a sorted
cell array. WP-B keeps that definition and changes only where the integer comes
from, so the test that matters is not "is rank plausible" but **"is it the same
integer a `cumsum` would give"** — checked against brute force on every cell,
not on a sample.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidmesh.cellrank import CellRank


def _brute(flags: np.ndarray) -> np.ndarray:
    """Rank by the definition: set cells strictly below each index."""
    return np.concatenate(([0], np.cumsum(flags.astype(np.int64))[:-1]))


@pytest.mark.parametrize("cells", (1, 7, 8, 9, 63, 64, 65, 1000, 4096, 5001))
def test_rank_matches_a_cumsum_on_every_cell(cells: int) -> None:
    """Every id, not a sample: an off-by-one inside one byte is exactly the
    defect this structure could plausibly have, and it would move one vertex."""
    rng = np.random.default_rng(7)
    flags = rng.random(cells) < 0.37
    bits = CellRank.of(flags)
    ids = np.arange(cells, dtype=np.int64)
    assert np.array_equal(bits.rank(ids), _brute(flags))
    assert np.array_equal(bits.contains(ids), flags)
    assert bits.total == int(flags.sum())


@pytest.mark.parametrize("flags", (True, False))
def test_the_degenerate_sets_are_right(flags: bool) -> None:
    """All-set and empty both have to work: a station whose every cell meshes
    and one that meshes nothing are both real outcomes (CLAUDE.md §4 rule 7)."""
    dense = np.full(300, flags)
    bits = CellRank.of(dense)
    ids = np.arange(300, dtype=np.int64)
    assert np.array_equal(bits.rank(ids), _brute(dense))
    assert bits.total == (300 if flags else 0)


def test_rank_is_the_index_the_old_remap_assigned() -> None:
    """The equivalence stated as the pipeline used to compute it.

    `remap` was built by numbering the *sorted meshed cells* that survived, in
    order. Rank over the lattice gives the same integer because the surviving
    set is a subsequence of the meshed set in the same ascending cell order —
    which is the property `PHASE1-DETERMINISM-SPEC.md` §5(c) already depends on.
    """
    rng = np.random.default_rng(11)
    lattice = 4096
    meshed_flags = rng.random(lattice) < 0.25
    survives = rng.random(lattice) < 0.6
    final_flags = meshed_flags & survives

    meshed_cells = np.flatnonzero(meshed_flags)          # the old sorted `cells`
    keep = final_flags[meshed_cells]                     # the old `final`
    keep_idx = np.flatnonzero(keep)                      # the old `keep_idx`
    remap = np.full(meshed_cells.size, -1, np.int32)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=np.int32)

    bits = CellRank.of(final_flags)
    surviving_cells = meshed_cells[keep]
    assert np.array_equal(
        bits.rank(surviving_cells), remap[keep_idx].astype(np.int64)
    )
    assert bits.total == keep_idx.size


def test_the_structure_is_smaller_than_what_it_replaces() -> None:
    """The reason it exists, asserted rather than described.

    Against the three arrays WP-B removes for the same lattice: a sorted int64
    cell list, an int32 remap over it, and an int32 owning-tile id per surviving
    vertex. The bar is deliberately loose — the claim is an order of magnitude,
    and pinning a ratio would make this a test of the fixture's density.
    """
    cells = 1 << 20
    rng = np.random.default_rng(3)
    flags = rng.random(cells) < 0.94          # a dense station, the worst case
    bits = CellRank.of(flags)
    mine = int(bits.packed.nbytes + bits.cumulative.nbytes)
    meshed = np.flatnonzero(flags)
    theirs = int(meshed.nbytes) + 4 * int(meshed.size) + 4 * int(flags.sum())
    assert mine < theirs / 4, (mine, theirs)
