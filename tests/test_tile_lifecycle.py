"""
What is still resident when a tile is finalised — WP-11m.1, ITEM-022, DEC-012.

`tile_build.py` already writes one tile at a time, and `test_tiles.py` already
proves no whole-station `MeshData` is ever built. Neither observes the question
Round 11m asks: at the moment a tile is assembled, **which station-scale arrays
does `build_tiles` still hold?** WP-10m measured a tiles-stage tracemalloc
marginal of 3.41 GB on ordinal 20 against a largest tile whose own mesh is
425 MB, so the difference is arrays that are alive without being needed.

These tests pin two of those, and they pin them by observation rather than by
description:

* `build_tiles` holds no station-scale array at finalise beyond the ones it must
  — the two membership masks it returns, the vertex-indexed maps the assembler
  reads, and `cells`, which is the *caller's* array and outlives the call
  whatever this function does with its own binding.
* one tile's spool is never held twice while it is turned into an array.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed, and
no number here is a real-data result.
"""

from __future__ import annotations

import pathlib
import sys
import tracemalloc
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tile_spool import SPOOL_READ_BYTES, TileSpoolSet

# Names `build_tiles` is allowed to still hold at finalise, each with a reason.
#
#   before, final   returned in `TileBuildResult`; Pass B writes the observation
#                   store and counts `dropped_island` from them.
#   keep_idx        the assembler maps a global vertex index back through it.
#   vertex_tile     the assembler decides ownership with it.
#   cells           the *caller's* array. `pass_b_finalise` and the measurement
#                   harness both hold it across this call and release it on
#                   return, so dropping the parameter binding here would change
#                   the frame without freeing a byte. Releasing it needs a
#                   two-phase entry point and is not this package.
#: WP-B: nothing. `before`, `final`, `cells`, `keep_idx`, `remap` and
#: `vertex_tile` were the station-scale locals `build_tiles` used to hold, and
#: the lattice partition builds none of them — membership is a bitset, the
#: global index is its rank, and ownership is an integer divide. An empty set is
#: the strongest form of the claim this test was already making.
PERMITTED_AT_FINALISE: frozenset[str] = frozenset()


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    """Small enough to run in seconds, large enough to make several tiles."""
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="lifecycle",
    )


def _station_scale_locals_at_finalise(
    station: synthetic.SyntheticScan, out: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    """Names in `build_tiles`' frame bound to an array with one row per sample.

    `_finalise_tiles` imports `assemble_tile` from the module at call time, so
    replacing the module attribute is enough to get a hook that runs with the
    real frame stack in place. The wrapper walks up to `build_tiles` and reports
    what it is holding; it changes nothing.
    """
    import rapidmesh.tile_assemble as tile_assemble

    real = tile_assemble.assemble_tile
    seen: set[str] = set()

    def spy(*args: Any, **kwargs: Any) -> Any:
        frame: Any = sys._getframe()
        while frame is not None and frame.f_code.co_name != "build_tiles":
            frame = frame.f_back
        assert frame is not None, "assemble_tile was not called under build_tiles"
        # WP-B: "station-scale" can no longer be sized from a `retained`
        # local, because there is not one. The surviving-cell count is what the
        # removed arrays were one element per, and it comes off the bitset the
        # frame does hold. The bitset's own arrays are `packed` and `cumulative`
        # inside `CellRank`, not frame locals, and are 1.13 bits a cell rather
        # than one element a vertex — which is the distinction being drawn.
        surviving = frame.f_locals.get("surviving")
        assert surviving is not None, "build_tiles has no `surviving` local"
        sizes = {int(surviving.total), int(frame.f_locals["meshed"].total)}
        sizes.discard(0)
        for name, value in frame.f_locals.items():
            if (
                isinstance(value, np.ndarray)
                and value.shape
                and value.shape[0] in sizes
            ):
                seen.add(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(tile_assemble, "assemble_tile", spy)
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(out), tile_size=2.0,
    )
    assert result.tiles is not None
    assert len(result.tiles.tile_ids) > 0, "fixture produced no tile to finalise"
    return seen


def test_finalise_holds_no_station_scale_array_it_does_not_need(
    station: synthetic.SyntheticScan,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifecycle claim, stated as what is alive rather than what is written.

    Red before WP-11m.1: `remap` — one int32 per retained sample, 52.4 MB on
    ordinal 20 — is built for the spool phase and has no reader once the last
    block is filed, but stayed bound through finalise, which is where this
    function peaks.

    This fails if any *new* station-scale local appears too, which is the point:
    the constraint is on the whole frame, not on one name.
    """
    held = _station_scale_locals_at_finalise(station, tmp_path / "out", monkeypatch)

    assert "remap" not in held, (
        "build_tiles still holds `remap`, a station-scale array with no reader "
        f"at finalise; frame holds {sorted(held)}"
    )
    assert "cells" not in held and "keep_idx" not in held, (
        f"build_tiles still holds a station-scale index array: {sorted(held)}"
    )
    unexpected = held - PERMITTED_AT_FINALISE
    assert not unexpected, (
        f"build_tiles holds station-scale arrays that are not accounted for: "
        f"{sorted(unexpected)}"
    )


def test_a_tile_spool_is_never_held_twice(tmp_path: pathlib.Path) -> None:
    """Reading one tile's spool allocates the destination once.

    Red before WP-11m.1: `_finalise_tiles` collected every block into a list and
    called `np.concatenate`, so the tile's records existed twice at once — the
    list and the result — 282.2 MB of duplicate on ordinal 20's largest tile.

    The bound is expressed against the array's own size rather than as an
    absolute, so it means the same thing at any station size. A single read
    block (4 MB) is allowed on top; a second full copy is not.
    """
    from rapidmesh.tile_build import _SPOOL_FIELDS, _read_tile_records

    dtype = np.dtype(_SPOOL_FIELDS)
    count = 2_000_000
    spool = TileSpoolSet(tmp_path / "spool", "tri", dtype)
    records = np.zeros(count, dtype)
    records["v0"] = np.arange(count, dtype=np.int32)
    records["v1"] = records["v0"]
    records["v2"] = records["v0"]
    records["owner"] = 0
    step = 250_000
    for start in range(0, count, step):
        piece = records[start : start + step]
        spool.append(np.zeros(piece.shape[0], np.int64), piece)
    spool.finish()
    del records

    tracemalloc.start()
    try:
        got = _read_tile_records(spool, 0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert got.shape[0] == count
    assert np.array_equal(got["v0"], np.arange(count, dtype=np.int32))

    # One destination plus the read blocks in flight, never two destinations.
    # Two blocks are legitimately live at the top of each iteration — the
    # previous one is still bound when the next `read` returns — so the
    # allowance is expressed in blocks rather than as a round number.
    ceiling = got.nbytes + 3 * SPOOL_READ_BYTES
    assert peak <= ceiling, (
        f"reading one tile's spool peaked at {peak:,} B for a "
        f"{got.nbytes:,} B result; a second full copy was held"
    )

    # Sensitivity: the same ceiling against the shape this replaced. A green run
    # above only means something if the check can still go red, and the pre-11m.1
    # path is the property deliberately broken.
    tracemalloc.start()
    try:
        blocks = list(spool.read(0))
        old = np.concatenate(blocks) if blocks else np.empty(0, dtype)
        _, old_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.array_equal(old, got), "the two shapes must return the same records"
    assert old_peak > ceiling, (
        f"the list-and-concatenate shape peaked at {old_peak:,} B, inside the "
        f"{ceiling:,} B ceiling — this test can no longer tell the two apart"
    )


def test_reading_a_short_spool_is_an_error_not_a_short_tile(
    tmp_path: pathlib.Path,
) -> None:
    """A spool whose file is missing must not read back as an empty tile.

    `TileSpoolSet.read` returns without yielding when the path is gone, and its
    own count check never runs. Silently finalising that tile would drop real
    geometry and publish a manifest that looks complete.
    """
    from rapidmesh.tile_build import _SPOOL_FIELDS, _read_tile_records

    dtype = np.dtype(_SPOOL_FIELDS)
    spool = TileSpoolSet(tmp_path / "spool", "tri", dtype)
    block = np.zeros(4, dtype)
    spool.append(np.zeros(4, np.int64), block)
    spool.finish()
    spool.path(0).unlink()

    with pytest.raises(ValueError, match="records"):
        _read_tile_records(spool, 0)
