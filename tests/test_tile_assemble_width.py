"""
Index width inside `assemble_tile` — WP-11m.2, ITEM-022, DEC-012.

WP-11m.1 measured the tile stage and found its cost is not the station: it is
four temporaries built while a *single* tile is assembled, about 1,483 MB on
ordinal 20's largest tile against a tile whose finished mesh is 425 MB. Two of
them — `spooled` and `local`, both `(R,3)` — held vertex indices in int64.

Those indices cannot need int64. `tile_build._SPOOL_FIELDS` stores `v0`, `v1`
and `v2` as `<i4`, so every index reaching this function has already been
through int32 on its way to disk; widening them on arrival bought nothing and
doubled the two largest index arrays in the pipeline's hottest step.

Narrowing an index changes no arithmetic, and these tests are built to hold
that line rather than to assert it:

* the widths are observed inside the live call, not inferred from the source;
* positions, normals and triangles are pinned to digests **recorded from the
  pre-change code**, so a change in the last bit of a normal fails here. That
  is the T3 tier of `PHASE1-DETERMINISM-SPEC.md` §7 held at its strictest —
  bitwise, not the 1e-6 rad directed rule the tier would allow.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tile_build import _SPOOL_FIELDS
from rapidmesh.tile_io import read_tile

# Recorded from the pre-change code on 4 September 2026, before `spooled` and
# `local` were narrowed, with the fixture below at tile_size 2.0. These are the
# bits the index-width change must not move.
TILE_COUNT = 57
POSITIONS_SHA = "b8198c4c26ea68d586409e567ef5b24c6ed0d09f73278b53a03e0894d78837e3"
NORMALS_SHA = "e034199b3eaf601415a17e19d3931017049987d4df2d34e5826affa6a300bc29"
TRIANGLES_SHA = "9790686b2d19c4afb81522d82d6ec85c7a22ad5cbe0fb2a782ae58d7e4fa6b98"

# The arrays this package narrows, and the width they must now have.
NARROWED = ("spooled", "local", "used")


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="lifecycle",
    )


def _run(station: synthetic.SyntheticScan, out: pathlib.Path) -> Any:
    return mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(out), tile_size=2.0,
        # Pinned to the metric partition **on purpose**. These digests were
        # recorded from code that no longer exists, and that chain is the whole
        # evidence value of this test — regenerating them under WP-A's lattice
        # partition would silently reset the baseline to "whatever it does now".
        # DEC-021 moved production to lattice windows; it did not licence
        # throwing away the record of what the old assembler emitted.
        partition="metric",
    )


def test_the_spool_contract_is_what_makes_narrowing_safe() -> None:
    """int32 is not a new assumption — it is the width already on disk.

    If a later package widened the spool record, these indices would have to be
    widened with it, and this test is where that shows up rather than in a
    silent overflow on a station past 2.1 billion samples.
    """
    fields = dict((name, dtype) for name, dtype in _SPOOL_FIELDS)
    for column in ("v0", "v1", "v2"):
        assert fields[column] == "<i4", (
            f"spool column {column} is {fields[column]}, not '<i4'; "
            "assemble_tile's int32 index width follows from this"
        )


def test_assemble_holds_its_index_temporaries_at_int32(
    station: synthetic.SyntheticScan,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widths, observed inside the live call.

    Red before WP-11m.2: `spooled` and `local` are `(R,3)` int64 — 423.2 MB each
    on ordinal 20's largest tile — and `used` is int64 with them.

    `_accumulate_normals` is the last thing called while all three are alive, so
    a spy on it runs at exactly the moment the stage peaks.
    """
    import rapidmesh.tile_assemble as tile_assemble

    real = tile_assemble._accumulate_normals
    widths: dict[str, set[str]] = {name: set() for name in NARROWED}

    def spy(*args: Any, **kwargs: Any) -> Any:
        frame: Any = sys._getframe()
        while frame is not None and frame.f_code.co_name != "assemble_tile":
            frame = frame.f_back
        assert frame is not None, "_accumulate_normals was not called under assemble_tile"
        for name in NARROWED:
            value = frame.f_locals.get(name)
            if isinstance(value, np.ndarray):
                widths[name].add(value.dtype.name)
        return real(*args, **kwargs)

    monkeypatch.setattr(tile_assemble, "_accumulate_normals", spy)
    result = _run(station, tmp_path / "out")
    assert result.tiles is not None

    for name in NARROWED:
        assert widths[name], f"the spy never observed `{name}`"
        assert widths[name] == {"int32"}, (
            f"`{name}` is {sorted(widths[name])} inside assemble_tile, not int32; "
            "the index-width change did not reach it"
        )


def test_narrowing_the_index_moved_no_geometry_bit(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Positions, normals and triangles against digests from the pre-change code.

    This is a guard, not a red-first test: it passed before the change and must
    keep passing after it. That is the point — it is the evidence the narrowing
    is an index change and not an arithmetic one.

    Normals are compared **bitwise**, which is stricter than the T3 tier of
    `PHASE1-DETERMINISM-SPEC.md` §7 requires. An index width has no licence to
    move them at all, so the loose tier would hide exactly the defect worth
    catching.
    """
    result = _run(station, tmp_path / "out")
    assert result.tiles is not None
    ids = sorted(result.tiles.tile_ids)
    assert len(ids) == TILE_COUNT

    generation = tmp_path / "out" / "generations" / "00000000" / "tile"
    positions = hashlib.sha256()
    normals = hashlib.sha256()
    triangles = hashlib.sha256()
    for tile_id in ids:
        payload = read_tile(generation / f"{tile_id:012d}.rmtile")
        positions.update(np.ascontiguousarray(payload.positions).tobytes())
        normals.update(np.ascontiguousarray(payload.normals).tobytes())
        triangles.update(np.ascontiguousarray(payload.triangles).tobytes())

    assert positions.hexdigest() == POSITIONS_SHA, "tile positions moved"
    assert normals.hexdigest() == NORMALS_SHA, "tile normals moved (T3)"
    assert triangles.hexdigest() == TRIANGLES_SHA, "tile triangles moved"


def test_the_join_still_carries_int64_global_indices(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """`TileJoin.global_vertex_index` is I64 on disk and stays I64.

    The narrowing is internal to the assembler. The join is a written contract,
    so the cast back at the boundary is part of the change rather than an
    accident of dtype propagation.
    """
    from rapidmesh.tile_io import read_tile_join

    result = _run(station, tmp_path / "out")
    assert result.tiles is not None
    generation = tmp_path / "out" / "generations" / "00000000" / "tile"
    checked = 0
    for tile_id in sorted(result.tiles.tile_ids):
        join = read_tile_join(generation / f"{tile_id:012d}.rmtjoin")
        assert join.global_vertex_index.dtype == np.int64, (
            f"tile {tile_id} join global_vertex_index is "
            f"{join.global_vertex_index.dtype}, not int64"
        )
        assert join.source_sample_id.dtype == np.int64
        assert join.row.dtype == np.int32
        checked += 1
    assert checked == TILE_COUNT
