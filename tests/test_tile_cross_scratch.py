"""
The scratch `np.cross` writes on — WP-11m.t, ITEM-022.

WP-11m.s located the tiles stage's peak inside `_accumulate_normals` and sized
the transient at 1,923,076,080 B on ordinal 20. It attributed that to the
expression's *input* temporaries. Measured, that attribution was wrong: in
multiples of one `(R,3)` float64 array,

    np.cross(a.astype(f64) - c, b.astype(f64) - c)   5.33
      the two arguments np.cross needs                 2.00
      np.cross's own internal working set              2.33
      the finished `fn`                                1.00

Rewriting how the arguments are built saves 544 bytes. The scratch is inside
`np.cross`. This module holds two lines at once:

* the component-wise cross must stay **bit-identical** to `np.cross` — it is the
  T3 normal path, and `PHASE1-DETERMINISM-SPEC.md` §7 compares normals where a
  last-bit difference is a real angle;
* it must actually be smaller, or the change bought nothing.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import hashlib
import pathlib
import tracemalloc
from typing import Any

import numpy as np

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tile_assemble import _accumulate_normals
from rapidmesh.tile_io import read_tile

# Recorded from the pre-WP-11m.t code, whose normals came straight from
# `np.cross`. Normals derive from `fn`, so these are the production-level proof
# that `fn`'s bits did not move.
SMALL = (48, 192, 2.0)
SMALL_POSITIONS = "b8198c4c26ea68d586409e567ef5b24c6ed0d09f73278b53a03e0894d78837e3"
SMALL_NORMALS = "e034199b3eaf601415a17e19d3931017049987d4df2d34e5826affa6a300bc29"
SMALL_TRIANGLES = "9790686b2d19c4afb81522d82d6ec85c7a22ad5cbe0fb2a782ae58d7e4fa6b98"

# One (R,3) float64 array, the unit everything below is measured in.
BENCH_R = 400_000
BENCH_V = 180_000
BLOCK_BYTES = BENCH_R * 3 * 8


def _bench_inputs() -> tuple[Any, Any]:
    rng = np.random.default_rng(7)
    local_verts = (rng.random((BENCH_V, 3), dtype=np.float32) * 10.0)
    faces = rng.integers(0, BENCH_V, size=(BENCH_R, 3)).astype(np.int32)
    return local_verts, faces


def _peak_of(call: Any) -> tuple[Any, int]:
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        out = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak - before


class _Pose:
    """`rotate_local` is identity here: this module is about `fn`, not the pose."""

    @staticmethod
    def rotate_local(a: Any) -> Any:
        return a


def _accumulate_normals_pre_11mt(local_verts: Any, faces: Any, pose: Any) -> Any:
    """`_accumulate_normals` exactly as WP-11m.t found it.

    Kept here rather than described, so both claims below are measured against
    the real thing on the same inputs instead of against a remembered figure.
    """
    fn = np.cross(
        local_verts[faces[:, 1]].astype(np.float64) - local_verts[faces[:, 0]],
        local_verts[faces[:, 2]].astype(np.float64) - local_verts[faces[:, 0]],
    )
    acc = np.zeros((int(local_verts.shape[0]), 3), np.float64)
    for col in range(3):
        np.add.at(acc, faces[:, col], fn)
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc /= np.maximum(norm, 1e-12)
    facing = np.einsum("ij,ij->i", acc, -local_verts.astype(np.float64))
    acc[facing < 0] *= -1.0
    out: Any = pose.rotate_local(acc).astype(np.float32)
    return out


def test_normals_are_bitwise_what_np_cross_produced() -> None:
    """The line this package must not cross, held directly.

    `_accumulate_normals` no longer calls `np.cross`. This runs the original
    beside the shipped one on the same inputs and requires them to agree bit for
    bit — compared as raw `uint32`, not `allclose`, which would pass a real
    angle change at the T3 tier.
    """
    local_verts, faces = _bench_inputs()
    mine = _accumulate_normals(local_verts, faces, _Pose())
    theirs = _accumulate_normals_pre_11mt(local_verts, faces, _Pose())
    assert np.array_equal(
        mine.view(np.uint32), theirs.view(np.uint32)
    ), "the component-wise cross moved a normal bit"


def test_the_cross_scratch_is_gone() -> None:
    """Red before WP-11m.t: the two implementations peaked the same.

    Both are measured here, on the same inputs, in multiples of one `(R,3)`
    float64 array — so the claim is a *difference* between two things this test
    runs itself, not a bound I would have to keep re-tuning as the fixture
    changes. The whole function is measured, so `acc` and the tail are in both
    figures and cancel.
    """
    local_verts, faces = _bench_inputs()
    _, mine = _peak_of(lambda: _accumulate_normals(local_verts, faces, _Pose()))
    _, theirs = _peak_of(
        lambda: _accumulate_normals_pre_11mt(local_verts, faces, _Pose())
    )
    # One whole block. The isolated expression saves 2.00, but at function level
    # the peak relocates to the argument construction, so the measured saving is
    # ~1.38 — the threshold is set from that measurement, not from the 2.00 I
    # expected before running it.
    saved = (theirs - mine) / BLOCK_BYTES
    assert saved >= 1.0, (
        f"the component-wise cross saved {saved:.2f} blocks "
        f"({theirs - mine:,} B); np.cross's internal scratch is still there"
    )


def test_the_written_tiles_are_unchanged(tmp_path: pathlib.Path) -> None:
    """Positions, normals and triangles against digests from the pre-change code.

    A guard rather than a red-first test. Normals are the ones that matter here:
    they are `fn`'s only route into the written generation.
    """
    rows, cols, tile_size = SMALL
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=rows, cols=cols,
        dropout=0.01, range_noise=0.002, seed=11, station_id="cut",
    )
    out = tmp_path / "out"
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(out), tile_size=tile_size,
        # Metric on purpose: these are pre-WP-11m.t bits (see the header). The
        # cut under test is in `_accumulate_normals` and is partition-blind, so
        # holding the partition fixed keeps the comparison to the old code exact.
        partition="metric",
    )
    assert result.tiles is not None
    gen = out / "generations" / "00000000" / "tile"
    digests = {k: hashlib.sha256() for k in ("positions", "normals", "triangles")}
    for tile_id in sorted(result.tiles.tile_ids):
        payload = read_tile(gen / f"{tile_id:012d}.rmtile")
        digests["positions"].update(np.ascontiguousarray(payload.positions).tobytes())
        digests["normals"].update(np.ascontiguousarray(payload.normals).tobytes())
        digests["triangles"].update(np.ascontiguousarray(payload.triangles).tobytes())

    assert digests["positions"].hexdigest() == SMALL_POSITIONS
    assert digests["normals"].hexdigest() == SMALL_NORMALS, "tile normals moved"
    assert digests["triangles"].hexdigest() == SMALL_TRIANGLES
