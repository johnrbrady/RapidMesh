"""
`TileStore` must honour QA's block caps — WP-13b, ITEM-022.

WP-13a measured the `qa` stage and found the cause of its +4.13 GB RSS jump on
ordinal 20 was not a QA algorithm but a block size: `qa_stream` caps a block at
4 MB, `ResidentMesh` slices to fit, and `TileStore` did not — *"One tile is the
block."* Ordinal 20 packs its surface into 5 tiles, so one block was 18,491,098
triangles and its float64 corner array alone was 1,331,359,056 B.

The fix is a slice, not an algorithm. Every triangle and owned vertex is still
yielded exactly once, in exactly the same order; there are simply more, smaller
blocks. So the test that matters is not "is it smaller" but **"is it the same
data"**, and the digests below were recorded from the pre-change code on the
same fixture — one whose single tile is 8.0x the triangle cap and 1.4x the
vertex cap, so the change actually has something to slice.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.qa_stream import (
    QA_TRIANGLE_BLOCK_BYTES,
    QA_VERTEX_BLOCK_BYTES,
    surface_distances,
)
from rapidmesh.tile_equivalence import reconstitute_mesh

# A station whose whole surface lands in ONE tile, so the pre-change code yields
# one enormous block. `tile_size` is a parameter, not a constant, and this
# fixture is the degenerate end of it — the same shape ordinal 20 has at 5 tiles.
FIXTURE_ROWS = 240
FIXTURE_COLS = 960
FIXTURE_TILE_SIZE = 1000.0

# Recorded from the pre-change `TileStore` on 4 September 2026.
VERTEX_RECORDS = 227_239
TRIANGLE_RECORDS = 445_059
VERTEX_IDS_SHA = "bd9806764f3e43c64974587b45e22224e8fa3273ffa0dcc858f5a0753387637b"
VERTEX_XYZ_SHA = "ffcac2741d38008a06acaaa07d65aef142d4cf72bbe3e98e0b13d6de26034001"
TRIANGLE_IDS_SHA = "fe408fb9836c2c1083ab3203fdeafbeb572ef6b06ab8c929e1d4da4595fb902a"
TRIANGLE_CORNERS_SHA = "8ffb08a52cc9aa6d99cb644588797f2f0ad0e41c5df09e396b85b2ea0fd2996d"
SAMPLE_IDS_SHA = "32ecda97ce3c43572564f43d672226496de43f734cc43d19c7d0602f11f1aff4"
QUERIES_SHA = "c326af4a92e76ae9b9e7b57695130767ad9788ccb339d7c3678c66367d0eb86d"
DISTANCES_SHA = "b8bdf1d4b5c255aac6f5a8e816edc42a1e1643f1c611b4cce0f2042ffbc313d4"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("qa-blocks")
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=FIXTURE_ROWS, cols=FIXTURE_COLS,
        dropout=0.01, range_noise=0.002, seed=11, station_id="qablocks",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(root / "out"), tile_size=FIXTURE_TILE_SIZE,
        # Metric on purpose. This fixture exists to be *degenerate* — one tile
        # holding the whole station, 8.0x the triangle cap — which is precisely
        # what a lattice window cannot produce any more. The slice under test is
        # in `TileStore`, not in the partition, so the pathological input is
        # constructed rather than hoped for, and the WP-13b digests stay valid.
        partition="metric",
    )
    assert result.tiles is not None
    assert len(result.tiles.tile_ids) == 1, "the fixture must land in one tile"
    return result.tiles


def _digest(blocks: Any, field: str) -> tuple[str, int, int]:
    """SHA-256 over one field across every block, in yield order."""
    h = hashlib.sha256()
    records = 0
    largest = 0
    for block in blocks:
        array = np.ascontiguousarray(getattr(block, field))
        h.update(array.tobytes())
        records += int(array.shape[0])
        largest = max(largest, int(array.nbytes))
    return h.hexdigest(), records, largest


def test_tilestore_blocks_respect_the_qa_byte_caps(store: Any) -> None:
    """The cap applies to both `GeometrySource` implementations, or to neither.

    Red before WP-13b: this fixture's single tile yields one triangle block of
    32,044,248 B — 8.01x `QA_TRIANGLE_BLOCK_BYTES` — and one vertex block of
    5,453,736 B, 1.36x `QA_VERTEX_BLOCK_BYTES`. `ResidentMesh` has always
    sliced; `TileStore` yielded the tile whole.
    """
    triangle_sizes = [b.corners.nbytes for b in store.triangle_blocks()]
    vertex_sizes = [b.xyz.nbytes for b in store.vertex_blocks()]
    identified_sizes = [b.corners.nbytes for b in store.identified_triangle_blocks()]

    assert max(triangle_sizes) <= QA_TRIANGLE_BLOCK_BYTES, (
        f"largest triangle block is {max(triangle_sizes):,} B, "
        f"{max(triangle_sizes) / QA_TRIANGLE_BLOCK_BYTES:.2f}x the "
        f"{QA_TRIANGLE_BLOCK_BYTES:,} B cap"
    )
    assert max(identified_sizes) <= QA_TRIANGLE_BLOCK_BYTES, (
        f"largest identified block is {max(identified_sizes):,} B"
    )
    assert max(vertex_sizes) <= QA_VERTEX_BLOCK_BYTES, (
        f"largest vertex block is {max(vertex_sizes):,} B, "
        f"{max(vertex_sizes) / QA_VERTEX_BLOCK_BYTES:.2f}x the "
        f"{QA_VERTEX_BLOCK_BYTES:,} B cap"
    )
    # Not vacuous: the tile is big enough that honouring the cap must split it.
    assert len(triangle_sizes) > 1
    assert len(vertex_sizes) > 1


def test_slicing_yields_the_same_records_in_the_same_order(store: Any) -> None:
    """Same data, more blocks — against digests from the pre-change code.

    A guard, not a red-first test: it passed before the change and must keep
    passing. It is the one that would catch a slice that dropped a record,
    duplicated one at a boundary, or reordered them.
    """
    ids_sha, ids_n, _ = _digest(store.vertex_blocks(), "ids")
    xyz_sha, xyz_n, _ = _digest(store.vertex_blocks(), "xyz")
    assert ids_n == xyz_n == VERTEX_RECORDS
    assert ids_sha == VERTEX_IDS_SHA, "vertex ids moved"
    assert xyz_sha == VERTEX_XYZ_SHA, "vertex positions moved"

    tid_sha, tid_n, _ = _digest(store.triangle_blocks(), "ids")
    cor_sha, cor_n, _ = _digest(store.triangle_blocks(), "corners")
    assert tid_n == cor_n == TRIANGLE_RECORDS
    assert tid_sha == TRIANGLE_IDS_SHA, "triangle corner ids moved"
    assert cor_sha == TRIANGLE_CORNERS_SHA, "triangle corner positions moved"

    sid_sha, sid_n, _ = _digest(store.identified_triangle_blocks(), "sample_ids")
    icor_sha, _, _ = _digest(store.identified_triangle_blocks(), "corners")
    assert sid_n == TRIANGLE_RECORDS
    assert sid_sha == SAMPLE_IDS_SHA, "source sample ids moved"
    assert icor_sha == TRIANGLE_CORNERS_SHA, (
        "identified corners must match the plain triangle corners exactly"
    )


def test_forward_qa_distances_are_unchanged(store: Any) -> None:
    """The consumer's numbers, not just the blocks handed to it.

    Queries are pushed off the surface deliberately: a mesh vertex measured
    against its own surface returns 0.0, and a digest of four thousand zeros
    would be identical for any mesh and would prove nothing.
    """
    mesh = reconstitute_mesh(store)
    rng = np.random.default_rng(3)
    index = np.sort(rng.choice(mesh.vertices.shape[0], size=4000, replace=False))
    base = mesh.vertices[index].astype(np.float64)
    queries = base + rng.normal(0.0, 0.02, size=base.shape)
    assert hashlib.sha256(np.ascontiguousarray(queries).tobytes()).hexdigest() == (
        QUERIES_SHA
    ), "the fixture's query set drifted; the distance digest below is not comparable"

    distances = surface_distances(store, queries, k=2, workers=1, block=25_000)
    assert distances.size == 4000
    assert float((distances > 0).mean()) == 1.0, "queries must be off the surface"
    assert hashlib.sha256(
        np.ascontiguousarray(distances).tobytes()
    ).hexdigest() == DISTANCES_SHA, "forward QA distances moved"


def test_reverse_v2_selects_the_same_triangles(tmp_path: pathlib.Path) -> None:
    """Reverse QA's *selection*, not just the blocks it was built from.

    This is the one the block digests do not cover. `_build_reverse_runs` turns
    each `identified_triangle_blocks` block into one sorted run, so slicing a
    tile changes how many runs exist and how the records are distributed across
    them. The merge is supposed to make that invisible — the canonical order is
    a property of the keys, not of the run boundaries — but "supposed to" is
    what a test is for, and the area recurrence that picks the samples walks
    that merged order strictly left to right.

    Every figure below was recorded from the pre-change `TileStore` on the same
    one-tile fixture, so a selection that shifted by a single triangle moves
    `rms` and fails here.
    """
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=FIXTURE_ROWS, cols=FIXTURE_COLS,
        dropout=0.01, range_noise=0.002, seed=11, station_id="qablocks",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=True, measure_samples=20_000,
        out_dir=str(tmp_path / "out"), tile_size=FIXTURE_TILE_SIZE,
    )
    reverse = result.mesh_to_source
    evidence = result.reverse_qa_evidence
    assert reverse is not None and evidence is not None

    assert reverse.sampled_points == 20_000
    assert reverse.population == TRIANGLE_RECORDS
    assert reverse.rms == 0.01834356807002254, "reverse-v2 selection moved"
    assert reverse.mean == 0.015292574743553092
    assert reverse.p95 == 0.035484387347565295
    assert reverse.p99_9 == 0.0683835784143599
    assert reverse.maximum == 0.08535343423698667
    assert reverse.within_2mm == 0.01775
    assert reverse.within_5mm == 0.09975

    # The selection's own shape, so a change that happened to leave `rms` alone
    # would still be caught.
    assert evidence.samples_selected == 20_000
    assert evidence.samples_measured == 20_000
    assert evidence.samples_unmatched == 0
    assert evidence.windows_used == 239
    assert evidence.largest_window_candidates == 17_124
    assert evidence.positive_area_triangles == TRIANGLE_RECORDS


def test_a_tile_smaller_than_one_record_still_terminates(
    tmp_path: pathlib.Path,
) -> None:
    """`_records_per_block` floors at one record; the slice must not stall.

    A cap below one record would make a naive `range(0, n, 0)` raise, and an
    empty tile must still be skipped rather than yielding a zero-length block.
    """
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="small",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(tmp_path / "out"), tile_size=2.0,
    )
    assert result.tiles is not None
    for block in result.tiles.triangle_blocks():
        assert block.corners.shape[0] > 0
    for block in result.tiles.vertex_blocks():
        assert block.xyz.shape[0] > 0
