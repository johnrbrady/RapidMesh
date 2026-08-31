"""
Observation store v0 — WP-1.G1b Part B2, PLAN.md §5 item 12.

Phase 5's QC comparison must be fed **observations**, never a display mesh
(`docs/adr/ADR-007`). This store is where the observations come from, and its
contract has three halves that are easy to let drift apart:

* it must **round-trip** — what Pass B wrote is what a later stage reads, field
  for field, not approximately;
* it must **agree with the ledger** — the store's record count is the ledger's
  `retained`, and `FilterStats` is the thing that has to balance;
* it must stay **server-side** — every record carries `source_sample_id`, which
  DEC-004 says never leaves the server, so the store lives outside the
  client-eligible tile payload and a reader holding only `.rmtile` bytes cannot
  reach it.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.obs_store import (
    DISP_RETAINED,
    OBS_MAGIC,
    ObservationStoreError,
    build_records,
    read_store,
    write_store,
)
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tile_io import read_tile


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="obs",
    )


@pytest.fixture(scope="module")
def run(
    station: synthetic.SyntheticScan, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Any, pathlib.Path]:
    """One tiled run that also writes the store, kept for the whole module."""
    out = tmp_path_factory.mktemp("obs-run") / "out"
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(out), tile_size=2.0,
    )
    return result, out


def _store_path(out: pathlib.Path) -> pathlib.Path:
    return out / "generations" / "00000000" / "obs" / "observations.rmobs"


# ---------------------------------------------------------------------------
# it is written, and it round-trips
# ---------------------------------------------------------------------------


def test_the_pipeline_writes_a_store_beside_the_generation(
    run: tuple[Any, pathlib.Path]
) -> None:
    _, out = run
    path = _store_path(out)
    assert path.exists()
    assert path.read_bytes()[:8] == OBS_MAGIC
    assert not list(out.rglob("*.tmp"))


def test_the_store_round_trips_field_for_field(
    run: tuple[Any, pathlib.Path]
) -> None:
    """Round-trip is asserted with `np.array_equal` per field, never a
    tolerance: a store that read back "close enough" would be evidence that had
    quietly moved between being written and being believed."""
    result, out = run
    store = read_store(_store_path(out))
    assert store.count == result.stats.retained
    for name in store.records.dtype.names:
        assert store.records[name].shape[0] == store.count, name
    assert np.array_equal(
        store.records["disposition"],
        np.full(store.count, DISP_RETAINED, np.uint8),
    )
    # The lattice geometry travels with the store, so a later stage can
    # re-derive directions without being handed a pose it might apply
    # differently.
    assert store.rows == result.lattice.rows
    assert store.cols == result.lattice.cols
    assert store.az_step == result.lattice.az_step
    assert store.el_step == result.lattice.el_step


def test_a_written_store_reads_back_exactly_what_was_built(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """The narrow round-trip, independent of the pipeline: build records, write,
    read, compare every field. If this and the pipeline test ever disagree the
    fault is in the hook, not the format."""
    scan = station.scan
    keep = np.zeros(len(scan), bool)
    keep[::3] = True
    expected = build_records(scan, keep, scan.pose)
    path = tmp_path / "obs" / "observations.rmobs"
    size = write_store(path, scan, keep, scan.pose)
    assert size == path.stat().st_size > 0

    store = read_store(path)
    assert store.count == int(keep.sum()) == expected.shape[0]
    for name in expected.dtype.names:
        assert np.array_equal(store.records[name], expected[name]), name
    assert np.array_equal(store.origin, np.asarray(scan.pose.translation))


def test_the_angles_come_from_the_lattice_not_from_the_positions(
    station: synthetic.SyntheticScan
) -> None:
    """Pinned because the alternative is silently plausible.

    Re-deriving azimuth and elevation with `arctan2` over a narrowed float32
    position gives *almost* the same angle, and would hand a later stage a
    direction the scanner never sampled. The store stores the lattice's angle.
    """
    scan = station.scan
    keep = np.ones(len(scan), bool)
    records = build_records(scan, keep, scan.pose)
    lattice = scan.lattice
    want_az = (lattice.az0 + scan.col.astype(np.int64) * lattice.az_step).astype(np.float32)
    want_el = (lattice.el0 + scan.row.astype(np.int64) * lattice.el_step).astype(np.float32)
    assert np.array_equal(records["azimuth"], want_az)
    assert np.array_equal(records["elevation"], want_el)


# ---------------------------------------------------------------------------
# it agrees with the ledger
# ---------------------------------------------------------------------------


def test_the_store_count_is_the_ledgers_retained(
    run: tuple[Any, pathlib.Path]
) -> None:
    """The store's whole contract with `FilterStats`."""
    result, out = run
    result.stats.require_balanced()
    store = read_store(_store_path(out))
    assert store.count == result.stats.retained
    assert store.count > 0


def test_the_envelope_records_the_stores_size_and_count(
    run: tuple[Any, pathlib.Path]
) -> None:
    """A by-product nobody sizes is a by-product nobody notices growing."""
    result, out = run
    streaming = dict(result.evidence_report("0f" * 32).metadata.streaming)
    assert int(streaming["observation_count"]) == result.stats.retained
    assert int(streaming["observation_bytes"]) == _store_path(out).stat().st_size
    assert int(streaming["observation_bytes"]) > 0


def test_the_resident_branch_writes_no_store(
    station: synthetic.SyntheticScan
) -> None:
    """Without an output directory there is no generation to write beside, and
    the diagnostics say zero rather than pretending."""
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3, measure=False
    )
    assert result.diagnostics is not None
    assert result.diagnostics.observation_bytes == 0
    assert result.diagnostics.observation_count == 0


# ---------------------------------------------------------------------------
# DEC-004: it stays server-side
# ---------------------------------------------------------------------------


def test_source_identity_is_in_the_store_and_not_in_the_tile_payload(
    run: tuple[Any, pathlib.Path]
) -> None:
    """The redaction boundary is the file split, asserted from both sides.

    The store carries `sample_id`; the client-eligible `.rmtile` has no field
    that could carry one. A reader handed only tile bytes cannot rebuild the
    evidence join, and that is structural rather than a runtime check.
    """
    result, out = run
    store = read_store(_store_path(out))
    assert "sample_id" in store.records.dtype.names

    tile_id = result.tiles.tile_ids[0]
    payload = read_tile(result.tiles.tile_path(tile_id))
    assert set(vars(payload)) == {
        "tile_id", "origin", "bounds", "positions", "normals", "triangles", "rgb"
    }


def test_the_store_lives_outside_the_tile_directory(
    run: tuple[Any, pathlib.Path]
) -> None:
    """Serving the tile directory must not serve the evidence join."""
    _, out = run
    generation = out / "generations" / "00000000"
    assert _store_path(out).parent == generation / "obs"
    assert not list((generation / "tile").glob("*.rmobs"))


# ---------------------------------------------------------------------------
# it fails closed
# ---------------------------------------------------------------------------


def test_a_corrupt_store_fails_closed(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """`segments_io`'s rule applied to the evidence tier: a short or edited read
    that silently yields fewer observations would drop survey evidence and
    report success."""
    scan = station.scan
    keep = np.ones(len(scan), bool)
    path = tmp_path / "observations.rmobs"
    write_store(path, scan, keep, scan.pose)

    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ObservationStoreError):
        read_store(path)


def test_a_truncated_store_fails_closed(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    scan = station.scan
    keep = np.ones(len(scan), bool)
    path = tmp_path / "observations.rmobs"
    write_store(path, scan, keep, scan.pose)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ObservationStoreError):
        read_store(path)


def test_a_foreign_file_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "observations.rmobs"
    path.write_bytes(b"NOTASTORE" + b"\x00" * 200)
    with pytest.raises(ObservationStoreError):
        read_store(path)
