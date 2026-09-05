"""
Phase snapshots inside the tile build — WP-11m.s, ITEM-022.

WP-11m.r named 67.3% of the tiles stage's live allocation on ordinal 20 and left
~1.15 GB unnamed. This package reads the two instruments `memory.py` already
keeps apart *at* the phases that can peak, rather than at the end of the stage.

A diagnostic that changes the thing it measures is worthless, so the tests here
are mostly about what the instrumentation must **not** do:

* it is off unless the environment asks, and off means nothing is recorded;
* it never holds an array — only `nbytes` — so it cannot keep alive what it is
  measuring, which is the one failure mode that would not look like a failure;
* the written generation is byte-identical with snapshots on and off.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.memory import TILE_SNAPSHOT_ENV, drain_phases, snapshots_enabled
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tile_io import read_tile

# Phases the tile build must report when asked. `normals_peak` is the one that
# matters: it is the only point at which `fn` and `acc` are both live.
EXPECTED_PHASES = {
    "assign", "spooled", "tile_records", "assemble_arrays",
    "normals_peak", "finalised",
}


def _build(out: pathlib.Path) -> Any:
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="phases",
    )
    return mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, out_dir=str(out), tile_size=2.0,
    )


def _generation_digest(out: pathlib.Path, result: Any) -> str:
    h = hashlib.sha256()
    gen = out / "generations" / "00000000" / "tile"
    for tile_id in sorted(result.tiles.tile_ids):
        payload = read_tile(gen / f"{tile_id:012d}.rmtile")
        for array in (payload.positions, payload.normals, payload.triangles):
            h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def test_snapshots_are_off_unless_the_environment_asks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path records nothing at all.

    Not "records a little" — nothing. An instrument that is cheap rather than
    absent still has to be argued about every time the stage is measured.
    """
    monkeypatch.delenv(TILE_SNAPSHOT_ENV, raising=False)
    drain_phases()
    assert not snapshots_enabled()

    result = _build(tmp_path / "out")
    assert result.tiles is not None
    assert drain_phases() == []


def test_asking_records_every_phase_that_can_peak(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red before WP-11m.s: `memory` has no `record_phase` to import."""
    monkeypatch.setenv(TILE_SNAPSHOT_ENV, "1")
    drain_phases()

    result = _build(tmp_path / "out")
    assert result.tiles is not None
    phases = drain_phases()
    seen = {entry["phase"] for entry in phases}
    assert seen >= EXPECTED_PHASES, f"missing {sorted(EXPECTED_PHASES - seen)}"

    # Exactly one `tracemalloc` ranking, and only if tracing was on. The arm is
    # one-shot precisely so a 364-tile station does not take 364 snapshots.
    rankings = [e for e in phases if "top" in e]
    assert len(rankings) <= 1, f"{len(rankings)} rankings; the arm is one-shot"

    # One `tile_records` per tile that had records, so the per-tile phases are
    # not silently collapsing onto one another.
    per_tile = [e for e in phases if e["phase"] == "tile_records"]
    assert len(per_tile) == len(result.tiles.tile_ids)


def test_the_sink_never_holds_an_array(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every recorded value is a scalar, a string, or a list of those.

    This is the test that keeps the instrument honest: a snapshot holding a
    reference to `acc` or `fn` would keep the largest arrays in the stage alive
    past their release, and every phase figure after it would be too high.
    """
    monkeypatch.setenv(TILE_SNAPSHOT_ENV, "1")
    drain_phases()
    _build(tmp_path / "out")

    def scalar(value: Any) -> bool:
        return isinstance(value, (int, float, str, bool)) or value is None

    for entry in drain_phases():
        for key, value in entry.items():
            if key == "top":
                assert isinstance(value, list)
                for item in value:
                    assert all(scalar(v) for v in item.values()), item
                continue
            assert scalar(value), f"{key} is {type(value).__name__}, not a scalar"
            assert not isinstance(value, np.ndarray)


def test_the_generation_is_identical_with_snapshots_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instrumentation must not move a single bit of the written tiles.

    Built twice from the same fixture — once with the flag off, once on — and
    the tile payloads compared by digest. If a snapshot ever perturbed the
    assembly arithmetic this is where it would surface.
    """
    monkeypatch.delenv(TILE_SNAPSHOT_ENV, raising=False)
    drain_phases()
    off_dir = tmp_path / "off"
    off = _generation_digest(off_dir, _build(off_dir))

    monkeypatch.setenv(TILE_SNAPSHOT_ENV, "1")
    on_dir = tmp_path / "on"
    on = _generation_digest(on_dir, _build(on_dir))
    recorded = drain_phases()

    assert recorded, "the on-run must actually have recorded something"
    assert on == off, "snapshots changed the written generation"
