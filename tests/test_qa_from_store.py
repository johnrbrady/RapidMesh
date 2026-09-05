"""
QA reads the observation store, not a station-scale array — WP-C, ITEM-022 T4.

WP-13a sized T4 at **768,720,960 B on ordinal 20**: the `offsets` float32 array
(165 MB), its float64 copy inside the query (330 MB), the `rows` int64 (110 MB),
the `argsort` permutation (110 MB) and the sorted copy (55 MB). Every one of
them exists so that QA can answer two bounded questions:

* *forward* — give me 300,000 observations chosen by `(n, seed)`;
* *reverse* — give me the observations whose lattice row is in `[lo, hi]`.

The store already holds exactly those observations, in ascending lattice-cell
order. Two facts make reading them back **bit-identical** rather than merely
equivalent:

1. `rs.choice(n, 300_000, replace=False)` depends only on `n` and the seed, so
   the sample *set and its order* can be reproduced without holding the
   population it indexes;
2. ascending cell order means the store's `row` column is **non-decreasing**, so
   `np.argsort(rows, kind="stable")` is the identity and a row window is a
   **contiguous slice**. Candidates are then the same points in the same order,
   which is what makes the KD-tree return the same float64 distances rather than
   ones that agree to a tolerance.

The second is asserted below rather than assumed, because the whole bounded
reverse path rests on it.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed

BAND_ROWS = 16
CHUNK_POINTS = 5_000
HALO = 3
SEED = 0
SAMPLES = 2_000


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One streamed generation, kept so every test reads the same store."""
    root = tmp_path_factory.mktemp("qa-store")
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="qastore",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=BAND_ROWS, chunk_points=CHUNK_POINTS, halo=HALO,
        measure=False, out_dir=str(root / "out"), window=(16, 64),
    )
    assert result.tiles is not None
    path = root / "out" / "generations" / "00000000" / "obs" / "observations.rmobs"
    assert path.exists()
    return station, result, path


def _resident(path: pathlib.Path) -> tuple[Any, Any]:
    """The two station-scale arrays QA used to be handed, for comparison only."""
    from rapidmesh.obs_store import read_store

    store = read_store(path)
    offsets = np.stack(
        [store.records["x"], store.records["y"], store.records["z"]], axis=1
    ).astype(np.float32)
    return offsets, store.records["row"].astype(np.int32)


# ---------------------------------------------------------------------------
# the property the bounded reverse path rests on
# ---------------------------------------------------------------------------


def test_the_store_is_in_non_decreasing_row_order(built: Any) -> None:
    """A guard, not a red-first test, and the load-bearing one.

    If the store were ever written in another order, `argsort` would stop being
    the identity, a row window would stop being contiguous, and the bounded
    reverse path would silently build its tree from a different candidate set —
    the one failure mode that would still produce plausible distances.
    """
    _station, _result, path = built
    _offsets, rows = _resident(path)
    assert rows.size > 0
    assert np.all(np.diff(rows.astype(np.int64)) >= 0), "store rows are not sorted"
    order = np.argsort(rows.astype(np.int64), kind="stable")
    assert np.array_equal(order, np.arange(rows.size)), (
        "np.argsort(rows) is not the identity; a row window is no longer a "
        "contiguous slice of the store"
    )


# ---------------------------------------------------------------------------
# forward: the sample set from (n, seed), gathered rather than indexed
# ---------------------------------------------------------------------------


def test_the_store_reader_gathers_exactly_what_indexing_would(built: Any) -> None:
    """Red before WP-C: `obs_store` has no `ObservationWindows` to import.

    The comparison is bitwise on the raw float32 bits. These are positions, and
    a QA figure computed from positions that differ in the last bit is a
    different figure — the tolerance question does not arise.
    """
    from rapidmesh.obs_windows import ObservationWindows

    _station, _result, path = built
    offsets, _rows = _resident(path)
    n = int(offsets.shape[0])

    windows = ObservationWindows.open(path)
    assert windows.count == n

    rs = np.random.default_rng(SEED)
    idx = rs.choice(n, min(SAMPLES, n), replace=False)
    got = windows.gather(idx)
    assert got.shape == (idx.size, 3)
    assert np.array_equal(
        np.ascontiguousarray(got).view(np.uint32),
        np.ascontiguousarray(offsets[idx]).view(np.uint32),
    ), "gathered offsets differ from indexing the resident array"


def test_a_row_window_is_a_contiguous_slice_of_the_store(built: Any) -> None:
    """Red before WP-C: there is no row offset table to ask.

    The bounded reverse path replaces `searchsorted` over a station-scale sorted
    copy with a lookup in a table of one entry per lattice row — 19 KB on the
    high-resolution lattice against 165 MB for the two arrays it retires.
    """
    from rapidmesh.obs_windows import ObservationWindows

    _station, _result, path = built
    _offsets, rows = _resident(path)
    windows = ObservationWindows.open(path)
    rows64 = rows.astype(np.int64)

    for lo, hi in ((0, 3), (5, 5), (10, 47), (-4, 6), (0, 10_000)):
        start, stop = windows.row_range(lo, hi)
        expected = np.flatnonzero((rows64 >= lo) & (rows64 <= hi))
        if expected.size == 0:
            assert stop <= start, (lo, hi)
            continue
        assert (start, stop) == (int(expected[0]), int(expected[-1]) + 1), (lo, hi)


# ---------------------------------------------------------------------------
# both directions, against the arrays they replace
# ---------------------------------------------------------------------------


def test_forward_qa_from_the_store_is_the_same_report(built: Any) -> None:
    """Every field of the report, to the last digit."""
    from rapidmesh.obs_windows import ObservationWindows
    from rapidmesh.qa import deviation_report

    _station, result, path = built
    offsets, _rows = _resident(path)
    source = result.tiles

    resident = deviation_report(
        source, offsets, max_samples=SAMPLES, seed=SEED, workers=1
    )
    bounded = deviation_report(
        source, ObservationWindows.open(path),
        max_samples=SAMPLES, seed=SEED, workers=1,
    )
    for name in ("rms", "mean", "p95", "p99_9", "maximum", "sampled_points",
                 "population", "exact", "metric"):
        assert getattr(bounded, name) == getattr(resident, name), name


def test_reverse_qa_from_the_store_is_the_same_report(
    built: Any, tmp_path: pathlib.Path
) -> None:
    """The distances themselves, bitwise, and then the report.

    The candidate set is the same points in the same order, so this is an
    equality and not an approximation. If it ever became an approximation the
    window would have started coalescing groups, which SPEC §5(h) step 6
    forbids because it can only shorten a distance.
    """
    from rapidmesh.obs_windows import ObservationWindows
    from rapidmesh.reverse_qa import mesh_to_source_report_v2

    _station, result, path = built
    offsets, rows = _resident(path)
    source = result.tiles

    resident, _ = mesh_to_source_report_v2(
        source, offsets, source_rows=rows, max_samples=SAMPLES, seed=SEED,
        work_dir=tmp_path / "a",
    )
    bounded, _ = mesh_to_source_report_v2(
        source, ObservationWindows.open(path), source_rows=None,
        max_samples=SAMPLES, seed=SEED, work_dir=tmp_path / "b",
    )
    for name in ("rms", "mean", "p95", "p99_9", "maximum", "sampled_points",
                 "population", "exact", "metric"):
        assert getattr(bounded, name) == getattr(resident, name), name


def test_the_production_qa_path_holds_no_station_scale_offsets(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structural claim: `read_store_offsets` leaves the production path.

    Red before WP-C: `pass_b_finalise` calls it to build `offsets` and `qa_rows`
    before either QA direction runs. A detonator is stronger than a peak, for
    the reason `test_tiles.py` gives — a peak can be small because the fixture
    is small, but a call is a call.
    """
    import rapidmesh.obs_store as obs_store

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("QA rebuilt a station-scale offsets array")

    monkeypatch.setattr(obs_store, "read_store_offsets", forbidden)
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="qanostore",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=BAND_ROWS, chunk_points=CHUNK_POINTS, halo=HALO,
        measure=True, measure_samples=SAMPLES,
        out_dir=str(tmp_path / "out"), window=(16, 64),
    )
    assert result.tiles is not None
    assert result.deviation is not None
    assert result.mesh_to_source is not None
