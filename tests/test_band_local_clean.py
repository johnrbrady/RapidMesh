"""Band-local despeckle -> carve -> restore equals the in-memory `clean` path.

PLAN.md §5 item 6. Design authority: `PHASE1-HALO-CALCULUS.md` §5-§7
(composition, encoding, judge-once) and `PHASE1-DETERMINISM-SPEC.md` §7
(comparison is exact array equality, never `allclose`).

Scope of this file is the **filtering stage only**. Triangulation, island
culling, incremental output and the full streamed pipeline are PLAN.md §5
items 7 and 8; the streamed matrix in `test_stream_equivalence.py` stays
required-red until they land.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.e57_reader import RawChunk, iter_row_bands
from rapidmesh.filters import (
    COMPOSED_FILTER_HALO,
    BandPlan,
    _judged_bands,
    clean,
    clean_bands,
    iter_clean_bands,
)
from rapidmesh.grid import CoarseRangeGrid
from rapidmesh.types import (
    FilterStats,
    LatticeInfo,
    LatticeSource,
    ScanPose,
    StructuredScan,
)

# The WP-1.1 harness fixture, unchanged: 12 x 48 with the mover enabled.
# `test_stream_equivalence.py` sizes its matrix against these, so the two
# files describe the same configurations.
HARNESS_ROWS, HARNESS_COLS = 12, 48
BAND_ROWS_MATRIX = (3, 4, 6)
CHUNK_POINTS_MATRIX = (100, 200)

# A second, finer fixture. The 12 x 48 lattice carves but never restores
# (measured: restored_from_carve == 0), so restoration would go untested if
# this file stopped at the harness size.
FINE_ROWS, FINE_COLS = 40, 180


def _station(
    scanner: tuple[float, float, float],
    *,
    rows: int,
    cols: int,
    mover: bool,
    seed: int,
    station_id: str,
) -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(scanner=scanner, mover=mover),
        rows=rows,
        cols=cols,
        dropout=0.0,
        range_noise=0.0,
        seed=seed,
        station_id=station_id,
    )


@pytest.fixture(scope="module")
def harness_scan() -> synthetic.SyntheticScan:
    return _station(
        (0.35, -0.2, 0.0),
        rows=HARNESS_ROWS,
        cols=HARNESS_COLS,
        mover=True,
        seed=17,
        station_id="harness",
    )


@pytest.fixture(scope="module")
def harness_neighbours() -> list[CoarseRangeGrid]:
    b = _station(
        (-2.6, 1.7, 0.0), rows=HARNESS_ROWS, cols=HARNESS_COLS,
        mover=False, seed=11, station_id="B",
    )
    c = _station(
        (2.7, 1.4, 0.0), rows=HARNESS_ROWS, cols=HARNESS_COLS,
        mover=False, seed=13, station_id="C",
    )
    return [CoarseRangeGrid.build(b.scan), CoarseRangeGrid.build(c.scan)]


@pytest.fixture(scope="module")
def fine_scan() -> synthetic.SyntheticScan:
    return _station(
        (0.35, -0.2, 0.0), rows=FINE_ROWS, cols=FINE_COLS,
        mover=True, seed=17, station_id="fine",
    )


@pytest.fixture(scope="module")
def fine_neighbours() -> list[CoarseRangeGrid]:
    b = _station(
        (-2.6, 1.7, 0.0), rows=FINE_ROWS, cols=FINE_COLS,
        mover=False, seed=11, station_id="B",
    )
    c = _station(
        (2.7, 1.4, 0.0), rows=FINE_ROWS, cols=FINE_COLS,
        mover=False, seed=13, station_id="C",
    )
    return [CoarseRangeGrid.build(b.scan), CoarseRangeGrid.build(c.scan)]


def assert_same_scan(left: StructuredScan, right: StructuredScan) -> None:
    """Exact equality, field by field. `np.allclose` would pass a divergence."""
    assert len(left) == len(right)
    for name in ("row", "col", "xyz", "rng", "sample_id", "rgb", "intensity"):
        lv, rv = getattr(left, name), getattr(right, name)
        assert (lv is None) == (rv is None), name
        if lv is not None:
            assert np.array_equal(lv, rv), name
    assert left.lattice == right.lattice
    assert left.station_id == right.station_id


# --------------------------------------------------------------------------
# 1. Band-local == full-scan, exact
# --------------------------------------------------------------------------


def test_band_local_matches_full_scan_on_harness_fixture(
    harness_scan: synthetic.SyntheticScan,
    harness_neighbours: list[CoarseRangeGrid],
) -> None:
    """The 12 x 48 mover fixture, carving on, at the composed halo.

    48 columns at 2*pi/48 makes this a full sweep, so `columns_wrap` is true
    and the azimuth seam is inside every band — the scan-level predicate of
    `PHASE1-HALO-CALCULUS.md` §3 is exercised here rather than assumed.
    """
    from rapidmesh.filters import columns_wrap

    assert columns_wrap(harness_scan.scan) is True

    reference, ref_stats = clean(harness_scan.scan, others=harness_neighbours)
    banded, band_stats = clean_bands(
        harness_scan.scan,
        others=harness_neighbours,
        band_rows=min(BAND_ROWS_MATRIX),
        halo=COMPOSED_FILTER_HALO,
    )
    assert band_stats == ref_stats
    assert_same_scan(reference, banded)
    # The fixture must actually reach the carve stage, or this proves nothing.
    assert ref_stats.dropped_mover_carve > 0


def test_band_local_matches_full_scan_with_restoration(
    fine_scan: synthetic.SyntheticScan,
    fine_neighbours: list[CoarseRangeGrid],
) -> None:
    """A fixture fine enough that `restore_parallax_carve` actually fires.

    Restoration is the stage the composed halo exists for: it is the only one
    that reads a *post-carve* neighbourhood, so a fixture where it never runs
    cannot distinguish a correct halo from an absent one.
    """
    reference, ref_stats = clean(fine_scan.scan, others=fine_neighbours)
    assert ref_stats.restored_from_carve > 0

    banded, band_stats = clean_bands(
        fine_scan.scan,
        others=fine_neighbours,
        band_rows=7,
        halo=COMPOSED_FILTER_HALO,
    )
    assert band_stats == ref_stats
    assert_same_scan(reference, banded)


# --------------------------------------------------------------------------
# 2. Band-size invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("band_rows", (*BAND_ROWS_MATRIX, HARNESS_ROWS, 256))
def test_band_rows_invariance(
    harness_scan: synthetic.SyntheticScan,
    harness_neighbours: list[CoarseRangeGrid],
    band_rows: int,
) -> None:
    """Output is independent of `band_rows` (PLAN.md §5, Gate 1).

    `band_rows=3` forces four bands on a 12-row lattice; `band_rows=256`
    forces one. A one-band configuration proves nothing on its own — it has no
    band edges at all (`PHASE1-HALO-CALCULUS.md` §1) — so it is included only
    as the degenerate end of the sweep.
    """
    n_bands = len(_judged_bands(HARNESS_ROWS, band_rows, COMPOSED_FILTER_HALO))
    if band_rows == min(BAND_ROWS_MATRIX):
        assert n_bands >= 3

    reference, ref_stats = clean(harness_scan.scan, others=harness_neighbours)
    banded, band_stats = clean_bands(
        harness_scan.scan,
        others=harness_neighbours,
        band_rows=band_rows,
        halo=COMPOSED_FILTER_HALO,
    )
    assert band_stats == ref_stats, f"band_rows={band_rows} n_bands={n_bands}"
    assert_same_scan(reference, banded)


# --------------------------------------------------------------------------
# 3. The chunk axis, separated from the band axis
# --------------------------------------------------------------------------


def _raw_chunks(scan: StructuredScan, chunk_points: int) -> list[RawChunk]:
    """A row-major raw stream of this scan's lattice addresses."""
    out: list[RawChunk] = []
    n = len(scan)
    for start in range(0, n, chunk_points):
        stop = min(start + chunk_points, n)
        out.append(
            RawChunk(
                offset=start,
                count=stop - start,
                data={
                    "rowIndex": scan.row[start:stop].astype(np.int64).copy(),
                    "columnIndex": scan.col[start:stop].astype(np.int64).copy(),
                },
            )
        )
    return out


@pytest.mark.parametrize("chunk_points", CHUNK_POINTS_MATRIX)
@pytest.mark.parametrize("band_rows", BAND_ROWS_MATRIX)
def test_assembler_bands_at_composed_halo_match_the_plan(
    harness_scan: synthetic.SyntheticScan, band_rows: int, chunk_points: int
) -> None:
    """`iter_row_bands(halo=3)` emits exactly the windows `_judged_bands` plans.

    `PHASE1-HALO-CALCULUS.md` §6 claims no change to `iter_row_bands` is needed
    for halo 3 — its readiness and retention expressions already generalise.
    This measures that claim, and separates the chunk axis from the band axis
    while doing it: chunk boundaries are deliberately not band boundaries.
    """
    scan = harness_scan.scan
    assert chunk_points % (band_rows * HARNESS_COLS) != 0

    bands = list(
        iter_row_bands(
            iter(_raw_chunks(scan, chunk_points)),
            row_min=0,
            row_stop=HARNESS_ROWS,
            band_rows=band_rows,
            halo=COMPOSED_FILTER_HALO,
        )
    )
    planned = _judged_bands(HARNESS_ROWS, band_rows, COMPOSED_FILTER_HALO)
    assert [
        (b.core_row_start, b.core_row_stop, b.data_row_start, b.data_row_stop)
        for b in bands
    ] == [
        (p.core_row_start, p.core_row_stop, p.data_row_start, p.data_row_stop)
        for p in planned
    ]

    # Judge-once at the assembler: every lattice row owned by exactly one core.
    owned = [r for b in bands for r in range(b.core_row_start, b.core_row_stop)]
    assert owned == list(range(HARNESS_ROWS))


@pytest.mark.parametrize("chunk_points", CHUNK_POINTS_MATRIX)
@pytest.mark.parametrize("band_rows", BAND_ROWS_MATRIX)
def test_filtering_driven_by_assembler_bands_matches_full_scan(
    harness_scan: synthetic.SyntheticScan,
    harness_neighbours: list[CoarseRangeGrid],
    band_rows: int,
    chunk_points: int,
) -> None:
    """Filter using the bands the assembler actually produced from a chunk stream.

    This is the chunk axis of `PHASE1-DETERMINISM-SPEC.md` §7 configuration 3
    applied to the filtering stage: the band composition comes from a chunked
    row-major stream, and the result still equals the in-memory reference.
    """
    scan = harness_scan.scan
    bands = [
        BandPlan(
            core_row_start=b.core_row_start,
            core_row_stop=b.core_row_stop,
            data_row_start=b.data_row_start,
            data_row_stop=b.data_row_stop,
        )
        for b in iter_row_bands(
            iter(_raw_chunks(scan, chunk_points)),
            row_min=0,
            row_stop=HARNESS_ROWS,
            band_rows=band_rows,
            halo=COMPOSED_FILTER_HALO,
        )
    ]
    reference, ref_stats = clean(scan, others=harness_neighbours)
    banded, band_stats = clean_bands(scan, others=harness_neighbours, bands=bands)
    assert band_stats == ref_stats, f"band_rows={band_rows} chunk={chunk_points}"
    assert_same_scan(reference, banded)


# --------------------------------------------------------------------------
# 4. Judge-once: halo evidence is read, never counted
# --------------------------------------------------------------------------


def test_each_sample_is_owned_by_exactly_one_band(
    fine_scan: synthetic.SyntheticScan, fine_neighbours: list[CoarseRangeGrid]
) -> None:
    """`PHASE1-HALO-CALCULUS.md` §7: one band writes each sample's disposition.

    Bands overlap by design, so the failure this guards against is a retained
    sample appearing twice, or a per-band count including a halo row. Both
    would inflate the ledger in the direction that looks like more evidence.
    """
    reference, ref_stats = clean(fine_scan.scan, others=fine_neighbours)

    seen: list[int] = []
    despeckled = carved = restored = 0
    for result in iter_clean_bands(
        fine_scan.scan,
        fine_neighbours,
        band_rows=7,
        halo=COMPOSED_FILTER_HALO,
    ):
        rows = result.retained.row
        assert np.all(rows >= result.plan.core_row_start)
        assert np.all(rows < result.plan.core_row_stop)
        assert result.retained.sample_id is not None
        seen.extend(int(v) for v in result.retained.sample_id)
        despeckled += result.dropped_despeckle
        carved += result.dropped_mover_carve
        restored += result.restored_from_carve

    assert len(seen) == len(set(seen))
    assert reference.sample_id is not None
    assert seen == [int(v) for v in reference.sample_id]
    assert despeckled == ref_stats.dropped_despeckle
    assert carved == ref_stats.dropped_mover_carve
    assert restored == ref_stats.restored_from_carve


def test_band_local_ledger_balances(
    fine_scan: synthetic.SyntheticScan, fine_neighbours: list[CoarseRangeGrid]
) -> None:
    """The ledger balances once `retained` is filled in, exactly as it does today.

    `clean` and `clean_bands` both leave `retained` at 0 — `mesh_station` is
    what knows how many samples survived meshing. Filling it the same way here
    checks that the band-local counts are an exclusive partition of the input
    and not merely equal to the reference's.
    """
    _, ref_stats = clean(fine_scan.scan, others=fine_neighbours)
    banded, band_stats = clean_bands(
        fine_scan.scan, others=fine_neighbours, band_rows=7,
        halo=COMPOSED_FILTER_HALO,
    )
    assert band_stats == ref_stats

    from dataclasses import replace

    balanced = replace(band_stats, retained=len(banded))
    balanced.require_balanced()
    assert balanced.accounted == band_stats.input_points
    # Nothing has been counted twice: every disposition sums back to the input.
    assert isinstance(balanced, FilterStats)


# --------------------------------------------------------------------------
# 5. The composed halo, measured
# --------------------------------------------------------------------------

# A purpose-built falsifier for the filter chain (`PHASE1-HALO-CALCULUS.md`
# §9). One lattice, one carved sample, one supporting sample placed so that
# the support crosses `min_support = 5` only when the band can see two rows
# above its core.
#
#   X = (3, 5) at 3.0 m — kept by full-scan despeckle on exactly two agreeing
#       neighbours: (2, 5) above it, and Y diagonally below.
#   Y = (4, 6) at 3.0 m — carved by the neighbour grid, then restored on
#       exactly five agreeing survivors, one of which is X.
#
# With band_rows = 4 the core boundary falls at row 4, so X is a halo row for
# the band that owns Y. At halo 1 that band cannot see row 2, judges X as
# isolated, drops it from the evidence, and Y's support falls to four: Y is
# not restored. At halo 2 and above the band sees row 2 and agrees with the
# full-scan result. That is the 1 + 0 + 1 composition of §5 made observable.
WITNESS_ROWS, WITNESS_COLS = 8, 12
WITNESS_BAND_ROWS = 4
WITNESS_AZ0, WITNESS_AZ_STEP = math.radians(-12.0), math.radians(2.0)
WITNESS_EL0, WITNESS_EL_STEP = math.radians(-8.0), math.radians(2.0)
WITNESS_WALL, WITNESS_NEAR = 5.0, 3.0
WITNESS_X = (3, 5)
WITNESS_Y = (4, 6)
WITNESS_NEAR_CELLS = (WITNESS_X, WITNESS_Y, (2, 5), (3, 7), (4, 7), (5, 6), (5, 7))
WITNESS_DECISION = "restore_keep_at_band_boundary"


def _witness_scan() -> StructuredScan:
    rows = np.repeat(np.arange(WITNESS_ROWS, dtype=np.int32), WITNESS_COLS)
    cols = np.tile(np.arange(WITNESS_COLS, dtype=np.int32), WITNESS_ROWS)
    rng = np.full(rows.size, WITNESS_WALL, dtype=np.float64)
    for r, c in WITNESS_NEAR_CELLS:
        rng[r * WITNESS_COLS + c] = WITNESS_NEAR
    az = WITNESS_AZ0 + cols.astype(np.float64) * WITNESS_AZ_STEP
    el = WITNESS_EL0 + rows.astype(np.float64) * WITNESS_EL_STEP
    direction = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1
    )
    return StructuredScan(
        row=rows,
        col=cols,
        xyz=(direction * rng[:, None]).astype(np.float32),
        rng=rng.astype(np.float32),
        pose=ScanPose(
            translation=np.zeros(3, np.float64), rotation=np.eye(3, dtype=np.float64)
        ),
        lattice=LatticeInfo(
            rows=WITNESS_ROWS,
            cols=WITNESS_COLS,
            az_step=WITNESS_AZ_STEP,
            el_step=WITNESS_EL_STEP,
            az0=WITNESS_AZ0,
            el0=WITNESS_EL0,
            source=LatticeSource.SYNTHETIC,
        ),
        station_id="halo-witness-filter",
        sample_id=np.arange(rows.size, dtype=np.int64),
        source_sample_count=int(rows.size),
    )


def _witness_carve_grid(scan: StructuredScan) -> CoarseRangeGrid:
    """A neighbour that saw straight through Y and nothing else.

    Built from a single world point on the far side of Y, so every other
    angular bin reads `inf` and carves nothing. `from_world_points` rather
    than `build`, because `build` fills empty bins and would widen the carve.
    """
    index = WITNESS_Y[0] * WITNESS_COLS + WITNESS_Y[1]
    target = scan.pose.local_to_world(scan.xyz[index : index + 1])[0]
    origin = np.array([0.0, 4.0, 0.0], dtype=np.float64)
    ray = target - origin
    behind = (origin + ray / np.linalg.norm(ray) * 12.0).reshape(1, 3)
    return CoarseRangeGrid.from_world_points(behind, origin, width=2048, height=1024)


def test_halo_witness_carves_exactly_the_intended_sample() -> None:
    """The witness is only a witness if the carve is the one it was built for."""
    scan = _witness_scan()
    grid = _witness_carve_grid(scan)
    carved = grid.seen_through(scan.pose.local_to_world(scan.xyz))
    hits = [(int(scan.row[i]), int(scan.col[i])) for i in np.flatnonzero(carved)]
    assert hits == [WITNESS_Y]

    _, stats = clean(scan, others=[grid])
    # Full-scan: the carve is undone by restoration, so Y survives.
    assert stats.restored_from_carve == 1
    assert stats.dropped_mover_carve == 0


@pytest.mark.parametrize(
    ("halo", "matches_in_memory"),
    ((0, False), (1, False), (2, True), (3, True), (4, True)),
)
def test_composed_filter_halo_is_two(halo: int, matches_in_memory: bool) -> None:
    """Measured halo table for the filter chain (`PHASE1-HALO-CALCULUS.md` §9).

    §5 derives 2 rows for despeckle -> carve -> restore and §6 raises the
    streamed figure to 3 to cover triangulation's down-reach row. At
    filtering-only scope the third row has nothing to do, so the smallest
    exact halo measured here is 2 — which confirms §5 rather than
    contradicting §6. The named decision that flips is
    `restore_keep_at_band_boundary`.
    """
    scan = _witness_scan()
    grid = _witness_carve_grid(scan)
    reference, ref_stats = clean(scan, others=[grid])
    banded, band_stats = clean_bands(
        scan, others=[grid], band_rows=WITNESS_BAND_ROWS, halo=halo
    )

    y_id = WITNESS_Y[0] * WITNESS_COLS + WITNESS_Y[1]
    assert reference.sample_id is not None and banded.sample_id is not None
    y_retained = y_id in set(int(v) for v in banded.sample_id)

    if matches_in_memory:
        assert band_stats == ref_stats, f"halo={halo}"
        assert np.array_equal(reference.sample_id, banded.sample_id), f"halo={halo}"
        assert y_retained
    else:
        assert (band_stats != ref_stats) or not np.array_equal(
            reference.sample_id, banded.sample_id
        ), f"halo={halo} did not diverge"
        if halo == 1:
            # The single named decision, and only that one.
            assert not y_retained
            assert band_stats.dropped_mover_carve == 1
            assert band_stats.restored_from_carve == 0
            assert band_stats.dropped_despeckle == ref_stats.dropped_despeckle
            assert WITNESS_DECISION == "restore_keep_at_band_boundary"
