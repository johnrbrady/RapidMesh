"""
The extent ladder — WP-1.G0, closing Round 3.

WP-3.2's density ladder grew the *sampling* of a fixed room. That made it good
evidence for a per-tile bound and useless for a station-wide one: at a 4 m tile
the tile count was **16 at every rung**, so the tiles densified instead of
multiplying, and the three per-sample slopes came out 413 then 171 B/sample —
a factor of 2.4 apart. No line through those describes a bigger station.

This ladder grows the **room** and holds the lattice at 500 x 2000. The scene is
scaled uniformly, scanner position included, so the two ladders vary exactly one
thing each:

| | density ladder (WP-3.2) | extent ladder (this) |
|---|---|---|
| samples | grows | **fixed** |
| room extent | fixed | grows |
| tiles at 4 m | fixed at 16 | grows |

Because the scanner scales with the room, the ladder is a **geometric
similarity**: every ray leaves at the same angle and hits the same surface at a
scaled range. The lattice hit pattern, the dropout draw and the noise draws are
all functions of `rows * cols` and the seed, none of which move. So the sample
count is not merely within the gate's ±5 % — it is *identical*, and a test
asserts that rather than the looser bound, because the weaker assertion would
pass for a ladder that had quietly stopped being a similarity.

**What this ladder can and cannot settle.** It separates the two terms in the
tiled working set: whatever moves across it is the extent/tile term, and
whatever holds still is the sample-store floor, because the sample count is
pinned. It does **not** license an extrapolation to the reference station, which
differs from every rung here in extent *and* density *and* geometry. Nothing in
this module or its report claims one.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.memory import measure_in_child
from rapidmesh.synthetic import RoomScene

TOOLS = str(pathlib.Path(__file__).resolve().parent.parent / "tools")

# The ladder the WP-1.G0 kickoff fixes. Lattice is held; only the scene grows.
EXTENT_SCALES = (1.0, 2.0, 4.0)
LADDER_ROWS, LADDER_COLS = 500, 2000
GATE_TILE_SIZE = 4.0

# Gate G0-3. The assertion below is tighter — exact equality — and this is the
# number the kickoff actually requires, kept so a future scene change that
# breaks similarity without breaking the gate still reports honestly.
SAMPLE_COUNT_TOLERANCE = 0.05

# A small lattice for the in-process behavioural tests. The memory ladder runs
# at the real size in child processes; these only need the geometry to be right.
FAST_ROWS, FAST_COLS = 60, 240


# ---------------------------------------------------------------------------
# RoomScene.scaled — the fixture plumbing
# ---------------------------------------------------------------------------


def test_scaled_multiplies_every_metre_valued_field() -> None:
    base = RoomScene()
    big = base.scaled(3.0)
    assert big.half_x == base.half_x * 3.0
    assert big.half_y == base.half_y * 3.0
    assert big.floor_z == base.floor_z * 3.0
    assert big.ceil_z == base.ceil_z * 3.0
    assert big.rail_radius == base.rail_radius * 3.0
    assert big.column_radius == base.column_radius * 3.0
    assert big.door_depth == base.door_depth * 3.0
    # Tuples scale component-wise, including the scanner: the scanner is *in*
    # the room, so leaving it put would move it relative to the walls and stop
    # the scene being a similarity.
    assert big.scanner == tuple(c * 3.0 for c in base.scanner)
    assert big.rail_centre == tuple(c * 3.0 for c in base.rail_centre)
    assert big.mover_from == tuple(c * 3.0 for c in base.mover_from)


def test_scaled_accounts_for_every_field_on_the_scene() -> None:
    """Totality, so a field added later cannot be silently left unscaled.

    The failure this prevents is quiet: a new metre-valued field that `scaled`
    does not touch makes the ladder stop being a similarity, the sample count
    drifts, and the separation the ladder exists to draw becomes invalid — with
    nothing raising.
    """
    names = {f.name for f in dataclasses.fields(RoomScene)}
    assert names, "RoomScene has no fields; the reflection below proves nothing"
    assert names >= synthetic.NON_METRIC_SCENE_FIELDS
    base = RoomScene()
    scaled = base.scaled(2.0)
    for name in names - synthetic.NON_METRIC_SCENE_FIELDS:
        before, after = getattr(base, name), getattr(scaled, name)
        if isinstance(before, tuple):
            assert after == tuple(c * 2.0 for c in before), name
        else:
            assert after == before * 2.0, name
    for name in synthetic.NON_METRIC_SCENE_FIELDS:
        assert getattr(scaled, name) == getattr(base, name), name


def test_scaled_leaves_the_original_alone() -> None:
    base = RoomScene()
    before = dataclasses.asdict(base)
    base.scaled(7.0)
    assert dataclasses.asdict(base) == before


@pytest.mark.parametrize("factor", (0.0, -1.0, float("nan"), float("inf")))
def test_scaled_rejects_a_factor_that_is_not_a_scale(factor: float) -> None:
    with pytest.raises(ValueError):
        RoomScene().scaled(factor)


def test_scaling_by_one_is_the_identity() -> None:
    assert dataclasses.asdict(RoomScene().scaled(1.0)) == dataclasses.asdict(RoomScene())


# ---------------------------------------------------------------------------
# the ladder is a similarity: extent moves, sampling does not
# ---------------------------------------------------------------------------


def _fast(scale: float) -> synthetic.SyntheticScan:
    return synthetic.generate(
        RoomScene(mover=True).scaled(scale),
        rows=FAST_ROWS, cols=FAST_COLS, dropout=0.01, range_noise=0.002,
        seed=7, station_id="extent",
    )


@pytest.fixture(scope="module")
def rungs() -> dict[float, synthetic.SyntheticScan]:
    return {scale: _fast(scale) for scale in EXTENT_SCALES}


def test_the_sample_count_is_identical_across_the_ladder(
    rungs: dict[float, synthetic.SyntheticScan]
) -> None:
    """G0-3, asserted exactly rather than to ±5 %.

    Every ray leaves at the same angle, the dropout mask is drawn from the same
    seed over the same `rows * cols`, and the near-range cutoff (`t > 0.3` m)
    sits far below the closest surface at every rung. So the valid set is the
    same set, not merely the same size.
    """
    counts = {scale: len(scan.scan) for scale, scan in rungs.items()}
    baseline = counts[EXTENT_SCALES[0]]
    assert baseline > 0
    assert set(counts.values()) == {baseline}, counts
    for scale, count in counts.items():
        assert abs(count - baseline) <= SAMPLE_COUNT_TOLERANCE * baseline, scale


def test_the_same_lattice_cells_are_sampled_at_every_rung(
    rungs: dict[float, synthetic.SyntheticScan]
) -> None:
    """The stronger form of the claim above: same cells, not just as many."""
    first = rungs[EXTENT_SCALES[0]].scan
    for scale in EXTENT_SCALES[1:]:
        other = rungs[scale].scan
        assert np.array_equal(first.row, other.row), scale
        assert np.array_equal(first.col, other.col), scale


def test_the_extent_really_grows(rungs: dict[float, synthetic.SyntheticScan]) -> None:
    """Guard on the guard: a ladder whose extent did not move would pass every
    invariance test above and prove nothing at all."""
    spans = {
        scale: float(
            np.ptp(scan.scan.xyz.astype(np.float64), axis=0).max()
        )
        for scale, scan in rungs.items()
    }
    ordered = [spans[s] for s in EXTENT_SCALES]
    assert ordered[0] < ordered[1] < ordered[2], spans
    # Similarity, so the span should track the scale closely. Loose because the
    # range noise is absolute and does not scale with the room.
    for scale in EXTENT_SCALES[1:]:
        ratio = spans[scale] / spans[EXTENT_SCALES[0]]
        assert abs(ratio - scale) < 0.05 * scale, (scale, ratio)


def test_the_ranges_scale_but_the_noise_does_not(
    rungs: dict[float, synthetic.SyntheticScan]
) -> None:
    """A property of the ladder worth pinning, because it is a real confound.

    `range_noise` is an instrument parameter and is deliberately **not** scaled
    with the scene — a bigger room does not make a scanner noisier. The
    consequence is that relative noise falls as the room grows, and the absolute
    thresholds downstream (`noise_floor`, `min_component_area`) therefore see a
    different scene. The report says what that does to the triangle count; this
    pins the cause so the effect is attributable.
    """
    first = rungs[EXTENT_SCALES[0]].scan
    for scale in EXTENT_SCALES[1:]:
        ratio = float(np.median(rungs[scale].scan.rng) / np.median(first.rng))
        assert abs(ratio - scale) < 0.01 * scale, (scale, ratio)


# ---------------------------------------------------------------------------
# the point of the ladder: tiles multiply instead of densifying
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ladder(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """One tiled run per extent rung, each in its own child process.

    `measure=False`, so the row is the tiling and not QA — the same workload
    WP-3.2's memory rows used, at the same 4 m tile and 64-row band.
    """
    root = tmp_path_factory.mktemp("extent-ladder")
    rows: list[dict[str, Any]] = []
    for scale in EXTENT_SCALES:
        fixture = str(root / f"scan-x{scale:g}.npz")
        made = measure_in_child(
            "measure_peak_memory", "prepare_fixture",
            {"path": fixture, "rows": LADDER_ROWS, "cols": LADDER_COLS,
             "extent_scale": scale},
            label=f"fixture x{scale:g}", sys_path=[TOOLS], timeout=1800.0,
        )
        run = measure_in_child(
            "measure_peak_memory", "tiled_station",
            {"fixture": fixture, "band_rows": 64, "tile_size": GATE_TILE_SIZE,
             "measure": False},
            label=f"tiled x{scale:g}", sys_path=[TOOLS], timeout=3600.0,
        )
        rows.append({
            "scale": scale,
            "samples": int(made.detail["samples"]),
            "tiles": int(run.detail["tiles"]),
            "triangles": int(run.detail["triangles"]),
            "vertices": int(run.detail["vertices"]),
            "working": run.working_set_delta_bytes,
            "peak": run.peak_rss_bytes,
            "tile_bytes": int(run.detail["tile_bytes"]),
            "seconds": run.seconds,
        })
    return rows


def test_the_ladder_has_all_three_rungs(ladder: list[dict[str, Any]]) -> None:
    """G0-1."""
    assert len(ladder) == len(EXTENT_SCALES)
    assert [row["scale"] for row in ladder] == list(EXTENT_SCALES)


def test_tile_count_strictly_increases_with_extent(
    ladder: list[dict[str, Any]]
) -> None:
    """G0-2, and the whole reason this ladder exists.

    WP-3.2's density ladder held the tile count at 16 across a 11x range of
    sample counts. If this one does not move the tile count, the two ladders
    vary the same thing and the separation in the report is not supported.
    """
    tiles = [row["tiles"] for row in ladder]
    assert tiles[0] < tiles[1] < tiles[2], tiles


def test_the_sample_count_is_pinned_across_the_measured_ladder(
    ladder: list[dict[str, Any]]
) -> None:
    """G0-3 on the measured rungs, not only on the fast in-process ones."""
    counts = [row["samples"] for row in ladder]
    baseline = counts[0]
    for count in counts:
        assert abs(count - baseline) <= SAMPLE_COUNT_TOLERANCE * baseline, counts


def test_the_measured_rungs_stay_inside_the_adr_006_budgets(
    ladder: list[dict[str, Any]]
) -> None:
    """Recorded, not gated — the kickoff is explicit that passing 512 MB and
    1.5 GB on this ladder is not a G0 criterion. Asserted anyway at a bar wide
    enough to catch a regression rather than to certify a budget."""
    from rapidmesh.memory import PEAK_RSS_BUDGET_BYTES, WORKING_MEMORY_BUDGET_BYTES

    for row in ladder:
        assert row["working"] <= WORKING_MEMORY_BUDGET_BYTES, row
        assert row["peak"] <= PEAK_RSS_BUDGET_BYTES, row
        assert row["tile_bytes"] > 0, row


def test_the_triangle_count_barely_moves_across_the_ladder(
    ladder: list[dict[str, Any]]
) -> None:
    """The confound, bounded rather than assumed away.

    `range_noise` does not scale with the scene, so the absolute thresholds
    downstream — `noise_floor` at 12 mm, `min_component_area` at 0.005 m² — meet
    a different scene at each rung and the mesh can change size. If it changed
    much, the working-set fall in the test below would be partly "fewer
    triangles" rather than "more tiles", and the separation the ladder exists to
    draw would not hold.

    Measured at 0.36 % across a 4x extent range. The bar is 2 %, wide enough for
    the measurement and far too tight to admit a real confound: a future change
    that makes those thresholds bite would fire this before it could quietly
    corrupt the report's conclusion.
    """
    counts = [row["triangles"] for row in ladder]
    baseline = counts[0]
    assert baseline > 0
    for count in counts:
        assert abs(count - baseline) <= 0.02 * baseline, counts


def test_the_working_set_falls_as_the_tiles_multiply(
    ladder: list[dict[str, Any]]
) -> None:
    """The finding this ladder exists to produce, pinned.

    Samples are held identical and triangles move by under half a percent, so
    what falls between E1 and E3 is the per-tile term and nothing else. That
    single fact is what lets the report bound the per-tile term from below and
    the station-wide floor from above, without arithmetic and without a fitted
    line.

    Not one of the kickoff's gates — it is the result, and a result is worth
    pinning. If a future change flattens this, the separation in the report is
    no longer supported and has to be re-derived rather than re-quoted.
    """
    working = [row["working"] for row in ladder]
    assert working[-1] < working[0], working
    # Measured 28.9 MB of fall across the ladder; require a margin far above
    # run-to-run spread (~0.5 MB) so this is a finding and not a coin toss.
    assert working[0] - working[-1] >= 5_000_000, working
