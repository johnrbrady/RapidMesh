"""
Bounded reverse QA — `reverse-qa-v2`, PLAN.md §5 item 9, SPEC §5(h).

The property v1 lacked is the one every test here is about: **the figure must
be a function of the mesh, not of the order its triangles happen to sit in.**
`test_v1_moves_with_triangle_order_and_v2_does_not` measures both on the same
mesh and is the reason the reverse report was excluded from the equivalence
comparison until now.

The window has its own two claims, and both are checked rather than asserted:
it must not change the answer (calibration against the unbounded result on the
*same* samples), and when it is made too small it must change it — otherwise
the calibration proves nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rapidmesh import qa, synthetic
from rapidmesh.filters import clean
from rapidmesh.grid import ScanGrid
from rapidmesh.pipeline import mesh_station, mesh_station_streamed
from rapidmesh.reverse_qa import (
    CANONICAL_ORDER_VERSION,
    DEFAULT_QA_WINDOW_ROWS,
    REVERSE_QA_METRIC,
    REVERSE_QA_VERSION,
    _windowed_distances,
    calibrate_window,
    mesh_to_source_report_v2,
    run_reverse_qa,
)
from rapidmesh.triangulate import build_mesh, cull_islands, triangulate, used_vertices
from rapidmesh.types import MeshData

SAMPLES = 20_000


class Fixture:
    """A meshed synthetic station plus the arrays reverse QA needs."""

    def __init__(self, rows: int, cols: int, seed: int = 7) -> None:
        syn = synthetic.generate(
            synthetic.RoomScene(mover=True), rows=rows, cols=cols,
            dropout=0.01, range_noise=0.002, seed=seed, station_id="rev",
        )
        cleaned, _ = clean(syn.scan)
        grid = ScanGrid.build(cleaned)
        tris = cull_islands(grid.scan.xyz, triangulate(grid), min_area=0.005)
        used = used_vertices(len(grid.scan), tris)
        self.mesh = build_mesh(grid.scan, tris)
        self.points = grid.scan.pose.rotate_local(
            grid.scan.xyz[used]
        ).astype(np.float32)
        self.rows = grid.scan.row[used]
        self.lattice = cleaned.lattice


@pytest.fixture(scope="module")
def small() -> Fixture:
    return Fixture(60, 240)


@pytest.fixture(scope="module")
def medium() -> Fixture:
    return Fixture(150, 600)


# ---------------------------------------------------------------------------
# the property v1 lacked
# ---------------------------------------------------------------------------


def _shuffled(mesh: MeshData, seed: int = 3) -> MeshData:
    order = np.random.default_rng(seed).permutation(mesh.triangle_count)
    return MeshData(
        origin=mesh.origin, vertices=mesh.vertices, triangles=mesh.triangles[order],
        source_pose=mesh.source_pose, normals=mesh.normals, rgb=mesh.rgb,
        source_sample_id=mesh.source_sample_id,
    )


def test_v1_moves_with_triangle_order_and_v2_does_not(small: Fixture) -> None:
    """The measurement that blocked the comparison, and the one that lifts it.

    Nothing about the mesh changes — the same triangles, the same vertices,
    only the order of the array. v1's inverse-CDF selection walks a probability
    vector in that order and picks different triangles; v2's selection is a
    function of canonical triple identity and cannot.
    """
    shuffled = _shuffled(small.mesh)

    v1_a = qa.mesh_to_source_report(small.mesh, small.points, max_samples=SAMPLES)
    v1_b = qa.mesh_to_source_report(shuffled, small.points, max_samples=SAMPLES)
    assert v1_a.rms != v1_b.rms, "v1 was expected to move with array order"

    v2_a, _ = mesh_to_source_report_v2(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    v2_b, _ = mesh_to_source_report_v2(
        shuffled, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    for field in ("sampled_points", "rms", "mean", "p95", "p99_9", "maximum"):
        assert getattr(v2_a, field) == getattr(v2_b, field), field


def test_the_same_mesh_measured_twice_is_bitwise_identical(small: Fixture) -> None:
    """No RNG, so no reseeding and no process-global state to depend on."""
    first, _ = run_reverse_qa(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    second, _ = run_reverse_qa(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    assert np.array_equal(first.view(np.uint64), second.view(np.uint64))


def test_a_different_seed_moves_the_interior_points(small: Fixture) -> None:
    """The seed is real, not decorative: it selects where inside each triangle
    the sample falls, so a different seed must give a different figure."""
    a, _ = run_reverse_qa(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES, seed=0
    )
    b, _ = run_reverse_qa(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES, seed=1
    )
    assert a.size == b.size
    assert not np.array_equal(a, b)


def test_selection_produces_exactly_n_samples_by_the_spec_rule(small: Fixture) -> None:
    """SPEC §5(h) step 3 keeps v1's count rule; step 4 makes it systematic, so
    the count must be exact rather than approximate."""
    for max_samples in (1_000, SAMPLES):
        _, evidence = run_reverse_qa(
            small.mesh, small.points, source_rows=small.rows, max_samples=max_samples
        )
        triangles = evidence.positive_area_triangles
        expected = min(max_samples, max(triangles, min(10_000, max_samples)))
        assert evidence.samples_selected == expected
        assert evidence.samples_measured == expected      # nothing unmatched here


# ---------------------------------------------------------------------------
# the window: it must not change the answer, and too small a one must
# ---------------------------------------------------------------------------


def test_the_window_does_not_change_the_answer(medium: Fixture) -> None:
    """Calibration against the unbounded result, on the *same* samples.

    "Match" is bitwise-equal float64 distance, not "close". Only the candidate
    set differs between the two runs, so any disagreement would be the window's
    and nothing else's.
    """
    result = calibrate_window(
        medium.mesh, medium.points, source_rows=medium.rows,
        qa_window_rows=DEFAULT_QA_WINDOW_ROWS, max_samples=SAMPLES,
    )
    assert result["identical"] == result["samples"], result
    assert result["max_overestimate_m"] == 0.0
    assert result["unmatched"] == 0.0
    # And it is genuinely bounded: the candidate set is a fraction of the mesh.
    assert result["largest_candidate_set"] < 0.5 * result["unbounded_candidate_set"]


def test_too_small_a_window_does_change_the_answer(small: Fixture) -> None:
    """The falsifier. A calibration that passes at every window size would be
    telling us nothing about the window.

    At zero rows the candidate set is only the triangle's own rows, and on this
    fixture exactly one sample of 20,000 then finds a worse nearest neighbour —
    over-estimating by about 6.4 mm.
    """
    starved = calibrate_window(
        small.mesh, small.points, source_rows=small.rows,
        qa_window_rows=0, max_samples=SAMPLES,
    )
    assert starved["identical"] < starved["samples"], starved
    assert starved["max_overestimate_m"] > 0.001, starved

    fine = calibrate_window(
        small.mesh, small.points, source_rows=small.rows,
        qa_window_rows=DEFAULT_QA_WINDOW_ROWS, max_samples=SAMPLES,
    )
    assert fine["identical"] == fine["samples"], fine


def test_a_bounded_distance_is_never_an_under_estimate(small: Fixture) -> None:
    """The direction of the error is the safety property.

    A window can only remove candidates, so a windowed nearest-neighbour
    distance is greater than or equal to the global one. A metric that could
    silently *under*-report deviation would flatter the mesh.
    """
    for window in (0, 1, 2, DEFAULT_QA_WINDOW_ROWS):
        result = calibrate_window(
            small.mesh, small.points, source_rows=small.rows,
            qa_window_rows=window, max_samples=SAMPLES,
        )
        assert result["min_difference_m"] >= 0.0, (window, result)


def test_the_window_of_eight_covers_the_reach_the_triangulator_allows(
    small: Fixture, medium: Fixture
) -> None:
    """Why 8 rows, structurally rather than by fitting the fixture.

    `source_points` are the mesh's own vertices, so a triangle's nearest
    observation is never further than its own longest edge. Triangulation
    accepts an edge only below `range * step * tan(max_incidence)`, and one
    lattice row subtends `range * el_step`, so an accepted edge spans at most
    `tan(82 deg) * step / el_step` rows — 7.12 where the elevation step is the
    coarser one. A window of 8 covers that with a row to spare.

    The condition is stated because it is a condition: on a lattice whose
    azimuth step is coarser, the reach scales by `az_step / el_step` and 8 must
    be re-derived rather than assumed.
    """
    for fixture in (small, medium):
        lattice = fixture.lattice
        step = max(abs(lattice.az_step), abs(lattice.el_step))
        reach = math.tan(math.radians(82.0)) * step / abs(lattice.el_step)
        assert reach <= DEFAULT_QA_WINDOW_ROWS, (reach, lattice.describe())

        rows = fixture.rows[np.asarray(fixture.mesh.triangles, np.int64)]
        span = int((rows.max(axis=1) - rows.min(axis=1)).max())
        assert span <= DEFAULT_QA_WINDOW_ROWS


def test_an_empty_window_is_excluded_and_counted_not_invented() -> None:
    """Out-of-window behaviour, named.

    Under the production contract — `source_points` are the mesh's own retained
    observations — a triangle's own vertices are always candidates, so this
    branch is unreachable there. It is exercised directly because it becomes
    reachable the moment the observation store (PLAN.md §5 item 12) supplies a
    source set the mesh does not span, and because the alternative to excluding
    an unmatched sample is inventing a match.
    """
    samples = np.zeros((3, 3), np.float64)
    points = np.zeros((4, 3), np.float64)
    source_rows = np.array([0, 0, 1, 1], np.int64)
    lo = np.array([0, 500, 0], np.int64)          # middle sample is far away
    hi = np.array([0, 500, 1], np.int64)

    distances, unmatched, windows, largest = _windowed_distances(
        samples, points, source_rows, lo, hi, 8
    )
    assert distances.size == 2
    assert unmatched == 1
    assert windows == 3
    assert largest == 4


# ---------------------------------------------------------------------------
# evidence and versioning
# ---------------------------------------------------------------------------


def test_the_report_and_evidence_state_the_window_and_the_version(
    small: Fixture,
) -> None:
    report, evidence = mesh_to_source_report_v2(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    assert report.metric == REVERSE_QA_METRIC
    assert report.metric != "mesh-to-retained-source"      # v1 is a different metric
    assert report.exact is False                           # a sampled interior
    assert report.population == small.mesh.triangle_count
    assert evidence.metric_version == REVERSE_QA_VERSION
    assert evidence.canonical_order_version == CANONICAL_ORDER_VERSION
    assert evidence.qa_window_rows == DEFAULT_QA_WINDOW_ROWS
    assert dict(evidence.as_settings()) == {
        "reverse_qa_version": REVERSE_QA_VERSION,
        "qa_window_rows": str(DEFAULT_QA_WINDOW_ROWS),
    }
    assert evidence.bounded


def test_an_empty_mesh_reports_nothing_rather_than_zero_deviation() -> None:
    empty = MeshData(
        origin=np.zeros(3), vertices=np.zeros((0, 3), np.float32),
        triangles=np.zeros((0, 3), np.uint32), source_pose=None,
        source_sample_id=np.zeros(0, np.int64),
    )
    report, evidence = mesh_to_source_report_v2(
        empty, np.zeros((0, 3), np.float32), source_rows=np.zeros(0, np.int32)
    )
    assert report.sampled_points == 0
    assert report.metric == REVERSE_QA_METRIC
    assert evidence.samples_selected == 0
    assert evidence.positive_area_triangles == 0


def test_the_pipeline_line_by_line_matches_the_direct_call(small: Fixture) -> None:
    """The pipeline must not be running some other variant of the metric."""
    direct, _ = mesh_to_source_report_v2(
        small.mesh, small.points, source_rows=small.rows, max_samples=SAMPLES
    )
    syn = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=60, cols=240,
        dropout=0.01, range_noise=0.002, seed=7, station_id="rev",
    )
    result = mesh_station(syn.scan, measure=True, measure_samples=SAMPLES)
    assert result.mesh_to_source is not None
    assert result.mesh_to_source.rms == direct.rms
    assert result.reverse_qa_evidence is not None
    assert result.reverse_qa_evidence.qa_window_rows == DEFAULT_QA_WINDOW_ROWS
    assert result.settings["qa_window_rows"] == DEFAULT_QA_WINDOW_ROWS
    assert result.settings["reverse_qa_version"] == REVERSE_QA_VERSION


def test_both_pipelines_produce_the_same_reverse_figure() -> None:
    """§4.4: reverse QA runs in Pass B, against final dispositions, under the
    same contract as the in-memory path — so the figures must agree exactly."""
    syn = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=12, cols=48,
        dropout=0.0, range_noise=0.0, seed=17, station_id="harness",
    )
    reference = mesh_station(syn.scan, measure=True, measure_samples=2_000)
    for band_rows in (3, 4, 6):
        streamed = mesh_station_streamed(
            syn.scan, band_rows=band_rows, chunk_points=100, halo=3,
            measure=True, measure_samples=2_000,
        )
        left, right = reference.mesh_to_source, streamed.mesh_to_source
        assert left is not None and right is not None
        for field in ("sampled_points", "rms", "mean", "p95", "p99_9", "maximum"):
            assert getattr(left, field) == getattr(right, field), (band_rows, field)
        assert streamed.reverse_qa_evidence is not None
        assert reference.reverse_qa_evidence is not None
        assert (
            streamed.reverse_qa_evidence.samples_selected
            == reference.reverse_qa_evidence.samples_selected
        )


def test_the_pipeline_b_reservoir_fallback_is_not_implemented() -> None:
    """PLAN.md §5 item 9 names the reservoir strategy as the Pipeline B
    fallback, *later*. It is not built, so it must not be reachable by name —
    a deferred capability that can be called by accident is a claimed one."""
    import rapidmesh.reverse_qa as module

    names = [n for n in dir(module) if "reservoir" in n.lower()]
    assert names == []
    assert "not implemented" in module.__doc__.lower()
