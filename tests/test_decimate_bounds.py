"""The guaranteed bound and the error-bounded stop rule — WP-3.5.

Three separable claims, and each is tested against something other than the code
that makes it:

* the bound **holds** — checked by recomputing plane distances from the input
  geometry, with none of the sweep's bookkeeping involved;
* the stop rule **binds** — no accepted collapse exceeds the budget, and the
  budget changes the output, so a rule that silently did nothing would be red;
* `sqrt(Q)` **is not** a distance — the defect `decimate_bounds` documents, held
  by a test so that a later change cannot quietly restore the old claim.

`tests/test_decimate.py` covers one patch's mechanics and `test_decimate_kernel`
covers the two sweeps agreeing; neither is repeated here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rapidmesh.decimate import DecimationSettings, decimate_patch
from rapidmesh.decimate_bounds import face_min_areas, plane_bound, verify_plane_bound
from rapidmesh.decimate_nearest import audit_exhaustive, closest_points_certified
from rapidmesh.decimate_tiles import lock_boundary


def ridge(n: int = 31, spacing: float = 0.05, amplitude: float = 0.15):
    """A curved grid patch: fine enough to collapse, curved enough to cost error.

    The height is a function of the **lattice index**, not of the metric
    coordinate, so that scaling `spacing` and `amplitude` together produces a
    geometrically *similar* patch rather than the same wave sampled differently.
    `test_sqrt_of_the_quadric_cost_is_not_a_distance` depends on that
    similarity: a scaling law can only be read off shapes that are the same
    shape.
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    gx, gy = i * spacing, j * spacing
    gz = amplitude * np.sin(i * 0.2) * np.cos(j * 0.2)
    positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append((a, a + 1, a + n + 1))
            faces.append((a, a + n + 1, a + n))
    triangles = np.asarray(faces, np.int64)
    return positions, triangles, lock_boundary(triangles, positions.shape[0],
                                               positions.shape[0])


@pytest.mark.parametrize("budget_mm", [0.5, 1.0, 3.2, 10.0])
def test_the_reported_bound_holds_against_recomputed_plane_distances(budget_mm):
    """Every surviving vertex is inside the bound of every plane it absorbed.

    `verify_plane_bound` rebuilds the planes from the *input* triangles and
    measures the *emitted* float32 positions against them, so nothing the sweep
    recorded is trusted — only the geometry that went in and the geometry that
    came out.
    """
    positions, triangles, locked = ridge()
    patch = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(max_plane_deviation_m=budget_mm / 1000.0),
        kernel="python",
    )
    worst, holds = verify_plane_bound(
        positions, triangles, patch.positions, patch.representative,
        patch.plane_deviation_bound_m,
    )
    assert worst > 0.0, "the check found no plane to measure against"
    assert holds, (
        f"a plane distance of {worst * 1000:.6f} mm exceeded the reported bound "
        f"of {patch.plane_deviation_bound_m * 1000:.6f} mm"
    )


@pytest.mark.parametrize("budget_mm", [0.5, 1.0, 3.2, 10.0])
def test_the_stop_rule_binds_and_the_budget_is_never_exceeded(budget_mm):
    """The reported bound is under the budget, and the budget did something.

    The second half matters as much as the first: a rule wired up but never
    consulted would satisfy "never exceeded" trivially, so the test also
    requires the sweep to have refused candidates and to have stopped short of
    what it does unbounded.
    """
    positions, triangles, locked = ridge()
    budget = budget_mm / 1000.0
    bounded = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(max_plane_deviation_m=budget), kernel="python",
    )
    assert bounded.plane_deviation_bound_m <= budget
    assert bounded.rejected_deviation > 0
    unbounded = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=1), kernel="python",
    )
    assert unbounded.triangle_count < bounded.triangle_count
    assert unbounded.plane_deviation_bound_m > budget


def test_an_unbounded_ratio_sweep_accepts_a_collapse_far_past_any_budget():
    """Round 10 D6, as a test: the ratio rule takes whatever is left.

    This is the behaviour the error-bounded rule exists to replace, so it is
    pinned rather than left as prose in a report — if a later change made the
    ratio rule safe, this test would say so.
    """
    positions, triangles, locked = ridge()
    patch = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 16),
        kernel="python",
    )
    assert patch.plane_deviation_bound_m > 0.01     # 10 mm, three times the p99.9 bar
    assert patch.rejected_deviation == 0            # nothing was ever refused


def test_sqrt_of_the_quadric_cost_is_not_a_distance():
    """`max_accepted_error_m` scales with area, so it is not in metres.

    The same shape at two scales must move by the same *ratio* if the quantity
    is a length. `sqrt(Q)` moves by the square of it, which is the whole of
    `decimate_bounds`'s Part 1 and the reason `max_error_m` is not the stop
    rule. Written as a ratio rather than against absolute numbers so that it
    tests the scaling law and not one machine's arithmetic.
    """
    factor = 10.0
    reported, bounded = [], []
    for scale in (0.05, 0.05 * factor):
        positions, triangles, locked = ridge(spacing=scale, amplitude=3.0 * scale)
        patch = decimate_patch(
            positions, triangles, locked,
            DecimationSettings(target_triangles=triangles.shape[0] // 8),
            kernel="python",
        )
        reported.append(patch.max_accepted_error_m)
        bounded.append(patch.plane_deviation_bound_m)
    # The honest bound scales linearly with the geometry, as a distance must.
    assert bounded[1] / bounded[0] == pytest.approx(factor, rel=0.05)
    # `sqrt(Q)` scales as the square, which is exactly why it is not one.
    assert reported[1] / reported[0] == pytest.approx(factor * factor, rel=0.05)


def test_amin_ranges_over_the_same_faces_the_quadric_summed():
    """A degenerate face carries no plane, so it must carry no area either.

    If the two disagreed the bound would be divided by an area whose plane is
    not in the quadric, and it would be too small — the one direction in which a
    bound must never be wrong.
    """
    positions, triangles, _ = ridge(n=6)
    # Collapse one triangle onto a line: zero area, no plane, no contribution.
    positions[triangles[0, 2]] = positions[triangles[0, 0]]
    amin = np.frombuffer(
        face_min_areas(positions, triangles, positions.shape[0]), np.float64
    )
    corners = positions[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    assert float(area[0]) == 0.0
    for slot in triangles[0]:
        incident = np.flatnonzero(np.any(triangles == slot, axis=1))
        live = area[incident] > 0.0
        expected = float(area[incident][live].min()) if bool(live.any()) else math.inf
        assert amin[slot] == expected


def test_plane_bound_does_not_return_nan_on_a_negative_rounded_cost():
    """A flat quadric at its own minimiser can round below zero.

    An unclamped square root would make the budget test `NaN > limit`, which is
    `False` — so the collapse the rule meant to refuse would be accepted. The
    clamp is the difference between a stop rule and a silent hole in one.
    """
    assert plane_bound([0.0] * 9 + [-1e-18], 1.0, 0.0, 0.0, 0.0) == 0.0
    assert not math.isnan(plane_bound([0.0] * 10, 1.0, 1.0, 2.0, 3.0))


def test_the_certified_metric_agrees_with_brute_force_over_every_triangle():
    """The index's proof, checked against no index at all.

    `audit_exhaustive` measures the same queries against every triangle of the
    tier. A missed candidate can only make the indexed distance longer, so the
    gap is one-sided and any positive value is a real miss.
    """
    positions, triangles, locked = ridge()
    patch = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 12),
        kernel="python",
    )
    verts = patch.positions.astype(np.float64)
    faces = patch.triangles.astype(np.int64)
    _, _, _, certified = closest_points_certified(verts, faces, positions)
    assert bool(certified.all())
    gap, checked = audit_exhaustive(
        verts, faces, positions, rows=np.arange(0, positions.shape[0], 13)
    )
    assert checked > 0
    assert gap == 0.0


def test_the_audit_gives_the_same_answer_however_it_is_blocked():
    """Blocking bounds the audit's memory; it must not touch the answer.

    `AUDIT_TRIANGLE_BLOCK` is half a million, so on any fixture the loop runs
    exactly once and the blocked path is never exercised by the tests above.
    Forcing several blocks is the only way to know that a minimum over blocks is
    still the minimum — which is the whole reason blocking was safe to add on a
    station too large to audit unblocked.
    """
    positions, triangles, locked = ridge()
    patch = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 12),
        kernel="python",
    )
    verts = patch.positions.astype(np.float64)
    faces = patch.triangles.astype(np.int64)
    rows = np.arange(0, positions.shape[0], 29)
    whole, _ = audit_exhaustive(verts, faces, positions, rows=rows)
    # 7 and 64 divide this tier into many blocks; the last is larger than the
    # tier and exercises the single-block path the default always takes.
    assert faces.shape[0] > 64, "the fixture must span several of the blocks below"
    for block in (7, 64, 1 << 19):
        blocked, _ = audit_exhaustive(
            verts, faces, positions, rows=rows, block=block
        )
        assert blocked == whole


def test_the_vertex_k_rule_overstates_the_maximum_that_the_exact_rule_pins():
    """Round 10 D3, as a test rather than as a table in a report.

    The narrow candidate rule misses the large triangle a query sits under, and
    a missed candidate can only make the distance longer — so its maximum is an
    upper bound above the exact one, and that is the failure that made every
    Round 10 error figure a bound.
    """
    from rapidmesh.decimate_closest import closest_points

    positions, triangles, locked = ridge()
    patch = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 12),
        kernel="python",
    )
    verts = patch.positions.astype(np.float64)
    faces = patch.triangles.astype(np.int64)
    exact, _, _, _ = closest_points_certified(verts, faces, positions)
    narrow, _, _ = closest_points(verts, faces, positions, k=2)
    assert narrow.max() > exact.max()
    assert np.all(narrow >= exact - 1e-12)
