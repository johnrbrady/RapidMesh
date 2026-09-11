"""
The signed deviation metric — WP-3.4, DEC-013 amendment 2, Round 10.

Four things have to be true before a bias figure is worth quoting:

1. **The distance is `qa.py`'s distance.** `closest_on_triangle` is
   `qa._point_triangle_distance` with the norm removed, so the two must agree
   bit for bit. If they ever drift, every Round 10 figure stops being comparable
   with every Phase 1 figure, silently.
2. **The normal points at the instrument.** The sign has no meaning until the
   orientation does, and it is the same rule `tile_assemble` applies.
3. **The sign reads the right way round.** A target displaced towards the
   scanner must give a *positive* mean.
4. **Bias sees what RMS cannot.** A surface wrong by the same amount everywhere
   and a surface wrong by the same amount in alternating directions have the
   same RMS. Only one of them is a building that has moved, and the bias budget
   exists to tell them apart — so the test asserts that it does.

Honesty condition (standing rule 8)
------------------------------------
Property 1 is a genuine equivalence test against code that exists on the
pre-change tree, and would fail today if the branch chain had been retyped
rather than reused. Properties 2 to 4 are regression tests on new code, expected
to pass on first run; property 4 carries its own red case, since the alternating
surface is constructed to have the same RMS and a bias near zero, so a "bias"
that were really a magnitude would fail it.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import numpy as np

from rapidmesh.decimate_closest import closest_on_triangle, closest_points
from rapidmesh.decimate_qa import (
    sample_surface,
    summarise,
    triangle_normals,
)
from rapidmesh.qa import _point_triangle_distance


def sheet(rows: int, cols: int, height: float, step: float = 0.02) -> tuple:
    """A flat sheet at `height` above the origin, centred on it."""
    xs, ys = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    verts = np.stack(
        [
            (xs.ravel() - rows / 2.0) * step,
            (ys.ravel() - cols / 2.0) * step,
            np.full(xs.size, height),
        ],
        axis=1,
    ).astype(np.float64)
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    tris = np.concatenate(
        [np.stack([a, a + 1, a + cols + 1], 1), np.stack([a, a + cols + 1, a + cols], 1)]
    ).astype(np.int64)
    return verts, tris


def test_the_closest_point_reproduces_qa_distance_bit_for_bit() -> None:
    rs = np.random.default_rng(3)
    n = 20_000
    p = rs.standard_normal((n, 3))
    a = rs.standard_normal((n, 3))
    b = a + rs.standard_normal((n, 3)) * 0.5
    c = a + rs.standard_normal((n, 3)) * 0.5
    # Degenerate cases matter more than the general one: the branch chain exists
    # for them, so a slice of the batch is made a sliver, an edge and a point.
    b[:200] = a[:200]
    c[:100] = a[:100]
    c[200:400] = a[200:400] + (b[200:400] - a[200:400]) * 0.5

    mine = np.linalg.norm(p - closest_on_triangle(p, a, b, c), axis=1)
    theirs = _point_triangle_distance(p, a, b, c)
    assert mine.tobytes() == theirs.tobytes()


def test_face_normals_point_at_the_instrument() -> None:
    verts, tris = sheet(6, 6, height=3.0)
    normals = triangle_normals(verts, tris)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # The station origin is the scanner, so every normal on a sheet above it
    # must have a negative z.
    assert np.all(normals[:, 2] < 0.0)

    below, tris_below = sheet(6, 6, height=-3.0)
    assert np.all(triangle_normals(below, tris_below)[:, 2] > 0.0)


def test_a_target_displaced_towards_the_scanner_reads_positive() -> None:
    reference, ref_tris = sheet(12, 12, height=2.0)
    # A wider target so every query's closest point is interior to it and the
    # rim cannot contribute an oblique closest point.
    target, tgt_tris = sheet(40, 40, height=2.0)
    shift = 0.0012
    target = target.copy()
    target[:, 2] -= shift                       # towards the origin

    queries, normals = sample_surface(reference, ref_tris, 4_000, seed=1)
    distance, closest, _ = closest_points(target, tgt_tris, queries)
    signed = np.einsum("ij,ij->i", closest - queries, normals)
    report = summarise(
        distance, signed, baseline="fixture", metric="fixture", population=1
    )
    assert np.isclose(report.mean_signed, shift, rtol=0, atol=1e-9)
    assert np.isclose(report.rms, shift, rtol=0, atol=1e-9)


def test_bias_separates_a_drift_from_a_wobble_that_rms_cannot() -> None:
    """The whole reason the signed mean is reported beside the magnitude."""
    reference, ref_tris = sheet(12, 12, height=2.0)
    amount = 0.0008
    queries, normals = sample_surface(reference, ref_tris, 4_000, seed=2)

    drift, drift_tris = sheet(40, 40, height=2.0 - amount)
    wobble, wobble_tris = sheet(40, 40, height=2.0)
    wobble = wobble.copy()
    # Alternate whole rows so the surface stays a height field and every query
    # still lands on a face that is displaced by exactly `amount`.
    rows = np.arange(wobble.shape[0]) // 40
    wobble[:, 2] += np.where(rows % 2 == 0, -amount, amount)

    def figures(verts: np.ndarray, tris: np.ndarray) -> tuple[float, float]:
        distance, closest, _ = closest_points(verts, tris, queries)
        signed = np.einsum("ij,ij->i", closest - queries, normals)
        out = summarise(
            distance, signed, baseline="fixture", metric="fixture", population=1
        )
        return out.rms, out.mean_signed

    drift_rms, drift_bias = figures(drift, drift_tris)
    wobble_rms, wobble_bias = figures(wobble, wobble_tris)

    # Comparable magnitude, and only one of them is a building that moved.
    assert 0.5 < wobble_rms / drift_rms < 2.0
    assert np.isclose(drift_bias, amount, rtol=0, atol=1e-9)
    assert abs(wobble_bias) < 0.25 * abs(drift_bias)


def test_surface_samples_are_weighted_by_area_and_not_by_face() -> None:
    verts = np.array(
        [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [3.0, 3.0, 1.0],
         [3.1, 3.0, 1.0], [3.0, 3.1, 1.0]],
        np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], np.int64)   # areas 4.5 and 0.005
    points, _ = sample_surface(verts, tris, 20_000, seed=5)
    in_big = points[:, 0] + points[:, 1] <= 3.0 + 1e-9
    assert 0.995 < in_big.mean() <= 1.0


def test_an_empty_report_is_zeroed_rather_than_undefined() -> None:
    empty = summarise(
        np.empty(0), np.empty(0), baseline="none", metric="none", population=7
    )
    assert empty.samples == 0
    assert empty.population == 7
    assert empty.rms == 0.0
    assert empty.mean_signed == 0.0
