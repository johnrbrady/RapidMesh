"""
How much geometry the exact-area fallback actually buffers — WP-12a, ITEM-022.

WP-10mb established that `needs_exact` fires on ordinal 20 and on nothing else,
for **one** component of 1,295. What it did not record is how big that component
is, and the size is the whole question: `exact_component_areas` buffers one
Python float and one list slot per triangle *of each flagged component*, so the
buffer is a function of the flagged components' triangle counts, not of the
station's.

`AreaResult` already carries `counts` and `fallback`. `fallback_triangle_count`
is their inner product and costs nothing to compute — this package adds the
integer, not a measurement.

Nothing here changes accumulation. The two paths deliberately flag on
*different* thresholds — `needs_exact` applies `BOUNDARY_SAFETY`, the resident
`component_area_v1` does not — so these tests pin each path against its own
rule rather than against the other.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import numpy as np

from rapidmesh.pass_b_area import (
    BOUNDARY_SAFETY,
    EXACTNESS_RATIO_LIMIT,
    StreamedAreaAccumulator,
)


def _accumulate(blocks: list[tuple[list[int], list[float]]]) -> StreamedAreaAccumulator:
    accumulator = StreamedAreaAccumulator.empty()
    for roots, areas in blocks:
        accumulator.add_block(
            np.array(roots, np.int64), np.array(areas, np.float64)
        )
    return accumulator


def test_nothing_flagged_gives_a_zero_count() -> None:
    """The field exists and is 0 when the streamed total is trusted everywhere.

    Red before WP-12a: `AreaResult` has no `fallback_triangle_count`.

    Two ordinary components. Every ratio is far below the limit, so
    `needs_exact` returns the empty set, `exact_component_areas` takes its early
    return, and nothing is buffered — which the integer must say.
    """
    accumulator = _accumulate([([0, 0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0, 5.0])])
    flagged = accumulator.needs_exact()
    assert flagged == set()

    result = accumulator.as_area_result(flagged)
    assert result.fallback_components == 0
    assert result.fallback_triangle_count == 0


def test_the_count_is_the_flagged_components_triangles_and_only_those() -> None:
    """One flagged component among three, and the integer names its triangles.

    A component earns the fallback by *span*, not by size: one triangle many
    orders of magnitude smaller than the total drives
    `total / smallest_positive` past `2**29 * 0.99`. So the flagged component
    here is deliberately not the one with the most triangles — an integer that
    reported the largest component, or the station's total, would pass a test
    built the other way round and fail this one.
    """
    tiny = 1.0e-9
    accumulator = _accumulate([(
        # component 0: 4 triangles, ordinary span -> not flagged
        [0, 0, 0, 0]
        # component 1: 3 triangles, one of them minute -> flagged
        + [1, 1, 1]
        # component 2: 6 triangles, ordinary span -> not flagged
        + [2] * 6,
        [1.0, 1.0, 1.0, 1.0]
        + [tiny, 1.0, 1.0]
        + [0.5] * 6,
    )])

    ratios = accumulator.ratios()
    limit = EXACTNESS_RATIO_LIMIT * BOUNDARY_SAFETY
    assert ratios[1] > limit, "fixture must actually trip the exactness condition"
    assert ratios[0] < limit and ratios[2] < limit

    flagged = accumulator.needs_exact()
    assert flagged == {1}

    result = accumulator.as_area_result(flagged)
    assert result.fallback_components == 1
    # Component 1's three triangles, not component 2's six and not all thirteen.
    assert result.fallback_triangle_count == 3
    assert int(result.counts.sum()) == 13


def test_the_count_agrees_with_the_arrays_it_is_derived_from() -> None:
    """`fallback_triangle_count` is `counts[fallback].sum()`, on any input.

    Stated as an identity rather than a literal so the field cannot drift from
    the two arrays a reader would check it against.
    """
    tiny = 1.0e-9
    accumulator = _accumulate([(
        [0, 0, 1, 1, 1, 2, 2, 2, 2],
        [tiny, 1.0, 1.0, 1.0, 1.0, tiny, 2.0, 2.0, 2.0],
    )])
    result = accumulator.as_area_result(accumulator.needs_exact())

    assert result.fallback_triangle_count == int(result.counts[result.fallback].sum())
    assert result.fallback_components == int(result.fallback.sum())
    # Not vacuous: something was flagged and something was not.
    assert 0 < result.fallback_components < result.root_ids.size


def test_the_resident_path_reports_the_field_too() -> None:
    """`component_area_v1` is the reference the streamed path is checked against.

    If only the streamed path carried the integer, the two `AreaResult`s would
    no longer be comparable field for field, which is how `test_pass_b_bound`
    checks them.
    """
    from rapidmesh.triangulate import component_area_v1

    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        np.float32,
    )
    tris = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    roots = np.array([0, 0], np.int64)
    cells = np.array([0, 1, 2, 3], np.int64)

    result = component_area_v1(verts, tris, roots, cells)
    assert result.fallback_components == 0
    assert result.fallback_triangle_count == 0
    assert result.fallback_triangle_count == int(result.counts[result.fallback].sum())
