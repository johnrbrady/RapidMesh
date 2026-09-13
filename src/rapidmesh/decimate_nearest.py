"""Closest point on a target surface, with a **certificate** — WP-3.5.

`decimate_closest.closest_points` answers the same question with the candidate
rule `qa.deviation_report` uses: the triangles incident to a query's `k` nearest
target *vertices*. Round 10 measured what that rule does to a decimated tier and
raised it as D3. It is one-sided — a missed candidate can only make a distance
longer — so every figure it produces is an upper bound rather than a
measurement; on ordinal 6 at 16x it is still falling 11% between k=16 and k=64;
on ordinal 16 at 16x it is falling 19% in RMS and 22% in p99.9 between k=64 and
k=128 and has converged at no `k` tested; and **the maximum is not pinned at any
`k` at either tier**, so `00-PRODUCT-DEFINITION.md` §4.1's requirement to report
a maximum could not be met at all.

The cause is a premise, not a tuning value. `qa.py` justifies a small `k` on an
**undecimated** mesh, where every vertex is a source observation and every
triangle is tiny, so the closest triangle is certainly incident to one of the
nearest vertices. Decimation removes that premise: on a coarse tier a query can
be closest to the **interior** of a large triangle all three of whose corners are
further away than two corners of some other triangle. No `k` fixes that, because
the rule is indexed by the wrong thing — and so this module indexes by the
triangle.

Why this converges: it does not converge, it terminates
--------------------------------------------------------
Each triangle `t` has a centroid `c_t` and a radius `r_t`, the distance from the
centroid to its furthest corner, so every point of `t` lies within `r_t` of
`c_t`. For a query `q` and any point `x` on `t`,

    |q - x|  >=  |q - c_t| - r_t

Therefore, if the best distance found so far is `d` and every triangle not yet
examined has `|q - c_t| - r_t > d`, **no unexamined triangle can hold a closer
point** and `d` is exactly the distance to the surface. That is a proof, not a
convergence argument, and it is what the returned `certified` flag records. The
test only ever gets easier as `d` falls, so certifying against a running best is
sound even though the best may improve afterwards.

Triangles are grouped into size classes and the test is applied per class with
that class's own largest radius, because one large triangle in a tier of small
ones would otherwise widen the search radius for every query. A class whose
every member has been examined is certified by exhaustion, so the loop
terminates at worst having looked at everything.

**A query can therefore be reported as exact rather than as an upper bound**, and
a maximum can be pinned. Where certification is not reached — it should not
happen, and the caller is told rather than left to assume — the distance is
still an upper bound and is labelled one.

What this does not change
-------------------------
`decimate_closest.closest_points` stays in the tree, unchanged and still the
default in `decimate_qa`, because every Round 10 and Round 11 figure was measured
with it and a metric that replaced it silently would make those figures
incomparable rather than superseded. The two are run against the same tiers and
the difference is reported (Round 12 §3). The tie-break on an exactly equal
distance is the lower triangle index in both, so the two agree on `which` where
they agree on the distance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .decimate_closest import closest_on_triangle
from .qa import QA_QUERY_BLOCK

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

#: How many centroids a class is asked for on the first pass, and the factor the
#: ask grows by when a query is not yet certified against that class. Four
#: rather than two because an escalation costs a fresh tree query over the
#: uncertified set, and the measured shape is that almost every query certifies
#: on the first pass while a handful need a much wider look.
FIRST_K = 8
ESCALATION = 4

#: A class small enough to examine in full is examined in full: the tree query
#: and the certification test together cost more than the distances do.
EXHAUSTIVE_BELOW = 64

#: The most centroids one class will be asked for before a query is given up on
#: and reported as an upper bound instead of a measurement. A guard against a
#: pathological tier — degenerate slivers sharing a centroid, say — rather than
#: an operating parameter; every run in Round 12 certified 100% of queries well
#: inside it, and the count is reported so that a run which did not would say
#: so rather than quietly print bounds as though they were exact.
ESCALATION_CAP = 8192

#: How many triangles `audit_exhaustive` measures against at once. The audit is
#: the one place that deliberately touches every triangle, and
#: `closest_on_triangle` holds about a dozen `(block, 3)` float64 temporaries
#: while it runs — so this is the knob that keeps a whole-tier audit inside a few
#: hundred megabytes on a high-resolution station instead of a few gigabytes.
AUDIT_TRIANGLE_BLOCK = 1 << 19

#: Class boundaries are powers of two in the triangle radius. A class is merged
#: into the next larger one when it holds fewer triangles than this share of the
#: tier, so that a handful of outliers do not each pay for a tree.
MERGE_BELOW_FRACTION = 0.002


def closest_points_certified(
    vertices: F64 | F32,
    triangles: Any,
    queries: F64,
    *,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
    cap: int = ESCALATION_CAP,
) -> tuple[F64, F64, I64, BOOL]:
    """Distance, closest point, winning triangle, and whether each is exact.

    The target is resident, and deliberately so: it is a **decimated** tier,
    which is small enough to hold by construction and whose size is a figure
    this package reports rather than hides. `CLAUDE.md` §11's rule is about the
    full-resolution mesh, which nothing here loads.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(triangles, np.int64)
    n = int(queries.shape[0])
    if n == 0 or tris.shape[0] == 0:
        return (
            np.empty(0, np.float64),
            np.empty((0, 3), np.float64),
            np.empty(0, np.int64),
            np.empty(0, bool),
        )

    # Centroid and radius are derived in blocks and the `(T, 3, 3)` corner
    # array is never held: it is 72 bytes a triangle, a gigabyte on a
    # high-resolution tier, and only the few candidates a query actually
    # examines are ever needed. `_consider` gathers those from `verts` and
    # `tris` directly.
    centroid = np.empty((tris.shape[0], 3), np.float64)
    radius = np.empty(tris.shape[0], np.float64)
    for lo in range(0, tris.shape[0], AUDIT_TRIANGLE_BLOCK):
        face = tris[lo : lo + AUDIT_TRIANGLE_BLOCK]
        corner = verts[face]
        mid = corner.mean(axis=1)
        centroid[lo : lo + face.shape[0]] = mid
        radius[lo : lo + face.shape[0]] = np.linalg.norm(
            corner - mid[:, None, :], axis=2
        ).max(axis=1)

    classes = _size_classes(radius)
    trees = [(cKDTree(centroid[members]), members, float(radius[members].max()))
             for members in classes]
    del centroid, radius

    best = np.full(n, np.inf, np.float64)
    point = np.zeros((n, 3), np.float64)
    which = np.full(n, -1, np.int64)
    certified = np.ones(n, bool)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        _block(
            queries[lo:hi], trees, verts, tris, workers, cap,
            best[lo:hi], point[lo:hi], which[lo:hi], certified[lo:hi],
        )
    if not np.all(np.isfinite(best)):
        raise ValueError("a query found no candidate triangle on the target surface")
    return best, point, which, certified


def _size_classes(radius: F64) -> list[I64]:
    """Triangle indices grouped so that each group's radii share a scale."""
    import numpy as np

    count = radius.shape[0]
    # `log2` of a zero-radius (degenerate) triangle is -inf; it belongs in the
    # smallest class and `floor` of -inf would not put it there, so it is
    # floored to the smallest finite exponent present instead.
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log2(np.maximum(radius, np.finfo(np.float64).tiny)))
    order = np.argsort(exponent, kind="stable")
    bounds = np.flatnonzero(
        np.r_[True, exponent[order][1:] != exponent[order][:-1], True]
    )
    floor = max(EXHAUSTIVE_BELOW, int(MERGE_BELOW_FRACTION * count))

    out: list[I64] = []
    pending: list[I64] = []
    held = 0
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        members = order[start:stop]
        pending.append(members)
        held += members.shape[0]
        # Merging runs from small radii upward, so a flushed class's own maximum
        # radius is the largest it holds — which is the `R_c` the certification
        # test needs and is tight for that class by construction.
        if held >= floor:
            out.append(np.concatenate(pending))
            pending, held = [], 0
    if pending:
        # The tail is too small to stand alone; it joins the previous class,
        # whose `R_c` then rises to cover it. Correct either way, and one tree
        # cheaper.
        if out:
            out[-1] = np.concatenate([out[-1], *pending])
        else:
            out.append(np.concatenate(pending))
    return out


def _block(
    q: F64, trees: list[tuple[Any, I64, float]], verts: F64, tris: I64,
    workers: int, cap: int, best: F64, point: F64, which: I64, certified: BOOL,
) -> None:
    """One batch of queries against every class, escalating until each is proved."""
    import numpy as np

    rows = q.shape[0]
    for tree, members, class_radius in trees:
        size = members.shape[0]
        active = np.arange(rows, dtype=np.int64)
        k = min(FIRST_K, size) if size > EXHAUSTIVE_BELOW else size
        while True:
            # One query, both halves. The centroid distances do not change as
            # `best` falls — only the test against them does — so asking twice
            # would be the same answer at twice the cost.
            far, near = tree.query(q[active], k=k, workers=workers)
            near = np.asarray(near, np.int64).reshape(active.shape[0], -1)
            edge = np.asarray(far, np.float64).reshape(active.shape[0], -1)[:, -1]
            _consider(q, active, members[near], verts, tris, best, point, which)
            if k >= size:
                break                  # examined in full: certified by exhaustion
            # Open: some unexamined triangle of this class is still near enough
            # to hold a closer point, so this query is not yet proved.
            open_rows = edge <= best[active] + class_radius
            if not bool(open_rows.any()):
                break
            active = active[open_rows]
            if k >= cap:
                # The guard, and it has never fired in measurement. A query left
                # here keeps a distance that is still a valid **upper bound**;
                # what it loses is the proof, which is exactly what the flag
                # records so that a caller reports a bound rather than a
                # measurement for it.
                certified[active] = False
                break
            k = min(k * ESCALATION, size)


def _consider(
    q: F64, active: I64, candidates: I64, verts: F64, tris: I64,
    best: F64, point: F64, which: I64,
) -> None:
    """Take the best of these candidate triangles for each active query."""
    import numpy as np

    per = candidates.shape[1]
    owner = np.repeat(active, per)
    tri_ids = candidates.ravel()
    face = tris[tri_ids]
    near = closest_on_triangle(
        q[owner], verts[face[:, 0]], verts[face[:, 1]], verts[face[:, 2]]
    )
    d = np.linalg.norm(q[owner] - near, axis=1)
    # `np.minimum.at` reduces the distance but cannot carry the point that
    # produced it, so the winner is found first and then written once. The
    # lexsort key puts the lowest triangle index first among equal distances,
    # which is `decimate_closest`'s tie-break and keeps `which` comparable
    # between the two metrics.
    order = np.lexsort((tri_ids, d, owner))
    owner, d, near, tri_ids = owner[order], d[order], near[order], tri_ids[order]
    first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    winners = owner[first]
    better = d[first] < best[winners]
    take = first[better]
    best[winners[better]] = d[take]
    point[winners[better]] = near[take]
    which[winners[better]] = tri_ids[take]


def audit_exhaustive(
    vertices: F64 | F32,
    triangles: Any,
    queries: F64,
    *,
    rows: I64 | None = None,
    block: int = AUDIT_TRIANGLE_BLOCK,
) -> tuple[float, int]:
    """Brute-force the same queries against **every** triangle: `(worst_gap, n)`.

    The certificate above is an argument, and an argument is checked rather than
    believed. This recomputes the distance with no index, no class and no
    candidate rule at all — every query against every triangle — and returns the
    largest amount by which the indexed answer exceeded it. A correct index
    returns 0.0 exactly, and a positive number is a missed candidate.

    It is O(Q x T) and is meant for a subsample: `rows` selects which queries,
    and a few hundred is enough to catch a rule that is wrong, because a rule
    that misses candidates misses them everywhere.

    Triangles are swept in blocks of `block`. `closest_on_triangle` is
    vectorised over its whole input and holds around a dozen `(T, 3)` float64
    temporaries at once, so running it over a whole tier unblocked would want
    several gigabytes on a high-resolution station — an audit that ran out of
    memory on exactly the tiers worth auditing. Blocking bounds it by the block
    and not by the tier, and changes no answer: a minimum over blocks is the
    minimum.
    """
    import numpy as np

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(triangles, np.int64)
    index = (
        np.arange(queries.shape[0], dtype=np.int64) if rows is None
        else np.asarray(rows, np.int64)
    )
    if index.size == 0 or tris.shape[0] == 0:
        return 0.0, 0
    indexed, _, _, _ = closest_points_certified(verts, tris, queries[index])
    worst = 0.0
    for row in range(index.size):
        truth = math.inf
        for lo in range(0, tris.shape[0], block):
            face = tris[lo : lo + block]
            p = np.repeat(queries[index[row]][None, :], face.shape[0], axis=0)
            near = closest_on_triangle(
                p, verts[face[:, 0]], verts[face[:, 1]], verts[face[:, 2]]
            )
            truth = min(truth, float(np.linalg.norm(p - near, axis=1).min()))
        gap = float(indexed[row]) - truth
        if gap > worst:
            worst = gap
    return worst, int(index.size)
