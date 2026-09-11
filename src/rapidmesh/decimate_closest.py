"""
Closest point on a resident target surface — WP-3.4.

`qa_stream.surface_distances` answers "how far", in bounded blocks, over a
generation that may be far too large to hold. This answers "how far, **where**,
and on which triangle", over a target small enough to be resident — which a
decimated tier is by construction, and which the full-resolution mesh is not
(`CLAUDE.md` §11, and nothing here ever loads one).

The extra return is the whole reason the module exists. A signed deviation needs
the closest *point*, not only its distance, and so does any later stage that
wants to know which face a sample landed on. `closest_on_triangle` is
`qa._point_triangle_distance`'s branch chain verbatim with the norm removed, so
the two cannot disagree by accident, and `tests/test_decimate_qa.py` holds them
to bit equality rather than to inspection.

The candidate rule is `qa.deviation_report`'s, unchanged: the triangles incident
to a query's `k` nearest target vertices. Narrowing it would be a different
metric (WP-3.0 D3), and these figures have to stay comparable with that one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .qa import QA_QUERY_BLOCK

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

#: How many nearest target vertices contribute candidate triangles.
#:
#: **Not `qa.py`'s 2, and the difference is measured rather than assumed.**
#: `qa.deviation_report` justifies `k = 2` on an undecimated mesh, where a
#: query's closest triangle is incident to one of its two nearest vertices
#: because every vertex is a source observation and the triangles are tiny.
#: Decimation removes that premise: on a coarse tier a query can be closest to
#: the *interior* of a large triangle whose three corners are all further away
#: than two corners of some other triangle, and the rule then misses it.
#:
#: The error is **one-sided** — a missed candidate can only make the reported
#: distance longer, never shorter — so it inflates exactly the tail a p99.9 bar
#: is read from. Measured on ordinal 6 at 4x reduction, 299,999 area-weighted
#: samples (`MEASUREMENTS/2026-09-08/data/r10-ord6-ksens-4x.json`):
#:
#:     k        1       2       4       8      16      32
#:     rms   2.637   1.521   0.813   0.430   0.414   0.412  mm
#:     p99.9 29.190  17.261   5.573   2.089   1.975   1.961  mm
#:     max  121.394 121.394 109.634  24.988  12.731   5.962  mm
#:
#: RMS and p99.9 are within 0.7% of their `k = 32` values by `k = 16`, so 16 is
#: the default. **The maximum is not converged at any k tested** and every
#: figure this rule produces is an upper bound, not an estimate — see
#: `rm_r10_ksens.py` and the Round 10 report, which quotes it as one.
QA_NEAREST_K = 16


def closest_points(
    vertices: F64 | F32,
    triangles: Any,
    queries: F64,
    *,
    k: int = QA_NEAREST_K,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
) -> tuple[F64, F64, I64]:
    """Distance, closest point and winning triangle for every query.

    The target is resident, and deliberately so: it is a **decimated** tier,
    which is small enough to hold by construction and whose size is a figure
    this package reports rather than hides. `CLAUDE.md` §11's rule is about the
    full-resolution mesh, which nothing here loads.

    Queries are evaluated in blocks for the same measured reason `qa_reference`
    gives: the candidate gather is sized by queries times local incidence, not
    by the mesh, so doing them all at once is a large transient for no change in
    the answer.
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
        )
    tree = cKDTree(verts)
    start, incident = _vertex_triangle_map(tris, verts.shape[0])

    best = np.full(n, np.inf, np.float64)
    point = np.zeros((n, 3), np.float64)
    which = np.full(n, -1, np.int64)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        _block(
            queries[lo:hi], tree, verts, tris, start, incident, k, workers,
            best[lo:hi], point[lo:hi], which[lo:hi],
        )
    if not np.all(np.isfinite(best)):
        raise ValueError("a query found no candidate triangle on the target surface")
    return best, point, which


def _block(
    q: F64, tree: Any, verts: F64, tris: I64, start: I64, incident: I64,
    k: int, workers: int, best: F64, point: F64, which: I64,
) -> None:
    """One batch of queries against the whole target, written in place."""
    import numpy as np

    _, nearest = tree.query(q, k=k, workers=workers)
    nearest = nearest.reshape(q.shape[0], -1)
    for col in range(nearest.shape[1]):
        v = nearest[:, col]
        cnt = (start[v + 1] - start[v]).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            continue
        owner = np.repeat(np.arange(q.shape[0], dtype=np.int64), cnt)
        pos = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tri_ids = incident[np.repeat(start[v], cnt) + pos]
        t = tris[tri_ids]
        near = closest_on_triangle(
            q[owner], verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
        )
        d = np.linalg.norm(q[owner] - near, axis=1)
        # `np.minimum.at` reduces the distance but cannot carry the point that
        # produced it, so the winner is found first and then written once.
        order = np.lexsort((d, owner))
        owner, d, near, tri_ids = (
            owner[order], d[order], near[order], tri_ids[order]
        )
        first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        rows = owner[first]
        better = d[first] < best[rows]
        take = first[better]
        best[rows[better]] = d[take]
        point[rows[better]] = near[take]
        which[rows[better]] = tri_ids[take]


def closest_on_triangle(p: F64, a: F64, b: F64, c: F64) -> F64:
    """Ericson's closest point on a triangle, elementwise and vectorised.

    `qa._point_triangle_distance` is this function followed by a norm; the
    branch chain, the masked overwrites and their order are the same, so the two
    agree bit for bit and `tests/test_decimate_qa.py` holds them to it.
    """
    import numpy as np

    ab, ac = b - a, c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = np.maximum(va + vb + vc, 1e-30)
    v = (vb / denom)[:, None]
    w = (vc / denom)[:, None]
    closest = a + ab * v + ac * w

    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)          # edge BC
    t6 = ((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30))[:, None]
    closest = np.where(m6[:, None], b + (c - b) * t6, closest)

    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                        # edge AC
    t5 = (d2 / np.maximum(d2 - d6, 1e-30))[:, None]
    closest = np.where(m5[:, None], a + ac * t5, closest)

    m4 = (d6 >= 0) & (d5 <= d6)                                   # vertex C
    closest = np.where(m4[:, None], c, closest)

    m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                        # edge AB
    t3 = (d1 / np.maximum(d1 - d3, 1e-30))[:, None]
    closest = np.where(m3[:, None], a + ab * t3, closest)

    m2 = (d3 >= 0) & (d4 <= d3)                                   # vertex B
    closest = np.where(m2[:, None], b, closest)

    m1 = (d1 <= 0) & (d2 <= 0)                                    # vertex A
    out: F64 = np.where(m1[:, None], a, closest)
    return out


def _vertex_triangle_map(tris: I64, n_verts: int) -> tuple[I64, I64]:
    """CSR map from vertex index to the triangles incident to it."""
    import numpy as np

    flat = tris.ravel()
    counts = np.bincount(flat, minlength=n_verts)
    start = np.zeros(n_verts + 1, np.int64)
    np.cumsum(counts, out=start[1:])
    order = np.argsort(flat, kind="stable")
    return start, (order // 3).astype(np.int64)


