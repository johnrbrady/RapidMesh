"""
The global k-nearest-vertex search, from a partition — WP-3.1.

Forward QA's candidate set is fixed by `00-PRODUCT-DEFINITION.md`'s fidelity
metric and is not a tuning parameter: the triangles incident to each query
point's **global** two nearest mesh vertices. Narrowing it to a tile, a band or
a radius is a different metric and needs a version bump (WP-3.0 D3). So this
module answers exactly that question — it just refuses to hold a whole-station
KD-tree to do it.

**Why a partition is enough.** The k nearest over a partition of the vertices is
the k nearest overall. Each block contributes its own k best and a running k best
absorbs them; a block is skipped only when its bounding box is *strictly* farther
than a bound already achieved by real vertices, which cannot discard a candidate
that ties.

**The one place this and `cKDTree` may disagree.** With two vertices exactly
equidistant, `cKDTree` returns whichever its traversal reached first; this
returns the lower global id. The distances are equal either way, so a reported
figure moves only if two tied vertices carry different incident triangles.
`tests/test_qa_bound.py` asserts the whole-station equality and counts the ties
it saw, so the exposure is a measured number rather than an assumption.

`QA_SEED_VERTICES` and the radius it produces are an accelerator and nothing
else. The measurement that forced them is in `nearest_vertices`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .qa_stream import GeometrySource

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

# Vertices in the stride subsample that seeds the neighbour search radius.
# 65,536 float64 positions is 1.6 MB, a fixed cost, and spread widely enough over
# a station's lattice to put every query within a few sample steps of a seed. It
# is an accelerator: it bounds the search, never the answer.
QA_SEED_VERTICES = 65_536


def nearest_vertices(
    source: GeometrySource,
    queries: F64,
    *,
    k: int,
    workers: int = 1,
    block: int,
    seed_cap: int = QA_SEED_VERTICES,
) -> tuple[F64, I64]:
    """The global `k` nearest vertices per query, without a station-wide tree.

    Returns `(distance, global id)`, both `(Q, k)` and both ordered by ascending
    distance. Ties are broken by ascending global id so the answer is a function
    of the mesh's content rather than of the order blocks happened to arrive in;
    `cKDTree` over the whole station breaks them by traversal order instead, and
    the two agree on every distance by construction.

    `block` batches the queries against each block's tree. It is WP-1.10's
    `QA_QUERY_BLOCK` and keeps its meaning: it bounds a query-sized transient.

    **Why the search radius is seeded, measured rather than assumed.** A block of
    a row-major mesh is an elevation annulus (`PHASE1-TILE-CONTRACT-V0.md` §2),
    so a query from another band sits *inside* that block's bounding box while
    being far from every one of its points — the case a KD-tree is worst at,
    because nothing prunes and the search degenerates towards exhaustive. On the
    990,045-sample fixture one half-mesh tree answered its own 150,180 queries in
    **0.40 s** and the other half's 149,820 in **65.80 s**; the same foreign
    queries under a 50 mm `distance_upper_bound` took **0.05 s**. So every query
    is given a finite radius before any block is visited: a bounded stride
    subsample of the vertices, spread over the whole station, is queried once and
    its k-th distance is an upper bound on the true k-th distance.

    None of that can move the answer. The radius is an upper bound derived from
    real vertices, a block is skipped only when its bounding box is *strictly*
    farther, and a candidate refused by `distance_upper_bound` is strictly worse
    than one already held. The bound is widened by a slack step so a vertex tying
    the radius exactly is still returned and can still win the lower-id
    tie-break.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    if block < 1:
        raise ValueError(f"block must be a positive query count, got {block}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    n = int(queries.shape[0])
    best_d = np.full((n, k), np.inf, np.float64)
    best_i = np.full((n, k), -1, np.int64)
    if n == 0:
        return best_d, best_i

    radius = _seed_radius(
        source, queries, k=k, workers=workers, block=block, cap=seed_cap
    )
    for chunk in source.vertex_blocks():
        size = int(chunk.ids.shape[0])
        if size == 0:
            continue
        take = min(k, size)
        # Strict prune: a block whose bounding box is farther than the current
        # bound cannot hold a closer vertex. Strict, so an exact tie is still
        # visited and the lower-id rule still decides it.
        live = np.nonzero(_box_distance(queries, chunk.xyz) <= radius)[0]
        if live.size == 0:
            continue
        tree = cKDTree(chunk.xyz)
        # Ordered by current bound so each batch's shared cap is tight: one
        # loose query cannot then charge its cost to every tight one beside it.
        live = live[np.argsort(radius[live], kind="stable")]
        for lo in range(0, live.size, block):
            rows = live[lo : lo + block]
            found, where = tree.query(
                queries[rows], k=take, workers=workers,
                distance_upper_bound=_slack(float(radius[rows].max())),
            )
            found = _as_columns(found, np.float64)
            ids = chunk.ids[np.clip(_as_columns(where, np.int64), 0, size - 1)]
            ids[~np.isfinite(found)] = -1
            _merge_best(best_d, best_i, rows, found, ids, k)
        np.minimum(radius, best_d[:, k - 1], out=radius)
    return best_d, best_i


def _as_columns(values: Any, dtype: Any) -> Any:
    """`cKDTree.query` returns `(Q,)` for k=1 and `(Q,k)` otherwise."""
    import numpy as np

    array = np.asarray(values, dtype)
    if array.ndim > 1:
        return array.reshape(array.shape[0], -1)
    return array.reshape(-1, 1)


def _slack(bound: float) -> float:
    """One step wider than `bound`, so a candidate that ties it is returned.

    `cKDTree` does not specify whether its upper bound is inclusive, and the
    difference matters exactly where two vertices are equidistant — the one case
    where the lower-id tie-break has anything to decide. Widening is free: a
    candidate admitted by the slack and worse than the k-th best is dropped by
    the merge a moment later.
    """
    import numpy as np

    if not np.isfinite(bound):
        return float(np.inf)
    return float(np.nextafter(bound * (1.0 + 1e-9) + 1e-12, np.inf))


def _seed_radius(
    source: GeometrySource,
    queries: F64,
    *,
    k: int,
    workers: int,
    block: int,
    cap: int,
) -> F64:
    """An upper bound on each query's true k-th nearest distance.

    From a stride subsample of the vertices, capped at `cap` records, so the tree
    it builds is a fixed cost whatever the station's size. The stride is taken
    **within** each block, so the sample is a function of how the source blocks
    its vertices; that is deliberate and harmless, because the sample only sets a
    bound and never contributes an answer. Nothing is merged into the running
    best from here — a seed vertex that genuinely is one of the k nearest is
    found again in its own block, and merging it twice could let one vertex
    occupy two of the k slots.

    Infinite — no bound at all — when the subsample cannot supply `k` points,
    because only the k-th column bounds the k-th nearest and a shorter answer
    would over-prune.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    n = int(queries.shape[0])
    out = np.full(n, np.inf, np.float64)
    total = source.vertex_count
    if total < k or cap < k:
        return out
    stride = max(total // cap, 1)
    # `ascontiguousarray`, not the strided view. A view keeps its whole parent
    # block alive, so collecting one per block held every float64 vertex block
    # at once — measured at 29.5 MB of the seed stage's 40.1 MB on the 990,045
    # sample fixture, for a sample that is 1.6 MB. The copy is the fix, and it is
    # the sort of leak that looks like a constant until someone attributes it.
    pieces = [
        np.ascontiguousarray(chunk.xyz[::stride])
        for chunk in source.vertex_blocks()
        if chunk.ids.shape[0]
    ]
    if not pieces:
        return out
    sample = np.concatenate(pieces)
    del pieces
    if sample.shape[0] < k:
        return out
    tree = cKDTree(sample)
    for lo in range(0, n, block):
        found = _as_columns(
            tree.query(queries[lo : lo + block], k=k, workers=workers)[0], np.float64
        )
        out[lo : lo + block] = found[:, -1]
    return out


def _box_distance(queries: F64, points: F64) -> F64:
    """Distance from each query to the axis-aligned box enclosing `points`.

    A lower bound on the distance to every point in the block, and the whole
    basis of the prune. Zero inside the box.
    """
    import numpy as np

    low = points.min(axis=0)
    high = points.max(axis=0)
    outside = np.maximum(np.maximum(low - queries, queries - high), 0.0)
    out: F64 = np.sqrt(np.einsum("ij,ij->i", outside, outside))
    return out


def _merge_best(
    best_d: F64, best_i: I64, rows: I64, found: F64, ids: I64, k: int
) -> None:
    """Fold one block's candidates into the running k best, in place."""
    import numpy as np

    joint_d = np.concatenate((best_d[rows], found), axis=1)
    joint_i = np.concatenate((best_i[rows], ids), axis=1)
    order = np.lexsort((joint_i, joint_d), axis=1)[:, :k]
    at = np.arange(rows.size, dtype=np.int64)[:, None]
    best_d[rows] = joint_d[at, order]
    best_i[rows] = joint_i[at, order]


def tied_neighbour_count(found: F64) -> int:
    """Queries whose k-th and (k-1)-th neighbour distances are exactly equal.

    The measured exposure of the one place a streamed neighbour search and a
    whole-station `cKDTree` are allowed to disagree. Reported, not assumed.
    """
    import numpy as np

    if found.shape[1] < 2:
        return 0
    return int(np.count_nonzero(found[:, -1] == found[:, -2]))


