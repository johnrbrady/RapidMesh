"""
The resident forward-QA implementation, kept as the reference — WP-3.1.

`qa.distances` streams its geometry now (`qa_stream.py`), and the claim that
streaming did not move any number is only checkable against something. This is
that something: the implementation WP-1.10 shipped, unchanged, holding the whole
station's float64 vertices, int64 triangles and vertex→triangle CSR map at once.

It is production code in the sense that it must keep working, and it is not
production code in the sense that nothing calls it: `pipeline.py` and `pass_b.py`
both go through `qa.distances`. `tests/test_qa_bound.py` asserts the two agree
bitwise across the fixture ladder, exactly as `triangulate.component_area_v1`
was kept so `pass_b`'s streamed area reduction could be pinned to it.

**Do not "optimise" this file.** Its value is that it is the older answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .qa import QA_QUERY_BLOCK, _point_triangle_distance

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .types import MeshData

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


def resident_distances(
    mesh: MeshData,
    points: F32,
    max_samples: int | None = None,
    k: int = 2,
    seed: int = 0,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
) -> F64:
    """Per-point distance to the nearest mesh surface, whole station resident.

    Signature-compatible with `qa.distances` so the two can be swapped in a
    test without any other difference between the runs.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    if block < 1:
        raise ValueError(f"block must be a positive query count, got {block}")
    if mesh.triangle_count == 0 or points.shape[0] == 0:
        return np.empty(0, np.float64)

    q = points
    if max_samples is not None and q.shape[0] > max_samples:
        rs = np.random.default_rng(seed)
        q = q[rs.choice(q.shape[0], max_samples, replace=False)]
    q64 = q.astype(np.float64)

    verts = mesh.vertices.astype(np.float64)
    tris = mesh.triangles.astype(np.int64)

    tree = cKDTree(verts)
    start, incident = _vertex_triangle_map(tris, verts.shape[0])

    # `full(inf)` rather than `empty`: if a future edit ever left a query
    # unwritten, uninitialised memory can hold a plausible-looking finite
    # distance and be believed. An infinity cannot.
    best = np.full(q64.shape[0], np.inf, np.float64)
    for lo in range(0, q64.shape[0], block):
        best[lo : lo + block] = _block_distances(
            q64[lo : lo + block], tree, verts, tris, start, incident, k, workers
        )
    return best


def _block_distances(
    q64: F64, tree: Any, verts: F64, tris: I64, start: I64, incident: I64,
    k: int, workers: int,
) -> F64:
    """One batch of queries against the whole mesh.

    Split out of `distances` for one measured reason: the ragged gather below
    costs about **2,955 bytes per query** — the arrays it builds are sized by
    queries times incident triangles, not by the mesh — so evaluating every
    query at once put ~886 MB on the peak at the default 300,000 samples,
    whatever the station's size (`REPORTS/2026-08-31-WP-3.0-report.md` §1.5).

    Splitting cannot move the answer, and the claim is not left as an argument:
    each query's candidate triangle set is the same triangles either way,
    `_point_triangle_distance` is elementwise, and the reduction is a minimum,
    which does not depend on how the queries were grouped. `tests/
    test_forward_qa_blocked.py` asserts bitwise equality against the
    single-batch result at four block sizes.
    """
    import numpy as np

    _, nearest = tree.query(q64, k=k, workers=workers)
    nearest = np.atleast_2d(nearest.T).T if k > 1 else nearest.reshape(-1, 1)

    best = np.full(q64.shape[0], np.inf)
    for col in range(nearest.shape[1]):
        v = nearest[:, col]
        cnt = (start[v + 1] - start[v]).astype(np.int64)
        if not cnt.sum():
            continue
        owner = np.repeat(np.arange(q64.shape[0], dtype=np.int64), cnt)
        # Ragged gather: position within each query's own run of triangles.
        pos = np.arange(owner.size, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tri_ids = incident[np.repeat(start[v], cnt) + pos]
        t = tris[tri_ids]
        d = _point_triangle_distance(
            q64[owner], verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
        )
        np.minimum.at(best, owner, d)

    # A query whose nearest vertices carry no triangles (an isolated vertex
    # that survived compaction) falls back to vertex distance rather than
    # silently reporting inf and poisoning the RMS.
    missing = ~np.isfinite(best)
    if np.any(missing):
        dv, _ = tree.query(q64[missing], k=1, workers=workers)
        best[missing] = dv
    return best


def _vertex_triangle_map(tris: I64, n_verts: int) -> tuple[I64, I64]:
    """CSR map from vertex index to the triangles incident to it."""
    import numpy as np

    flat = tris.ravel()
    counts = np.bincount(flat, minlength=n_verts)
    start = np.zeros(n_verts + 1, np.int64)
    np.cumsum(counts, out=start[1:])
    order = np.argsort(flat, kind="stable")
    return start, (order // 3).astype(np.int64)


