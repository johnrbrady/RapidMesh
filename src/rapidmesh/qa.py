"""
Accuracy measurement — the artefact that makes "survey-grade" a fact.

Nobody in this market ships a deviation figure with their mesh. For a product
sold to surveying firms that is the gap worth walking into: a number beats an
adjective, and a number is checkable.

Two measurements, and the difference between them matters
---------------------------------------------------------
`deviation_report` measures **mesh against the source points**. That is what
can be computed for a real client scan, and it is what goes in the delivered
report. It answers "how faithfully did we represent what the scanner
measured", and its floor is the instrument's own noise — a perfect mesh through
noisy points still reads about 1 sigma, because it is fitting the noise.

`truth_report` measures **mesh against analytic geometry**, and only works on
the synthetic fixtures in `synthetic.py`. It answers the question the first one
cannot: "how accurate are we *actually*", with the noise removed. A mesh can
score well against the points and badly against the truth — that is
over-fitting to noise — and without a synthetic harness you would never see it.

Method
------
Exact point-to-triangle distance, not point-to-vertex. Point-to-vertex is much
easier and systematically over-reports on a decimated mesh, where the closest
point on the surface is usually in the middle of a triangle. Since the whole
point of the decimation stage is to remove vertices while staying inside a
tolerance, a metric that penalises vertex removal would make the tolerance
meaningless.

Candidate triangles come from a KD-tree over mesh vertices: for each query
point, take the `k` nearest vertices and test every triangle incident to them.
`k = 2` covers the case where the true closest triangle is incident to the
second-nearest vertex, which happens on elongated triangles near edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import DeviationReport, MeshData

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


def deviation_report(
    mesh: MeshData,
    points: F32,
    max_samples: int = 500_000,
    k: int = 2,
    seed: int = 0,
) -> DeviationReport:
    """Point-to-mesh distance statistics, in metres.

    `points` are offsets from the same origin as `mesh.vertices` — scanner-
    local, not project coordinates. Mixing the two frames produces a deviation
    of several hundred kilometres, which is at least an unmistakable failure.

    Subsampled to `max_samples` because the statistics converge long before the
    cost does; half a million points pins an RMS to well under a tenth of a
    millimetre. Sampling is uniform random rather than a lattice stride, so a
    periodic artefact cannot hide between the samples.
    """
    import numpy as np

    d = distances(mesh, points, max_samples=max_samples, k=k, seed=seed)
    if d.size == 0:
        return DeviationReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return DeviationReport(
        sampled_points=int(d.size),
        rms=float(np.sqrt(np.mean(d * d))),
        mean=float(np.mean(d)),
        p95=float(np.percentile(d, 95.0)),
        p99_9=float(np.percentile(d, 99.9)),
        maximum=float(d.max()),
        within_2mm=float(np.mean(d <= 0.002)),
        within_5mm=float(np.mean(d <= 0.005)),
    )


def truth_report(
    mesh: MeshData,
    truth_points: F64,
    max_samples: int = 500_000,
    k: int = 2,
    seed: int = 0,
) -> DeviationReport:
    """Deviation against noise-free analytic surface points.

    Same computation as `deviation_report`; the separate name exists so a
    report never conflates "matches the scan" with "matches reality". Only
    meaningful for `synthetic.py` fixtures.
    """
    import numpy as np

    return deviation_report(mesh, truth_points.astype(np.float32), max_samples, k, seed)


def distances(
    mesh: MeshData,
    points: F32,
    max_samples: int | None = None,
    k: int = 2,
    seed: int = 0,
) -> F64:
    """Per-point distance to the nearest mesh surface. Unaggregated, so a
    caller can map the error back onto the lattice and *see* where it is."""
    import numpy as np
    from scipy.spatial import cKDTree

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
    _, nearest = tree.query(q64, k=k, workers=-1)
    nearest = np.atleast_2d(nearest.T).T if k > 1 else nearest.reshape(-1, 1)

    start, incident = _vertex_triangle_map(tris, verts.shape[0])

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
        dv, _ = tree.query(q64[missing], k=1, workers=-1)
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


def _point_triangle_distance(p: F64, a: F64, b: F64, c: F64) -> F64:
    """Exact distance from points to triangles, elementwise and vectorised.

    Ericson's closest-point-on-triangle (Real-Time Collision Detection §5.1.5),
    with the branch chain expressed as masked overwrites applied in reverse
    priority order so the first matching region wins, exactly as the original
    if-chain does.
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

    # Edge BC
    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t6 = ((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30))[:, None]
    closest = np.where(m6[:, None], b + (c - b) * t6, closest)

    # Edge AC
    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t5 = (d2 / np.maximum(d2 - d6, 1e-30))[:, None]
    closest = np.where(m5[:, None], a + ac * t5, closest)

    # Vertex C
    m4 = (d6 >= 0) & (d5 <= d6)
    closest = np.where(m4[:, None], c, closest)

    # Edge AB
    m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t3 = (d1 / np.maximum(d1 - d3, 1e-30))[:, None]
    closest = np.where(m3[:, None], a + ab * t3, closest)

    # Vertex B
    m2 = (d3 >= 0) & (d4 <= d3)
    closest = np.where(m2[:, None], b, closest)

    # Vertex A
    m1 = (d1 <= 0) & (d2 <= 0)
    closest = np.where(m1[:, None], a, closest)

    out: F64 = np.linalg.norm(p - closest, axis=1)
    return out


# --------------------------------------------------------------------------
# filter scoring — only possible against a labelled fixture
# --------------------------------------------------------------------------


def score_mover_filter(
    is_mover: npt.NDArray[np.bool_], kept: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Recall and false-positive rate for a mover filter, against ground truth.

    `is_mover` and `kept` are both per-input-sample. Returns:

    ``recall``  — fraction of true mover samples removed. The headline number.
    ``false_positive`` — fraction of genuine static surface samples removed.
        This is the one that matters more in practice: a mesh with a ghost car
        in it is embarrassing, but a mesh missing a handrail is wrong.
    ``kept_movers`` — absolute count still in the mesh, because a percentage
        of a large number can still be a visible object.
    """
    import numpy as np

    movers = int(np.count_nonzero(is_mover))
    static = int(np.count_nonzero(~is_mover))
    removed_movers = int(np.count_nonzero(is_mover & ~kept))
    removed_static = int(np.count_nonzero(~is_mover & ~kept))
    return {
        "recall": removed_movers / movers if movers else 1.0,
        "false_positive": removed_static / static if static else 0.0,
        "kept_movers": float(movers - removed_movers),
        "mover_samples": float(movers),
    }
