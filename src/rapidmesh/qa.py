"""
Mesh-fidelity measurement — evidence about the derived surface, not survey accuracy.

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

from typing import TYPE_CHECKING, Any

from .types import DeviationReport, MeshData

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

# Queries per batch in `distances`. Chosen by measurement, and the measurement
# corrected a first guess: WP-3.0 read the floor as 50,000, but that reading was
# taken with the mesh built in the same process, so the pipeline's own
# high-water mark sat above the query transient and hid it. Measured against a
# mesh merely loaded from disk, the peak is 432 MB at 50,000 and 386 MB at both
# 25,000 and 10,000 — so **25,000 is the largest batch that reaches the floor**,
# and 50,000 leaves 46 MB of transient that WP-3.1 would expose the moment it
# lowers everything around it. Iteration cost between the two is nil (5.45 s vs
# 5.49 s on that fixture).
#
# A setting under `PHASE1-DETERMINISM-SPEC.md` §2, recorded in the evidence
# envelope, because a reader must be able to see it did not move the output.
QA_QUERY_BLOCK = 25_000


def sha256_file(path: str, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash a source incrementally without materialising it or exposing its path."""
    import hashlib

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def deviation_report(
    mesh: MeshData,
    points: F32,
    max_samples: int = 500_000,
    k: int = 2,
    seed: int = 0,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
) -> DeviationReport:
    """Point-to-mesh distance statistics, in metres.

    `points` are project-axis offsets from the same origin as `mesh.vertices`.
    Scanner-local points must be rotated through their `ScanPose` first. This
    convention matches `SPATIAL-CONTRACT.md` and makes a dropped rotation fail
    as an alignment error rather than remaining internally self-consistent.

    Subsampled to `max_samples` because the statistics converge long before the
    cost does; half a million points pins an RMS to well under a tenth of a
    millimetre. Sampling is uniform random rather than a lattice stride, so a
    periodic artefact cannot hide between the samples.
    """
    d = distances(
        mesh, points, max_samples=max_samples, k=k, seed=seed, workers=workers,
        block=block,
    )
    return _summarise(
        d,
        metric="retained-source-to-mesh",
        population=int(points.shape[0]),
        exact=points.shape[0] <= max_samples,
        source_of_truth="retained source observations",
    )


def mesh_to_source_report(
    mesh: MeshData,
    source_points: F32,
    max_samples: int = 500_000,
    seed: int = 0,
    workers: int = 1,
) -> DeviationReport:
    """Sample finished triangle interiors and measure to source observations.

    Point-to-mesh alone cannot detect a triangle invented across a doorway or
    occlusion: all three vertices may be original samples while the triangle's
    interior represents empty space. This reverse-direction measure exposes
    that failure. Each selected triangle contributes a deterministic uniform
    interior sample; when the triangle count exceeds ``max_samples``, triangle
    selection is area-weighted so the result estimates surface-area error.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    population = mesh.triangle_count
    if population == 0 or source_points.shape[0] == 0 or max_samples <= 0:
        return _summarise(
            np.empty(0, np.float64),
            metric="mesh-to-retained-source",
            population=population,
            exact=False,
            source_of_truth="retained source observations",
        )

    verts = mesh.vertices.astype(np.float64)
    tris = mesh.triangles.astype(np.int64)
    tri_verts = verts[tris]
    cross = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    good = area > 0.0
    tri_ids = np.nonzero(good)[0]
    if tri_ids.size == 0:
        return _summarise(
            np.empty(0, np.float64),
            metric="mesh-to-retained-source",
            population=population,
            exact=False,
            source_of_truth="retained source observations",
        )

    rs = np.random.default_rng(seed)
    # Draw with replacement in proportion to triangle area. This is uniform
    # over the continuous mesh surface; one sample per triangle would
    # overweight a dense patch of tiny faces and underweight a large invented
    # bridge. Small meshes still receive enough interior samples to make a
    # single bad face observable, bounded by the caller's cap.
    n = min(max_samples, max(int(tri_ids.size), min(10_000, max_samples)))
    weights = area[tri_ids]
    chosen = rs.choice(tri_ids, size=n, replace=True, p=weights / weights.sum())

    # sqrt transform gives a uniform point over triangle area.
    u = np.sqrt(rs.random(chosen.size))
    v = rs.random(chosen.size)
    tv = tri_verts[chosen]
    samples = (
        (1.0 - u)[:, None] * tv[:, 0]
        + (u * (1.0 - v))[:, None] * tv[:, 1]
        + (u * v)[:, None] * tv[:, 2]
    )
    tree = cKDTree(source_points.astype(np.float64))
    d, _ = tree.query(samples, k=1, workers=workers)
    return _summarise(
        np.asarray(d, np.float64),
        metric="mesh-to-retained-source",
        population=population,
        # Even one sample per triangle is a sampled interior, not an exact
        # supremum over continuous surface area.
        exact=False,
        source_of_truth="retained source observations",
    )


def _summarise(
    d: F64,
    *,
    metric: str,
    population: int,
    exact: bool,
    source_of_truth: str,
) -> DeviationReport:
    """Build one consistently-labelled deviation report."""
    import numpy as np

    if d.size == 0:
        return DeviationReport(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            metric=metric,
            population=population,
            exact=exact,
            source_of_truth=source_of_truth,
        )
    return DeviationReport(
        sampled_points=int(d.size),
        rms=float(np.sqrt(np.mean(d * d))),
        mean=float(np.mean(d)),
        p95=float(np.percentile(d, 95.0)),
        p99_9=float(np.percentile(d, 99.9)),
        maximum=float(d.max()),
        within_2mm=float(np.mean(d <= 0.002)),
        within_5mm=float(np.mean(d <= 0.005)),
        metric=metric,
        population=population,
        exact=exact,
        source_of_truth=source_of_truth,
    )


def truth_report(
    mesh: MeshData,
    truth_points: F64,
    max_samples: int = 500_000,
    k: int = 2,
    seed: int = 0,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
) -> DeviationReport:
    """Deviation against noise-free analytic surface points.

    Same computation as `deviation_report`; the separate name exists so a
    report never conflates "matches the scan" with "matches reality". Only
    meaningful for `synthetic.py` fixtures.
    """
    import numpy as np

    points = truth_points.astype(np.float32)
    d = distances(
        mesh, points, max_samples=max_samples, k=k, seed=seed, workers=workers,
        block=block,
    )
    return _summarise(
        d,
        metric="analytic-truth-to-mesh",
        population=int(points.shape[0]),
        exact=points.shape[0] <= max_samples,
        source_of_truth="analytic fixture geometry",
    )


def distances(
    mesh: MeshData,
    points: F32,
    max_samples: int | None = None,
    k: int = 2,
    seed: int = 0,
    workers: int = 1,
    block: int = QA_QUERY_BLOCK,
) -> F64:
    """Per-point distance to the nearest mesh surface. Unaggregated, so a
    caller can map the error back onto the lattice and *see* where it is.

    `workers` is the KD-tree query thread count and defaults to **1**, not to
    SciPy's `-1`. `PHASE1-DETERMINISM-SPEC.md` §3 requires every thread pool in
    an evidence run to be an explicit, recordable count, and "all processors on
    whatever machine this is" is not one. Measured on an 88,395-point fixture:
    `workers=1` and `workers=-1` return bitwise-identical distances and
    indices, so this is a reproducibility change and not a numerical one.

    `block` is the query batch size. The three structures that scale with the
    *station* — the float64 vertex copy, the int64 triangle copy and the
    vertex-to-triangle map — are built once, outside the loop, exactly as
    before; only the per-query work is split. Bounding those is `WP-3.1`, and
    is deliberately not attempted here.
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
