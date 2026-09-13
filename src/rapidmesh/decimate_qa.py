"""
Signed surface-to-surface deviation for a decimated tier — WP-3.4.

Phase 3's gate asks for magnitude **and bias**: "the bias budget flat and tiny at
every level, so switching can never reveal systematic movement"
(`00-PRODUCT-DEFINITION.md` §8). `qa.py` measures magnitude only, and a mean of
unsigned distances cannot see a drift — every triangle moving 0.5 mm outward and
every triangle moving alternately in and out read the same. This module adds the
sign, and it is the sign DEC-013 calls the figure that matters most.

Two baselines, and neither is ever reported alone
--------------------------------------------------
DEC-013 amendment 2, 8 September 2026:

**Baseline 1 — the pre-decimation surface.** Decimation's own contribution, and
what the bar is judged on. It escapes the ITEM-015 degeneracy: the
full-resolution mesh is built *from* the source observations, so forward QA
reads 0.000 mm by construction, whereas a simplified mesh against the mesh it
came from is a genuine surface-to-surface measurement.

**Baseline 2 — the source observations.** The end-to-end figure, and the one a
viewer actually sees. Baseline 1 alone proves only that decimation is faithful
to the full-resolution mesh, never that the full-resolution mesh is faithful to
the scan. Reporting the flattering half alone would breach `CLAUDE.md` §4 rule 3.

`DecimationDeviation.baseline` names which one a report is, and there is no
default: a caller has to say.

The sign convention, stated once
---------------------------------
    signed = dot(closest_on_target - query, n)

`n` is the unit normal oriented **towards the scanner**, exactly as
`tile_assemble._accumulate_normals` orients a vertex normal — for baseline 1 it
is the pre-decimation triangle's normal at the query, and for baseline 2, where
the query is an observation and has no surface of its own, it is the target
mesh's normal at the closest point. So a **positive** mean says the measured
surface has moved towards the instrument relative to its reference, and a
negative mean says it has moved away. The magnitude is the bias.

What is measured, and what is not
----------------------------------
This is the **one-sided** deviation from the reference surface to the target:
points are drawn on the reference and measured to the target. For a decimator
that only collapses, that is the displacement question — no sheet is invented,
so there is nothing on the target with no reference beneath it. It is **not** a
Hausdorff distance, and a figure from it is never described as one.

Distances reuse `qa._point_triangle_distance`'s branch chain verbatim, returning
the closest point instead of discarding it, and `tests/test_decimate_qa.py`
asserts the two agree bit for bit rather than by inspection.

Two candidate rules, and which one a figure came from is recorded
------------------------------------------------------------------
`metric="vertex-k"` is Round 10's: the triangles incident to a query's `k`
nearest target vertices. A candidate it misses can only make a distance longer,
so every figure it produces is an **upper bound** and never an estimate — and on
a coarse tier that bound is not tight and its maximum is not pinned at any `k`
(`decimate_closest.QA_NEAREST_K`; Round 10 §3.3 and §3.11). It stays the default
so that the recorded Round 10 and Round 11 figures remain reproducible.

`metric="exact"` is WP-3.5's `decimate_nearest.closest_points_certified`: a
triangle-side index that **proves** it has found the closest triangle rather
than converging towards it. A figure from it is the distance, not a bound on it,
and `DecimationDeviation.certified_fraction` records the share of queries the
proof covered — 1.0 when every one of them is exact.

Both are one-sided and both are a **sample** of the surface. An exact per-query
distance does not make the RMS or the p99.9 over 300,000 draws a guarantee about
the points not drawn, and nothing here is described as one. The guaranteed
quantity is `decimate_bounds.py`'s, it is about planes rather than the surface,
and it is reported beside these rather than folded into them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .decimate_closest import QA_NEAREST_K, closest_points
from .decimate_nearest import closest_points_certified

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .tiles import TileStore

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DecimationDeviation:
    """One baseline's figures, in metres, with the sign kept.

    `zero_fraction` is reported because it is the shape of the ITEM-015 trap: a
    query that is still a vertex of the target reads exactly 0.000, and on
    baseline 2 a large share of the observations are still mesh vertices after
    decimation. The RMS over all of them is the honest end-to-end figure and is
    also diluted by that spike, so both are stated.
    """

    baseline: str
    metric: str
    samples: int
    population: int
    rms: float
    mean_absolute: float
    mean_signed: float
    p95: float
    p99_9: float
    maximum: float
    within_1mm: float
    within_3_2mm: float
    zero_fraction: float
    certified_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.baseline}: n={self.samples}/{self.population}  "
            f"rms={self.rms * 1000:.3f} mm  p99.9={self.p99_9 * 1000:.3f} mm  "
            f"bias={self.mean_signed * 1000:+.4f} mm  "
            f"max={self.maximum * 1000:.3f} mm"
        )


def summarise(
    distance: F64, signed: F64, *, baseline: str, metric: str, population: int,
    certified: Any = None,
) -> DecimationDeviation:
    """Turn per-query distances into one consistently-labelled report.

    `certified` is the per-query proof flag from `metric="exact"`. It is `None`
    for the vertex-k rule, which proves nothing and whose figures are upper
    bounds — recorded as a certified fraction of 0.0 rather than as a missing
    field, so that a reader comparing two rows cannot mistake "not proved" for
    "not measured".
    """
    import numpy as np

    if distance.size == 0:
        return DecimationDeviation(
            baseline, metric, 0, population,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    return DecimationDeviation(
        baseline=baseline,
        metric=metric,
        samples=int(distance.size),
        population=int(population),
        rms=float(np.sqrt(np.mean(distance * distance))),
        mean_absolute=float(np.mean(distance)),
        mean_signed=float(np.mean(signed)),
        p95=float(np.percentile(distance, 95.0)),
        p99_9=float(np.percentile(distance, 99.9)),
        maximum=float(distance.max()),
        within_1mm=float(np.mean(distance <= 0.001)),
        within_3_2mm=float(np.mean(distance <= 0.0032)),
        zero_fraction=float(np.mean(distance == 0.0)),
        certified_fraction=(
            0.0 if certified is None else float(np.mean(np.asarray(certified, bool)))
        ),
    )


# ---------------------------------------------------------------------------
# drawing points on a surface
# ---------------------------------------------------------------------------


def triangle_normals(vertices: F64 | F32, triangles: Any) -> F64:
    """Unit face normals, oriented towards the station origin.

    The station origin *is* the scanner (`tile_io`: every tile's `origin` is the
    pose translation), so "towards the origin" is "towards the instrument", and
    it is the same orientation rule `tile_assemble._accumulate_normals` applies
    to vertex normals. A degenerate face gets a zero normal and contributes
    nothing to a signed mean rather than a random direction.
    """
    import numpy as np

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(triangles, np.int64)
    a = verts[tris[:, 0]]
    n = np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    unit = np.divide(n, length, out=np.zeros_like(n), where=length > 0.0)
    centroid = (a + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3.0
    flip = np.einsum("ij,ij->i", unit, -centroid) < 0.0
    unit[flip] *= -1.0
    return unit


def sample_surface(
    vertices: F64 | F32, triangles: Any, count: int, seed: int
) -> tuple[F64, F64]:
    """`count` uniform points over triangle area, with the face normal at each.

    Area-weighted with replacement, as `qa.mesh_to_source_report` draws: one
    sample per triangle would overweight a dense patch of small faces and
    underweight a large one, which is the opposite of a surface measurement.
    """
    import numpy as np

    verts = np.asarray(vertices, np.float64)
    tris = np.asarray(triangles, np.int64)
    if tris.shape[0] == 0 or count <= 0:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)
    corners = verts[tris]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    live = np.flatnonzero(area > 0.0)
    if live.size == 0:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)
    rs = np.random.default_rng(seed)
    chosen = live[
        rs.choice(live.size, size=int(count), replace=True, p=area[live] / area[live].sum())
    ]
    # The sqrt transform makes (u, v) uniform over the triangle rather than
    # bunched at one corner.
    u = np.sqrt(rs.random(chosen.size))[:, None]
    v = rs.random(chosen.size)[:, None]
    tv = corners[chosen]
    points = (1.0 - u) * tv[:, 0] + (u * (1.0 - v)) * tv[:, 1] + (u * v) * tv[:, 2]
    return points, triangle_normals(verts, tris[chosen])


def sample_generation(
    store: TileStore, count: int, seed: int
) -> tuple[F64, F64, float]:
    """Area-weighted samples over a whole written generation, tile by tile.

    Two passes and no resident station: the first reads each tile's total area,
    the second draws that tile's share of the budget. A tile holding a tenth of
    the surface contributes a tenth of the samples, which is what makes the
    result a sample of the station rather than of the tiles.

    Returns the points, their normals, and the station's total triangle area —
    the last because it is the denominator of every per-area figure a reader
    might want to recompute.
    """
    import numpy as np

    areas: dict[int, float] = {}
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        areas[int(tile_id)] = float(_area(payload.positions, payload.triangles))
        del payload
    total = sum(areas.values())
    if total <= 0.0:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64), 0.0

    points: list[Any] = []
    normals: list[Any] = []
    for tile_id in store.tile_ids:
        share = int(round(count * areas[int(tile_id)] / total))
        if share <= 0:
            continue
        payload = store.payload(tile_id)
        # The seed is per tile and derived from the tile id, so the draw does
        # not depend on how many tiles came before it or on what order they
        # were read in — a station's sample is reproducible tile by tile.
        p, n = sample_surface(
            payload.positions, payload.triangles, share, seed * 1_000_003 + int(tile_id)
        )
        points.append(p)
        normals.append(n)
        del payload
    if not points:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64), total
    return np.concatenate(points), np.concatenate(normals), total


def _area(positions: F32, triangles: Any) -> float:
    import numpy as np

    verts = np.asarray(positions, np.float64)
    tris = np.asarray(triangles, np.int64)
    if tris.shape[0] == 0:
        return 0.0
    a = verts[tris[:, 0]]
    n = np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a)
    return float(0.5 * np.linalg.norm(n, axis=1).sum())


# ---------------------------------------------------------------------------
# the two baselines
# ---------------------------------------------------------------------------


def against_pre_decimation(
    store: TileStore,
    target_vertices: F32,
    target_triangles: Any,
    *,
    samples: int,
    seed: int,
    k: int = QA_NEAREST_K,
    workers: int = 1,
    metric: str = "vertex-k",
) -> DecimationDeviation:
    """Baseline 1 — the decimated surface against the surface it came from.

    Queries are drawn on the pre-decimation generation and measured to the
    decimated tier, and the sign is taken along the **pre-decimation** face
    normal, so a positive mean is decimation having pulled the surface towards
    the instrument.
    """
    import numpy as np

    queries, normals, _ = sample_generation(store, samples, seed)
    distance, closest, _, certified, label = _closest(
        target_vertices, target_triangles, queries, k, workers, metric
    )
    signed = np.einsum("ij,ij->i", closest - queries, normals)
    return summarise(
        distance, signed,
        baseline="pre-decimation surface",
        metric=f"pre-decimation-surface-to-decimated-mesh ({label})",
        population=int(store.triangle_count),
        certified=certified,
    )


def against_observations(
    observations: Any,
    target_vertices: F32,
    target_triangles: Any,
    *,
    samples: int,
    seed: int,
    k: int = QA_NEAREST_K,
    workers: int = 1,
    metric: str = "vertex-k",
) -> DecimationDeviation:
    """Baseline 2 — the decimated surface against the source observations.

    `observations` is an `obs_windows.ObservationWindows` over the generation's
    own store: the retained samples the scanner measured, persisted by Pass B
    (`obs_store.py`). The sign is taken along the **target** face normal at the
    closest point, because an observation is a point and has no surface of its
    own to carry one.

    This is the end-to-end figure of DEC-013 amendment 2 and carries both terms
    — decimation's own contribution and the full-resolution mesh's fidelity to
    the scan. It is never reported without baseline 1 beside it, and never
    reported *as* baseline 1.
    """
    import numpy as np

    total = int(observations.count)
    index = (
        np.random.default_rng(seed).choice(total, samples, replace=False)
        if total > samples
        else np.arange(total, dtype=np.int64)
    )
    index.sort()
    queries = observations.gather(index).astype(np.float64)
    distance, closest, which, certified, label = _closest(
        target_vertices, target_triangles, queries, k, workers, metric
    )
    normals = triangle_normals(target_vertices, np.asarray(target_triangles, np.int64))
    signed = np.einsum("ij,ij->i", closest - queries, normals[which])
    return summarise(
        distance, signed,
        baseline="source observations",
        metric=f"retained-observations-to-decimated-mesh ({label})",
        population=total,
        certified=certified,
    )


def _closest(
    vertices: F32, triangles: Any, queries: F64, k: int, workers: int, metric: str
) -> tuple[F64, F64, I64, Any, str]:
    """Dispatch the two candidate rules, and carry back which one answered.

    The label goes into `DecimationDeviation.metric`, so a row on the record
    always says which rule produced it and a reader never has to infer it from
    the date of the run.
    """
    if metric == "vertex-k":
        distance, closest, which = closest_points(
            vertices, triangles, queries, k=k, workers=workers
        )
        return distance, closest, which, None, f"k={k}"
    if metric == "exact":
        distance, closest, which, certified = closest_points_certified(
            vertices, triangles, queries, workers=workers
        )
        return distance, closest, which, certified, "exact, triangle-side index"
    raise ValueError(f"unknown metric {metric!r}")


def open_observations(generation_root: Path) -> Any:
    """The observation store Pass B wrote beside a generation's tiles."""
    from .obs_windows import ObservationWindows

    return ObservationWindows.open(generation_root / "obs" / "observations.rmobs")
