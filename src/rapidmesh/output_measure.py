"""Measuring an output partition — WP-3.6.

`output_tiles.py` decides what a tile *is*; this decides what to say about one.
The two are separate because ADR-006 asks different questions of them: the
partition is a parameter, and the measurements below are what the Phase 3b
benchmark records across a range of values of it.

Three things live here, and each answers a named line of that benchmark.

**Occupancy** answers the question DEC-021 settled for the *intermediate* and
which this package must not inherit for the *output*. Sample density on a range
image goes as 1/r^2, so a fixed square of ground holds an unbounded number of
full-resolution samples — measured, not argued. An error-bounded sweep flattens
density, so the same square may or may not hold a bounded number of *decimated*
vertices. `metric_occupancy` counts the first, `occupancy_of` the second, at the
same cell size on the same station, so the comparison is like for like.

**Seam-local deviation** answers ADR-006's "Report seam-local maximum deviation,
not just seam-local mean". A site-wide average hides a crack completely, and a
locked seam is exactly where a partition's cost concentrates, so the sample is
drawn in a band around the boundary rather than over the surface.

**Normal discontinuity across seams** answers "cracks or normal discontinuities
detected". The crack half is already exact and cheap — `decimate_tiles.check_seams`
compares boundary edge sets in global ids with no tolerance — so what is added
here is the second half: how far the surface *kinks* where two tiles meet, which
a boundary-set check cannot see because the edges are all still there.

Nothing here is a fidelity claim (ITEM-018) and nothing here is the guaranteed
bound. Per ITEM-027 the two are different quantities: `decimate_bounds` bounds a
vertex's distance to the **planes** merged into it, and every figure in this
module is a **sampled surface** distance. They are never quoted as one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .output_tiles import metric_cell

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .tiles import TileStore

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

#: The bands a seam figure is reported in, in metres, narrowest first. Several
#: rather than one, because "seam-local" has no obvious width and picking one
#: hides the choice: measured on a 12-tile medium station a 0.25 m band caught
#: 73% of the drawn surface. Each band is a column, and the whole-surface figure
#: is always reported beside them.
SEAM_BANDS_M = (0.01, 0.02, 0.05, 0.10, 0.25)

#: How many points are drawn in the seam band. Fixed rather than proportional to
#: seam length, so that a coarse partition with few seams and a fine one with
#: many are each measured to the same precision instead of the fine one getting
#: a better-resolved answer for free.
SEAM_SAMPLES = 100_000


def metric_occupancy(
    store: TileStore, cell_m: float
) -> tuple[int, int, int, float]:
    """`(cells, max owned vertices in one cell, total owned, mean per cell)`.

    The measurement DEC-021 made on the **intermediate**, available here for any
    generation and — via `occupancy_of`— for any decimated tier, so the two can
    be compared at the same cell size on the same station. Counting only; no
    patch is assembled and nothing station-sized is held but the cell histogram,
    which is bounded by the number of occupied cells and not by the surface.
    """
    import numpy as np

    counts: dict[tuple[int, int], int] = {}
    total = 0
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        owned = np.asarray(join.owned, bool)
        if bool(owned.any()):
            cell = metric_cell(np.asarray(payload.positions)[owned], cell_m)
            keys, freq = np.unique(cell, axis=0, return_counts=True)
            for (i, j), n in zip(keys.tolist(), freq.tolist(), strict=True):
                key = (int(i), int(j))
                counts[key] = counts.get(key, 0) + int(n)
            total += int(owned.sum())
        del payload, join
    if not counts:
        return 0, 0, 0, 0.0
    return len(counts), max(counts.values()), total, total / len(counts)


def occupancy_of(positions: Any, cell_m: float) -> tuple[int, int, int, float]:
    """`metric_occupancy` for a resident vertex array — a decimated tier's.

    Same statistic over the same cell size, so the full-resolution and decimated
    numbers are comparable without either being recomputed differently.
    """
    import numpy as np

    if int(np.asarray(positions).shape[0]) == 0:
        return 0, 0, 0, 0.0
    cell = metric_cell(positions, cell_m)
    _, freq = np.unique(cell, axis=0, return_counts=True)
    total = int(freq.sum())
    return int(freq.shape[0]), int(freq.max()), total, total / int(freq.shape[0])


# ---------------------------------------------------------------------------
# seam-local deviation, and the kink a boundary-set check cannot see
# ---------------------------------------------------------------------------


def seam_vertices(out_dir: Path) -> Any:
    """Every locked vertex position across a written decimated tier, deduplicated.

    The locked set *is* the seam: `decimate_tiles.lock_boundary` locks a vertex
    because another tile owns it or because it sits on a once-used edge, and
    those are exactly the vertices a partition refused permission to move. So a
    band around them is the band a partition's cost lives in, and it needs no
    separate definition of "near a boundary".
    """
    import numpy as np

    rows: list[Any] = []
    for path in sorted(out_dir.glob("*.npz")):
        with np.load(path) as data:
            locked = np.asarray(data["locked"], bool)
            if bool(locked.any()):
                rows.append(np.asarray(data["positions"], np.float64)[locked])
    if not rows:
        return np.empty((0, 3), np.float64)
    out: Any = np.unique(np.concatenate(rows), axis=0)
    return out


def seam_local_deviation(
    store: TileStore,
    target_vertices: Any,
    target_triangles: Any,
    seams: Any,
    *,
    samples: int = SEAM_SAMPLES,
    seed: int = 0,
    bands_m: tuple[float, ...] = SEAM_BANDS_M,
) -> dict[str, Any]:
    """Sampled surface deviation, banded by distance to the nearest seam.

    ADR-006 asks for "seam-local maximum deviation, not just seam-local mean",
    because a site-wide average hides a crack completely. It does not say how
    wide "local" is, and the answer is not obvious: a 0.25 m band on a 12-tile
    medium station caught **73% of the drawn surface**, which restricts nothing
    and would have been reported as a seam figure anyway.

    So rather than pick a width, every drawn point is measured once and then
    **bucketed** by its distance to the nearest seam vertex. One pass yields
    every band and the whole-surface figure together, so a seam band is always
    quoted beside the station-wide number it has to be compared with, and the
    width is a column rather than a hidden constant.

    Points are drawn exactly as `decimate_qa.against_pre_decimation` draws them —
    area-weighted over the pre-decimation generation — so the whole-surface row
    is the same quantity Round 12 reported. Distances are exact:
    `decimate_nearest.closest_points_certified` proves it found the closest
    triangle rather than converging towards it.

    Every figure here is a **sampled surface** distance. It is not the guaranteed
    plane bound, and the two are never reported as one quantity (ITEM-027).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from .decimate_nearest import closest_points_certified
    from .decimate_qa import sample_generation, summarise

    queries, normals, _ = sample_generation(store, int(samples), seed)
    if queries.shape[0] == 0:
        return {"drawn": 0, "note": "no surface drawn"}
    if int(np.asarray(seams).shape[0]) == 0:
        return {"drawn": int(queries.shape[0]), "note": "no locked vertices"}

    distance, closest, _, certified = closest_points_certified(
        target_vertices, target_triangles, queries
    )
    signed = np.einsum("ij,ij->i", closest - queries, normals)
    to_seam = cKDTree(np.asarray(seams, np.float64)).query(queries, k=1)[0]

    def block(mask: Any, label: str) -> dict[str, Any]:
        rows = summarise(
            distance[mask], signed[mask],
            baseline="pre-decimation surface",
            metric=label,
            population=int(queries.shape[0]),
            certified=certified[mask],
        ).to_dict()
        rows["share_of_draw"] = float(mask.mean())
        return rows

    out: dict[str, Any] = {
        "drawn": int(queries.shape[0]),
        "seam_vertices": int(np.asarray(seams).shape[0]),
        "whole_surface": block(np.ones(queries.shape[0], bool), "whole surface (exact)"),
        "bands": {},
    }
    for band in bands_m:
        mask = to_seam <= float(band)
        out["bands"][f"{band:g}"] = (
            block(mask, f"within {band:g} m of a seam (exact)")
            if bool(mask.any()) else {"samples": 0, "share_of_draw": 0.0}
        )
    return out


def seam_dihedral_change(store: TileStore, out_dir: Path) -> dict[str, Any]:
    """Do two tiles meet worse than the surface meets itself? — ADR-006 quantity 5.

    `decimate_tiles.check_seams` answers the **crack** question exactly and with
    no tolerance. It cannot see a **kink**: two tiles can agree on every boundary
    edge and still meet at an angle, because each simplified its own side
    independently.

    Two ways of measuring that were tried here and both were wrong, so both are
    recorded rather than quietly replaced.

    *The raw angle.* The baseline 512x512 partition — byte-for-byte Round 12's
    output, seams clean — reads a maximum cross-tile angle of **89.9 degrees**,
    because a wall meets a floor at ninety degrees and a seam running along that
    corner inherits it. The angle is the building's, not the partition's.

    *The change in angle across decimation.* That looked like the isolation
    `CLAUDE.md` §11 asks for, and it is not: it read an 87.8 degree **increase**
    on that same clean baseline. Decimation makes triangles larger, so a dihedral
    between two coarse faces spans much more curvature than one between two
    full-resolution faces — everywhere, seam or not. The measure conflated the
    partition with the coarsening.

    So the control is taken on the **same mesh**: the dihedral distribution at
    **cross-tile** edges against the distribution at **interior** edges of that
    same tier. Both are post-decimation, both are between equally coarse faces,
    so triangle size is common to the two and cancels. What is left is the
    question ADR-006 is actually asking — whether a locked seam leaves the
    surface meeting itself worse than it does anywhere else.

    A ratio near 1 means the partition introduced no discontinuity the surface
    did not already have.
    """
    import numpy as np

    cross, interior = _tier_dihedrals(out_dir)
    if not cross.size:
        return {"seam_edges": 0}

    def block(values: Any) -> dict[str, Any]:
        return {
            "edges": int(values.shape[0]),
            "max_deg": float(values.max()),
            "p99_deg": float(np.percentile(values, 99.0)),
            "median_deg": float(np.median(values)),
            "mean_deg": float(values.mean()),
            "over_1_deg": int(np.count_nonzero(values > 1.0)),
            "over_5_deg": int(np.count_nonzero(values > 5.0)),
        }

    out: dict[str, Any] = {
        "seam_edges": int(cross.shape[0]),
        "cross_tile": block(cross),
        "interior": block(interior) if interior.size else None,
    }
    if interior.size:
        out["ratio_p99_cross_over_interior"] = (
            float(np.percentile(cross, 99.0) / np.percentile(interior, 99.0))
            if np.percentile(interior, 99.0) > 0 else None
        )
        out["ratio_mean_cross_over_interior"] = (
            float(cross.mean() / interior.mean()) if interior.mean() > 0 else None
        )
    del store
    return out


def _tier_dihedrals(out_dir: Path) -> tuple[Any, Any]:
    """`(cross-tile angles, interior angles)` over one written tier, in degrees.

    One pass, so the two populations come off the same mesh with the same
    winding convention and the same arithmetic — which is the whole point of
    using the interior as the control.
    """
    import numpy as np

    faces: list[Any] = []
    owner: list[Any] = []
    gids: list[Any] = []
    xyz: list[Any] = []
    for index, path in enumerate(sorted(out_dir.glob("*.npz"))):
        with np.load(path) as data:
            gid = np.asarray(data["global_vertex_index"], np.int64)
            tri = gid[np.asarray(data["triangles"], np.int64)]
            gids.append(gid)
            xyz.append(np.asarray(data["positions"], np.float64))
        faces.append(tri)
        owner.append(np.full(tri.shape[0], index, np.int64))
    if not faces:
        return np.empty(0), np.empty(0)

    keys, first = np.unique(np.concatenate(gids), return_index=True)
    table = np.concatenate(xyz)[first]
    tri = np.concatenate(faces)
    own = np.concatenate(owner)
    if tri.shape[0] == 0:
        return np.empty(0), np.empty(0)

    pairs = np.concatenate((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]), axis=0)
    pairs.sort(axis=1)
    face_of = np.tile(np.arange(tri.shape[0], dtype=np.int64), 3)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, face_of = pairs[order], face_of[order]
    same = np.flatnonzero(
        (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] == pairs[:-1, 1])
    )
    if same.size == 0:
        return np.empty(0), np.empty(0)
    left, right = face_of[same], face_of[same + 1]

    corner_l = table[np.searchsorted(keys, tri[left])]
    corner_r = table[np.searchsorted(keys, tri[right])]

    def unit(corner: Any) -> Any:
        n = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        out: Any = np.divide(n, length, out=np.zeros_like(n), where=length > 0.0)
        return out

    # `abs` before the arccos: two tiles wind a shared edge in opposite
    # directions, so their face normals are anti-parallel where the surface is
    # flat. The unsigned angle is the kink; the signed one would read 180. The
    # same convention is applied to the interior population, so the control and
    # the measurement cannot differ by a winding.
    dot = np.clip(
        np.abs(np.einsum("ij,ij->i", unit(corner_l), unit(corner_r))), 0.0, 1.0
    )
    angle = np.degrees(np.arccos(dot))
    across = own[left] != own[right]
    return angle[across], angle[~across]


def _dihedrals(
    tri: Any, owner: Any, keys: Any, table: Any, *, cross_only: bool
) -> dict[tuple[int, int], float]:
    """Angle between the two faces at each shared edge, keyed by global id pair.

    `cross_only` keeps just the edges whose two faces have different owners —
    the partition's seams — and is how the decimated side is read. The source
    side takes every shared edge and lets the caller select, because the source's
    own owners are the intermediate windows, which are not the output tiles.
    """
    import numpy as np

    if tri.shape[0] == 0:
        return {}
    pairs = np.concatenate((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]), axis=0)
    pairs.sort(axis=1)
    face_of = np.tile(np.arange(tri.shape[0], dtype=np.int64), 3)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, face_of = pairs[order], face_of[order]
    same = np.flatnonzero(
        (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] == pairs[:-1, 1])
    )
    if same.size == 0:
        return {}
    left, right, edges = face_of[same], face_of[same + 1], pairs[same]
    if cross_only:
        keep = owner[left] != owner[right]
        if not bool(keep.any()):
            return {}
        left, right, edges = left[keep], right[keep], edges[keep]

    def unit(rows: Any) -> Any:
        corner = table[np.searchsorted(keys, tri[rows])]
        n = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        out: Any = np.divide(n, length, out=np.zeros_like(n), where=length > 0.0)
        return out

    # `abs` before the arccos: two tiles wind a shared edge in opposite
    # directions, so their face normals are anti-parallel where the surface is
    # flat. The unsigned angle is the kink; the signed one would read 180.
    dot = np.clip(np.abs(np.einsum("ij,ij->i", unit(left), unit(right))), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return {
        (int(a), int(b)): float(v)
        for (a, b), v in zip(edges.tolist(), angle.tolist(), strict=True)
    }


def _cross_tile_dihedrals(out_dir: Path) -> dict[tuple[int, int], float]:
    """Seam dihedrals of a written decimated tier, keyed by global id pair."""
    import numpy as np

    faces: list[Any] = []
    owner: list[Any] = []
    gids: list[Any] = []
    xyz: list[Any] = []
    for index, path in enumerate(sorted(out_dir.glob("*.npz"))):
        with np.load(path) as data:
            gid = np.asarray(data["global_vertex_index"], np.int64)
            tri = gid[np.asarray(data["triangles"], np.int64)]
            gids.append(gid)
            xyz.append(np.asarray(data["positions"], np.float64))
        faces.append(tri)
        owner.append(np.full(tri.shape[0], index, np.int64))
    if not faces:
        return {}
    keys, first = np.unique(np.concatenate(gids), return_index=True)
    return _dihedrals(
        np.concatenate(faces), np.concatenate(owner), keys,
        np.concatenate(xyz)[first], cross_only=True,
    )


def _source_dihedrals(
    store: TileStore, wanted: set[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """The same angles on the **pre-decimation** surface, for the same edges.

    Only triangles with at least two corners among the wanted edges' endpoints
    are carried. Both faces at an edge contain both of its endpoints, so that
    filter keeps everything needed and nothing else — which is what stops this
    materialising a full-resolution station mesh to measure a seam
    (`CLAUDE.md` §11). What is held is bounded by the seam, not by the surface.
    """
    import numpy as np

    if not wanted:
        return {}
    endpoints = np.unique(np.asarray(sorted(wanted), np.int64).ravel())
    faces: list[Any] = []
    gids: list[Any] = []
    xyz: list[Any] = []
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        gid = np.asarray(join.global_vertex_index, np.int64)
        tri = gid[np.asarray(payload.triangles, np.int64)]
        if tri.shape[0]:
            touching = np.isin(tri, endpoints).sum(axis=1) >= 2
            if bool(touching.any()):
                tri = tri[touching]
                faces.append(tri)
                keep = np.isin(gid, np.unique(tri))
                gids.append(gid[keep])
                xyz.append(np.asarray(payload.positions, np.float64)[keep])
        del payload, join
    if not faces:
        return {}
    keys, first = np.unique(np.concatenate(gids), return_index=True)
    tri = np.concatenate(faces)
    angles = _dihedrals(
        tri, np.arange(tri.shape[0], dtype=np.int64), keys,
        np.concatenate(xyz)[first], cross_only=False,
    )
    return {edge: value for edge, value in angles.items() if edge in wanted}
