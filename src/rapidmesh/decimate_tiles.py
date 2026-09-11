"""
Decimation across a written tile generation, one tile at a time — WP-3.4.

`decimate.py` simplifies one patch and knows nothing else. This module is
everything else: which vertices a tile may not move, how a decimated tile is
written, and how the seam guarantee is *measured* on the finished station rather
than argued for in a comment.

Why the tiles are the unit, and why the tiling is not redesigned
----------------------------------------------------------------
DEC-021's lattice window already bounds a tile at `window_rows x window_cols`
owned vertices whatever the geometry does, and Gate 1 rests on it. Decimation
inherits that bound for free: one tile is resident at a time, so the working set
follows the window and not the station, and `CLAUDE.md` §11's "do not
materialise a full-resolution mesh in memory" is satisfied by construction
rather than by care.

The locked set — three rules, and only the first is about tiles
---------------------------------------------------------------
A vertex is locked when it is

1. **not owned by this tile** — it is a corner of an emitted triangle that a
   neighbouring tile owns, stored here only so the triangle is resolvable
   (`tile_io`'s boundary ring);
2. an endpoint of an edge **used by exactly one** of this tile's triangles —
   the patch boundary, which is either a seam with a neighbour or a true rim of
   the surface, and this module deliberately does not distinguish them; or
3. an endpoint of an edge used by **more than two** — a non-manifold edge, where
   a collapse has no well-defined meaning.

Rules 2 and 3 are computed from the tile's own triangles, with no cross-tile
lookup, which is what lets a tile be decimated in isolation.

The seam guarantee, and how it is checked
------------------------------------------
Because every patch-boundary edge has two locked endpoints, no such edge can be
collapsed; because `decimate.py` enforces the link condition, no new one can be
created. So **the set of boundary edges is invariant across decimation**, and
that is checkable exactly, in global vertex ids, with no tolerance:

* a crack — one tile dropping a seam edge its neighbour keeps — removes that
  edge from one side, so the union sees it once instead of twice and it appears
  as a *gained* boundary edge;
* a T-junction — one tile collapsing a mid-seam vertex `m` so it holds `(a,b)`
  where its neighbour still holds `(a,m)` and `(m,b)` — turns all three into
  once-used edges, and all three appear as gained boundary edges.

`union_boundary` computes the set; `check_seams` compares before against after.
A station passes when the symmetric difference is empty **and** every vertex
held by more than one tile has identical position bits in all of them.

A third failure is invisible to both and is refused in `decimate.py` instead:
two tiles independently closing the same fan onto the same pair of ring
vertices emit the same face twice, which doubles the surface rather than
tearing it, so the boundary set is unchanged and only a union edge-use count
sees it. `decimate._would_join_locked` refuses the collapse that would do it.
This was a real defect, found on the synthetic station fixture at 4x reduction
(11 edges used by four triangles instead of two), and
`tests/test_decimate_generation.py` holds the rule red with it removed.
`tests/test_decimate_seams.py` drives the same functions against a two-tile
fixture with the locking deliberately removed, and they come back red.

Output
------
The prototype writes one `.npz` per tile and a manifest, **not** a `.rmtile`
generation. That is a deliberate limit of Round 10 and not an oversight: a
finished tile carries per-vertex normals whose area-weighted sums are only
complete in the tile that owns the vertex, and reconstituting that after
decimation is production work this package is not authorised to do. What is
written is the geometry and the join needed to measure and to reproduce.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .decimate import DecimationSettings, decimate_patch

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .tiles import TileStore

    F32 = npt.NDArray[np.float32]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

#: The prototype's own on-disk contract. Bumped if the key set changes, so a
#: harness reading an older run fails loudly rather than reading a field that
#: has moved.
DECIMATED_CONTRACT = "rapidmesh-decimated-prototype-v0"


@dataclass(frozen=True)
class TileDecimation:
    """What one tile did. Counts and seconds only — no identifier, no coordinate."""

    tile_id: int
    vertices_in: int
    triangles_in: int
    owned_in: int
    locked: int
    vertices_out: int
    triangles_out: int
    collapses: int
    rejected_link: int
    rejected_seam: int
    rejected_turn: int
    rejected_error: int
    max_accepted_error_m: float
    read_seconds: float
    decimate_seconds: float
    write_seconds: float


@dataclass
class GenerationDecimation:
    """The station-level ledger, including the seam verdict."""

    tiles: list[TileDecimation] = field(default_factory=list)
    triangles_in: int = 0
    triangles_out: int = 0
    vertices_in: int = 0
    vertices_out: int = 0
    seam_boundary_edges_before: int = 0
    seam_boundary_edges_after: int = 0
    seam_edges_lost: int = 0
    seam_edges_gained: int = 0
    shared_vertex_position_conflicts: int = 0
    read_seconds: float = 0.0
    decimate_seconds: float = 0.0
    write_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def seams_clean(self) -> bool:
        """No crack, no T-junction, and every shared vertex identical in bits."""
        return (
            self.seam_edges_lost == 0
            and self.seam_edges_gained == 0
            and self.shared_vertex_position_conflicts == 0
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["seams_clean"] = self.seams_clean
        return out


# ---------------------------------------------------------------------------
# the locked set and the boundary
# ---------------------------------------------------------------------------


def lock_boundary(
    triangles: Any, vertex_count: int, owned_count: int, *, lock_edges: bool = True
) -> BOOL:
    """Which vertices of one tile may not move.

    `lock_edges=False` removes rule 2 and rule 3 and is **only** for the
    sensitivity test: it is the deliberately broken seam that `check_seams` must
    report red. Nothing in the pipeline may call it that way.
    """
    import numpy as np

    locked = np.zeros(int(vertex_count), bool)
    locked[int(owned_count) :] = True
    if not lock_edges:
        return locked
    edges, counts = _edge_counts(triangles)
    odd = counts != 2
    if bool(odd.any()):
        locked[np.unique(edges[odd])] = True
    return locked


def boundary_edges(triangles: Any, global_ids: I64) -> I64:
    """This patch's once-used edges, as ascending `(E,2)` pairs of global ids.

    Global rather than tile-local, because the whole point is to compare one
    tile's answer against its neighbour's, and a tile-local slot means nothing
    outside the tile that allocated it.
    """
    import numpy as np

    edges, counts = _edge_counts(triangles)
    if edges.shape[0] == 0:
        return np.empty((0, 2), np.int64)
    once: I64 = np.asarray(global_ids, np.int64)[edges[counts == 1]]
    once.sort(axis=1)
    return once


def union_boundary(per_tile: list[I64]) -> I64:
    """Edges used exactly once across the whole station, ascending.

    Only per-tile boundary edges need to be offered: an edge used twice inside
    one tile is already interior in the union, so it can never be a union
    boundary and never needs to be held. That is what keeps this bounded by the
    seam rather than by the surface.
    """
    import numpy as np

    rows = [e for e in per_tile if e.shape[0]]
    if not rows:
        return np.empty((0, 2), np.int64)
    pairs = np.concatenate(rows, axis=0)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    out: I64 = unique[counts == 1]
    return out


def check_seams(before: list[I64], after: list[I64]) -> tuple[int, int, int, int]:
    """`(before_count, after_count, lost, gained)` over the union boundary.

    A clean run has `lost == gained == 0`: the boundary is exactly what it was,
    on both sides of every seam.
    """
    bset = {(int(u), int(v)) for u, v in union_boundary(before).tolist()}
    aset = {(int(u), int(v)) for u, v in union_boundary(after).tolist()}
    return len(bset), len(aset), len(bset - aset), len(aset - bset)


def _edge_counts(triangles: Any) -> tuple[Any, Any]:
    """Every undirected edge of a patch with how many triangles use it."""
    import numpy as np

    tris = np.asarray(triangles, np.int64)
    if tris.shape[0] == 0:
        return np.empty((0, 2), np.int64), np.empty(0, np.int64)
    pairs = np.concatenate(
        (tris[:, (0, 1)], tris[:, (1, 2)], tris[:, (2, 0)]), axis=0
    )
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0, return_counts=True)


# ---------------------------------------------------------------------------
# the sweep over a generation
# ---------------------------------------------------------------------------


def decimate_generation(
    store: TileStore,
    out_dir: Path,
    settings: DecimationSettings,
    *,
    reduction: float | None = None,
    lock_edges: bool = True,
) -> GenerationDecimation:
    """Decimate every tile of a written generation, writing one `.npz` each.

    `reduction` spreads a station-level target over the tiles: each is asked for
    `ceil(its own triangles / reduction)`, so a tile that holds a tenth of the
    surface is asked to give up a tenth of the triangles rather than an equal
    share. It overrides `settings.target_triangles`, which is the per-patch form
    of the same rule and is what `decimate.py` actually sees.

    One tile is resident at a time and nothing station-sized is accumulated
    except the seam ledger, which is bounded by the boundary and not by the
    surface.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    report = GenerationDecimation()
    before: list[I64] = []
    after: list[I64] = []
    shared_ids: list[I64] = []
    shared_xyz: list[F32] = []
    started = time.perf_counter()

    for tile_id in store.tile_ids:
        t0 = time.perf_counter()
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        gids = np.asarray(join.global_vertex_index, np.int64)
        read_seconds = time.perf_counter() - t0

        locked = lock_boundary(
            payload.triangles, payload.vertex_count, payload.owned_count,
            lock_edges=lock_edges,
        )
        before.append(boundary_edges(payload.triangles, gids))

        tile_settings = settings
        if reduction is not None:
            target = -(-payload.triangle_count // int(round(reduction)))
            tile_settings = DecimationSettings(
                max_error_m=settings.max_error_m,
                target_triangles=target,
                placement=settings.placement,
                max_normal_turn_deg=settings.max_normal_turn_deg,
            )

        t0 = time.perf_counter()
        patch = decimate_patch(payload.positions, payload.triangles, locked, tile_settings)
        decimate_seconds = time.perf_counter() - t0

        out_gids = gids[patch.source_index]
        out_locked = locked[patch.source_index]
        after.append(boundary_edges(patch.triangles, out_gids))
        rows = np.flatnonzero(out_locked)
        shared_ids.append(out_gids[rows])
        shared_xyz.append(patch.positions[rows])

        t0 = time.perf_counter()
        np.savez(
            out_dir / f"{tile_id:012d}.npz",
            positions=patch.positions,
            triangles=patch.triangles,
            global_vertex_index=out_gids,
            locked=out_locked,
            moved=patch.moved,
        )
        write_seconds = time.perf_counter() - t0

        report.tiles.append(
            TileDecimation(
                tile_id=int(tile_id),
                vertices_in=payload.vertex_count,
                triangles_in=payload.triangle_count,
                owned_in=payload.owned_count,
                locked=patch.locked_vertices,
                vertices_out=patch.vertex_count,
                triangles_out=patch.triangle_count,
                collapses=patch.collapses,
                rejected_link=patch.rejected_link,
                rejected_seam=patch.rejected_seam,
                rejected_turn=patch.rejected_turn,
                rejected_error=patch.rejected_error,
                max_accepted_error_m=patch.max_accepted_error_m,
                read_seconds=read_seconds,
                decimate_seconds=decimate_seconds,
                write_seconds=write_seconds,
            )
        )
        report.triangles_in += payload.triangle_count
        report.triangles_out += patch.triangle_count
        # Each vertex is owned by exactly one tile, so counting the survivors
        # among this tile's owned slots counts the station's vertices once.
        report.vertices_in += payload.owned_count
        report.vertices_out += int(
            np.count_nonzero(patch.source_index < payload.owned_count)
        )
        report.read_seconds += read_seconds
        report.decimate_seconds += decimate_seconds
        report.write_seconds += write_seconds
        del payload, join, patch, locked, gids, out_gids, out_locked, rows

    counts = check_seams(before, after)
    report.seam_boundary_edges_before = counts[0]
    report.seam_boundary_edges_after = counts[1]
    report.seam_edges_lost = counts[2]
    report.seam_edges_gained = counts[3]
    report.shared_vertex_position_conflicts = _position_conflicts(
        shared_ids, shared_xyz
    )
    report.total_seconds = time.perf_counter() - started

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "contract": DECIMATED_CONTRACT,
                "settings": settings.describe(),
                "reduction": reduction,
                "lock_edges": lock_edges,
                "source_contract": store.manifest.get("contract"),
                "report": report.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _position_conflicts(ids: list[I64], xyz: list[F32]) -> int:
    """Global ids two tiles disagree about, bit for bit, over locked vertices.

    Only locked vertices are offered: a vertex present in two tiles is on both
    patches' boundaries and so is locked in both, and carrying the interior ones
    as well would make this the station-sized table the tiling exists to avoid.

    Compared on the float32 bit patterns rather than by equality, so a seam that
    agrees to a hair and not to a bit is a conflict — which is the difference
    ADR-006 2a's locked boundary exists to remove.
    """
    import numpy as np

    rows = [i for i in ids if i.shape[0]]
    if not rows:
        return 0
    gid = np.concatenate(rows)
    bits = np.concatenate([x for x in xyz if x.shape[0]]).view(np.uint32)
    order = np.lexsort((gid,))
    gid, bits = gid[order], bits[order]
    same_id = gid[1:] == gid[:-1]
    differs = np.any(bits[1:] != bits[:-1], axis=1)
    return int(np.count_nonzero(same_id & differs))


def read_decimated(out_dir: Path) -> tuple[F32, Any, I64]:
    """Every decimated tile joined into one station mesh, for measurement only.

    This *is* a resident mesh, and that is the point: a decimated tier small
    enough to hold is what the whole stage exists to produce, so its size is a
    reported figure rather than a hidden one. It is never on the production
    path — `CLAUDE.md` §11 forbids the full-resolution equivalent, and nothing
    here reads one.

    Vertices are keyed by global index, so a seam vertex held by two tiles
    becomes one vertex, which is what makes the joined mesh watertight when the
    seam check passes.
    """
    import numpy as np

    paths = sorted(out_dir.glob("*.npz"))
    ids: list[Any] = []
    xyz: list[Any] = []
    faces: list[Any] = []
    for path in paths:
        with np.load(path) as data:
            ids.append(data["global_vertex_index"])
            xyz.append(data["positions"])
            faces.append(data["triangles"].astype(np.int64))
    if not paths:
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.int64),
            np.empty(0, np.int64),
        )
    unique, first = np.unique(np.concatenate(ids), return_index=True)
    positions = np.concatenate(xyz)[first]
    joined = [
        np.searchsorted(unique, tile_ids)[tile_faces]
        for tile_ids, tile_faces in zip(ids, faces, strict=True)
    ]
    return positions, np.concatenate(joined), unique
