"""
Bounded geometry access for QA — WP-3.1, the QA half of DEC-009 step 3.

`REPORTS/2026-08-31-WP-3.0-report.md` §4 measured forward QA holding about
**3.06 GB of station-scaling structure** at the reference station, none of which
is the answer: a float64 copy of every vertex, an int64 copy of every triangle,
and a CSR vertex→triangle map whose three int64 arrays are each three entries
per triangle. WP-1.10 bounded the *query* transient; this module bounds what the
query is run against.

What must not move
------------------
The candidate set is fixed by `00-PRODUCT-DEFINITION.md`'s fidelity metric and
is **not** a tuning parameter: for each query point, the triangles incident to
its **global** two nearest mesh vertices. Narrowing it to a tile, a band or a
radius is a different metric and needs a version bump (WP-3.0 D3). So nothing
here is an approximation.

The neighbour half of that lives in `qa_neighbours.py`, which explains why a
partition still answers a global question. The triangle half is here, and rests
on one fact: **the reduction is a minimum.** `_point_triangle_distance` is
elementwise and `min` is associative, commutative and exact in float64, so which
block a triangle arrived in cannot move the result. That is the same argument
WP-1.10 used for splitting the query loop, applied to the other axis.

Sources, not meshes
-------------------
Both stages read a `GeometrySource`, never a `MeshData`. `ResidentMesh` is the
adapter for a mesh that is already in memory; WP-3.3 adds the tile-store
adapter, and neither QA direction has to know which it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from .types import MeshData

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

# Byte caps, not record counts, for the same reason `pass_b_merge` uses byte
# caps: the figure must not silently change when a record gains a field.
#
# B14  vertex block, float64 positions        QA_VERTEX_BLOCK_BYTES    4,000,000 B
# B15  triangle block, float64 corners        QA_TRIANGLE_BLOCK_BYTES  4,000,000 B
# B16  ragged query x candidate expansion     QA_PAIR_BLOCK             (records)
#
# `QA_PAIR_BLOCK` is a record count because the thing it bounds is a *pair* —
# one query against one candidate triangle — and WP-1.10 measured that pair's
# cost directly at about 246 B (2,955 B per query over roughly twelve candidate
# triangles). `_point_triangle_distance` allocates about twenty arrays over the
# expanded pair set, so the transient is a large multiple of the pairs
# themselves: measured on the 990,045-sample fixture, moving this from 100,000
# to 25,000 took the whole forward marginal from 102,461,440 B to 82,919,424 B,
# while 40,000 and 10,000 landed within 600 KB of 25,000. The knee is at 40,000
# and 25,000 sits below it with room, at no measurable time cost (6.88 s against
# 7.31 s).
#
# The two byte caps were measured the same way and are deliberately not larger:
# past ~4 MB each they stop showing up in the peak at all (73,453,568 B at 3 MB
# against 73,863,168 B at 4 MB, inside the run-to-run spread), and smaller blocks
# only buy more KD-tree builds.
QA_VERTEX_BLOCK_BYTES = 4_000_000
QA_TRIANGLE_BLOCK_BYTES = 4_000_000
QA_PAIR_BLOCK = 25_000

_F64_BYTES = 8
_VERTEX_RECORD_BYTES = 3 * _F64_BYTES        # one float64 position
_TRIANGLE_RECORD_BYTES = 9 * _F64_BYTES      # three float64 corners


@dataclass(frozen=True)
class VertexBlock:
    """A bounded run of mesh vertices, with their **global** ids.

    The ids are carried explicitly rather than implied by position because a
    tile store hands out vertices in tile order, not in global order, and the
    neighbour search has to report a global identity either way.
    """

    ids: I64                    # (B,)
    xyz: F64                    # (B,3) project-axis offsets, float64


@dataclass(frozen=True)
class TriangleBlock:
    """A bounded run of triangles, corner ids and corner positions together.

    Carrying the positions removes the one thing a streamed pass cannot do
    cheaply — random access into a vertex array that is no longer resident. A
    tile already stores its own corners, so this costs the tile source nothing
    and saves the resident source only a gather it was doing anyway.
    """

    ids: I64                    # (B,3) global vertex id per corner
    corners: F64                # (B,3,3) float64 corner positions


@dataclass(frozen=True)
class IdentifiedTriangleBlock:
    """A triangle block that also carries per-corner `source_sample_id`.

    Reverse QA identifies a triangle by its canonical oriented sample-id triple
    (`PHASE1-DETERMINISM-SPEC.md` §4), never by an array position, so it needs
    the ids alongside the geometry. They are a separate block type rather than
    an optional field on `TriangleBlock` because forward QA must not be able to
    reach them by accident: per-vertex source identity is DEC-004's redaction
    boundary, and a field that is usually there is a field that leaks.
    """

    sample_ids: I64             # (B,3) source_sample_id per corner
    ids: I64                    # (B,3) global vertex id per corner
    corners: F64                # (B,3,3) float64 corner positions


class GeometrySource(Protocol):
    """Bounded read access to a finished surface, whatever holds it."""

    @property
    def vertex_count(self) -> int: ...

    @property
    def triangle_count(self) -> int: ...

    def vertex_blocks(self) -> Iterator[VertexBlock]: ...

    def triangle_blocks(self) -> Iterator[TriangleBlock]: ...

    def identified_triangle_blocks(self) -> Iterator[IdentifiedTriangleBlock]: ...


@dataclass(frozen=True)
class ResidentMesh:
    """`GeometrySource` over a `MeshData` that is already in memory.

    The widening to float64 happens **per block**, never for the station:
    `vertices[block].astype(float64)` and `vertices.astype(float64)[block]` hold
    the same bits, because every float32 is exactly representable in float64, so
    this is a memory change and not a numerical one.
    """

    mesh: MeshData
    vertex_block_bytes: int = QA_VERTEX_BLOCK_BYTES
    triangle_block_bytes: int = QA_TRIANGLE_BLOCK_BYTES

    @property
    def vertex_count(self) -> int:
        return self.mesh.vertex_count

    @property
    def triangle_count(self) -> int:
        return self.mesh.triangle_count

    def vertex_blocks(self) -> Iterator[VertexBlock]:
        import numpy as np

        step = _records_per_block(self.vertex_block_bytes, _VERTEX_RECORD_BYTES)
        for lo in range(0, self.vertex_count, step):
            hi = min(lo + step, self.vertex_count)
            yield VertexBlock(
                ids=np.arange(lo, hi, dtype=np.int64),
                xyz=self.mesh.vertices[lo:hi].astype(np.float64),
            )

    def triangle_blocks(self) -> Iterator[TriangleBlock]:
        import numpy as np

        step = _records_per_block(self.triangle_block_bytes, _TRIANGLE_RECORD_BYTES)
        for lo in range(0, self.triangle_count, step):
            hi = min(lo + step, self.triangle_count)
            ids = self.mesh.triangles[lo:hi].astype(np.int64)
            yield TriangleBlock(
                ids=ids, corners=self.mesh.vertices[ids].astype(np.float64)
            )

    def identified_triangle_blocks(self) -> Iterator[IdentifiedTriangleBlock]:
        import numpy as np

        sample = self.mesh.source_sample_id
        if sample is None:
            raise ValueError(
                "reverse QA needs source_sample_id; this mesh carries none"
            )
        ids64 = np.asarray(sample, np.int64)
        for block in self.triangle_blocks():
            yield IdentifiedTriangleBlock(
                sample_ids=ids64[block.ids], ids=block.ids, corners=block.corners
            )


def _records_per_block(byte_cap: int, record_bytes: int) -> int:
    """At least one record, so a cap smaller than a record still terminates."""
    return max(int(byte_cap // record_bytes), 1)


def vertex_block_step() -> int:
    """Owned vertices per QA block, from `QA_VERTEX_BLOCK_BYTES`.

    Public because `TileStore` slices to the same cap (WP-13b), and a second
    copy of this arithmetic living in `tiles.py` is exactly how the two
    `GeometrySource` implementations drifted apart in the first place.
    `ResidentMesh` keeps its per-instance override, which is what the QA tests
    vary; this is the default the two sources share.
    """
    return _records_per_block(QA_VERTEX_BLOCK_BYTES, _VERTEX_RECORD_BYTES)


def triangle_block_step() -> int:
    """Triangles per QA block, from `QA_TRIANGLE_BLOCK_BYTES`. See above."""
    return _records_per_block(QA_TRIANGLE_BLOCK_BYTES, _TRIANGLE_RECORD_BYTES)


# ---------------------------------------------------------------------------
# stage two — distance to the triangles incident to those vertices
# ---------------------------------------------------------------------------


def incident_distances(
    source: GeometrySource,
    queries: F64,
    neighbours: I64,
    fallback: F64,
    *,
    pair_block: int = QA_PAIR_BLOCK,
) -> F64:
    """Nearest-surface distance over each query's own candidate triangles.

    `neighbours` is `(Q, k)` global vertex ids from `nearest_vertices` and
    `fallback` is their distance column 0, used for a query whose neighbours
    carry no triangle at all — an isolated vertex that survived compaction. The
    resident implementation re-queries the tree for that case; the distance to
    the nearest vertex is exactly what that query returns, and it is already in
    hand, so it is reused rather than recomputed.
    """
    import numpy as np

    n = int(queries.shape[0])
    best = np.full(n, np.inf, np.float64)
    if n == 0 or source.triangle_count == 0:
        return best

    wanted, first, span, owner = _wanted_vertex_index(neighbours)
    if wanted.size:
        for chunk in source.triangle_blocks():
            _accumulate_block(
                best, queries, chunk, wanted, first, span, owner, pair_block
            )

    missing = ~np.isfinite(best)
    if bool(np.any(missing)):
        best[missing] = fallback[missing]
    return best


def _wanted_vertex_index(neighbours: I64) -> tuple[I64, I64, I64, I64]:
    """CSR from a wanted vertex id to the queries that want it.

    Bounded by the query count, never by the mesh: there are exactly `Q x k`
    entries whatever the station's size. This is the structure that replaces the
    whole-station vertex→triangle map, and the inversion is what makes that
    possible — the map was indexed by the mesh, this is indexed by the queries.
    """
    import numpy as np

    k = int(neighbours.shape[1])
    flat = np.asarray(neighbours, np.int64).ravel()
    if flat.size == 0 or not bool((flat >= 0).any()):
        empty = np.empty(0, np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    # The negative slots are the `distance_upper_bound` misses and an empty
    # block's padding, and on a normal station there are none. Skipping the
    # filter when there are none avoids two more arrays the size of the query
    # set; the intermediates are released as they stop being needed, because
    # this runs once and its transient is charged to the same peak as the
    # triangle loop that follows it.
    if bool((flat >= 0).all()):
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]
    else:
        keep = np.flatnonzero(flat >= 0)
        ids = flat[keep]
        inner = np.argsort(ids, kind="stable")
        order = keep[inner]
        sorted_ids = ids[inner]
        del keep, ids, inner
    owner = (order // k).astype(np.int64)
    del order
    wanted, first, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    del sorted_ids
    return wanted, first.astype(np.int64), counts.astype(np.int64), owner


def _accumulate_block(
    best: F64,
    queries: F64,
    chunk: TriangleBlock,
    wanted: I64,
    first: I64,
    span: I64,
    owner: I64,
    pair_block: int,
) -> None:
    """One triangle block against every query that wants one of its corners."""
    import numpy as np

    from .qa import _point_triangle_distance

    flat = chunk.ids.ravel()
    slot = np.searchsorted(wanted, flat)
    np.clip(slot, 0, wanted.size - 1, out=slot)
    hit = np.nonzero(wanted[slot] == flat)[0]
    if hit.size == 0:
        return

    slot = slot[hit]
    triangle = (hit // 3).astype(np.int64)
    counts = span[slot]
    # Split the ragged expansion so the transient is a cap and not a product of
    # the block size and the local incidence. `searchsorted` over the running
    # total finds the largest prefix that fits.
    edges = _pair_edges(np.cumsum(counts), pair_block)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        count = counts[lo:hi]
        total = int(count.sum())
        if total == 0:
            continue
        rows = np.repeat(triangle[lo:hi], count)
        intra = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(count) - count, count
        )
        who = owner[np.repeat(first[slot[lo:hi]], count) + intra]
        corners = chunk.corners[rows]
        d = _point_triangle_distance(
            queries[who], corners[:, 0], corners[:, 1], corners[:, 2]
        )
        np.minimum.at(best, who, d)


def _pair_edges(running: I64, pair_block: int) -> list[int]:
    """Cut points over `running` totals so no slice expands past `pair_block`.

    A single hit whose own count exceeds the cap still forms one slice: refusing
    to make progress would be worse than briefly exceeding a transient bound,
    and the case needs a vertex with more incident triangles than the cap.
    """
    import numpy as np

    edges = [0]
    size = int(running.shape[0])
    while edges[-1] < size:
        base = 0 if edges[-1] == 0 else int(running[edges[-1] - 1])
        nxt = int(np.searchsorted(running, base + pair_block, side="right"))
        edges.append(max(nxt, edges[-1] + 1))
    return edges


# ---------------------------------------------------------------------------
# the two stages, composed
# ---------------------------------------------------------------------------


def surface_distances(
    source: GeometrySource,
    queries: F64,
    *,
    k: int,
    workers: int,
    block: int,
    pair_block: int = QA_PAIR_BLOCK,
) -> F64:
    """Exact point-to-surface distance for every query, bounded throughout.

    Nothing resident here is sized by the station: the running neighbour table
    and its inversion are sized by the query count, and the vertex and triangle
    blocks by their own byte caps.
    """
    import numpy as np

    from .qa_neighbours import nearest_vertices

    if source.triangle_count == 0 or queries.shape[0] == 0:
        return np.empty(0, np.float64)
    found, ids = nearest_vertices(
        source, queries, k=k, workers=workers, block=block
    )
    return incident_distances(
        source, queries, ids, found[:, 0], pair_block=pair_block
    )


def as_source(mesh: MeshData | GeometrySource) -> GeometrySource:
    """Accept either a `MeshData` or something that is already a source."""
    from .types import MeshData as _MeshData

    if isinstance(mesh, _MeshData):
        return ResidentMesh(mesh)
    return mesh
