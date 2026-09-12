"""
Quadric error metric decimation over one indexed triangle patch — WP-3.4.

This is the first stage in RapidMesh that changes the geometry **on purpose**.
Everything before it preserved the surface exactly, so the question was "did
anything move?"; from here it is "how much did it move, and can we bound it?" —
which is why every knob that can move a figure is a recorded setting and why the
result carries the accepted error bound rather than only a triangle count.

Garland & Heckbert, *Surface Simplification Using Quadric Error Metrics*
(SIGGRAPH '97). Each vertex carries the sum of the area-weighted squared
distances to the planes of the triangles merged into it, as a symmetric 4x4 held
by its ten unique entries. Collapsing an edge adds the two quadrics and places
the survivor where that sum is least.

What this module is, and is not
-------------------------------
It is **one patch**: positions, triangles indexing them, and a boolean saying
which vertices may not move. It knows nothing about tiles, files, stations or
poses — `decimate_tiles.py` owns all of that, and the split is what makes the
seam rule testable on a six-triangle fixture instead of on a station. It is also
**not** the production decimator: Round 10 is a prototype and a measurement
(DEC-013 as amended 8 September 2026), so normals, colour and the client
container are deliberately absent and `DecimatedPatch` carries geometry and
provenance only.

The locked set is the seam guarantee, and it is the caller's decision
--------------------------------------------------------------------
A locked vertex never moves and is never removed; an edge with two locked
endpoints is never collapsed. Those two rules are the whole of the crack and
T-junction argument. A patch's boundary edges are the edges used by exactly one
of its triangles; `decimate_tiles.lock_boundary` locks both endpoints of every
one, so no boundary edge can be collapsed, and the link condition below stops
one being created. The boundary edge set therefore leaves decimation identical
to the set that went in, on both sides of every seam — which
`decimate_tiles.boundary_edges` measures rather than assumes.

Two rejections that are not optional
------------------------------------
**The link condition.** Collapsing edge (u,v) preserves a 2-manifold only if the
vertices adjacent to both u and v are exactly the vertices opposite the edge.
Without it a collapse can weld two sheets together into a non-manifold edge that
no downstream stage is prepared for. It is checked over *this patch*, which is
why `_would_join_locked` exists beside it: a locked vertex's star may continue
into a neighbouring tile, so an edge that is new here is not necessarily new
anywhere, and two tiles closing the same fan would each emit the same face.

**The normal turn.** The quadric is a distance metric and says nothing about
orientation, so a collapse can fold a triangle through its own plane at very low
cost. `max_normal_turn_deg` rejects a collapse that turns any surviving incident
triangle further than that, which at the 90 degree default is exactly "no
triangle may flip".

Storage, and determinism
------------------------
The vectorised half — quadrics, plane fits, the initial edge set — is numpy over
the whole patch at once; the collapse loop is scalar Python, because a numpy row
access costs more than the three-float arithmetic it carries. Positions and
quadrics therefore sit in `array.array('d')` and the queue is compacted when it
fills with superseded entries, so that one patch of half a million triangles is
bounded by its own size rather than by how many collapses it took.

`PHASE1-DETERMINISM-SPEC.md` §3: queue entries are ``(cost, u, v, versions)``
and are distinct by construction, so the pop order is a total order and does not
depend on heap internals or on compaction; every rejection is re-tested rather
than remembered; and the output is compacted in ascending input-index order. Two
runs at the same settings produce the same bytes, and `tests/test_decimate.py`
asserts it rather than this docstring claiming it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .decimate_quadrics import (
    doubles,
    quadric_error,
    solve_optimal,
    vertex_quadrics,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    U32 = npt.NDArray[np.uint32]
    BOOL = npt.NDArray[np.bool_]

#: Reject a collapse that turns a surviving incident triangle by more than this.
#: At 90 degrees it is exactly a flip test, the classical guard; a smaller angle
#: also refuses folds that stop short of inverting. A recorded setting.
DEFAULT_MAX_NORMAL_TURN_DEG = 90.0

#: The queue is compacted when it exceeds this multiple of the live edge count
#: (estimated as 1.5 per live triangle). Superseded entries are dropped and the
#: rest re-heapified; entries are distinct, so this cannot change the pop order.
_HEAP_SLACK = 3.0


@dataclass(frozen=True)
class DecimationSettings:
    """Everything that can move the output, in one recordable object.

    `max_error_m` caps ``sqrt(Q(v*))``: the root of the accumulated
    area-weighted squared distance from the placed vertex to every plane merged
    into it, which bounds the distance from that vertex to each of those planes
    individually. So it is a *guaranteed* per-collapse bound and is reported as
    one — and it is **not** a bound on surface-to-surface deviation, which is
    sampled and measured separately (`decimate_qa.py`). The two are never
    reported as though they were one quantity.

    `target_triangles` stops the sweep at a triangle count. Either rule alone is
    a legitimate operating point; given both, whichever bites first stops it.
    """

    max_error_m: float = math.inf
    target_triangles: int | None = None
    placement: str = "optimal"
    max_normal_turn_deg: float = DEFAULT_MAX_NORMAL_TURN_DEG

    def __post_init__(self) -> None:
        if self.placement not in ("optimal", "endpoint"):
            raise ValueError(f"unknown placement {self.placement!r}")
        if not (self.max_error_m > 0.0):
            raise ValueError(f"max_error_m must be positive, got {self.max_error_m}")
        if self.target_triangles is not None and self.target_triangles < 0:
            raise ValueError("target_triangles must not be negative")
        if not (0.0 < self.max_normal_turn_deg <= 180.0):
            raise ValueError("max_normal_turn_deg must be in (0, 180]")

    def describe(self) -> dict[str, Any]:
        return {
            "max_error_m": (
                None if math.isinf(self.max_error_m) else float(self.max_error_m)
            ),
            "target_triangles": self.target_triangles,
            "placement": self.placement,
            "max_normal_turn_deg": float(self.max_normal_turn_deg),
        }


@dataclass(frozen=True)
class DecimatedPatch:
    """One decimated patch, plus the ledger of what the sweep did.

    `source_index` maps each output vertex back to the input slot it survived
    from, so a caller can carry a global vertex id across decimation without
    this module knowing what a global vertex id is. `moved` says whether that
    survivor's position changed — no locked vertex is ever in it, and a reader
    can check that rather than trust it.
    """

    positions: F32              # (V', 3)
    triangles: U32              # (T', 3)
    source_index: I64           # (V',) input slot of each surviving vertex
    moved: BOOL                 # (V',) whether its position changed
    collapses: int
    rejected_link: int
    rejected_seam: int
    rejected_turn: int
    rejected_error: int
    max_accepted_error_m: float
    locked_vertices: int

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


def _use_kernel(choice: str) -> bool:
    """Resolve `decimate_patch`'s `kernel` argument to a yes or a no.

    `"rust"` is a demand and fails loudly when the extension is not built;
    `"auto"` is a preference and falls back in silence, because a tree with no
    crate compiled must behave exactly as it did before there was one.
    """
    from rapidmesh import decimate_kernel

    if choice == "auto":
        choice = decimate_kernel.default_choice()
    if choice == "python":
        return False
    if choice == "rust":
        if not decimate_kernel.available():
            raise RuntimeError(
                "kernel='rust' requested but unavailable: "
                f"{decimate_kernel.unavailable_reason()}"
            )
        return True
    if choice == "auto":
        return decimate_kernel.available()
    raise ValueError(f"unknown kernel {choice!r}")


def decimate_patch(
    positions: F32 | F64,
    triangles: Any,
    locked: BOOL,
    settings: DecimationSettings | None = None,
    *,
    kernel: str = "auto",
) -> DecimatedPatch:
    """Collapse edges of one patch, cheapest first, until a stop rule bites.

    `positions` are offsets from the caller's own origin, in metres, returned at
    float32 — a locked vertex's bits round-trip exactly, which is what lets two
    tiles agree on a seam rather than nearly agree (`SPATIAL-CONTRACT.md` §2.4).

    `kernel` picks the sweep. `"python"` is the implementation in this module,
    which is the **equivalence reference** and is never removed; `"rust"` is
    `decimate_kernel`'s native one and raises if the crate is not built;
    `"auto"` consults `RAPIDMESH_DECIMATE_KERNEL`, which defaults to `"python"`.

    **The native sweep is opt-in**, so a tree that merely has the crate built
    runs exactly the code it ran before it did — see
    `decimate_kernel.DEFAULT_CHOICE` for why that matters and what broke when it
    did not. The two sweeps are required to produce byte-identical output and
    `tests/test_decimate_kernel.py` holds them to it.
    """
    import numpy as np

    settings = DecimationSettings() if settings is None else settings
    pos64 = np.asarray(positions, np.float64)
    tris = np.asarray(triangles, np.int64)
    lock = np.asarray(locked, bool)
    if pos64.ndim != 2 or pos64.shape[1] != 3:
        raise ValueError("positions must be (V,3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must be (T,3)")
    if lock.shape != (pos64.shape[0],):
        raise ValueError("locked must be one flag per vertex")
    if tris.size and (int(tris.max()) >= pos64.shape[0] or int(tris.min()) < 0):
        raise ValueError("a triangle names a vertex outside the patch")

    if _use_kernel(kernel):
        from rapidmesh import decimate_kernel

        return decimate_kernel.sweep(pos64, tris, lock, settings).finish()
    state = _PatchState(pos64, tris, lock, settings)
    state.run()
    return state.finish()


# ---------------------------------------------------------------------------


class _PatchState:
    """The mutable half of one sweep, kept off the module's public surface."""

    def __init__(
        self, pos64: F64, tris: I64, lock: BOOL, settings: DecimationSettings
    ) -> None:
        self.settings = settings
        self.count = count = int(pos64.shape[0])
        faces = int(tris.shape[0])

        self.px = doubles(pos64[:, 0])
        self.py = doubles(pos64[:, 1])
        self.pz = doubles(pos64[:, 2])
        self.quad = vertex_quadrics(pos64, tris, count)
        self.locked = lock.tolist()
        self.moved = [False] * count
        self.alive = [True] * count
        self.version = [0] * count

        # One int object per index, shared by `tv`, `vtris` and the queue, so
        # the same integer in four places is four references and not four
        # boxed objects. It matters at half a million triangles per tile.
        self.iv = list(range(count))
        self.it = list(range(faces))
        self.tv: list[int] = [self.iv[i] for i in tris.ravel().tolist()]
        self.tri_alive = bytearray(b"\x01") * faces
        self.live_triangles = faces

        self.vtris: list[set[int]] = [set() for _ in range(count)]
        for t in self.it:
            base = 3 * t
            self.vtris[self.tv[base]].add(t)
            self.vtris[self.tv[base + 1]].add(t)
            self.vtris[self.tv[base + 2]].add(t)

        self.cos_limit = math.cos(math.radians(settings.max_normal_turn_deg))
        self.error_limit = (
            math.inf
            if math.isinf(settings.max_error_m)
            else settings.max_error_m * settings.max_error_m
        )
        self.heap: list[tuple[float, int, int, int]] = []
        self.heap_limit = self._heap_limit()
        self.collapses = 0
        self.rejected_link = 0
        self.rejected_seam = 0
        self.rejected_turn = 0
        self.rejected_error = 0
        self.max_accepted = 0.0

        edges = _unique_edges(tris)
        for u, v in zip(edges[:, 0].tolist(), edges[:, 1].tolist(), strict=True):
            self._push(u, v)
        del edges

    # -- the sweep ----------------------------------------------------------

    def run(self) -> None:
        """Pop the cheapest live candidate until the queue or a stop rule ends it."""
        target = self.settings.target_triangles
        heap = self.heap
        while heap:
            if target is not None and self.live_triangles <= target:
                return
            _, u, v, versions = heapq.heappop(heap)
            if not (self.alive[u] and self.alive[v]):
                continue
            if versions != self._versions(u, v):
                continue               # superseded; the live entry is still queued
            self._try_collapse(u, v)

    def _try_collapse(self, u: int, v: int) -> None:
        shared = self.vtris[u] & self.vtris[v]
        if not 1 <= len(shared) <= 2:
            self.rejected_link += 1
            return
        nu, nv = self._neighbours(u), self._neighbours(v)
        if len(nu & nv) != len(shared):
            self.rejected_link += 1
            return
        keep, drop = (u, v) if not self.locked[v] else (v, u)
        if self.locked[drop]:
            self.rejected_link += 1    # both ends locked: the edge is a seam
            return
        if self.locked[keep] and self._would_join_locked(keep, nu if keep == u else nv,
                                                         nv if keep == u else nu):
            self.rejected_seam += 1
            return

        q = self._quadric_sum(u, v)
        x, y, z = self._place(q, u, v)
        if not self._turn_ok(keep, drop, shared, x, y, z):
            self.rejected_turn += 1
            return
        cost = quadric_error(q, x, y, z)
        self._commit(keep, drop, shared, x, y, z)
        self.collapses += 1
        if cost > self.max_accepted:
            self.max_accepted = cost

    def _commit(
        self, keep: int, drop: int, shared: set[int], x: float, y: float, z: float
    ) -> None:
        tv = self.tv
        for t in shared:
            base = 3 * t
            for slot in (base, base + 1, base + 2):
                self.vtris[tv[slot]].discard(t)
            self.tri_alive[t] = 0
        self.live_triangles -= len(shared)

        for t in self.vtris[drop]:
            base = 3 * t
            for slot in (base, base + 1, base + 2):
                if tv[slot] == drop:
                    tv[slot] = self.iv[keep]
            self.vtris[keep].add(t)
        self.vtris[drop] = set()
        self.alive[drop] = False

        self.px[keep], self.py[keep], self.pz[keep] = x, y, z
        if not self.locked[keep]:
            self.moved[keep] = True
        base_k, base_d = 10 * keep, 10 * drop
        for i in range(10):
            self.quad[base_k + i] += self.quad[base_d + i]
        self.version[keep] += 1

        for w in self._neighbours(keep):
            self._push(keep, w)
        if len(self.heap) > self.heap_limit:
            self._compact()

    # -- geometry -----------------------------------------------------------

    def _would_join_locked(
        self, keep: int, keep_nbrs: set[int], drop_nbrs: set[int]
    ) -> bool:
        """Would this collapse create an edge between two locked vertices?

        A locked vertex's star may run on past this patch — that is what being
        on the boundary means — so the patch cannot see whether an edge between
        two locked vertices already exists somewhere else. Two tiles that each
        legally close a fan onto the same pair of ring vertices then emit the
        same face twice, which is a doubled surface rather than a crack and so
        slips past a boundary-set check.

        The rule is therefore local and exact: an edge that would be **new
        here** and joins two locked vertices is refused, because "new here" is
        only the same as "new anywhere" when at least one end's star is
        complete, and an unlocked vertex is precisely one whose star is.
        """
        locked = self.locked
        return any(
            w != keep and locked[w] and w not in keep_nbrs for w in drop_nbrs
        )

    def _neighbours(self, u: int) -> set[int]:
        out: set[int] = set()
        tv = self.tv
        for t in self.vtris[u]:
            base = 3 * t
            out.add(tv[base])
            out.add(tv[base + 1])
            out.add(tv[base + 2])
        out.discard(u)
        return out

    def _quadric_sum(self, u: int, v: int) -> list[float]:
        q, a, b = self.quad, 10 * u, 10 * v
        return [
            q[a] + q[b], q[a + 1] + q[b + 1], q[a + 2] + q[b + 2],
            q[a + 3] + q[b + 3], q[a + 4] + q[b + 4], q[a + 5] + q[b + 5],
            q[a + 6] + q[b + 6], q[a + 7] + q[b + 7], q[a + 8] + q[b + 8],
            q[a + 9] + q[b + 9],
        ]

    def _place(
        self, q: list[float], u: int, v: int
    ) -> tuple[float, float, float]:
        """Where the survivor goes: the locked end, the optimum, or the ladder.

        A locked endpoint decides the answer outright, which is why a seam
        vertex comes out carrying the bits it went in with. Otherwise
        `placement="optimal"` takes the quadric's minimiser where the solve is
        well conditioned and the cheapest of the two endpoints and the midpoint
        where it is not, and `placement="endpoint"` takes the cheaper endpoint
        and nothing else.
        """
        px, py, pz = self.px, self.py, self.pz
        if self.locked[u]:
            return px[u], py[u], pz[u]
        if self.locked[v]:
            return px[v], py[v], pz[v]
        ends = ((px[u], py[u], pz[u]), (px[v], py[v], pz[v]))
        if self.settings.placement == "endpoint":
            # A true subset placement: every surviving vertex is still one of
            # the scan-derived positions the patch arrived with, so the tier
            # invents no geometry. It costs error against `optimal` and is
            # offered because that trade is a product decision, not this
            # module's.
            return min(ends, key=lambda c: quadric_error(q, *c))
        best = solve_optimal(q)
        if best is not None:
            return best
        midpoint = (
            0.5 * (px[u] + px[v]), 0.5 * (py[u] + py[v]), 0.5 * (pz[u] + pz[v])
        )
        return min((*ends, midpoint), key=lambda c: quadric_error(q, *c))

    def _turn_ok(
        self, keep: int, drop: int, shared: set[int], x: float, y: float, z: float
    ) -> bool:
        """No surviving incident triangle may turn past the limit or degenerate."""
        px, py, pz, tv = self.px, self.py, self.pz, self.tv
        moving = (keep, drop)
        for source in moving:
            for t in self.vtris[source]:
                if t in shared:
                    continue
                base = 3 * t
                i, j, m = tv[base], tv[base + 1], tv[base + 2]
                ax, ay, az = px[i], py[i], pz[i]
                bx, by, bz = px[j], py[j], pz[j]
                cx, cy, cz = px[m], py[m], pz[m]
                nb = _normal(ax, ay, az, bx, by, bz, cx, cy, cz)
                if nb is None:
                    continue           # already degenerate; nothing to turn
                # A triangle holding *both* ends is in `shared` and was skipped,
                # so exactly one corner moves and the chain is exhaustive.
                if i in moving:
                    ax, ay, az = x, y, z
                elif j in moving:
                    bx, by, bz = x, y, z
                else:
                    cx, cy, cz = x, y, z
                na = _normal(ax, ay, az, bx, by, bz, cx, cy, cz)
                if na is None:
                    return False       # collapsed to a line or a point
                if nb[0] * na[0] + nb[1] * na[1] + nb[2] * na[2] < self.cos_limit:
                    return False
        return True

    # -- queue --------------------------------------------------------------

    def _versions(self, u: int, v: int) -> int:
        """Both endpoints' versions as one integer, so an entry is one tuple.

        A collapse bumps only the survivor's version, and every edge whose cost
        it changed is incident to that vertex, so this invalidates exactly the
        stale entries. The shift is 32 bits and not the vertex count: a version
        counts collapses, and packing it against a bound it can exceed would
        alias two different states onto one integer.
        """
        return (self.version[u] << 32) | self.version[v]

    def _push(self, u: int, v: int) -> None:
        if not (self.alive[u] and self.alive[v]):
            return
        if self.locked[u] and self.locked[v]:
            return
        a, b = (u, v) if u < v else (v, u)
        q = self._quadric_sum(a, b)
        cost = quadric_error(q, *self._place(q, a, b))
        if cost < 0.0:
            cost = 0.0                 # rounding under a flat quadric
        if cost > self.error_limit:
            self.rejected_error += 1
            return
        heapq.heappush(
            self.heap, (cost, self.iv[a], self.iv[b], self._versions(a, b))
        )

    def _heap_limit(self) -> int:
        return max(1 << 16, int(_HEAP_SLACK * 1.5 * self.live_triangles))

    def _compact(self) -> None:
        """Drop superseded entries. Entries are distinct, so the order is kept."""
        alive, version = self.alive, self.version
        self.heap = [
            e
            for e in self.heap
            if alive[e[1]]
            and alive[e[2]]
            and e[3] == (version[e[1]] << 32) | version[e[2]]
        ]
        heapq.heapify(self.heap)
        self.heap_limit = max(self._heap_limit(), 2 * len(self.heap))

    # -- output -------------------------------------------------------------

    def finish(self) -> DecimatedPatch:
        """Compact survivors in ascending input order, then re-index triangles."""
        import numpy as np

        keep = np.flatnonzero(np.asarray(self.alive, bool))
        remap = np.full(self.count, -1, np.int64)
        remap[keep] = np.arange(keep.size, dtype=np.int64)
        faces = np.asarray(self.tv, np.int64).reshape(-1, 3)
        live = np.flatnonzero(np.frombuffer(bytes(self.tri_alive), np.uint8))
        faces = remap[faces[live]]
        if faces.size and int(faces.min()) < 0:
            raise ValueError("a surviving triangle names a removed vertex")
        xyz = np.empty((keep.size, 3), np.float64)
        xyz[:, 0] = np.frombuffer(self.px, np.float64)[keep]
        xyz[:, 1] = np.frombuffer(self.py, np.float64)[keep]
        xyz[:, 2] = np.frombuffer(self.pz, np.float64)[keep]
        return DecimatedPatch(
            positions=xyz.astype(np.float32),
            triangles=faces.astype(np.uint32),
            source_index=keep.astype(np.int64),
            moved=np.asarray(self.moved, bool)[keep],
            collapses=self.collapses,
            rejected_link=self.rejected_link,
            rejected_seam=self.rejected_seam,
            rejected_turn=self.rejected_turn,
            rejected_error=self.rejected_error,
            max_accepted_error_m=math.sqrt(self.max_accepted),
            locked_vertices=int(np.count_nonzero(np.asarray(self.locked, bool))),
        )


def _normal(
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
    cx: float, cy: float, cz: float,
) -> tuple[float, float, float] | None:
    """Unit triangle normal, or `None` when the triangle has no area."""
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 0.0:
        return None
    return nx / length, ny / length, nz / length


def _unique_edges(tris: I64) -> Any:
    """Every undirected edge once, as an ascending ``(E,2)`` array of pairs."""
    import numpy as np

    if tris.shape[0] == 0:
        return np.empty((0, 2), np.int64)
    pairs = np.concatenate(
        (tris[:, (0, 1)], tris[:, (1, 2)], tris[:, (2, 0)]), axis=0
    )
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)
