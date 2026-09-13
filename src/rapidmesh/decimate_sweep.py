"""The collapse sweep itself — the mutable half of `decimate.decimate_patch`.

Split out of `decimate.py` in WP-3.5 (DEC-010 Rule 6; Round 10 raised it as D5
and Round 11's dispatch grew the file further). The boundary is the one the
module already had rather than a line drawn to hit a number: `decimate.py` is
the **contract** — what a caller may ask for, what it gets back, and which of
the two sweeps runs — and this file is the **mechanics** of one sweep: the
queue, the two rejections, the placement, the commit and the compaction.
`decimate_bounds.py` is the third piece, the quantity the stop rule consults,
and it is separate because it is the part with a proof attached.

Nothing here is new work except the error-bounded stop rule. Every other line is
Round 10's, moved: `sweep.rs` is a transcription of this class and the two are
required to agree to the last bit, so a restructuring while moving would have
cost that claim for nothing.

The two stop rules, and why both are kept
-----------------------------------------
`target_triangles` stops at a count. `max_plane_deviation_m` stops at a
**guaranteed** distance to every plane merged into the placed vertex, which is
what DEC-013's routing means by error-bounded and what `decimate_bounds.py`
proves and, as carefully, delimits. `max_error_m` is a third and is a
quadric-cost threshold, not a distance — see `decimate_bounds` for the
measurement that shows it is not one. It is kept because every recorded Round 10
and Round 11 figure was produced with it at infinity, and a knob removed is a run
that cannot be reproduced.

With `max_plane_deviation_m` at infinity this class does exactly what it did at
Round 11: the bound accumulator is written, and read by nothing that can change a
collapse decision, so the output is byte-identical and Round 10's tiers still
digest to what the record says they do. That is checked on the real stations
rather than argued here.
"""

from __future__ import annotations

import heapq
import math
from array import array
from typing import TYPE_CHECKING, Any

from .decimate import DecimatedPatch, DecimationSettings
from .decimate_bounds import emitted_bound, face_min_areas, plane_bound
from .decimate_quadrics import doubles, quadric_error, solve_optimal, vertex_quadrics

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

#: The queue is compacted when it exceeds this multiple of the live edge count
#: (estimated as 1.5 per live triangle). Superseded entries are dropped and the
#: rest re-heapified; entries are distinct, so this cannot change the pop order.
_HEAP_SLACK = 3.0


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
        # The guaranteed plane bound, and the smallest merged face area it is
        # taken against. An untouched vertex still lies on every plane merged
        # into it, so its bound starts — and, if it is never collapsed into,
        # stays — at exactly zero. `decimate_bounds` carries the derivation.
        self.bound = array("d", bytes(8 * count))
        self.amin = face_min_areas(pos64, tris, count)
        # The absorption forest. A vertex is its own parent until it is dropped
        # into another, and `finish()` resolves the chains — so this costs one
        # store per collapse and nothing per candidate.
        self.parent = array("i", range(count))

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
        self.deviation_limit = settings.max_plane_deviation_m
        self.heap: list[tuple[float, int, int, int]] = []
        self.heap_limit = self._heap_limit()
        self.collapses = 0
        self.rejected_link = 0
        self.rejected_seam = 0
        self.rejected_turn = 0
        self.rejected_error = 0
        self.rejected_deviation = 0
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
        # Recomputed rather than carried on the queue entry. The entry's
        # `versions` field is exactly the guarantee that neither end has moved,
        # changed quadric or changed `amin` since it was pushed, so this
        # reproduces the value `_push` tested against the budget — and
        # `tests/test_decimate_bounds.py` asserts the reproduction rather than
        # this comment claiming it.
        bound = plane_bound(q, self._amin_pair(u, v), x, y, z)
        self._commit(keep, drop, shared, x, y, z, bound)
        self.collapses += 1
        if cost > self.max_accepted:
            self.max_accepted = cost

    def _commit(
        self, keep: int, drop: int, shared: set[int],
        x: float, y: float, z: float, bound: float,
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
        self.parent[drop] = keep

        self.px[keep], self.py[keep], self.pz[keep] = x, y, z
        # Both written before the re-pushes below, so that every candidate
        # queued against the survivor is judged against the plane set it now
        # carries and not the one it carried a moment ago.
        self.bound[keep] = bound
        self.amin[keep] = self._amin_pair(keep, drop)
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

    def _amin_pair(self, u: int, v: int) -> float:
        """The merged plane set's smallest face area: `min` unions two sets."""
        a, b = self.amin[u], self.amin[v]
        return a if a < b else b

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

        **The placement does not consult the displacement budget**, deliberately.
        Choosing the point that minimises `r*` instead of the quadric would be a
        different decimator, would change every figure on the record, and would
        cost the bit-identity `sweep.rs` holds. The budget filters collapses; it
        does not move vertices.
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

        It is also what makes the displacement budget safe to test at push
        time: a live entry's endpoints have not moved, changed quadric or
        changed radius since it was pushed, so the value tested then is the
        value that would be tested now.
        """
        return (self.version[u] << 32) | self.version[v]

    def _push(self, u: int, v: int) -> None:
        if not (self.alive[u] and self.alive[v]):
            return
        if self.locked[u] and self.locked[v]:
            return
        a, b = (u, v) if u < v else (v, u)
        q = self._quadric_sum(a, b)
        place = self._place(q, a, b)
        cost = quadric_error(q, *place)
        if cost < 0.0:
            cost = 0.0                 # rounding under a flat quadric
        if cost > self.error_limit:
            self.rejected_error += 1
            return
        # The error-bounded stop rule (Round 10 D6). A candidate that would
        # carry some original vertex further than the budget never enters the
        # queue, so no accepted collapse can exceed it and the per-patch bound
        # `finish()` reports is the budget's own guarantee rather than a
        # statistic gathered afterwards.
        if plane_bound(q, self._amin_pair(a, b), *place) > self.deviation_limit:
            self.rejected_deviation += 1
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
        representative = remap[_resolve(np.frombuffer(self.parent, np.int32))]
        faces = np.asarray(self.tv, np.int64).reshape(-1, 3)
        live = np.flatnonzero(np.frombuffer(bytes(self.tri_alive), np.uint8))
        faces = remap[faces[live]]
        if faces.size and int(faces.min()) < 0:
            raise ValueError("a surviving triangle names a removed vertex")
        xyz = np.empty((keep.size, 3), np.float64)
        xyz[:, 0] = np.frombuffer(self.px, np.float64)[keep]
        xyz[:, 1] = np.frombuffer(self.py, np.float64)[keep]
        xyz[:, 2] = np.frombuffer(self.pz, np.float64)[keep]
        positions = xyz.astype(np.float32)
        swept = np.frombuffer(self.bound, np.float64)[keep]
        bound = emitted_bound(swept, xyz, positions)
        return DecimatedPatch(
            positions=positions,
            triangles=faces.astype(np.uint32),
            source_index=keep.astype(np.int64),
            representative=representative,
            moved=np.asarray(self.moved, bool)[keep],
            collapses=self.collapses,
            rejected_link=self.rejected_link,
            rejected_seam=self.rejected_seam,
            rejected_turn=self.rejected_turn,
            rejected_error=self.rejected_error,
            rejected_deviation=self.rejected_deviation,
            max_accepted_error_m=math.sqrt(self.max_accepted),
            plane_deviation_bound_m=float(bound.max()) if bound.size else 0.0,
            plane_deviation_swept_m=float(swept.max()) if swept.size else 0.0,
            locked_vertices=int(np.count_nonzero(np.asarray(self.locked, bool))),
        )


def _resolve(parent: Any) -> Any:
    """Follow the absorption forest to its roots, by pointer jumping.

    `parent[i] == i` exactly at a survivor, so the roots are the live vertices.
    Doubling rather than walking: a collapse chain can be thousands deep on a
    real tile and a per-vertex Python walk over half a million of them is the
    kind of loop this module exists to keep out of the sweep.
    """
    import numpy as np

    out = np.asarray(parent, np.int64).copy()
    while True:
        nxt = out[out]
        if np.array_equal(nxt, out):
            return out
        out = nxt


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
