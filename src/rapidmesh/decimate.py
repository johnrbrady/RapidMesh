"""
Quadric error metric decimation over one indexed triangle patch — WP-3.4/3.5.

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
It is the **contract**: what a caller may ask for (`DecimationSettings`), what
comes back (`DecimatedPatch`), and which of the two sweeps runs. The mechanics
live in `decimate_sweep.py` and the quantity the error-bounded stop rule
consults lives in `decimate_bounds.py`. That three-way split is WP-3.5's, taken
at the boundary the module already had: contract, mechanics, guarantee.

It is **one patch**: positions, triangles indexing them, and a boolean saying
which vertices may not move. It knows nothing about tiles, files, stations or
poses — `decimate_tiles.py` owns all of that, and the split is what makes the
seam rule testable on a six-triangle fixture instead of on a station. It is also
**not** the production decimator: normals, colour and the client container are
deliberately absent and `DecimatedPatch` carries geometry and provenance only.

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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class DecimationSettings:
    """Everything that can move the output, in one recordable object.

    Three stop rules, and given more than one whichever bites first stops the
    sweep. They are not interchangeable and the difference is the substance of
    WP-3.5:

    `max_plane_deviation_m` is the **error-bounded** rule (Round 10 D6). It caps
    the guaranteed quantity `decimate_bounds.py` derives: no collapse is
    accepted that would place a vertex further than this from the plane of any
    original triangle merged into it. It is in metres, it is a maximum and not a
    percentile, and `DecimatedPatch.plane_deviation_bound_m` reports what the
    sweep actually reached under it. Read `decimate_bounds` for what it does
    **not** cover — it is a bound about planes, not a Hausdorff distance to the
    surface, and that difference is stated there rather than glossed here.

    `max_error_m` caps ``sqrt(Q(v*))``, the root of the accumulated
    area-weighted squared distance from the placed vertex to the planes merged
    into it. **It is not a distance and it is not a bound.** The weight is the
    triangle area, so the quantity carries a factor of `sqrt(area)`: on a patch
    whose shape is held fixed while its scale changes, the true displacement
    scales linearly and this scales quadratically, understating the movement by
    953x at the triangle sizes a real station carries. `decimate_bounds.py`
    holds the measurement. It is kept, unchanged in behaviour, because every
    recorded Round 10 and Round 11 figure was produced with it at infinity and a
    knob removed is a run that cannot be reproduced — but it is a quadric-cost
    threshold, it is reported as one, and it is not the error bound.

    `target_triangles` stops at a triangle count. It is what Round 10 drove and
    it is unbounded in error by construction: once the cheap collapses run out
    it accepts whatever is left to reach its number.
    """

    max_error_m: float = math.inf
    max_plane_deviation_m: float = math.inf
    target_triangles: int | None = None
    placement: str = "optimal"
    max_normal_turn_deg: float = DEFAULT_MAX_NORMAL_TURN_DEG

    def __post_init__(self) -> None:
        if self.placement not in ("optimal", "endpoint"):
            raise ValueError(f"unknown placement {self.placement!r}")
        if not (self.max_error_m > 0.0):
            raise ValueError(f"max_error_m must be positive, got {self.max_error_m}")
        if not (self.max_plane_deviation_m > 0.0):
            raise ValueError(
                "max_plane_deviation_m must be positive, got "
                f"{self.max_plane_deviation_m}"
            )
        if self.target_triangles is not None and self.target_triangles < 0:
            raise ValueError("target_triangles must not be negative")
        if not (0.0 < self.max_normal_turn_deg <= 180.0):
            raise ValueError("max_normal_turn_deg must be in (0, 180]")

    def describe(self) -> dict[str, Any]:
        return {
            "max_error_m": (
                None if math.isinf(self.max_error_m) else float(self.max_error_m)
            ),
            "max_plane_deviation_m": (
                None
                if math.isinf(self.max_plane_deviation_m)
                else float(self.max_plane_deviation_m)
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

    `representative` is the absorption map: for every input vertex, the output
    vertex that now stands for it. It is what makes the guarantee *checkable* —
    the plane set a survivor's bound covers is exactly the faces incident to the
    input vertices mapping to it, and without this array a reader can only take
    the bound on trust. `decimate_bounds.verify_plane_bound` recomputes the
    whole claim from it and from the input geometry.

    Two bounds, and the difference between them is one rounding step. The
    stop rule governs the **swept** float64 positions, so
    `plane_deviation_swept_m` is the figure that is at or under the budget and
    is the rule's own guarantee. `finish()` then rounds the positions to
    float32, which moves each survivor by up to half a ULP per axis, and
    `plane_deviation_bound_m` is the guarantee for **the geometry that ships**
    — so it can sit a few nanometres above the budget, and it is the one to
    quote. A bound on geometry nobody emits would not be a bound; a budget that
    silently absorbed the rounding would not be a budget. Both are reported and
    neither is adjusted to make the other look tidy.

    `plane_deviation_bound_m` is the **guaranteed** figure: every surviving
    vertex of this patch is within it of the plane of every original triangle
    merged into it, including the float32 rounding of the emitted positions.
    `max_accepted_error_m` is the largest quadric cost accepted and is **not**
    in metres — see `DecimationSettings`. The two are never reported as the same
    quantity and never in the same column.
    """

    positions: F32              # (V', 3)
    triangles: U32              # (T', 3)
    source_index: I64           # (V',) input slot of each surviving vertex
    representative: I64         # (V,)  output vertex each *input* vertex is now in
    moved: BOOL                 # (V',) whether its position changed
    collapses: int
    rejected_link: int
    rejected_seam: int
    rejected_turn: int
    rejected_error: int
    rejected_deviation: int
    max_accepted_error_m: float
    plane_deviation_bound_m: float
    plane_deviation_swept_m: float
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

    `kernel` picks the sweep. `"python"` is `decimate_sweep._PatchState`, which
    is the **equivalence reference** and is never removed; `"rust"` is
    `decimate_kernel`'s native one and raises if the crate is not built;
    `"auto"` consults `RAPIDMESH_DECIMATE_KERNEL`, which defaults to `"python"`.

    **The native sweep is opt-in**, so a tree that merely has the crate built
    runs exactly the code it ran before it did — see
    `decimate_kernel.DEFAULT_CHOICE` for why that matters and what broke when it
    did not. The two sweeps are required to produce byte-identical output and
    `tests/test_decimate_kernel.py` holds them to it.
    """
    import numpy as np

    from .decimate_sweep import _PatchState

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
