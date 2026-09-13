"""The guaranteed error bound a collapse sweep can carry — WP-3.5.

Round 10 drove its sweeps with a **target ratio and no error cap**, and once the
cheap collapses ran out the sweep took whatever was left — a 52.6 mm collapse on
the control station at 16x (Round 10 §3.11). The fix DEC-013's routing names is
an error-bounded stop rule, and a stop rule is only worth the quantity it
consults. This module is that quantity, and it is separate from the sweep
because it is the part with a proof attached.

Part 1 — why `sqrt(Q(v*))` is not a bound, and the measurement that shows it
----------------------------------------------------------------------------
`DecimationSettings.max_error_m` caps the quadric cost, and until this package
that cap was documented as "a *guaranteed* per-collapse bound" in metres. It is
neither.

The vertex quadric is **area-weighted** (`decimate_quadrics.vertex_quadrics`:
the weight is `|n| / 2`), so

    Q(v) = sum over merged planes f of  area_f * d_f(v)**2

whose units are m**2 * m**2 = m**4. Its square root is in m**2, not metres, and
it carries a factor of `sqrt(area)` that no distance has. The consequence is not
academic. Hold the *shape* of a patch fixed and change only its scale, so that
the true displacement must scale linearly, and `sqrt(Q)` scales quadratically.
Measured on a 41x41 ridge decimated 16x, the same geometry at four scales
(`MEASUREMENTS/2026-09-12/harnesses/rm_r12_units.py`):

    mean triangle area   sqrt(Q) reported   true max displacement   true/reported
        52.35    m**2        43.553   m           41.520   m             0.95
         0.5235  m**2         0.4355  m            4.1520  m             9.5
         0.005235 m**2        0.004355 m           0.41520 m            95.3
         0.00005235 m**2      0.00004355 m         0.041520 m          953.3

The true column scales by 10 per row, as a distance must; the reported column
scales by 100. **At the triangle sizes a real station carries the quantity
understates the movement by two to three orders of magnitude**, and it
understates in the dangerous direction: a 1 mm cap on `sqrt(Q)` admits collapses
that move geometry by hundreds of millimetres. `max_error_m` is kept, unchanged
in behaviour, because every recorded Round 10 and Round 11 figure was produced
with it at infinity and a knob removed is a run that cannot be reproduced — but
it is a quadric-cost threshold, it is reported as one, and it is not a bound.

Part 2 — the bound this module does carry, and what exactly it guarantees
-------------------------------------------------------------------------
For a vertex `v` carrying quadric `Q_v` over the merged plane set `F(v)`, and
for any one plane `f0` in that set, every term of the sum is non-negative, so

    area_f0 * d_f0(x)**2  <=  Q_v(x)      hence      |d_f0(x)| <= sqrt(Q_v(x) / area_f0)

and therefore, writing `amin_v` for the smallest area in `F(v)`,

    max over f in F(v) of |d_f(x)|  <=  sqrt( Q_v(x) / amin_v )   =:   B(v, x)

`B` is in metres, it is a **maximum and not a percentile**, and it holds for
every plane merged into the vertex rather than for most of them. `amin` is one
scalar per vertex, initialised as the smallest area among the faces incident to
that vertex and combined on collapse by `min`, because adding two quadrics unions
their plane sets. Degenerate faces carry no plane in `vertex_quadrics` and so
carry no area here either — the two must agree or the bound is over a plane set
the quadric does not contain.

*What may be claimed.* After a sweep, for every surviving vertex `v` and every
original triangle `f` merged into it, `v` lies within `B` of the plane of `f`.
The per-patch figure is the largest such `B` over surviving vertices, so it is a
statement about all of them. A locked vertex never moves, so every plane merged
into it still passes through it and its bound is exactly 0.000 mm — a seam
contributes nothing, which is checkable rather than asserted.

*What may not be claimed, stated as plainly.* This is a bound on distance to the
**planes** of the original triangles, not to the original **surface**. A plane
is unbounded and a triangle is not, so a vertex that slid along a flat region and
off its rim reads zero against that region's plane while having left the
surface. It is therefore **not** a Hausdorff distance and not a bound on one, in
either direction, and no figure derived from it is described as one. What
constrains that failure in practice is the link condition and the normal-turn
test in `decimate_sweep.py`, neither of which is a proof; what measures it is the
sampled surface-to-surface deviation in `decimate_qa.py`, computed over a
**certified exhaustive** candidate set by `decimate_nearest.py`. The two are
reported side by side and never as one another: one is a guarantee about planes,
the other a sample of the surface.

*Why not the obvious alternative.* The first formulation tried here was a
triangle-inequality accumulator — per vertex, `r* = max(r_u + |p* - p_u|,
r_v + |p* - p_v|)`, an upper bound on how far the original input vertices it
represents have been carried. That **is** a true bound on distance to the output
*surface*, which is the stronger statement, and it was still discarded.

The reason is in the update rule: `|p* - p_u|` is the whole length of the
collapse, including the part that slides *along* the surface and moves nothing.
Collapsing an edge places the survivor near its midpoint, so the rule charges
roughly half an edge length — on a 50 mm lattice, about 25 mm — for a collapse
whose actual effect on the surface is the sagitta, three orders of magnitude
smaller on a gently curved patch. Applied to a fine mesh at a millimetre budget
it therefore refuses every collapse there is, and it does so more completely the
finer the mesh gets: the looseness is the ratio of edge length to sagitta, which
is unbounded. A guarantee nobody can operate under is not a usable stop rule,
and weakening the budget until it became usable would have been choosing a
number to fit an answer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from array import array as Array

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

#: A vertex with no non-degenerate incident face has an all-zero quadric and no
#: smallest area. Infinity is the identity for the `min` that combines two
#: vertices' areas and drives `B` to zero against a zero quadric, so the empty
#: case needs no branch anywhere else.
NO_AREA = math.inf


def plane_bound(
    q: list[float], amin: float, x: float, y: float, z: float
) -> float:
    """`B = sqrt(Q(x) / amin)` — the guaranteed distance to every merged plane.

    The cost is clamped at zero before the root for the reason `_push` clamps
    it: a flat quadric evaluated at its own minimiser can come out very slightly
    negative by rounding, and a negative under a square root would be a NaN that
    compares false against every budget and so would silently *accept* the
    collapse it was meant to judge.
    """
    from .decimate_quadrics import quadric_error

    cost = quadric_error(q, x, y, z)
    if cost <= 0.0:
        return 0.0
    return math.sqrt(cost / amin)


def face_min_areas(pos64: F64, tris: I64, count: int) -> Array[float]:
    """Per vertex, the smallest incident face area — `amin` at the sweep's start.

    Degenerate faces are skipped on exactly the test `vertex_quadrics` uses
    (`|n| > 0`), because `amin` has to range over the same plane set the quadric
    accumulated. A vertex with no surviving incident face gets `NO_AREA`.

    Vectorised with `np.minimum.at` rather than `np.bincount`: this is a minimum
    and not a sum, so there is no bincount form of it, and unlike a sum its
    result does not depend on accumulation order — so determinism costs nothing
    here.
    """
    import numpy as np

    from .decimate_quadrics import doubles

    acc = np.full(count, NO_AREA, np.float64)
    if tris.shape[0]:
        a = pos64[tris[:, 0]]
        n = np.cross(pos64[tris[:, 1]] - a, pos64[tris[:, 2]] - a)
        length = np.linalg.norm(n, axis=1)
        live = length > 0.0
        if bool(live.any()):
            area = 0.5 * length[live]
            np.minimum.at(acc, tris[live].ravel(), np.repeat(area, 3))
    return doubles(acc)


def emitted_bound(bound: F64, pos64: F64, pos32: F32) -> F64:
    """The per-vertex bound against the **emitted** float32 position.

    `finish()` rounds the swept float64 positions to float32, which moves each
    survivor by up to half a ULP per axis. That is small — about 7.6 um on a
    coordinate 80 m from a tile origin — but it is not nothing against a 1 mm
    budget, and a bound on geometry nobody ships is not a bound. A shift of `s`
    changes a distance to any plane by at most `|s|`, so adding the measured
    shift keeps the guarantee. The shift is exactly available rather than merely
    boundable, so it is measured and added instead of estimated.
    """
    import numpy as np

    delta = np.linalg.norm(
        np.asarray(pos64, np.float64) - np.asarray(pos32, np.float64), axis=1
    )
    out: F64 = np.asarray(bound, np.float64) + delta
    return out


def verify_plane_bound(
    source_positions: F64 | F32,
    source_triangles: Any,
    out_positions: F32,
    representative: I64,
    bound: float,
) -> tuple[float, bool]:
    """Recompute the whole claim from the geometry: `(worst, worst <= bound)`.

    A survivor's quadric holds the plane of face `f` exactly when `f` is
    incident to some input vertex that ended up in that survivor — quadrics are
    summed on collapse, so the merged plane set is the union of the absorbed
    vertices' incident faces. So the claim is checked by walking every
    (face, corner) pair: the survivor the corner maps to must lie within the
    bound of that face's plane.

    Nothing the sweep recorded is used except the absorption map itself. The
    planes are rebuilt from the **input** positions and the distances measured to
    the **emitted** float32 positions, which is the geometry that ships. No
    tolerance is applied: a bound that held only to a tolerance would not be one.
    """
    import numpy as np

    verts = np.asarray(source_positions, np.float64)
    tris = np.asarray(source_triangles, np.int64)
    out = np.asarray(out_positions, np.float64)
    rep = np.asarray(representative, np.int64)
    if tris.shape[0] == 0 or out.shape[0] == 0:
        return 0.0, True

    a = verts[tris[:, 0]]
    n = np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a)
    length = np.linalg.norm(n, axis=1)
    live = length > 0.0
    if not bool(live.any()):
        return 0.0, True
    faces, a, unit = tris[live], a[live], n[live] / length[live][:, None]
    offset = np.einsum("ij,ij->i", unit, a)

    # Each face against each of its three corners' survivors.
    owner = rep[faces.ravel()]
    plane = np.repeat(unit, 3, axis=0)
    distance = np.abs(
        np.einsum("ij,ij->i", plane, out[owner]) - np.repeat(offset, 3)
    )
    worst = float(distance.max())
    return worst, worst <= bound
