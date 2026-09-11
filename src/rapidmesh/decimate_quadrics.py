"""
The quadric algebra behind `decimate.py` — WP-3.4.

One symmetric 4x4 per vertex, held by its ten unique entries in the order

    [ a b c d ]
    [ b e f g ]   ->   (a, b, c, d, e, f, g, h, i, j)
    [ c f h i ]
    [ d g i j ]

so that ``v^T Q v`` for ``v = (x, y, z, 1)`` is the accumulated area-weighted
squared distance from the point to every plane merged into that vertex. That
one convention is the whole content of this module, which is why it is a module
and not four helpers at the bottom of the sweep: the layout is depended on by
the construction, the evaluation and the minimiser alike, and a change to it has
to be made in one place or it is a silent numerical bug in the other two.

Nothing here knows about edges, collapses, locking or seams. `decimate.py` owns
all of that.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

#: Relative determinant floor for the 3x3 optimal-placement solve. Below it the
#: quadric has no unique minimiser — a flat wall's optimum is a plane and a
#: crease's is a line — and the caller's endpoint/midpoint ladder is used
#: instead. Relative to the matrix scale, so it carries no unit.
SINGULAR_RELATIVE = 1e-10




def vertex_quadrics(pos64: F64, tris: I64, count: int) -> array[float]:
    """Area-weighted plane quadrics summed per vertex, ten entries per vertex.

    `np.bincount` rather than `np.add.at`: the same sum over ten independent
    components at a fraction of the cost, and its ordering is fixed by index
    rather than by arrival, which is what determinism needs of it.
    """
    import numpy as np

    acc = np.zeros((count, 10), np.float64)
    if tris.shape[0]:
        a = pos64[tris[:, 0]]
        n = np.cross(pos64[tris[:, 1]] - a, pos64[tris[:, 2]] - a)
        length = np.linalg.norm(n, axis=1)
        # The area weight is |n|/2 and the unit normal is n/|n|, so the weighted
        # outer product is n n^T / (2|n|). A degenerate face carries no plane
        # and is dropped rather than regularised into one.
        scale = np.zeros_like(length)
        np.divide(0.5, length, out=scale, where=length > 0.0)
        d = -np.einsum("ij,ij->i", n, a)
        plane = (n[:, 0], n[:, 1], n[:, 2], d)
        flat = tris.ravel()
        column = 0
        for i in range(4):
            for j in range(i, 4):
                w = plane[i] * plane[j] * scale
                acc[:, column] = np.bincount(flat, np.repeat(w, 3), minlength=count)
                column += 1
    return doubles(acc.ravel())


def doubles(values: F64) -> array[float]:
    """A float64 numpy array as an `array('d')`, without a list in between.

    The list would be the largest transient in the setup — 2.6 M boxed floats
    on a half-million-triangle tile — for a structure whose whole purpose is
    not to box them.
    """
    import numpy as np

    out = array("d")
    out.frombytes(np.ascontiguousarray(values, np.float64).tobytes())
    return out


def quadric_error(q: list[float], x: float, y: float, z: float) -> float:
    """``v^T Q v`` for ``v = (x, y, z, 1)``, written out."""
    a, b, c, d, e, f, g, h, i, j = q
    return (
        a * x * x + 2.0 * b * x * y + 2.0 * c * x * z + 2.0 * d * x
        + e * y * y + 2.0 * f * y * z + 2.0 * g * y
        + h * z * z + 2.0 * i * z
        + j
    )


def solve_optimal(q: list[float]) -> tuple[float, float, float] | None:
    """The point where the quadric is least, or `None` if it has no unique one.

    Solves the 3x3 upper block against ``-(d, g, i)`` by cofactors. The
    determinant is judged against the matrix's own scale, so a flat wall (whose
    optimum is a plane) and a crease (whose optimum is a line) both fall through
    to the caller's ladder instead of returning a point off at infinity.
    """
    a, b, c, d, e, f, g, h, i, _ = q
    c00 = e * h - f * f
    c01 = c * f - b * h
    c02 = b * f - c * e
    det = a * c00 + b * c01 + c * c02
    # The 3x3 block is a sum of w n n^T and so is positive semi-definite: its
    # trace is non-negative and bounds every entry, which makes it the matrix
    # scale without six absolute values per candidate edge.
    scale = a + e + h
    if scale <= 0.0 or abs(det) <= SINGULAR_RELATIVE * scale * scale * scale:
        return None
    c11 = a * h - c * c
    c12 = b * c - a * f
    c22 = a * e - b * b
    inv = 1.0 / det
    return (
        -inv * (c00 * d + c01 * g + c02 * i),
        -inv * (c01 * d + c11 * g + c12 * i),
        -inv * (c02 * d + c12 * g + c22 * i),
    )


