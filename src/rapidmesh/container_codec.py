"""The lossless byte filter, owned by neither candidate — WP-4.1.

Two transforms, both exactly invertible on integers and on float bit patterns,
and both cheap enough that a browser can undo them in JavaScript:

* `byte_shuffle` groups the k-th byte of every element together — the classic
  shuffle filter. Float32 coordinates over one tile share an exponent range, so
  the exponent plane becomes nearly constant and a generic compressor that could
  do nothing with interleaved floats has something to work on.
* `delta_rows` differences an index block row by row in wrapping uint32.
  Triangles emitted from a lattice window address nearby vertices, so
  consecutive rows differ by small numbers whose high bytes are zero.

This is meshopt-**class** in the sense that matters for bytes — separate the
byte planes so a generic compressor sees structure. It is **not** the meshopt
bitstream and is not claimed to be.

It lives in its own module, rather than inside the bespoke candidate that uses
it, so that the report can say what the filter is worth without that gain being
scored as an advantage of either container — which the Round 15 brief requires,
because a codec available to both flatters neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    U32 = npt.NDArray[np.uint32]


def byte_shuffle(array: Any) -> bytes:
    """Group the k-th byte of every element together, losslessly.

    The classic shuffle filter, and the half of a meshopt-class codec that is
    both cheap and provably exact. Float32 coordinates over one tile share an
    exponent range, so the exponent plane becomes nearly constant and a generic
    compressor that could do nothing with interleaved floats has something to
    work on. It is not the meshopt bitstream and is not claimed to be.
    """
    import numpy as np

    flat = np.ascontiguousarray(array)
    view = flat.reshape(-1).view(np.uint8).reshape(-1, flat.dtype.itemsize)
    return bytes(np.ascontiguousarray(view.T).tobytes())


def byte_unshuffle(blob: bytes, dtype: Any, shape: tuple[int, ...]) -> Any:
    """Inverse of `byte_shuffle`. Exact, and tested as exact rather than assumed."""
    import numpy as np

    kind = np.dtype(dtype)
    count = 1
    for dim in shape:
        count *= int(dim)
    view = np.frombuffer(blob, np.uint8, count=count * kind.itemsize)
    return np.ascontiguousarray(view.reshape(kind.itemsize, count).T).view(kind).reshape(shape)


def delta_rows(rows: U32) -> U32:
    """Row-wise difference of a (T,3) index block, in wrapping uint32.

    Triangles emitted from a lattice window address nearby vertices, so
    consecutive rows differ by small numbers whose high bytes are zero. Wrapping
    arithmetic makes the inverse an exact cumulative sum with no special case
    for a negative step.
    """
    import numpy as np

    block = np.ascontiguousarray(rows, np.uint32)
    if block.shape[0] < 2:
        return block
    out = block.copy()
    out[1:] = block[1:] - block[:-1]
    return out


def undelta_rows(rows: U32) -> U32:
    """Inverse of `delta_rows`, exact under the same wrapping arithmetic."""
    import numpy as np

    block = np.ascontiguousarray(rows, np.uint32)
    if block.shape[0] < 2:
        return block
    return np.cumsum(block, axis=0, dtype=np.uint32)
