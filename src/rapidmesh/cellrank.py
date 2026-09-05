"""
Lattice-cell membership as a bitset, with rank — WP-B, ITEM-022 T1.

Pass B used to answer two questions by holding station-scale integer arrays:

* *is this lattice cell meshed / did it survive the cull?* — answered by
  `searchsorted` into a sorted `int64` array of every meshed cell (110 MB on
  ordinal 20);
* *what is this cell's global vertex index?* — answered by `remap`, one `int32`
  per meshed cell (55 MB), built from `keep_idx` (110 MB).

Both are membership questions over the lattice, and a lattice cell is one bit.
`CellRank` stores that bit and the running count of set bits per byte, which is
all that "rank" needs:

| structure | ordinal 20 (2387 x 6096 = 14,551,152 cells) |
|:--|--:|
| packed bits | 1,818,894 B |
| per-byte cumulative counts (`int64`) | 14,551,152 B |
| **total** | **~16.4 MB**, against ~275 MB for the three arrays it replaces |

**Rank is the same integer the old `remap` held.** The global vertex index of a
surviving cell is the number of surviving cells strictly below it — ascending
cell order, which is `build_mesh`'s vertex compaction order
(`PHASE1-DETERMINISM-SPEC.md` §5(a), §5(c)) and exactly what
`remap[keep_idx] = arange(...)` assigned. This is a storage change, not a
renumbering, and `tests/test_cell_rank.py` holds it to a brute-force `cumsum`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    BOOL = npt.NDArray[np.bool_]
    I64 = npt.NDArray[np.int64]
    U8 = npt.NDArray[np.uint8]


def _popcount_table() -> Any:
    """256-entry byte popcount.

    A table rather than `np.bitwise_count` so this module does not acquire a
    numpy floor it does not otherwise need; the table is 256 bytes and the
    lookup is one gather.
    """
    import numpy as np

    return np.array([bin(value).count("1") for value in range(256)], np.uint8)


_POP = None


def _pop() -> Any:
    global _POP
    if _POP is None:
        _POP = _popcount_table()
    return _POP


@dataclass(frozen=True)
class CellRank:
    """Set membership over `[0, cells)` with O(1) rank, in ~1.13 bits per cell."""

    packed: Any          # np.packbits of the membership flags, big-endian
    cumulative: Any      # inclusive per-byte cumulative popcount
    cells: int           # lattice cell count the flags were built over
    total: int           # number of set cells

    @classmethod
    def of(cls, flags: BOOL) -> CellRank:
        """Build from a dense boolean array over the lattice.

        The caller is expected to release `flags` afterwards: it is 8x this
        structure's bit storage and the whole point is not to keep it.
        """
        import numpy as np

        packed = np.packbits(np.asarray(flags, bool))
        cumulative = np.cumsum(_pop()[packed], dtype=np.int64)
        return cls(
            packed=packed,
            cumulative=cumulative,
            cells=int(np.asarray(flags).shape[0]),
            total=int(cumulative[-1]) if cumulative.size else 0,
        )

    def contains(self, cell_ids: I64) -> BOOL:
        """Membership per cell id."""
        import numpy as np

        ids = np.asarray(cell_ids, np.int64)
        byte = ids >> 3
        bit = ids & 7
        out: BOOL = (((self.packed[byte] >> (7 - bit).astype(np.uint8)) & 1) != 0)
        return out

    def rank(self, cell_ids: I64) -> I64:
        """Set cells strictly below each id — the global vertex index.

        `np.packbits` is big-endian, so cell `8b + r` is bit `7 - r` of byte `b`
        and "strictly below within the byte" is the top `r` bits. The mask is
        built in 16-bit space because `0xFF << 8` does not survive `uint8`.
        """
        import numpy as np

        ids = np.asarray(cell_ids, np.int64)
        byte = ids >> 3
        bit = ids & 7
        # `cumulative[byte - 1]` reads the last element when `byte` is 0; the
        # `where` discards it, and indexing with -1 is defined, so no branch.
        below = np.where(byte > 0, self.cumulative[byte - 1], 0)
        mask = ((0xFF00 >> bit) & 0xFF).astype(np.uint8)
        out: I64 = below + _pop()[self.packed[byte] & mask].astype(np.int64)
        return out
