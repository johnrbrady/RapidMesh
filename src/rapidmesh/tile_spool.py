"""
Bounded per-tile append spools — `PHASE1-TILE-CONTRACT-V0.md` §2, WP-3.2.

§2's schedule for Pass B's spatial merge, verbatim: *"derives each triangle's
spatial tile id from the versioned tile-origin/extent rule, appends the record to
a bounded set of shard spools, closes least-recently-used spools when the
open-file cap is reached, then finalises one tile at a time."* This module is the
spool layer of that sentence and nothing else — it knows about records and tile
ids, and nothing about geometry.

Why a spool rather than a sort
------------------------------
Grouping by tile is a partition, not an ordering, so the external merge in
`pass_b_merge` is the wrong tool: it would sort what only needs to be filed. The
distinction matters because **append order is load-bearing here**. A tile's
records come back in exactly the order they were appended, which is the order the
producer walked the station in, so a per-tile reduction over them is a
subsequence of the global reduction. `tiles.py` depends on that to reproduce
`build_mesh`'s normal accumulation exactly; a sort would destroy it silently.

Caps
----
    B17  open spool handles                OPEN_SPOOL_LIMIT        64 files
    B18  per-spool write buffer            SPOOL_BUFFER_BYTES   1,000,000 B
    B19  read block                        SPOOL_READ_BYTES     4,000,000 B

The open-file cap is a *handle* cap, not a byte cap, because that is the resource
it protects; the bytes are bounded separately by the write buffer, and the two
multiply to a worst case of 64 MB with every spool hot. A station with more tiles
than the handle cap still works: the least-recently-used spool is flushed and
closed, and reopened for append if it is written to again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    I64 = npt.NDArray[np.int64]

OPEN_SPOOL_LIMIT = 64
SPOOL_BUFFER_BYTES = 1_000_000
SPOOL_READ_BYTES = 4_000_000


@dataclass
class _Open:
    """One spool's file handle and its pending buffer."""

    handle: Any
    pending: list[Any]
    pending_bytes: int


class TileSpoolSet:
    """Append records keyed by tile id; read one tile's records back in order.

    Not a context manager by accident: `close` must run before any read, and
    making that explicit keeps a half-flushed spool from being read as a short
    one. `finish()` is the single transition, and reading before it raises.
    """

    def __init__(
        self,
        root: Path,
        prefix: str,
        dtype: Any,
        *,
        open_limit: int = OPEN_SPOOL_LIMIT,
        buffer_bytes: int = SPOOL_BUFFER_BYTES,
    ) -> None:
        self.root = root
        self.prefix = prefix
        self.dtype = dtype
        self._open_limit = max(int(open_limit), 1)
        self._buffer_bytes = max(int(buffer_bytes), int(dtype.itemsize))
        self._open: dict[int, _Open] = {}
        self._counts: dict[int, int] = {}
        self._finished = False
        root.mkdir(parents=True, exist_ok=True)

    # -- writing ------------------------------------------------------------

    def append(self, tile_ids: I64, records: Any) -> None:
        """File each record under its tile. `tile_ids` is one id per record.

        Records are grouped by a stable sort, so the relative order of two
        records going to the same tile is the order they arrived in.
        """
        import numpy as np

        if self._finished:
            raise RuntimeError("cannot append to a finished spool set")
        if records.shape[0] == 0:
            return
        order = np.argsort(tile_ids, kind="stable")
        ordered_ids = tile_ids[order]
        ordered = records[order]
        edges = np.flatnonzero(np.diff(ordered_ids)) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [ordered_ids.size]))
        for start, stop in zip(starts, stops, strict=True):
            self._append_one(int(ordered_ids[start]), ordered[start:stop])

    def _append_one(self, tile_id: int, records: Any) -> None:
        import numpy as np

        entry = self._open.get(tile_id)
        if entry is None:
            entry = self._acquire(tile_id)
        entry.pending.append(np.ascontiguousarray(records))
        entry.pending_bytes += int(records.nbytes)
        self._counts[tile_id] = self._counts.get(tile_id, 0) + int(records.shape[0])
        if entry.pending_bytes >= self._buffer_bytes:
            _flush(entry)

    def _acquire(self, tile_id: int) -> _Open:
        """Open `tile_id`'s spool, evicting the least recently used if needed.

        `dict` preserves insertion order, so re-inserting on every touch makes
        the first key the least recently used — an LRU without a second
        structure to keep in step with this one.
        """
        while len(self._open) >= self._open_limit:
            victim = next(iter(self._open))
            self._release(victim)
        handle = open(self.path(tile_id), "ab")  # noqa: SIM115 - closed by _release
        entry = _Open(handle=handle, pending=[], pending_bytes=0)
        self._open[tile_id] = entry
        return entry

    def _release(self, tile_id: int) -> None:
        entry = self._open.pop(tile_id)
        _flush(entry)
        entry.handle.close()

    def touch(self, tile_id: int) -> None:
        """Mark a spool most-recently-used, so the LRU order is a real one."""
        if tile_id in self._open:
            self._open[tile_id] = self._open.pop(tile_id)

    def finish(self) -> None:
        """Flush and close every spool. Reads are only valid after this."""
        for tile_id in list(self._open):
            self._release(tile_id)
        self._finished = True

    # -- reading ------------------------------------------------------------

    def path(self, tile_id: int) -> Path:
        return self.root / f"{self.prefix}-{tile_id:012d}.rmspool"

    def tile_ids(self) -> list[int]:
        """Tiles that received at least one record, in ascending id order."""
        return sorted(self._counts)

    def count(self, tile_id: int) -> int:
        return self._counts.get(tile_id, 0)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def read(self, tile_id: int, *, block_bytes: int = SPOOL_READ_BYTES) -> Iterator[Any]:
        """One tile's records, in append order, in bounded blocks."""
        import numpy as np

        if not self._finished:
            raise RuntimeError("spool set must be finished before it is read")
        path = self.path(tile_id)
        if not path.exists():
            return
        per_read = max(int(block_bytes // self.dtype.itemsize), 1) * self.dtype.itemsize
        seen = 0
        with open(path, "rb") as handle:
            while raw := handle.read(per_read):
                if len(raw) % self.dtype.itemsize:
                    raise ValueError(
                        f"spool {path.name} ends mid-record; it was not flushed"
                    )
                seen += len(raw) // self.dtype.itemsize
                yield np.frombuffer(raw, dtype=self.dtype)
        expected = self._counts.get(tile_id, 0)
        if seen != expected:
            raise ValueError(
                f"spool {path.name} holds {seen} records, {expected} were appended"
            )

    def remove(self, tile_id: int) -> None:
        """Delete one finalised tile's spool. Its bytes are no longer needed."""
        self.path(tile_id).unlink(missing_ok=True)


def _flush(entry: _Open) -> None:
    import numpy as np

    if not entry.pending:
        return
    payload = (
        entry.pending[0] if len(entry.pending) == 1 else np.concatenate(entry.pending)
    )
    entry.handle.write(np.ascontiguousarray(payload).tobytes())
    entry.pending.clear()
    entry.pending_bytes = 0
