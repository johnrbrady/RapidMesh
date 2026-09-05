"""
Bounded random access to a written observation store — WP-C, ITEM-022 T4.

Split from `obs_store.py` for DEC-010 at a real boundary: that module owns the
format — what a record is, how a store is written and validated — and this one
owns the only thing QA needs from a written store, which is the ability to read
a little of it without reading all of it.

WP-13a sized T4 at **768,720,960 B on ordinal 20**: `offsets` float32 (165 MB),
its float64 copy inside the query (330 MB), `rows` int64 (110 MB), the `argsort`
permutation (110 MB) and the sorted copy (55 MB). Every one of them existed so
QA could answer two bounded questions about a population it therefore had to
hold. The store already holds that population, in ascending lattice-cell order,
and this reads the answers straight out of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .obs_store import (
    _FIXED,
    _HEADER,
    MAX_OBSERVATIONS,
    OBS_CONTRACT_VERSION,
    OBS_MAGIC,
    OBS_READ_BLOCK,
    ObservationStoreError,
    dtype,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ObservationWindows:
    """Bounded random access to a written store — WP-C, ITEM-022 T4.

    Both QA directions ask this object bounded questions and never hold the
    population they are asking about:

    * forward wants 300,000 observations chosen by `(n, seed)`. The draw depends
      only on `n` and the seed, so `gather` reads just those records;
    * reverse wants every observation whose lattice row is in `[lo, hi]`. The
      store is in ascending cell order, so that is a **contiguous slice**, and
      `row_range` turns the pair into byte offsets through the table written
      beside the records.

    What made the old path station-scale was not the questions, it was that they
    were answered by indexing arrays that had to exist first: `offsets`, its
    float64 copy, `rows`, `argsort(rows)` and the sorted copy — 768,720,960 B on
    ordinal 20 (WP-13a §2.3). None of them is built here.

    **Nothing is cached.** Each call reads what it needs and returns it; the
    object holds the row table (19 KB on the high-resolution lattice) and four
    integers. A cache would reintroduce exactly the array being removed, one
    window at a time.
    """

    path: Path
    count: int
    rows: int
    cols: int
    offsets: Any                 # (rows + 1,) int64, records with row < r
    _record_start: int

    @classmethod
    def open(cls, path: Path) -> ObservationWindows:
        import numpy as np

        with open(path, "rb") as handle:
            head = handle.read(_HEADER.size)
            if len(head) < _HEADER.size:
                raise ObservationStoreError(
                    f"observation store is truncated: {path.name}"
                )
            magic, contract, _flags, header_bytes, count, payload_bytes, _d = (
                _HEADER.unpack(head)
            )
            if magic != OBS_MAGIC:
                raise ObservationStoreError(f"not an observation store: {path.name}")
            if contract != OBS_CONTRACT_VERSION:
                raise ObservationStoreError(
                    f"observation store contract {contract} is not "
                    f"{OBS_CONTRACT_VERSION}; the row table is required"
                )
            if header_bytes != _HEADER.size or count > MAX_OBSERVATIONS:
                raise ObservationStoreError(
                    f"store header is not self-consistent: {path.name}"
                )
            fixed = _FIXED.unpack(handle.read(_FIXED.size))
            lattice_rows, lattice_cols = int(fixed[7]), int(fixed[8])
            record_start = _HEADER.size + _FIXED.size
            table_start = record_start + int(count) * dtype().itemsize
            if payload_bytes != (
                _FIXED.size + int(count) * dtype().itemsize + (lattice_rows + 1) * 8
            ):
                raise ObservationStoreError(
                    f"store length fields disagree: {path.name}"
                )
            handle.seek(table_start)
            raw = handle.read((lattice_rows + 1) * 8)
            offsets = np.frombuffer(raw, np.int64, count=lattice_rows + 1).copy()
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(count):
            raise ObservationStoreError(
                f"row offset table does not span the records: {path.name}"
            )
        return cls(
            path=path, count=int(count), rows=lattice_rows, cols=lattice_cols,
            offsets=offsets, _record_start=record_start,
        )

    def row_range(self, lo: int, hi: int) -> tuple[int, int]:
        """Records whose lattice row lies in `[lo, hi]`, as a half-open slice.

        Clamped rather than rejected: reverse QA widens a group's row span by
        `qa_window_rows` and the result runs off both ends of the lattice at the
        first and last bands, exactly as `searchsorted` over a sorted copy did.
        """
        low = min(max(int(lo), 0), self.rows)
        high = min(max(int(hi) + 1, 0), self.rows)
        if high <= low:
            return 0, 0
        return int(self.offsets[low]), int(self.offsets[high])

    def positions(self, start: int, stop: int) -> Any:
        """`[start, stop)` as `(n, 3)` float64, in store order.

        Float64 because that is what the KD-tree was built from before, and the
        widening is of the same float32 values — so the candidate set is the
        same points in the same order and the distances are the same bits.
        """
        import numpy as np

        take = max(int(stop) - int(start), 0)
        out = np.empty((take, 3), np.float64)
        if take == 0:
            return out
        kind = dtype()
        with open(self.path, "rb") as handle:
            handle.seek(self._record_start + int(start) * kind.itemsize)
            at = 0
            while at < take:
                step = min(OBS_READ_BLOCK, take - at)
                raw = handle.read(step * kind.itemsize)
                if len(raw) != step * kind.itemsize:
                    raise ObservationStoreError(
                        f"observation store payload is truncated: {self.path.name}"
                    )
                block = np.frombuffer(raw, dtype=kind, count=step)
                for axis, name in enumerate(("x", "y", "z")):
                    out[at : at + step, axis] = block[name]
                at += step
        return out

    def gather(self, index: Any) -> Any:
        """Positions at `index`, as float32, **in the order given**.

        Order is preserved because the forward sample is `q[rs.choice(...)]` and
        the choice is not sorted; returning the store's order instead would be a
        permutation of the same points, which no statistic here would notice and
        which would still be a different array than the one it replaces.
        """
        import numpy as np

        idx = np.asarray(index, np.int64)
        out = np.empty((idx.size, 3), np.float32)
        if idx.size == 0:
            return out
        order = np.argsort(idx, kind="stable")
        wanted = idx[order]
        kind = dtype()
        with open(self.path, "rb") as handle:
            at = 0
            while at < wanted.size:
                stop = at + 1
                # One read per contiguous stretch of wanted records, so a dense
                # draw costs one seek and a sparse one costs no more reads than
                # it has gaps.
                while (
                    stop < wanted.size
                    and int(wanted[stop]) - int(wanted[stop - 1]) == 1
                ):
                    stop += 1
                first = int(wanted[at])
                span = stop - at
                handle.seek(self._record_start + first * kind.itemsize)
                raw = handle.read(span * kind.itemsize)
                if len(raw) != span * kind.itemsize:
                    raise ObservationStoreError(
                        f"observation store payload is truncated: {self.path.name}"
                    )
                block = np.frombuffer(raw, dtype=kind, count=span)
                for axis, name in enumerate(("x", "y", "z")):
                    out[order[at:stop], axis] = block[name]
                at = stop
        return out

    def rows_at(self, index: Any) -> Any:
        """Lattice row per record index, as int64, in the order given.

        Reverse QA needs the rows of the corners of the triangles it selected —
        at most `3 x max_samples` of them — not the row of every observation.
        """
        import numpy as np

        idx = np.asarray(index, np.int64)
        flat = idx.reshape(-1)
        out = np.empty(flat.size, np.int64)
        kind = dtype()
        if flat.size:
            order = np.argsort(flat, kind="stable")
            wanted = flat[order]
            with open(self.path, "rb") as handle:
                at = 0
                while at < wanted.size:
                    stop = at + 1
                    while (
                        stop < wanted.size
                        and int(wanted[stop]) - int(wanted[stop - 1]) == 1
                    ):
                        stop += 1
                    first = int(wanted[at])
                    span = stop - at
                    handle.seek(self._record_start + first * kind.itemsize)
                    raw = handle.read(span * kind.itemsize)
                    block = np.frombuffer(raw, dtype=kind, count=span)
                    out[order[at:stop]] = block["row"].astype(np.int64)
                    at = stop
        return out.reshape(idx.shape)
