"""
Bounded external merge for Pass B — ITEM-009, DEC-009 step 1.

`PHASE1-TILE-CONTRACT-V0.md` §2: *"The merge is bounded, not an in-memory
station sort. Canonical triangle ordering uses per-segment bounded sorts
followed by k-way external merges."* This module is that layer, and it is the
one Pass B and `reverse-qa-v2` share.

WP-1.5 measured Pass B growing from 45% to 76% of the streamed pipeline's peak
as the station grew, because finalisation held the whole station's triangles
several times over: the stable-id triples, the index array, the component roots
and `component_area_v1`'s canonical-order working set, all live at once and all
O(triangles). None of them has to be.

The consumer that needed it — Pass B's streamed area reduction and cull — is
`pass_b.py`; this module knows nothing about triangles.

**Two merge views, never conflated.**
`(final_component_root, canonical_oriented_triple)` orders the area reduction;
`canonical_oriented_triple` alone orders `reverse-qa-v2`. §2 permits them to
share run machinery and forbids pretending one ordering is the other, so the
key tuple is an explicit argument, the run files carry the key in their name,
and a reader that asks for the wrong view gets a different file.

**Byte caps — the B10/B13 rows §6 left as skeleton.**
Every buffer here has an explicit byte cap rather than a record count, so the
figure does not silently change when a record type gains a field:

    B10  run-creation sort buffer        RUN_BUFFER_BYTES      8,000,000 B
    B13  k-way merge input, all runs     MERGE_BUFFER_BYTES   32,000,000 B
    B13  merge output block              OUTPUT_BLOCK_BYTES    8,000,000 B

The merge buffer is a *total*: the per-run share is the cap divided by the
number of runs, floored so a station with many bands still makes progress.
Peak for this layer is therefore bounded by the caps and not by the station —
`tests/test_pass_b_bound.py` asserts that against a fixture ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

RUN_BUFFER_BYTES = 8_000_000
MERGE_BUFFER_BYTES = 32_000_000
OUTPUT_BLOCK_BYTES = 8_000_000

# A floor so a station with very many bands still reads a useful slice per run
# rather than degenerating into per-record I/O.
MIN_RECORDS_PER_RUN = 1_024


@dataclass(frozen=True)
class RunSet:
    """Sorted runs on disk plus the key they are sorted by.

    The key travels with the files because §2 forbids treating one merge view
    as another; a caller that merges these runs must present the same key.
    """

    paths: list[Path]
    dtype: Any
    key: tuple[str, ...]
    record_count: int

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)


def write_runs(
    blocks: Iterable[Any],
    work_dir: Path,
    prefix: str,
    key: Sequence[str],
    *,
    buffer_bytes: int = RUN_BUFFER_BYTES,
) -> RunSet:
    """Sort each incoming block by `key` and write it as one run.

    Blocks arrive already bounded — one band's segment, one chunk of a mesh —
    so a run is never larger than the producer's own block, and the sort buffer
    is capped independently. A block bigger than the cap is split, so the cap
    holds even if a caller hands over something large.
    """
    import numpy as np

    root = work_dir / "merge"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    dtype: Any = None

    for block in blocks:
        array = np.asarray(block)
        if array.size == 0:
            continue
        dtype = array.dtype
        per_run = max(int(buffer_bytes // array.dtype.itemsize), MIN_RECORDS_PER_RUN)
        for start in range(0, array.shape[0], per_run):
            piece = array[start : start + per_run]
            order = np.lexsort(tuple(piece[name] for name in reversed(tuple(key))))
            path = root / f"{prefix}-{'_'.join(key)}-{len(paths):05d}.rmrun"
            _write_run(path, piece[order])
            paths.append(path)
            total += int(piece.shape[0])

    if dtype is None:
        dtype = np.dtype([(name, "<i8") for name in key])
    return RunSet(paths=paths, dtype=dtype, key=tuple(key), record_count=total)


def _write_run(path: Path, records: Any) -> None:
    import numpy as np

    with open(path, "wb") as handle:
        handle.write(np.ascontiguousarray(records).tobytes())


def merge_runs(
    runs: RunSet,
    *,
    merge_bytes: int = MERGE_BUFFER_BYTES,
    output_bytes: int = OUTPUT_BLOCK_BYTES,
) -> Iterator[Any]:
    """Yield the runs' records in `runs.key` order, in bounded blocks.

    Block-wise rather than record-wise: each run keeps a bounded slice
    resident, and every record whose key does not exceed the smallest
    still-buffered frontier is safe to emit. That is the standard k-way merge
    frontier argument, done with array operations so the cost is not one Python
    iteration per triangle.
    """
    import numpy as np

    if not runs.paths:
        return
    per_run = max(
        int(merge_bytes // (runs.itemsize * max(len(runs.paths), 1))),
        MIN_RECORDS_PER_RUN,
    )
    out_cap = max(int(output_bytes // runs.itemsize), MIN_RECORDS_PER_RUN)

    readers = [_RunReader(path, runs.dtype, per_run) for path in runs.paths]
    try:
        while True:
            live = [r for r in readers if r.refill()]
            if not live:
                return
            frontier = _frontier(live, runs.key)
            emit = [
                take
                for take in (r.take_upto(frontier, runs.key) for r in live)
                if take.size
            ]
            if not emit:
                # The run holding the frontier always emits its own buffer, so
                # this is unreachable while any run is live. Returning rather
                # than looping means a future change cannot spin here.
                return
            merged = np.concatenate(emit)
            order = np.lexsort(
                tuple(merged[name] for name in reversed(runs.key))
            )
            merged = merged[order]
            for start in range(0, merged.shape[0], out_cap):
                yield merged[start : start + out_cap]
    finally:
        for reader in readers:
            reader.close()


def _frontier(readers: list[_RunReader], key: tuple[str, ...]) -> tuple[int, ...] | None:
    """Largest key every run has already read past, or `None` when all are done.

    A run with unread bytes bounds the merge at the last key it has buffered;
    a run that has reached end of file bounds nothing.
    """
    bounds = [r.last_key(key) for r in readers if not r.exhausted]
    return min(bounds) if bounds else None


class _RunReader:
    """One run file, read in bounded slices."""

    __slots__ = ("_handle", "_dtype", "_per_run", "_buffer", "_offset", "exhausted")

    def __init__(self, path: Path, dtype: Any, per_run: int) -> None:
        import numpy as np

        self._handle = path.open("rb")  # noqa: SIM115 - closed by close()
        self._dtype = dtype
        self._per_run = per_run
        self._buffer: Any = np.empty(0, dtype=dtype)
        self._offset = 0
        self.exhausted = False

    def refill(self) -> bool:
        """Ensure the buffer holds records if any remain. False when spent."""
        import numpy as np

        if self._offset < self._buffer.shape[0]:
            return True
        if self.exhausted:
            return False
        raw = self._handle.read(self._per_run * self._dtype.itemsize)
        if not raw:
            self.exhausted = True
            return False
        self._buffer = np.frombuffer(raw, dtype=self._dtype)
        self._offset = 0
        if len(raw) < self._per_run * self._dtype.itemsize:
            self.exhausted = True
        return True

    def last_key(self, key: tuple[str, ...]) -> tuple[int, ...]:
        record = self._buffer[-1]
        return tuple(int(record[name]) for name in key)

    def take_upto(self, frontier: tuple[int, ...] | None, key: tuple[str, ...]) -> Any:
        """Records from the buffer whose key does not exceed `frontier`."""
        import numpy as np

        available = self._buffer[self._offset :]
        if frontier is None:
            self._offset = self._buffer.shape[0]
            return available
        columns = [available[name] for name in key]
        keep = _not_greater(columns, frontier)
        count = int(np.count_nonzero(keep))
        self._offset += count
        return available[:count]

    def close(self) -> None:
        self._handle.close()


def _not_greater(columns: list[Any], frontier: tuple[int, ...]) -> Any:
    """Mask of records whose composite key is <= `frontier`, lexicographically.

    The columns are already sorted, so the answer is a prefix; it is computed
    as a mask rather than a search because the composite comparison is easier
    to read this way and the block is bounded.
    """
    import numpy as np

    less = np.zeros(columns[0].shape[0], dtype=bool)
    equal = np.ones(columns[0].shape[0], dtype=bool)
    for column, bound in zip(columns, frontier, strict=True):
        less |= equal & (column < bound)
        equal &= column == bound
    result: Any = less | equal
    return result


# ---------------------------------------------------------------------------
# streamed component-area-v1
# ---------------------------------------------------------------------------
