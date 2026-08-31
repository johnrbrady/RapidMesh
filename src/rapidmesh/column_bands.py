"""
Column-major band assembly — row bands from column-grouped E57 streams.

Real structured E57 stores points grouped by `columnIndex` (`groupingByLine`).
`iter_row_bands` requires row-major order and rejects the native stream on its
first chunk. This module assembles the same `RawRowBand` objects without
reordering the whole station: each complete column is sliced for the current
band's row window and concatenated across columns until the lattice width is
covered, then the band is emitted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .e57_reader import RawChunk, RawRowBand


def detect_stream_order(chunk: RawChunk) -> str:
    """Return ``row-major`` or ``column-major`` from one chunk's index order.

    Row-major streams keep ``rowIndex`` non-decreasing; column-major streams
    cycle ``rowIndex`` within each column while ``columnIndex`` advances.
    ``columnIndex`` may decrease at row boundaries in row-major order, so only
    ``rowIndex`` monotonicity is tested.
    """
    import numpy as np

    if "rowIndex" not in chunk.data or "columnIndex" not in chunk.data:
        raise ValueError("stream-order detection requires rowIndex and columnIndex")
    row = np.asarray(chunk.data["rowIndex"], np.int64)
    if row.size < 2:
        return "row-major"
    if np.any(np.diff(row) < 0):
        return "column-major"
    return "row-major"


def iter_column_slices(
    chunks: Iterator[RawChunk],
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield each complete lattice column from a column-major chunk stream."""
    import numpy as np

    pending: dict[str, Any] | None = None
    pending_col: int | None = None

    def merge_slice(col_id: int, data: Mapping[str, Any]) -> None:
        nonlocal pending, pending_col
        if pending is None:
            pending = {field: np.asarray(values).copy() for field, values in data.items()}
            pending_col = col_id
            return
        if pending_col != col_id:
            raise ValueError("column slice order does not match stream order")
        for field, values in data.items():
            pending[field] = np.concatenate(
                (pending[field], np.asarray(values))
            )

    def flush() -> tuple[int, Mapping[str, Any]] | None:
        nonlocal pending, pending_col
        if pending is None or pending_col is None:
            return None
        col_id = pending_col
        data = {field: values.copy() for field, values in pending.items()}
        pending = None
        pending_col = None
        return col_id, data

    for chunk in chunks:
        col = np.asarray(chunk.data["columnIndex"], np.int64)
        if col.size == 0:
            continue
        boundaries = np.concatenate(
            ([0], np.flatnonzero(np.diff(col) != 0) + 1, [col.size])
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            col_id = int(col[start])
            data = {
                field: np.asarray(values)[start:stop].copy()
                for field, values in chunk.data.items()
            }
            if pending is None:
                merge_slice(col_id, data)
                continue
            if col_id == pending_col:
                merge_slice(col_id, data)
                continue
            finished = flush()
            assert finished is not None
            yield finished
            merge_slice(col_id, data)

    finished = flush()
    if finished is not None:
        yield finished


def iter_row_bands_column_major(
    chunk_factory: Callable[[], Iterator[RawChunk]],
    *,
    row_min: int,
    row_stop: int,
    col_count: int,
    band_rows: int = 256,
    halo: int = 1,
) -> Iterator[RawRowBand]:
    """Group a column-major raw stream into bounded, overlapping row bands.

    A column-major E57 visits each lattice column once for the whole station.
    Each row band therefore needs a fresh pass over the column stream; memory
    stays proportional to ``band_rows`` (plus halo), not station height.
    """
    import numpy as np

    if band_rows <= 0:
        raise ValueError("band_rows must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if row_stop < row_min:
        raise ValueError("row_stop must not precede row_min")
    if col_count <= 0:
        raise ValueError("col_count must be positive")

    core_start = row_min
    while core_start < row_stop:
        core_stop = min(core_start + band_rows, row_stop)
        data_start = max(row_min, core_start - halo)
        data_stop = min(row_stop, core_stop + halo)
        band_parts: dict[str, list[Any]] = {}
        points_buffered = 0
        columns_seen = 0

        for _col_idx, col_data in iter_column_slices(chunk_factory()):
            row = np.asarray(col_data["rowIndex"], np.int64)
            pick = (row >= data_start) & (row < data_stop)
            points_buffered += int(row.size)
            columns_seen += 1
            if bool(np.any(pick)):
                for field, values in col_data.items():
                    band_parts.setdefault(field, []).append(
                        np.asarray(values)[pick].copy()
                    )

        if columns_seen != col_count:
            raise ValueError(
                "column-major stream ended before every lattice column arrived"
            )

        if band_parts:
            yield RawRowBand(
                core_row_start=core_start,
                core_row_stop=core_stop,
                data_row_start=data_start,
                data_row_stop=data_stop,
                data={
                    field: np.concatenate(parts)
                    for field, parts in band_parts.items()
                },
                buffered_points_before_emit=points_buffered,
            )
        core_start = core_stop
