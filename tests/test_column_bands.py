"""
Column-major E57 band assembly — Round 4b gates CM-3 and CM-4.

The production path must accept the native column-grouped stream that every
authorised structured E57 carries, without reordering the whole station.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.column_bands import detect_stream_order, iter_row_bands_column_major
from rapidmesh.e57_reader import iter_row_bands
from rapidmesh.streaming import (
    StationMetadata,
    band_to_scan,
    iter_band_filter_results,
    scan_to_chunks,
    scan_to_chunks_column_major,
)


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=24, cols=180,
        dropout=0.0, range_noise=0.002, seed=17, station_id="col-major",
    )


def test_column_major_stream_is_detected(station: synthetic.SyntheticScan) -> None:
    chunk = next(scan_to_chunks_column_major(station.scan, chunk_points=997))
    assert detect_stream_order(chunk) == "column-major"


def test_row_major_stream_is_detected(station: synthetic.SyntheticScan) -> None:
    chunk = next(scan_to_chunks(station.scan, chunk_points=997))
    assert detect_stream_order(chunk) == "row-major"


def test_iter_row_bands_rejects_column_major_chunks(
    station: synthetic.SyntheticScan,
) -> None:
    """Required-red: the old path must still reject column-major input."""
    with pytest.raises(ValueError, match="not row-major"):
        list(
            iter_row_bands(
                scan_to_chunks_column_major(station.scan, chunk_points=997),
                row_min=0,
                row_stop=station.scan.lattice.rows,
                band_rows=5,
                halo=3,
            )
        )


def test_column_major_bands_match_row_major_reference(
    station: synthetic.SyntheticScan,
) -> None:
    """Column assembly must emit the same bands as row-major reconciliation."""
    metadata = StationMetadata.from_scan(station.scan)
    rows = station.scan.lattice.rows
    cols = station.scan.lattice.cols

    def summarise(bands: Any) -> list[tuple[int, int, int, int, int]]:
        return [
            (
                band.core_row_start,
                band.core_row_stop,
                band.data_row_start,
                band.data_row_stop,
                len(band_to_scan(band, metadata)),
            )
            for band in bands
        ]

    reference = summarise(
        iter_row_bands(
            scan_to_chunks(station.scan, 61),
            row_min=0,
            row_stop=rows,
            band_rows=5,
            halo=3,
        )
    )
    assembled = summarise(
        iter_row_bands_column_major(
            lambda: scan_to_chunks_column_major(station.scan, 61),
            row_min=0,
            row_stop=rows,
            col_count=cols,
            band_rows=5,
            halo=3,
        )
    )
    assert assembled == reference
    assert len(assembled) >= 4


@pytest.mark.parametrize("chunk_points", (43, 613, 997))
@pytest.mark.parametrize("band_rows", (3, 7))
def test_streamed_column_major_bands_equal_the_resident_filter(
    station: synthetic.SyntheticScan, chunk_points: int, band_rows: int
) -> None:
    """The driver must auto-detect column-major order and keep filter output."""
    from rapidmesh.filters import iter_clean_bands

    metadata = StationMetadata.from_scan(station.scan)
    resident = list(
        iter_clean_bands(station.scan, None, True, band_rows=band_rows, halo=3)
    )
    streamed = list(
        iter_band_filter_results(
            scan_to_chunks_column_major(station.scan, chunk_points),
            metadata,
            others=None,
            despeckle=True,
            band_rows=band_rows,
            halo=3,
            chunk_factory=lambda: scan_to_chunks_column_major(
                station.scan, chunk_points
            ),
        )
    )
    assert len(streamed) == len(resident) >= 3
    for left, right in zip(resident, streamed, strict=True):
        assert left.plan == right.plan
        assert np.array_equal(left.retained.row, right.retained.row)
        assert np.array_equal(left.retained.col, right.retained.col)
        assert np.array_equal(
            left.retained.xyz.view(np.uint32), right.retained.xyz.view(np.uint32)
        )
