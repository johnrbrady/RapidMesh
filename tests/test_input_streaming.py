"""
Input streaming — DEC-009 step 2, PLAN.md §5 items 6–8 driven from a stream.

The claim is that the whole scan is never resident during Pass A, and the risk
is that streaming quietly changes the answer. So the tests come in two kinds:

* **equality** — the streamed path must produce what the resident path
  produced, band for band and mesh for mesh, at chunk sizes chosen to split
  reads in awkward places; and
* **liveness** — `chunk_points` must actually do something. Until this package
  it was recorded in the diagnostics and otherwise ignored, so a test asserting
  "output does not move when chunk_points moves" was passing for the wrong
  reason. `test_chunk_size_actually_changes_the_read_pattern` pins that the
  axis is real before the invariance tests are allowed to mean anything.

The E57 tests write their own file into a temporary directory with `pye57`.
No reference-corpus data is read and none is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.e57_reader import iter_row_bands, read_scan
from rapidmesh.filters import iter_clean_bands
from rapidmesh.streaming import (
    StationMetadata,
    band_to_scan,
    iter_band_filter_results,
    scan_band_stream,
    scan_to_chunks,
)
from rapidmesh.streaming_e57 import (
    FramePolicy,
    UnsupportedStreamingInput,
    e57_band_filter_results,
    e57_band_to_scan,
    e57_chunks,
    e57_station_metadata,
)


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=24, cols=180,
        dropout=0.02, range_noise=0.002, seed=7, station_id="stream",
    )


@pytest.fixture(scope="module")
def e57_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Any]:
    """A structured E57 written here, so the production path is exercised.

    Carries `rowIndex`/`columnIndex` and a non-identity pose is deliberately
    *not* used: the frame heuristic needs a posed scan to have anything to
    decide, and this fixture's job is the ordinary case.
    """
    import pye57

    syn = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=24, cols=180,
        dropout=0.02, range_noise=0.002, seed=11, station_id="e57",
    )
    scan = syn.scan
    path = str(tmp_path_factory.mktemp("e57") / "station.e57")
    writer = pye57.E57(path, mode="w")
    writer.write_scan_raw(
        {
            "cartesianX": scan.xyz[:, 0].astype(np.float64),
            "cartesianY": scan.xyz[:, 1].astype(np.float64),
            "cartesianZ": scan.xyz[:, 2].astype(np.float64),
            "rowIndex": scan.row.astype(np.int64),
            "columnIndex": scan.col.astype(np.int64),
            "cartesianInvalidState": np.zeros(len(scan), np.int64),
        },
        name="station",
    )
    del writer
    return path, scan


# ---------------------------------------------------------------------------
# liveness: chunk_points must be a real axis
# ---------------------------------------------------------------------------


def test_chunk_size_actually_changes_the_read_pattern(
    station: synthetic.SyntheticScan,
) -> None:
    """Before this package `chunk_points` was recorded and never used, so the
    harness's invariance test was green for the wrong reason. This asserts the
    axis is real: different capacities produce different numbers of chunks and
    different boundaries."""
    sizes = [
        [chunk.count for chunk in scan_to_chunks(station.scan, points)]
        for points in (137, 1_000, 250_000)
    ]
    assert len(sizes[0]) > len(sizes[1]) > len(sizes[2]) == 1
    assert all(sum(counts) == len(station.scan) for counts in sizes)
    assert max(sizes[0]) == 137 and max(sizes[1]) == 1_000


def test_bands_are_assembled_identically_at_every_chunk_size(
    station: synthetic.SyntheticScan,
) -> None:
    """The two axes are separate: `chunk_points` decides where reads split,
    `band_rows` decides what a band owns, and `iter_row_bands` reconciles them
    (`PHASE1-HALO-CALCULUS.md` §6). Moving one must not move the other."""
    metadata = StationMetadata.from_scan(station.scan)
    reference = None
    for points in (61, 337, 5_000, 250_000):
        bands = [
            (band.core_row_start, band.core_row_stop,
             band.data_row_start, band.data_row_stop,
             len(band_to_scan(band, metadata)))
            for band in iter_row_bands(
                scan_to_chunks(station.scan, points), row_min=0,
                row_stop=station.scan.lattice.rows, band_rows=5, halo=3,
            )
        ]
        if reference is None:
            reference = bands
            assert len(bands) >= 4          # a one-band fixture proves nothing
        assert bands == reference, points


def test_a_partial_last_row_is_not_emitted_as_a_complete_core(
    station: synthetic.SyntheticScan,
) -> None:
    """The defect this package found in `iter_row_bands`.

    Readiness used to fire when the last needed row had merely *started*
    arriving. For interior bands that only truncated a halo row the filter
    chain never reads; for the **final** band the last needed row is its own
    core, so the core was emitted partial and the rest of that row was dropped
    once `core_start` had advanced past it. Every core row must be complete.
    """
    metadata = StationMetadata.from_scan(station.scan)
    rows = station.scan.lattice.rows
    for points in (37, 61, 500):
        owned: dict[int, int] = {}
        for band in iter_row_bands(
            scan_to_chunks(station.scan, points), row_min=0, row_stop=rows,
            band_rows=5, halo=3,
        ):
            band_scan = band_to_scan(band, metadata)
            core = (band_scan.row >= band.core_row_start) & (
                band_scan.row < band.core_row_stop
            )
            for row in np.asarray(band_scan.row)[core]:
                owned[int(row)] = owned.get(int(row), 0) + 1
        expected = {
            int(r): int(c)
            for r, c in zip(*np.unique(station.scan.row, return_counts=True), strict=True)
        }
        assert owned == expected, points


# ---------------------------------------------------------------------------
# equality with the resident path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_points", (43, 613, 250_000))
@pytest.mark.parametrize("band_rows", (3, 7))
def test_streamed_bands_equal_the_resident_filter(
    station: synthetic.SyntheticScan, chunk_points: int, band_rows: int
) -> None:
    """Same cores, same retained samples, same counts as `iter_clean_bands`
    driven from a whole scan — the filter implementation is shared, so this is
    checking the assembly around it."""
    resident = list(
        iter_clean_bands(station.scan, None, True, band_rows=band_rows, halo=3)
    )
    streamed = list(
        scan_band_stream(
            station.scan, chunk_points=chunk_points, others=None,
            despeckle=True, band_rows=band_rows, halo=3,
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
        assert left.dropped_despeckle == right.dropped_despeckle
        assert left.dropped_mover_carve == right.dropped_mover_carve


def test_the_metadata_handle_carries_what_pass_b_reads(
    station: synthetic.SyntheticScan,
) -> None:
    """Pass B takes a scan and reads only station-level facts from it. The
    handle must answer those and must not pretend to carry points."""
    metadata = StationMetadata.from_scan(station.scan)
    handle = metadata.handle_scan()
    assert len(handle) == 0
    assert handle.lattice == station.scan.lattice
    assert handle.pose is station.scan.pose
    assert handle.station_id == station.scan.station_id
    assert handle.source_sample_count == station.scan.source_sample_count
    assert (handle.rgb is None) == (station.scan.rgb is None)
    assert (handle.sample_id is None) == (station.scan.sample_id is None)


# ---------------------------------------------------------------------------
# the E57 production path
# ---------------------------------------------------------------------------


def test_streamed_e57_metadata_matches_a_whole_scan_read(
    e57_file: tuple[str, Any],
) -> None:
    """The lattice is the one `read_scan` would have produced.

    It is derived from a bounded strided subsample selected by valid-ordinal —
    the same set `idx[::stride]` picks out of the whole-scan valid index array
    — so this is an equality, not an approximation. Exact float equality is
    asserted because an angular step that drifts moves every incidence
    threshold in the pipeline with it.
    """
    path, _scan = e57_file
    metadata, policy = e57_station_metadata(path)
    whole = read_scan(path)

    assert metadata.lattice.rows == whole.lattice.rows
    assert metadata.lattice.cols == whole.lattice.cols
    assert metadata.lattice.az_step == whole.lattice.az_step
    assert metadata.lattice.el_step == whole.lattice.el_step
    assert metadata.lattice.az0 == whole.lattice.az0
    assert metadata.lattice.el0 == whole.lattice.el0
    assert metadata.lattice.source == whole.lattice.source
    assert isinstance(policy, FramePolicy)
    assert policy.shift_colour is False       # this export carries no colour


def test_streamed_e57_bands_reconstruct_the_scan_exactly(
    e57_file: tuple[str, Any],
) -> None:
    """Concatenating the cores reproduces `read_scan`'s samples, bit for bit.

    Positions are float32 and compared as bit patterns: a band converter that
    took a different route to the same coordinate would differ here even if it
    looked right.
    """
    path, _scan = e57_file
    metadata, policy = e57_station_metadata(path)
    whole = read_scan(path)

    parts = list(
        e57_band_filter_results(
            path, metadata, policy, chunk_points=997, despeckle=False,
            band_rows=6, halo=3,
        )
    )
    assert len(parts) >= 3
    rows = np.concatenate([p.retained.row for p in parts])
    cols = np.concatenate([p.retained.col for p in parts])
    xyz = np.concatenate([p.retained.xyz for p in parts])
    ids = np.concatenate([p.retained.sample_id for p in parts])

    assert np.array_equal(rows, whole.row)
    assert np.array_equal(cols, whole.col)
    assert np.array_equal(xyz.view(np.uint32), whole.xyz.view(np.uint32))
    assert np.array_equal(ids, whole.sample_id)


def test_streamed_e57_meshes_identically_to_the_resident_path(
    e57_file: tuple[str, Any],
) -> None:
    """End to end: the same station, from disk and from memory, compared with
    `compare_mesh_results` — exact SPEC §7, never `allclose`."""
    from rapidmesh.e57_reader import read_scan_with_frame_path
    from rapidmesh.equivalence import compare_mesh_results
    from rapidmesh.evidence import FRAME_PATH_UNRECORDED
    from rapidmesh.pipeline import mesh_station_from_chunks, mesh_station_streamed

    path, _scan = e57_file
    metadata, policy = e57_station_metadata(path)
    whole, frame_path = read_scan_with_frame_path(path)
    # Both runs read the same file, so both took the same frame branch. The
    # resident path has to be *told* it, because `StructuredScan` cannot carry
    # it; the streamed path decided it in `e57_station_metadata`. That the two
    # agree is part of what this test proves.
    assert frame_path == metadata.frame_path
    # And it is a real branch name, not the "we never looked" placeholder.
    assert frame_path != FRAME_PATH_UNRECORDED

    resident = mesh_station_streamed(
        whole, band_rows=6, chunk_points=997, halo=3, measure=True,
        measure_samples=2_000, frame_path=frame_path,
    )
    from_disk = mesh_station_from_chunks(
        e57_chunks(path, chunk_points=997),
        metadata,
        band_rows=6, chunk_points=997, halo=3, measure=True,
        measure_samples=2_000,
        converter=lambda band, meta: e57_band_to_scan(band, meta, policy),
    )
    assert compare_mesh_results(resident, from_disk, source_sha256="a" * 64).ok
    assert from_disk.mesh.triangle_count > 0
    # The counts came from the metadata pass, not from a resident scan.
    assert metadata.source_sample_count == whole.source_sample_count
    assert metadata.dropped_no_return == whole.dropped_no_return
    assert from_disk.stats.input_points == resident.stats.input_points


def test_a_scan_without_a_row_column_lattice_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A deferred capability that fails silently is a claimed one.

    The spherical and reprojected tiers derive their lattice from the whole
    scan's angle histogram, which a band cannot reproduce. Streaming refuses
    them and names what is missing rather than guessing a lattice per band.
    """
    import pye57

    syn = synthetic.generate(
        synthetic.RoomScene(mover=False), rows=12, cols=90, dropout=0.0,
        range_noise=0.0, seed=3, station_id="nolattice",
    )
    scan = syn.scan
    path = str(tmp_path / "nolattice.e57")
    writer = pye57.E57(path, mode="w")
    writer.write_scan_raw(
        {
            "cartesianX": scan.xyz[:, 0].astype(np.float64),
            "cartesianY": scan.xyz[:, 1].astype(np.float64),
            "cartesianZ": scan.xyz[:, 2].astype(np.float64),
            "cartesianInvalidState": np.zeros(len(scan), np.int64),
        },
        name="nolattice",
    )
    del writer

    with pytest.raises(UnsupportedStreamingInput, match="rowIndex/columnIndex"):
        e57_station_metadata(path)


# ---------------------------------------------------------------------------
# the driver holds nothing per station
# ---------------------------------------------------------------------------


def test_the_sweep_holds_one_band_at_a_time(
    station: synthetic.SyntheticScan,
) -> None:
    """The memory property, as a structural observable.

    Each band's points are handed over and dropped: the driver keeps no list of
    bands and the producer is a generator, so at any moment the only band scan
    reachable is the one being filtered. A change that accumulated bands would
    make more than one reachable here.
    """
    import gc
    import weakref

    metadata = StationMetadata.from_scan(station.scan)
    alive: list[weakref.ref[Any]] = []
    live_at_each_step: list[int] = []

    def tracking(band: Any, meta: StationMetadata) -> Any:
        gc.collect()
        live_at_each_step.append(sum(1 for ref in alive if ref() is not None))
        band_scan = band_to_scan(band, meta)
        alive.append(weakref.ref(band_scan))
        return band_scan

    results = list(
        iter_band_filter_results(
            scan_to_chunks(station.scan, 1_000), metadata, despeckle=True,
            band_rows=5, halo=3, converter=tracking,
        )
    )
    assert len(results) >= 4
    assert len(alive) == len(results)
    # The previous band's scan is unreachable by the time the next is built.
    assert live_at_each_step == [0] * len(alive), live_at_each_step
