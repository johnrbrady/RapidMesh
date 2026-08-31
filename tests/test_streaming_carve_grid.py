"""
Streaming carve-grid build, row-blocked hole closing and the cached artefact —
PLAN.md §5 item 7, designed in `PHASE1-TILE-CONTRACT-V0.md` §6.2, §6.3 and §7.

Two of these tests are equivalence tests and the standard is **bitwise**, not
`allclose`. A carve grid decides which survey returns get deleted from a
neighbour, so "nearly the same grid" is a different filter with a different
false-positive rate, and no downstream number would reveal the substitution.

Three tests exist only to prove the others can fail. `test_..._is_detected`
runs a deliberately broken variant — a chunk boundary that drops samples, an
in-place hole-closing sweep, a corrupted payload — and asserts the comparison
rejects it. A tolerance that never rejects is not a tolerance.
"""

from __future__ import annotations

import dataclasses
import gc
import math
import os
import tracemalloc
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import carvegrid as cg
from rapidmesh import carvegrid_io as cgio
from rapidmesh import synthetic
from rapidmesh.filters import clean
from rapidmesh.grid import CARVE_MAX_CELLS, CoarseRangeGrid
from rapidmesh.pipeline import carve_grids, carve_grids_streamed
from rapidmesh.types import LatticeInfo, LatticeSource, ScanPose, StructuredScan

# Small enough to keep the suite quick, large enough that a chunked build with
# `chunk_points` well below the sample count is genuinely exercised.
ROWS, COLS = 200, 800

# The worst lattice in the committed reference metadata
# (`PHASE1-TILE-CONTRACT-V0.md` §6.1). Dimensions only; no client identifier.
WORST_ROWS, WORST_COLS = 2387, 6096


def _bits(a: Any) -> Any:
    """Reinterpret f32 as u32 so `array_equal` means *bitwise* equal."""
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _same_grid(a: CoarseRangeGrid, b: CoarseRangeGrid) -> bool:
    return (
        a.width == b.width
        and a.height == b.height
        and np.array_equal(np.asarray(a.origin), np.asarray(b.origin))
        and np.array_equal(_bits(a.rng), _bits(b.rng))
    )


@pytest.fixture(scope="module")
def station_a() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(scanner=(0.35, -0.2, 0.0), mover=True),
        rows=ROWS, cols=COLS, station_id="A", seed=7,
    )


@pytest.fixture(scope="module")
def station_b() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(scanner=(-2.6, 1.7, 0.0), mover=False),
        rows=ROWS, cols=COLS, station_id="B", seed=11,
    )


@pytest.fixture(scope="module")
def station_c() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(scanner=(2.7, 1.4, 0.0), mover=False),
        rows=ROWS, cols=COLS, station_id="C", seed=13,
    )


@pytest.fixture(scope="module")
def rotated_scan() -> StructuredScan:
    """A non-identity rotation, so the pose matmul is genuinely exercised."""
    angle = 0.7
    pose = ScanPose(
        translation=np.array([12.5, -3.25, 1.75]),
        rotation=np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]),
    )
    return synthetic.generate(
        synthetic.RoomScene(scanner=(2.7, 1.4, 0.0), mover=False),
        rows=ROWS, cols=COLS, station_id="R", seed=13, pose=pose,
    ).scan


@pytest.fixture(scope="module")
def big_scan() -> StructuredScan:
    """Large enough that one whole-scan f64 copy is unmistakable in the peak."""
    return synthetic.generate(
        synthetic.RoomScene(scanner=(-2.6, 1.7, 0.0), mover=False),
        rows=500, cols=2000, station_id="BIG", seed=11,
    ).scan


# ---------------------------------------------------------------------------
# chunked build == legacy build, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_points", (3, 1_000, 50_000, 250_000))
@pytest.mark.parametrize("fill_holes", (True, False))
def test_chunked_build_is_bitwise_identical_to_legacy(
    station_b: synthetic.SyntheticScan, chunk_points: int, fill_holes: bool
) -> None:
    """The whole premise of item 7: same grid, bounded memory.

    `chunk_points=3` is in the matrix on purpose — it forces a chunk boundary
    between almost every pair of samples, so any dependence of the result on
    where the boundaries fall shows up immediately.
    """
    scan = station_b.scan
    legacy = CoarseRangeGrid.build(scan, fill_holes=fill_holes)
    chunked = cg.build_grid_chunked(
        scan, fill_holes=fill_holes, chunk_points=chunk_points
    )
    assert _same_grid(chunked, legacy)


def test_chunked_build_is_bitwise_identical_under_a_rotated_pose(
    rotated_scan: StructuredScan,
) -> None:
    """A second scan, with a non-identity rotation in the transform chain."""
    assert not np.allclose(rotated_scan.pose.rotation, np.eye(3))
    legacy = CoarseRangeGrid.build(rotated_scan)
    chunked = cg.build_grid_chunked(rotated_scan, chunk_points=7_000)
    assert _same_grid(chunked, legacy)


@pytest.mark.parametrize(
    ("rows", "cols", "max_cells"),
    (
        (200, 800, CARVE_MAX_CELLS),
        (2, 3, CARVE_MAX_CELLS),
        (WORST_ROWS, WORST_COLS, CARVE_MAX_CELLS),
        (WORST_ROWS, WORST_COLS, 1_000_000),
        (4000, 8000, 2_000_000),
    ),
)
def test_resolution_rule_matches_legacy(rows: int, cols: int, max_cells: int) -> None:
    """`resolve_grid_dimensions` is duplicated from `grid.py`; this is the guard.

    If the two ever diverge, every cached artefact silently claims a cell count
    it does not have, so the drift is caught here rather than in the cache.
    """
    height, width = cg.resolve_grid_dimensions(rows, cols, max_cells)
    lattice = LatticeInfo(
        rows=rows, cols=cols, az_step=1e-3, el_step=1e-3, az0=0.0, el0=0.0,
        source=LatticeSource.SYNTHETIC,
    )
    empty = StructuredScan(
        row=np.zeros(0, np.int32), col=np.zeros(0, np.int32),
        xyz=np.zeros((0, 3), np.float32), rng=np.zeros(0, np.float32),
        pose=ScanPose(translation=np.zeros(3), rotation=np.eye(3)), lattice=lattice,
    )
    legacy = CoarseRangeGrid.build(empty, max_cells=max_cells, fill_holes=False)
    assert (height, width) == (legacy.height, legacy.width)


def test_chunked_grids_carve_identically_in_a_multi_station_run(
    station_a: synthetic.SyntheticScan,
    station_b: synthetic.SyntheticScan,
    station_c: synthetic.SyntheticScan,
) -> None:
    """The consequence that matters: same neighbours, same deletions.

    Bitwise grid equality should imply this, but the filter is what the grid is
    *for*, so it is asserted rather than assumed.
    """
    others_legacy = [
        CoarseRangeGrid.build(station_b.scan), CoarseRangeGrid.build(station_c.scan)
    ]
    others_chunked = [
        cg.build_grid_chunked(station_b.scan, chunk_points=9_000),
        cg.build_grid_chunked(station_c.scan, chunk_points=9_000),
    ]
    ref, ref_stats = clean(station_a.scan, others=others_legacy)
    got, got_stats = clean(station_a.scan, others=others_chunked)

    assert np.array_equal(ref.row, got.row)
    assert np.array_equal(ref.col, got.col)
    assert np.array_equal(_bits(ref.rng), _bits(got.rng))
    assert ref_stats.dropped_mover_carve == got_stats.dropped_mover_carve
    assert ref_stats.restored_from_carve == got_stats.restored_from_carve
    assert ref_stats.dropped_mover_carve > 0  # the comparison is not vacuous


# ---------------------------------------------------------------------------
# row-blocked close_dropout == whole-grid close_dropout, bitwise
# ---------------------------------------------------------------------------


def _legacy_closed(source: Any) -> Any:
    grid = CoarseRangeGrid(
        origin=np.zeros(3), rng=np.array(source, np.float32, copy=True),
        width=int(source.shape[1]), height=int(source.shape[0]),
    )
    grid.close_dropout()
    return grid.rng


def _sparse_grid(height: int, width: int, seed: int, empty_fraction: float) -> Any:
    rng = np.random.default_rng(seed)
    values = rng.uniform(1.0, 40.0, (height, width)).astype(np.float32)
    values[rng.random((height, width)) < empty_fraction] = np.inf
    return values


@pytest.mark.parametrize("block_rows", (1, 2, 7, 64, 10_000))
@pytest.mark.parametrize("empty_fraction", (0.05, 0.6))
def test_row_blocked_close_dropout_is_bitwise_identical(
    block_rows: int, empty_fraction: float
) -> None:
    """Blocking must not change one bit, at any block size including 1 and
    'larger than the grid'. The 0.6 case makes large empty runs common, which
    is where an in-place sweep would visibly propagate."""
    source = _sparse_grid(37, 53, seed=5, empty_fraction=empty_fraction)
    assert np.array_equal(
        _bits(cg.close_dropout_blocked(source, block_rows=block_rows)),
        _bits(_legacy_closed(source)),
    )


def test_row_blocked_close_dropout_matches_on_a_real_scan_grid(
    station_b: synthetic.SyntheticScan,
) -> None:
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    assert np.array_equal(
        _bits(cg.close_dropout_blocked(grid.rng, block_rows=17)),
        _bits(_legacy_closed(grid.rng)),
    )


def test_row_blocked_close_dropout_leaves_its_source_untouched() -> None:
    """Immutable source, distinct output — the §6.3 requirement, asserted."""
    source = _sparse_grid(20, 24, seed=9, empty_fraction=0.4)
    before = source.copy()
    out = cg.close_dropout_blocked(source, block_rows=4)
    assert np.array_equal(_bits(source), _bits(before))
    assert out is not source
    assert not np.shares_memory(out, source)


# ---------------------------------------------------------------------------
# cache key, artefact naming and invalidation (§7)
# ---------------------------------------------------------------------------


def _base_digest_kwargs() -> dict[str, Any]:
    return {
        "max_cells": CARVE_MAX_CELLS,
        "fill_holes": True,
        "rotation": np.eye(3),
        "translation": np.array([1.0, 2.0, 3.0]),
        "lattice_rows": 200,
        "lattice_cols": 800,
        "frame_path": "",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_cells", 8_000_000),
        ("fill_holes", False),
        ("rotation", np.diag([1.0, -1.0, -1.0])),
        ("translation", np.array([1.0, 2.0, 3.0000001])),
        ("lattice_rows", 201),
        ("lattice_cols", 801),
        ("frame_path", "cartesian-fallback"),
    ),
)
def test_every_params_digest_input_changes_the_key(field: str, value: Any) -> None:
    """§9 falsification C: an effective-transform or frame-resolution change
    that did not change the key would be a silently stale grid."""
    base = cg.carve_params_digest(**_base_digest_kwargs())
    changed = dict(_base_digest_kwargs())
    changed[field] = value
    assert cg.carve_params_digest(**changed) != base


def test_artefact_name_carries_every_key_component() -> None:
    key = cg.CarveGridKey(
        source_identity="a" * 64, scan_index=3, grid_version=cg.GRID_VERSION,
        cells=14_551_152, params_digest="b" * 64,
    )
    name = key.filename()
    assert name == f"{'a' * 64}-3-v{cg.GRID_VERSION}-14551152-{'b' * 64}.rmgrid"


def test_synthetic_identity_is_refused_a_filename() -> None:
    """An in-memory fixture identity must never reach a file, because the
    filename is the whole invalidation mechanism."""
    key = cg.CarveGridKey(
        source_identity="station:B", scan_index=0, grid_version=0, cells=16,
        params_digest="b" * 64,
    )
    assert not key.is_file_nameable
    with pytest.raises(cg.CarveGridArtefactError):
        key.filename()


def _stored(tmp_path: Path, grid: CoarseRangeGrid, digest: str = "b" * 64) -> tuple[
    cgio.DirectoryGridStore, cg.CarveGridKey
]:
    store = cgio.DirectoryGridStore(root=tmp_path / "cg")
    key = cg.CarveGridKey(
        source_identity="a" * 64, scan_index=0, grid_version=cg.GRID_VERSION,
        cells=grid.height * grid.width, params_digest=digest,
    )
    store.store(key, grid)
    return store, key


def test_artefact_round_trip_is_a_hit_and_is_bitwise_identical(
    station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    grid = CoarseRangeGrid.build(station_b.scan)
    store = cgio.DirectoryGridStore(root=tmp_path / "cg")
    key = cg.CarveGridKey(
        source_identity="a" * 64, scan_index=0, grid_version=cg.GRID_VERSION,
        cells=grid.height * grid.width, params_digest="b" * 64,
    )
    assert store.load(key) is None                      # miss before the write
    store.store(key, grid)
    assert _same_grid(store.load(key), grid)            # hit after it


def test_a_key_that_differs_anywhere_is_a_miss(
    station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """Invalidation is by name only: no partial match, no timestamp."""
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    store, key = _stored(tmp_path, grid)
    for changed in (
        dataclasses.replace(key, params_digest="c" * 64),
        dataclasses.replace(key, scan_index=1),
        dataclasses.replace(key, grid_version=key.grid_version + 1),
        dataclasses.replace(key, cells=key.cells + 1),
        dataclasses.replace(key, source_identity="f" * 64),
    ):
        assert store.load(changed) is None


def test_corrupt_payload_fails_closed(
    station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """A flipped payload byte must raise, not silently carve with bad evidence."""
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    store, key = _stored(tmp_path, grid)
    path = store.path_for(key)

    raw = bytearray(path.read_bytes())
    offset = cgio._HEADER.size + 4_096
    raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(cg.CarveGridArtefactError, match="payload digest"):
        store.load(key)


def test_truncated_artefact_fails_closed(
    station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    store, key = _stored(tmp_path, grid)
    path = store.path_for(key)
    path.write_bytes(path.read_bytes()[: cgio._HEADER.size + 1_024])
    with pytest.raises(cg.CarveGridArtefactError, match="truncated"):
        store.load(key)


def test_renamed_artefact_fails_closed(
    station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """The header repeats the key, so a file moved onto another key's name is
    caught instead of being served as that key's grid."""
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    store, key = _stored(tmp_path, grid)
    other = dataclasses.replace(key, params_digest="c" * 64)
    store.path_for(other).write_bytes(store.path_for(key).read_bytes())
    with pytest.raises(cg.CarveGridArtefactError, match="does not match its name"):
        store.load(other)


def test_write_is_temp_then_replace_inside_the_destination_directory(
    station_b: synthetic.SyntheticScan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5.1: `os.replace`, with the temporary in the destination directory —
    `os.replace` is only atomic within one volume, and `os.rename` over an
    existing file raises on Windows."""
    grid = CoarseRangeGrid.build(station_b.scan, fill_holes=False)
    seen: list[tuple[Path, Path]] = []
    real = os.replace

    def spy(src: Any, dst: Any) -> None:
        seen.append((Path(src), Path(dst)))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    store, key = _stored(tmp_path, grid)

    assert len(seen) == 1
    src, dst = seen[0]
    assert src.parent == dst.parent == store.root
    assert src.name.endswith(".tmp")
    assert sorted(p.name for p in store.root.iterdir()) == [key.filename()]


# ---------------------------------------------------------------------------
# one-at-a-time orchestration
# ---------------------------------------------------------------------------


def _ref(scan: StructuredScan, log: list[str] | None = None) -> cg.StationRef:
    def load() -> StructuredScan:
        if log is not None:
            log.append(scan.station_id)
        return scan

    return dataclasses.replace(cg.StationRef.from_scan(scan), load=load)


def test_carve_grids_matches_the_legacy_orchestration_bitwise(
    station_a: synthetic.SyntheticScan,
    station_b: synthetic.SyntheticScan,
    station_c: synthetic.SyntheticScan,
) -> None:
    """`carve_grids` keeps its contract: same neighbours, same order, same bits.

    C is the nearer of the two setups to A, so the expected order is C then B.
    """
    scans = [station_a.scan, station_b.scan, station_c.scan]
    legacy = [CoarseRangeGrid.build(station_c.scan), CoarseRangeGrid.build(station_b.scan)]
    got = carve_grids(scans, exclude="A", nearest=2)
    assert len(got) == 2
    assert all(_same_grid(g, ref) for g, ref in zip(got, legacy, strict=True))
    assert carve_grids(scans, exclude="absent") == []


def test_selection_reads_origins_only_and_never_loads_the_rest(
    station_a: synthetic.SyntheticScan,
    station_b: synthetic.SyntheticScan,
    station_c: synthetic.SyntheticScan,
) -> None:
    """Selection from station origins is what lets a caller hold the whole
    project's metadata without holding anyone's points."""
    loaded: list[str] = []
    refs = [_ref(s.scan, loaded) for s in (station_a, station_b, station_c)]
    grids = carve_grids_streamed(refs, exclude="A", nearest=1, chunk_points=20_000)

    assert loaded == ["C"]  # B is ranked second and never loaded
    assert _same_grid(grids[0], CoarseRangeGrid.build(station_c.scan))


def test_only_one_neighbour_is_resident_at_a_time(
    station_a: synthetic.SyntheticScan, station_b: synthetic.SyntheticScan
) -> None:
    """The memory-safety property of the orchestration, as an observable.

    Each loader mints a fresh scan and keeps only a weak reference to it. By
    the time the next neighbour is loaded, the previous one must already be
    unreachable — if `build_or_load` held onto it, or a grid kept a view into
    its points, the weak reference would still be alive and this fails.
    """
    alive: list[weakref.ref[StructuredScan]] = []
    still_alive_at_load: list[int] = []

    def make(scan: StructuredScan) -> cg.StationRef:
        def load() -> StructuredScan:
            gc.collect()
            still_alive_at_load.append(sum(1 for r in alive if r() is not None))
            fresh = dataclasses.replace(scan)
            alive.append(weakref.ref(fresh))
            return fresh

        return dataclasses.replace(cg.StationRef.from_scan(scan), load=load)

    b = station_b.scan
    c = dataclasses.replace(b, station_id="C2")
    refs = [_ref(station_a.scan), make(b), make(c)]
    carve_grids_streamed(refs, exclude="A", nearest=2, chunk_points=20_000)

    assert len(alive) == 2
    assert still_alive_at_load == [0, 0]
    gc.collect()
    assert [r() for r in alive] == [None, None]


def test_directory_store_round_trips_through_the_orchestration(
    station_a: synthetic.SyntheticScan, station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """The production path end to end: build once, publish the artefact, hit it.

    Uses a real source digest, because a file-backed cache is named by one.
    """
    store = cgio.DirectoryGridStore(root=tmp_path / "cg", lattice_rows=ROWS, lattice_cols=COLS)
    neighbour = dataclasses.replace(_ref(station_b.scan), source_identity="9" * 64)
    refs = [_ref(station_a.scan), neighbour]

    first = carve_grids_streamed(refs, exclude="A", nearest=1, store=store, chunk_points=20_000)
    written = sorted(p.name for p in store.root.iterdir())
    assert written == [neighbour.grid_key().filename()]

    def explode() -> StructuredScan:
        raise AssertionError("a cache hit must not load the scan's points")

    hot = [refs[0], dataclasses.replace(neighbour, load=explode)]
    second = carve_grids_streamed(hot, exclude="A", nearest=1, store=store, chunk_points=20_000)
    assert _same_grid(second[0], first[0])
    assert _same_grid(second[0], CoarseRangeGrid.build(station_b.scan))


def test_a_file_cache_refuses_a_scan_with_no_source_digest(
    station_a: synthetic.SyntheticScan, station_b: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """`StationRef.from_scan` mints an in-memory identity, which cannot name a
    file. Refusing is deliberate: skipping the cache silently would hide the
    missing digest, and a grid filed under a made-up identity is exactly the
    stale-cache failure §7 exists to prevent."""
    store = cgio.DirectoryGridStore(root=tmp_path / "cg")
    refs = [_ref(station_a.scan), _ref(station_b.scan)]
    with pytest.raises(cg.CarveGridArtefactError, match="sha256 source digest"):
        carve_grids_streamed(refs, exclude="A", nearest=1, store=store)


def test_a_cache_hit_never_loads_the_points(
    station_b: synthetic.SyntheticScan, station_a: synthetic.SyntheticScan
) -> None:
    """The key is computable from pose and lattice alone, so a hit costs no I/O."""
    store = cgio.MemoryGridStore()
    refs = [_ref(station_a.scan), _ref(station_b.scan)]
    first = carve_grids_streamed(refs, exclude="A", nearest=1, store=store, chunk_points=20_000)
    assert len(store) == 1

    def explode() -> StructuredScan:
        raise AssertionError("a cache hit must not load the scan's points")

    hot = [refs[0], dataclasses.replace(refs[1], load=explode)]
    second = carve_grids_streamed(hot, exclude="A", nearest=1, store=store, chunk_points=20_000)
    assert _same_grid(second[0], first[0])


# ---------------------------------------------------------------------------
# memory: the claim item 7 exists to make true
# ---------------------------------------------------------------------------


class _WidthSpyPose(ScanPose):
    """Records the length of every array handed to the pose transform."""

    def __init__(self, pose: ScanPose, widths: list[int]) -> None:
        super().__init__(translation=pose.translation, rotation=pose.rotation)
        object.__setattr__(self, "widths", widths)

    def rotate_local(self, local: Any) -> Any:
        self.widths.append(int(np.asarray(local).shape[0]))  # type: ignore[attr-defined]
        return super().rotate_local(local)


def test_chunked_build_never_transforms_more_than_one_chunk(
    station_b: synthetic.SyntheticScan,
) -> None:
    """The direct observable for "no whole-scan f64 xyz".

    `rotate_local` starts with `np.asarray(local, float64)`, so the f64 copy is
    exactly as long as its argument. Recording every argument length therefore
    measures the largest f64 transform buffer the build ever allocates. The
    legacy call is measured with the same spy to prove the observable fires.
    """
    chunk = 20_000
    widths: list[int] = []
    scan = dataclasses.replace(
        station_b.scan, pose=_WidthSpyPose(station_b.scan.pose, widths)
    )
    n = len(scan)
    assert n > 4 * chunk

    cg.build_grid_chunked(scan, chunk_points=chunk)
    assert widths and max(widths) <= chunk
    assert sum(widths) == n            # every sample still binned exactly once

    widths.clear()
    CoarseRangeGrid.build(scan)
    assert max(widths) == n            # the legacy path the spy must catch


def test_chunked_build_peak_stays_below_one_whole_scan_f64_copy(
    big_scan: StructuredScan,
) -> None:
    """Measured with `tracemalloc`, which tracks NumPy's data allocator.

    The bound is one whole-scan f64 xyz copy, `len(scan) * 3 * 8` bytes: the
    chunked build must stay under it while the legacy build, which holds
    several such copies at once, must exceed it. Raising `chunk_points` toward
    the scan size legitimately raises the peak — the claim is that the peak is
    a function of the chunk, not of the scan.
    """
    n = len(big_scan)
    whole_scan_f64 = n * 3 * 8

    def peak_of(build: Any) -> int:
        tracemalloc.start()
        try:
            build()
            return int(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()

    chunked = peak_of(lambda: cg.build_grid_chunked(big_scan, chunk_points=50_000))
    legacy = peak_of(lambda: CoarseRangeGrid.build(big_scan))

    assert chunked < whole_scan_f64, (chunked, whole_scan_f64)
    assert legacy > 2 * whole_scan_f64, (legacy, whole_scan_f64)


def test_row_blocked_close_dropout_peak_is_one_output_grid_plus_blocks(
    big_scan: StructuredScan,
) -> None:
    """§6.3's replacement bound: one distinct output plus block temporaries,
    against the 5-to-7 whole-grid arrays the legacy sweep holds."""
    source = CoarseRangeGrid.build(big_scan, fill_holes=False).rng
    grid_bytes = int(source.size) * 4
    block_rows = 64
    block_bytes = block_rows * int(source.shape[1]) * 4

    def peak_of(run: Any) -> int:
        tracemalloc.start()
        try:
            run()
            return int(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()

    blocked = peak_of(lambda: cg.close_dropout_blocked(source, block_rows=block_rows))
    legacy = peak_of(lambda: _legacy_closed(source))

    assert blocked < grid_bytes + 8 * block_bytes, (blocked, grid_bytes, block_bytes)
    assert legacy > 4 * grid_bytes, (legacy, grid_bytes)


def test_carve_grid_resident_bytes_budget() -> None:
    """The §6.1 B1 line, as arithmetic. Not a measurement, and not claimed as
    one; the measured gate is PLAN.md §5 item 10."""
    assert CARVE_MAX_CELLS * 4 == 64_000_000
    assert 2 * CARVE_MAX_CELLS * 4 == 128_000_000

    height, width = cg.resolve_grid_dimensions(WORST_ROWS, WORST_COLS)
    assert (height, width) == (WORST_ROWS, WORST_COLS)   # no reduction applies
    assert height * width == 14_551_152
    assert height * width * 4 == 58_204_608
    assert 2 * height * width * 4 == 116_409_216


# ---------------------------------------------------------------------------
# sensitivity: the tests above must be able to go red
# ---------------------------------------------------------------------------


def test_a_chunk_boundary_that_drops_samples_is_detected(
    station_b: synthetic.SyntheticScan,
) -> None:
    """An off-by-one at the chunk boundary is the obvious way to break this,
    and it does not raise anywhere — it just bins fewer samples."""
    scan = station_b.scan
    height, width = cg.resolve_grid_dimensions(scan.lattice.rows, scan.lattice.cols)
    flat = np.full(height * width, np.inf, np.float32)
    chunk = 1_000
    for start in range(0, len(scan), chunk):
        stop = min(start + chunk, len(scan))
        cg._accumulate_chunk(  # last sample of each chunk silently skipped
            flat, scan.pose, scan.xyz[start : stop - 1],
            np.asarray(scan.pose.translation, np.float64), width, height,
        )
    broken = flat.reshape(height, width)
    legacy = CoarseRangeGrid.build(scan, fill_holes=False)
    assert not np.array_equal(_bits(broken), _bits(legacy.rng))


def test_an_in_place_hole_closing_sweep_is_detected() -> None:
    """The forbidden variant of §6.3, shown producing a different answer.

    Row 0 has returns and rows 1 to 3 are empty. The whole-grid operation fills
    row 1 only — one cell of reach. An in-place blocked sweep writes row 1 back
    into the array it is about to read, so block two sees a filled row 1 and
    propagates the fill into row 2, carving further than any evidence supports.
    """
    source = np.full((4, 4), np.inf, np.float32)
    source[0, :] = [7.0, 8.0, 9.0, 10.0]
    legacy = _legacy_closed(source)

    scratch = source.copy()
    for first in range(0, 4, 2):
        block = cg.close_dropout_blocked(scratch, block_rows=4)[first : first + 2]
        scratch[first : first + 2] = block          # writes into its own source

    assert np.array_equal(_bits(cg.close_dropout_blocked(source, block_rows=2)), _bits(legacy))
    assert np.isinf(legacy[2]).all()                # the reference reaches one row
    assert not np.array_equal(_bits(scratch), _bits(legacy))
