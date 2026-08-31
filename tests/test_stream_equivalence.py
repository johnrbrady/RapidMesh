"""Streamed ≡ in-memory equivalence harness (PLAN.md §5 item 5).

Comparison contract: PHASE1-DETERMINISM-SPEC.md §7.
Halo witness design: PHASE1-HALO-CALCULUS.md §9.

Until PLAN.md §5 items 6–8 existed, the matrix integration tests here were
**required-red**: they had to fail with StreamedMeshingNotImplemented, never
with ImportError, xfail or skip. Items 6, 7 and 8 are now implemented, so the
same tests are required-**green** and assert exact equivalence instead. Two
kept their names and inverted their meaning; one was renamed, because a test
called ``..._required_red`` that asserts a match reads as the opposite of what
it does. The mapping is recorded in the WP-1.4 report.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.equivalence import (
    NORMAL_ANGLE_TOL_RAD,
    REVERSE_QA_STATUS,
    EquivalenceMismatch,
    canonical_oriented_triples,
    compare_deviation_report_t1,
    compare_filter_stats_t1,
    compare_mesh_results,
    compare_normals_t3,
    compare_qa_metadata_t1,
    compare_source_id_triples_t2,
    compare_t1_array,
)
from rapidmesh.pipeline import (
    MeshResult,
    StreamedMeshingNotImplemented,
    mesh_station,
    mesh_station_streamed,
)
from rapidmesh.types import (
    DeviationReport,
    FilterStats,
    MeshData,
    QAReportMetadata,
    ScanPose,
)

# --------------------------------------------------------------------------
# Fixtures small enough to run both ways once item 6 exists
# --------------------------------------------------------------------------

# Force ≥3 bands under the smallest band_rows in the matrix.
HARNESS_ROWS = 12
HARNESS_COLS = 48

# band_rows: one value small enough that 12 rows produce ≥3 bands.
BAND_ROWS_MATRIX = (3, 4, 6)
# chunk_points: chosen so chunk boundaries do not coincide with band
# boundaries for any of the band_rows above (3*48=144, 4*48=192, 6*48=288).
CHUNK_POINTS_MATRIX = (100, 200)


@pytest.fixture
def harness_scan() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True),
        rows=HARNESS_ROWS,
        cols=HARNESS_COLS,
        dropout=0.0,
        range_noise=0.0,
        seed=17,
        station_id="harness",
    )


# --------------------------------------------------------------------------
# Equivalence helper unit tests (SPEC §7 / §9)
# --------------------------------------------------------------------------


def _tiny_mesh(
    *,
    origin: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    source_sample_id: np.ndarray | None = None,
) -> MeshData:
    pose = ScanPose(
        translation=np.zeros(3, np.float64),
        rotation=np.eye(3, dtype=np.float64),
    )
    return MeshData(
        origin=np.zeros(3, np.float64) if origin is None else origin,
        vertices=(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], np.float32)
            if vertices is None
            else vertices
        ),
        triangles=(
            np.array([[0, 1, 2]], np.uint32) if triangles is None else triangles
        ),
        source_pose=pose,
        normals=(
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.float32)
            if normals is None
            else normals
        ),
        rgb=rgb,
        source_sample_id=(
            np.array([10, 20, 30], np.int64)
            if source_sample_id is None
            else source_sample_id
        ),
    )


def _tiny_stats(**overrides: int) -> FilterStats:
    base = dict(
        input_points=10,
        retained=7,
        dropped_no_return=1,
        dropped_despeckle=1,
        dropped_mover_carve=1,
        dropped_island=0,
        dropped_other=0,
        restored_from_carve=0,
    )
    base.update(overrides)
    return FilterStats(**base)


def _tiny_deviation(**overrides: float | int | str | bool) -> DeviationReport:
    base: dict[str, float | int | str | bool] = dict(
        sampled_points=100,
        rms=0.0,
        mean=0.0,
        p95=0.0,
        p99_9=0.0,
        maximum=0.0,
        within_2mm=1.0,
        within_5mm=1.0,
        metric="point-to-mesh",
        population=100,
        exact=True,
        source_of_truth="source observations",
    )
    base.update(overrides)
    return DeviationReport(**base)  # type: ignore[arg-type]


def _tiny_result(
    mesh: MeshData | None = None,
    stats: FilterStats | None = None,
    deviation: DeviationReport | None = None,
    mesh_to_source: DeviationReport | None = None,
) -> MeshResult:
    syn = synthetic.generate(
        rows=4, cols=8, dropout=0.0, range_noise=0.0, seed=1, station_id="tiny"
    )
    return MeshResult(
        mesh=mesh if mesh is not None else _tiny_mesh(),
        stats=stats if stats is not None else _tiny_stats(),
        lattice=syn.scan.lattice,
        deviation=deviation if deviation is not None else _tiny_deviation(),
        mesh_to_source=mesh_to_source,
        timings={"clean": 0.01},
        settings={"despeckle": True, "measure": True},
        station_id="tiny",
    )


def test_t1_rejects_single_bit_float_difference() -> None:
    """T1 uses array_equal; a one-ULP float32 change must mismatch."""
    a = np.array([1.0], np.float32)
    b = a.copy()
    # Flip the least-significant bit of the float32 payload.
    b_view = b.view(np.uint32)
    b_view[0] ^= np.uint32(1)
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_t1_array("vertices", a, b)
    assert exc.value.field == "vertices"
    assert exc.value.index == 0


def test_t1_rejects_single_field_filter_stats_difference() -> None:
    left = _tiny_stats(retained=7)
    right = _tiny_stats(retained=6, dropped_other=1)
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_filter_stats_t1(left, right)
    assert exc.value.field == "stats.retained"


def test_t2_order_irrelevant_multiplicity_relevant() -> None:
    ids = np.array([10, 20, 30, 40], np.int64)
    # Same oriented triangles, different array order → equal.
    tris_a = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    tris_b = np.array([[0, 2, 3], [0, 1, 2]], np.uint32)
    compare_source_id_triples_t2(ids, tris_a, ids, tris_b)

    # Cyclic rotation of winding preserves orientation → equal.
    tris_rot = np.array([[1, 2, 0], [2, 3, 0]], np.uint32)
    compare_source_id_triples_t2(ids, tris_a, ids, tris_rot)

    # Multiplicity: two copies vs one → mismatch.
    tris_dup = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 3]], np.uint32)
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_source_id_triples_t2(ids, tris_a, ids, tris_dup)
    assert exc.value.field == "triangles"


def test_t2_canonical_rotation_count_reported() -> None:
    ids = np.array([1, 2, 3], np.int64)
    tris = np.array([[1, 2, 0]], np.uint32)  # needs one cyclic rotation
    canon, rotations = canonical_oriented_triples(ids, tris)
    assert rotations == 1
    assert canon.shape == (1, 3)
    assert tuple(canon[0]) == (1, 2, 3)


def test_t3_accepts_0_9x_bound_rejects_1_1x() -> None:
    """SPEC §9 boundary: 0.9× of 1e-6 rad accepts; 1.1× rejects."""
    base = np.array([[0.0, 0.0, 1.0]], np.float32)
    # Rotate about x by a directed angle theta in the yz plane.
    theta_ok = 0.9 * NORMAL_ANGLE_TOL_RAD
    theta_bad = 1.1 * NORMAL_ANGLE_TOL_RAD
    ok = np.array(
        [[0.0, float(np.sin(theta_ok)), float(np.cos(theta_ok))]], np.float32
    )
    bad = np.array(
        [[0.0, float(np.sin(theta_bad)), float(np.cos(theta_bad))]], np.float32
    )
    # Renormalise to unit length after float32 cast.
    ok /= np.linalg.norm(ok, axis=1, keepdims=True)
    bad /= np.linalg.norm(bad, axis=1, keepdims=True)
    compare_normals_t3(base, ok.astype(np.float32))
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_normals_t3(base, bad.astype(np.float32))
    assert exc.value.field == "normals"


def test_t1_origin_and_rgb_and_source_sample_id() -> None:
    mesh_a = _tiny_mesh(
        rgb=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.uint8),
    )
    mesh_b = _tiny_mesh(
        rgb=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.uint8),
    )
    compare_t1_array("origin", mesh_a.origin, mesh_b.origin)
    compare_t1_array("rgb", mesh_a.rgb, mesh_b.rgb)
    compare_t1_array(
        "source_sample_id", mesh_a.source_sample_id, mesh_b.source_sample_id
    )
    # Single-field rgb difference.
    mesh_bad = _tiny_mesh(
        rgb=np.array([[1, 2, 3], [4, 5, 7], [7, 8, 9]], np.uint8),
    )
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_t1_array("rgb", mesh_a.rgb, mesh_bad.rgb)
    assert exc.value.index == 1


def test_forward_deviation_t1_and_reverse_blocked() -> None:
    a = _tiny_deviation(rms=0.0)
    b = _tiny_deviation(rms=0.0)
    compare_deviation_report_t1("deviation", a, b)
    with pytest.raises(EquivalenceMismatch):
        compare_deviation_report_t1("deviation", a, _tiny_deviation(rms=1e-12))
    assert REVERSE_QA_STATUS == "blocked_until_reverse-qa-v2"


def test_qa_metadata_excludes_timing_and_rss() -> None:
    meta_a = QAReportMetadata(
        source_sha256="a" * 64,
        rapidmesh_version="0.1.0",
        settings=(("despeckle", "True"),),
        exclusions=("no-return",),
        processing_seconds=1.0,
        peak_rss_bytes=1000,
    )
    meta_b = QAReportMetadata(
        source_sha256="a" * 64,
        rapidmesh_version="0.1.0",
        settings=(("despeckle", "True"),),
        exclusions=("no-return",),
        processing_seconds=99.0,  # excluded
        peak_rss_bytes=9999,  # excluded
    )
    compare_qa_metadata_t1(meta_a, meta_b)
    meta_bad = replace(meta_b, rapidmesh_version="0.2.0")
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_qa_metadata_t1(meta_a, meta_bad)
    assert exc.value.field == "metadata.rapidmesh_version"


def test_compare_mesh_results_reports_first_mismatch_field() -> None:
    left = _tiny_result()
    right = _tiny_result(
        mesh=_tiny_mesh(
            source_sample_id=np.array([10, 20, 31], np.int64),
        )
    )
    with pytest.raises(EquivalenceMismatch) as exc:
        compare_mesh_results(left, right, source_sha256="b" * 64)
    assert "source_sample_id" in exc.value.field


def test_compare_identical_mesh_results_pass_with_reverse_blocked() -> None:
    left = _tiny_result(mesh_to_source=_tiny_deviation(metric="mesh-to-source"))
    right = _tiny_result(mesh_to_source=_tiny_deviation(metric="mesh-to-source"))
    # Mutate reverse figures so a naïve comparator would fail; harness must
    # record reverse as blocked and ignore the difference.
    right = MeshResult(
        mesh=right.mesh,
        stats=right.stats,
        lattice=right.lattice,
        deviation=right.deviation,
        mesh_to_source=_tiny_deviation(metric="mesh-to-source", rms=9.9),
        timings=right.timings,
        settings=right.settings,
        station_id=right.station_id,
    )
    report = compare_mesh_results(left, right, source_sha256="c" * 64)
    assert report.ok
    assert report.reverse_qa_status == REVERSE_QA_STATUS


# --------------------------------------------------------------------------
# Halo-witness fixture exists and documents the predicted decision
# --------------------------------------------------------------------------


def test_halo_witness_fixture_documents_predicted_decision() -> None:
    wit = synthetic.generate_halo_witness()
    assert wit.band_boundary_row > 0
    assert wit.predicted_decision_name
    assert wit.halo_2_changes_decision is True
    assert wit.halo_3_and_4_match is True
    assert wit.scan.lattice.rows > wit.band_boundary_row
    # Mover and thin feature must both be present in the labelled surfaces.
    assert wit.mover_count > 0
    rail = wit.surface == synthetic.SURF_RAIL
    assert np.any(rail)
    rail_rows = set(int(r) for r in wit.scan.row[rail])
    # Thin feature straddles the documented band boundary.
    assert any(r < wit.band_boundary_row for r in rail_rows)
    assert any(r >= wit.band_boundary_row for r in rail_rows)


# --------------------------------------------------------------------------
# Matrix runner — required-red until PLAN.md §5 items 6–8 exist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("band_rows", BAND_ROWS_MATRIX)
@pytest.mark.parametrize("chunk_points", CHUNK_POINTS_MATRIX)
def test_streamed_equals_in_memory_matrix(
    harness_scan: synthetic.SyntheticScan,
    band_rows: int,
    chunk_points: int,
) -> None:
    """Call mesh_station vs mesh_station_streamed for each matrix cell.

    The number of bands actually produced is asserted, not assumed: a matrix
    that silently ran one band per cell would pass while exercising no band
    boundary at all, which is the trap `PHASE1-HALO-CALCULUS.md` §1 records and
    `PHASE1-ISLANDS-FINALISATION.md` §7 closes with "any run must report the
    number of bands actually produced". The configuration is named in the
    assertion message so a red run identifies the cell.
    """
    n_bands = (HARNESS_ROWS + band_rows - 1) // band_rows
    assert n_bands >= 1
    if band_rows == min(BAND_ROWS_MATRIX):
        assert n_bands >= 3

    # Chunk boundaries must not coincide with band boundaries.
    band_point_stride = band_rows * HARNESS_COLS
    assert chunk_points % band_point_stride != 0

    config = (
        f"band_rows={band_rows} chunk_points={chunk_points} "
        f"n_bands={n_bands} halo=3"
    )
    ref = mesh_station(harness_scan.scan, measure=True, measure_samples=2_000)
    streamed = mesh_station_streamed(
        harness_scan.scan,
        band_rows=band_rows,
        chunk_points=chunk_points,
        halo=3,
        measure=True,
        measure_samples=2_000,
    )
    report = compare_mesh_results(ref, streamed, source_sha256="d" * 64)
    assert report.ok, config
    assert streamed.diagnostics is not None, config
    assert streamed.diagnostics.band_count == n_bands, config
    assert streamed.mesh.triangle_count > 0, config


def test_streamed_entry_does_not_call_mesh_station(
    monkeypatch: pytest.MonkeyPatch,
    harness_scan: synthetic.SyntheticScan,
) -> None:
    """The forbidden implementation, made impossible to hide.

    Inverted from its stub form: `mesh_station` is replaced with a function
    that raises, and the streamed entry point must now run to completion
    anyway. Reassembling the scan and delegating — the shortcut that would make
    every equivalence test above pass while proving nothing — cannot survive
    this, and neither can a partial delegation for one stage.

    `StreamedMeshingNotImplemented` stays imported and asserted-against because
    it remains the harness's named-failure channel; what changed is that the
    implemented path no longer uses it.
    """

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("mesh_station must not be called by the streamed path")

    monkeypatch.setattr("rapidmesh.pipeline.mesh_station", _boom)
    streamed = mesh_station_streamed(
        harness_scan.scan,
        band_rows=3,
        chunk_points=100,
        halo=3,
        measure=True,
        measure_samples=2_000,
    )
    assert streamed.mesh.triangle_count > 0
    assert streamed.deviation is not None
    assert streamed.diagnostics is not None
    assert streamed.diagnostics.band_count == 4
    assert issubclass(StreamedMeshingNotImplemented, NotImplementedError)


def test_halo_witness_streamed_matches_in_memory_with_neighbours() -> None:
    """HALO §9 witness, now with the neighbours it always needed.

    Renamed from ``test_halo_witness_streamed_matrix_required_red``: items 6–8
    exist, so this is required-green.

    The witness ships as a single scan, so carve and restore never ran on it
    (WP-1.2 report, finding 3) and the decision it is named for was
    unreachable. Two neighbour scans of the same scene — mover removed, since a
    neighbour that sees the mover cannot carve it — are generated here and
    turned into carve grids by the item-7 path, without changing `synthetic.py`.

    Two halos are exercised, and the second is what makes this a falsifier
    rather than a formality: at the specified composed halo of 3 the streamed
    result must match exactly, and at halo 0 it must **not**. A run where both
    matched would prove only that the fixture never crosses a band boundary
    that matters.
    """
    from rapidmesh.carvegrid import StationRef
    from rapidmesh.pipeline import carve_grids_streamed

    wit = synthetic.generate_halo_witness()
    rows, cols = wit.scan.lattice.rows, wit.scan.lattice.cols

    def neighbour(station_id: str, scanner: tuple[float, float, float], seed: int):  # type: ignore[no-untyped-def]
        scene = dataclasses.replace(wit.scene, scanner=scanner, mover=False)
        return synthetic.generate(
            scene, rows=rows, cols=cols, dropout=0.0, range_noise=0.0,
            seed=seed, station_id=station_id,
        ).scan

    scans = [
        wit.scan,
        neighbour("W-B", (-2.6, 1.7, 0.0), 31),
        neighbour("W-C", (2.7, 1.4, 0.0), 37),
    ]
    grids = carve_grids_streamed(
        [StationRef.from_scan(s) for s in scans],
        exclude=wit.scan.station_id,
        nearest=2,
        chunk_points=50_000,
    )
    ref = mesh_station(wit.scan, others=grids, measure=False)
    assert ref.stats.dropped_mover_carve > 0        # carving is live

    streamed = mesh_station_streamed(
        wit.scan,
        band_rows=wit.intended_band_rows,
        chunk_points=80,
        halo=3,
        others=grids,
        measure=False,
    )
    report = compare_mesh_results(ref, streamed, source_sha256="e" * 64)
    assert report.ok
    assert streamed.diagnostics is not None
    assert streamed.diagnostics.band_count == 2
    assert wit.predicted_decision_name == "restore_keep_at_band_boundary_rail"

    starved = mesh_station_streamed(
        wit.scan,
        band_rows=wit.intended_band_rows,
        chunk_points=80,
        halo=0,
        others=grids,
        measure=False,
    )
    with pytest.raises(EquivalenceMismatch):
        compare_mesh_results(ref, starved, source_sha256="e" * 64)
