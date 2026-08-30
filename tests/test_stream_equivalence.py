"""Streamed ≡ in-memory equivalence harness (PLAN.md §5 item 5).

Comparison contract: PHASE1-DETERMINISM-SPEC.md §7.
Halo witness design: PHASE1-HALO-CALCULUS.md §9.

This package delivers the harness and the streamed entry-point stub.
Band-local geometry (PLAN.md §5 items 6–8) is deliberately absent, so the
matrix integration tests are required-red: they must fail with
StreamedMeshingNotImplemented, not ImportError, not xfail, not skip.
"""

from __future__ import annotations

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

    Required-red until PLAN.md §5 items 6–8 exist: the streamed entry must
    raise StreamedMeshingNotImplemented (named failure), not ImportError,
    and this test must not xfail or skip. The configuration is named in the
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
    )
    # Unreachable until items 6–8 land; compare would run here.
    report = compare_mesh_results(ref, streamed, source_sha256="d" * 64)
    assert report.ok, config


def test_streamed_entry_does_not_call_mesh_station(
    monkeypatch: pytest.MonkeyPatch,
    harness_scan: synthetic.SyntheticScan,
) -> None:
    """Forbidden streamed implementation: reassemble + mesh_station.

    Green characterisation of the stub: it raises its dedicated exception
    without calling mesh_station.
    """

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("mesh_station must not be called by streamed stub")

    monkeypatch.setattr("rapidmesh.pipeline.mesh_station", _boom)
    with pytest.raises(StreamedMeshingNotImplemented) as exc:
        mesh_station_streamed(
            harness_scan.scan,
            band_rows=3,
            chunk_points=100,
            halo=3,
        )
    msg = str(exc.value)
    assert "PLAN.md" in msg
    assert "6" in msg and "7" in msg and "8" in msg


def test_halo_witness_streamed_matrix_required_red() -> None:
    """HALO §9 witness matrix stays required-red until item 6.

    Fixture loads and documents predictions; streamed comparison is not
    asserted green in this package.
    """
    wit = synthetic.generate_halo_witness()
    # Exercise halo=2 (predicted to change the named decision) — required-red.
    streamed = mesh_station_streamed(
        wit.scan,
        band_rows=wit.intended_band_rows,
        chunk_points=80,
        halo=2,
    )
    ref = mesh_station(wit.scan, measure=False)
    report = compare_mesh_results(ref, streamed, source_sha256="e" * 64)
    assert report.ok
    assert wit.predicted_decision_name == "restore_keep_at_band_boundary_rail"
