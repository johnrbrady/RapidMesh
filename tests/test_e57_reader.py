"""
E57 reader tests, run against real E57 files written on the fly.

Round-tripping through an actual file (rather than hand-feeding the reader a
dict) is the point: the failures worth catching here are in how a real file is
laid out — which frame the coordinates are in, whether the pose survives, how
colour is scaled — and none of those exist in a mock.

Skipped when pye57 is not installed, which is why it is an optional dependency:
the numeric core has to stay testable without a native build toolchain.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rapidmesh import e57_reader, synthetic
from rapidmesh.pipeline import mesh_station
from rapidmesh.types import LatticeSource, ScanPose

pye57 = pytest.importorskip("pye57", reason="E57 reading is an optional extra")

ROWS, COLS = 200, 800
TRUE_EL_STEP = np.radians(120.0) / (ROWS - 1)
TRUE_AZ_STEP = 2 * np.pi / COLS


@pytest.fixture(scope="module")
def fixture_scan() -> synthetic.SyntheticScan:
    return synthetic.generate(rows=ROWS, cols=COLS, seed=5)


ROTATED_QUATERNION = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
ROTATED_MATRIX = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
ROTATED_TRANSLATION = np.array([1000.125, -2000.25, 50.5], dtype=np.float64)


def _write(path, scan, points, rotation=None) -> None:  # type: ignore[no-untyped-def]
    """Write one structured scan with row/column indices and colour."""
    data = {
        "cartesianX": points[:, 0].astype(np.float64),
        "cartesianY": points[:, 1].astype(np.float64),
        "cartesianZ": points[:, 2].astype(np.float64),
        "rowIndex": scan.row.astype(np.int64),
        "columnIndex": scan.col.astype(np.int64),
        "colorRed": scan.rgb[:, 0].astype(np.int64),
        "colorGreen": scan.rgb[:, 1].astype(np.int64),
        "colorBlue": scan.rgb[:, 2].astype(np.int64),
    }
    f = pye57.E57(str(path), mode="w")
    f.write_scan_raw(
        data,
        name="STATION_A",
        rotation=np.array([1.0, 0.0, 0.0, 0.0]) if rotation is None else rotation,
        translation=np.asarray(scan.pose.translation, np.float64),
    )
    f.close()


@pytest.fixture(scope="module")
def local_file(tmp_path_factory, fixture_scan):  # type: ignore[no-untyped-def]
    """Spec-compliant: cartesian points in the SCAN-LOCAL frame."""
    p = tmp_path_factory.mktemp("e57") / "local.e57"
    _write(p, fixture_scan.scan, fixture_scan.scan.xyz.astype(np.float64))
    return p


@pytest.fixture(scope="module")
def world_file(tmp_path_factory, fixture_scan):  # type: ignore[no-untyped-def]
    """The known interoperability wart: already-transformed world coordinates
    written alongside a non-identity pose."""
    p = tmp_path_factory.mktemp("e57") / "world.e57"
    s = fixture_scan.scan
    _write(p, s, s.pose.local_to_world(s.xyz))
    return p


@pytest.fixture(scope="module")
def rotated_files(tmp_path_factory, fixture_scan):  # type: ignore[no-untyped-def]
    pose = ScanPose(translation=ROTATED_TRANSLATION, rotation=ROTATED_MATRIX)
    scan = replace(fixture_scan.scan, pose=pose)
    folder = tmp_path_factory.mktemp("e57-rotated")
    local_path = folder / "local.e57"
    project_path = folder / "project.e57"
    _write(local_path, scan, scan.xyz.astype(np.float64), ROTATED_QUATERNION)
    _write(project_path, scan, pose.local_to_world(scan.xyz), ROTATED_QUATERNION)
    return local_path, project_path


def test_probe_identifies_the_structured_grid(local_file) -> None:  # type: ignore[no-untyped-def]
    """The single fact that decides how good the mesh can be, reported without
    reading a point."""
    scans = e57_reader.probe(local_file)
    assert len(scans) == 1
    assert scans[0].source is LatticeSource.ROW_COL
    assert scans[0].has_rgb


def test_lattice_step_is_recovered_exactly(local_file) -> None:  # type: ignore[no-untyped-def]
    """Angular step to within a tenth of a percent.

    Regression guard for a real bug: azimuth has a branch cut at +/-pi, and
    least-squares fitting it after `np.unwrap` accumulates a spurious 2*pi at
    every row boundary because samples arrive in row-major order. That produced
    a step 3 % low — close enough to look plausible, wrong enough to matter at
    60 m. `_circular_step` differences within rows instead.
    """
    scan = e57_reader.read_scan(local_file, 0)
    assert scan.lattice.rows == ROWS
    assert scan.lattice.cols == COLS
    assert abs(scan.lattice.az_step - TRUE_AZ_STEP) / TRUE_AZ_STEP < 0.001
    assert abs(scan.lattice.el_step - TRUE_EL_STEP) / TRUE_EL_STEP < 0.001


def test_pose_survives_the_round_trip(local_file, fixture_scan) -> None:  # type: ignore[no-untyped-def]
    scan = e57_reader.read_scan(local_file, 0)
    assert np.allclose(scan.pose.translation, fixture_scan.scan.pose.translation, atol=1e-6)
    assert np.allclose(scan.pose.rotation, np.eye(3), atol=1e-9)


def test_world_coordinate_export_is_detected_and_corrected(world_file, local_file) -> None:  # type: ignore[no-untyped-def]
    """A non-compliant export must mesh identically to a compliant one.

    Writing world coordinates with a non-identity pose is catastrophic here and
    silent: every angle is measured from the scanner origin, so a shifted frame
    gives a wrong lattice, wrong incidence thresholds and a mesh in the wrong
    place, with nothing raising. `_resolve_frame` detects it from a physical
    property — a rotating-head scanner sweeps a vertical plane, so azimuth must
    be constant within a lattice column.
    """
    bad = e57_reader.read_scan(world_file, 0)
    good = e57_reader.read_scan(local_file, 0)

    assert abs(bad.lattice.az_step - good.lattice.az_step) < 1e-9
    assert abs(bad.lattice.el_step - good.lattice.el_step) < 1e-9
    # and the geometry itself, not just the lattice description
    assert np.allclose(bad.xyz, good.xyz, atol=1e-5)


def test_rotated_project_coordinate_export_uses_full_inverse_pose(rotated_files) -> None:  # type: ignore[no-untyped-def]
    local_path, project_path = rotated_files
    local = e57_reader.read_scan(local_path, 0)
    recovered = e57_reader.read_scan(project_path, 0)

    assert np.allclose(local.pose.rotation, ROTATED_MATRIX, atol=1e-12)
    assert np.allclose(recovered.pose.rotation, ROTATED_MATRIX, atol=1e-12)
    # The test writer re-encodes the transformed coordinates, so allow its
    # sub-0.1 mm round-trip quantisation while still catching a lost rotation.
    assert np.allclose(recovered.xyz, local.xyz, atol=1e-4)


def test_meshes_from_a_real_file(local_file) -> None:  # type: ignore[no-untyped-def]
    """End to end: file on disk in, mesh with a deviation figure out."""
    scan = e57_reader.read_scan(local_file, 0)
    res = mesh_station(scan, measure=True)
    assert res.mesh.triangle_count > 100_000
    assert res.mesh.normals is not None
    assert res.deviation is not None
    assert res.deviation.rms < 0.001


def test_lattice_stride_keeps_structure_not_random_points(local_file) -> None:  # type: ignore[no-untyped-def]
    """`max_points` must stride the lattice, never sample randomly.

    Random sampling would destroy the neighbour relationships the whole engine
    is built on and turn a structured scan into the unstructured cloud we
    refuse to work from. Striding leaves a coarser but still structured
    lattice, so the row and column indices stay on a regular grid.
    """
    scan = e57_reader.read_scan(local_file, 0, max_points=20_000)
    assert len(scan) <= 40_000
    rows = np.unique(scan.row)
    gaps = np.unique(np.diff(rows))
    assert gaps.size <= 2, f"row spacing is irregular: {gaps[:10]}"


def test_raw_chunk_reader_is_complete_ordered_and_bounded(local_file) -> None:  # type: ignore[no-untyped-def]
    chunk_points = 7_777
    chunks = list(
        e57_reader.iter_raw_chunks(
            local_file,
            chunk_points=chunk_points,
            fields=("cartesianX", "rowIndex", "columnIndex"),
        )
    )
    full = pye57.E57(str(local_file)).read_scan_raw(0)

    assert sum(chunk.count for chunk in chunks) == len(full["cartesianX"])
    assert all(chunk.count <= chunk_points for chunk in chunks)
    assert [chunk.offset for chunk in chunks] == [
        sum(c.count for c in chunks[:i]) for i in range(len(chunks))
    ]
    for field in ("cartesianX", "rowIndex", "columnIndex"):
        joined = np.concatenate([np.asarray(chunk.data[field]) for chunk in chunks])
        assert np.array_equal(joined, full[field])

    # The first chunk owns its data: later reads must not mutate it through a
    # reused libE57 destination buffer.
    first_before = np.asarray(chunks[0].data["cartesianX"]).copy()
    assert np.array_equal(chunks[0].data["cartesianX"], first_before)
