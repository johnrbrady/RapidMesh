"""Independent regression tests for the RapidMesh/Cairn spatial contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rapidmesh.e57_reader import quat_to_matrix
from rapidmesh.triangulate import build_mesh
from rapidmesh.types import LatticeInfo, LatticeSource, ScanPose, StructuredScan


def _rotation_z_90() -> np.ndarray:
    # Intentionally hard-coded rather than produced by the implementation's
    # quaternion helper: this is the independent expected answer.
    return np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _scan(pose: ScanPose) -> StructuredScan:
    return StructuredScan(
        row=np.array([0, 0, 1], dtype=np.int32),
        col=np.array([0, 1, 0], dtype=np.int32),
        xyz=np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        rng=np.array([1.0, math.sqrt(2.0), math.sqrt(2.0)], dtype=np.float32),
        pose=pose,
        lattice=LatticeInfo(2, 2, 1.0, 1.0, 0.0, 0.0, LatticeSource.SYNTHETIC),
        station_id="synthetic-rotated",
    )


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        ([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        ([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        ([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        ([0.0, 1.0, 0.0, 0.0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        ([0.0, 0.0, 1.0, 0.0], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ([0.0, 0.0, 0.0, 1.0], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    ],
)
def test_e57_wxyz_quaternion_convention(
    quaternion: list[float], expected: list[list[int]]
) -> None:
    assert np.allclose(quat_to_matrix(quaternion), np.asarray(expected), atol=1e-12)


@pytest.mark.parametrize(
    "translation",
    [
        np.array([1_000_000.125, -2_000_000.25, 100.5], dtype=np.float64),
        np.array([-1_000_000.125, 2_000_000.25, -100.5], dtype=np.float64),
    ],
)
def test_pose_round_trip_far_from_origin(translation: np.ndarray) -> None:
    pose = ScanPose(translation=translation, rotation=_rotation_z_90())
    local = np.array([[0.001, -0.002, 0.003], [123.456, -78.9, 4.25]], dtype=np.float32)

    restored = pose.world_to_local(pose.local_to_world(local))

    assert np.allclose(restored, local.astype(np.float64), atol=1e-9)


def test_mesh_applies_rotation_exactly_once_and_rotates_normals() -> None:
    origin = np.array([1_000_000.125, -2_000_000.25, 100.5], dtype=np.float64)
    pose = ScanPose(translation=origin, rotation=_rotation_z_90())
    scan = _scan(pose)

    mesh = build_mesh(scan, np.array([[0, 1, 2]], dtype=np.int64))
    reconstructed = mesh.origin + mesh.vertices.astype(np.float64)
    expected = np.array(
        [
            [1_000_000.125, -1_999_999.25, 100.5],
            [999_999.125, -1_999_999.25, 100.5],
            [1_000_000.125, -1_999_999.25, 101.5],
        ],
        dtype=np.float64,
    )

    assert np.allclose(reconstructed, expected, atol=1e-7)
    assert mesh.source_pose is pose
    assert mesh.normals is not None
    assert np.allclose(mesh.normals, np.array([[0.0, -1.0, 0.0]] * 3), atol=1e-7)


def test_mesh_storage_error_remains_below_contract_budget() -> None:
    pose = ScanPose(
        translation=np.array([-2_000_000.25, 1_000_000.125, -100.5], dtype=np.float64),
        rotation=_rotation_z_90(),
    )
    scan = _scan(pose)
    mesh = build_mesh(scan, np.array([[0, 1, 2]], dtype=np.int64), with_normals=False)

    error = np.abs((mesh.origin + mesh.vertices.astype(np.float64)) - pose.local_to_world(scan.xyz))

    assert float(error.max()) <= 0.0001


@pytest.mark.parametrize(
    "bad_rotation",
    [
        np.diag([-1.0, 1.0, 1.0]),
        np.diag([2.0, 1.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_pose_rejects_reflection_scale_and_non_finite_rotation(bad_rotation: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ScanPose(translation=np.zeros(3, dtype=np.float64), rotation=bad_rotation)

