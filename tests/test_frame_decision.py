"""
WP-F1 — the shared, fail-closed frame decision, and the F2 azimuth estimator.

These are the tests that were missing when F1 shipped. The independent review of
2 September 2026 found that the streamed path rewrote **every real station**
into a frame 5.8 x 10^6 m from where it belonged, and §3.5 explains why nothing
caught it: the only E57 fixture in the suite carried the *identity* pose, so the
rewrite branch never executed, and every synthetic fixture had perfectly planar
columns, so the rule's dependence on real instrument non-planarity was
invisible.

So the fixtures here are deliberately unlike the old ones:

* **posed** — a survey-magnitude translation and a genuine rotation, which is
  the only configuration in which the frame rule does anything at all;
* **jittered** — column azimuth noise of 0.01-0.05 degrees, the order a real
  instrument produces, because a perfectly planar column makes the discriminator
  look decisive when it is measuring floating-point residue;
* **column-major** — samples stored column by column, as every authorised
  structured export is, which is the storage order F2 was blind to.

No client data. Every file is written here with `pye57` into a temporary
directory, and no lattice, count or coordinate from the authorised sample
appears in this file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh.e57_reader import (
    FRAME_IDENTITY_POSE,
    FRAME_LEFT_AS_LOCAL,
    FRAME_REWRITTEN_TO_LOCAL,
    FRAME_SPHERICAL_IS_LOCAL,
    FRAME_UNDECIDED,
    MAX_PLAUSIBLE_RANGE_M,
    ImplausibleRange,
    _angular_steps_from_indices,
    _circular_step,
    assert_plausible_range,
    column_alignment_ceiling,
    decide_frame,
    read_scan_with_frame_path,
)
from rapidmesh.streaming_e57 import e57_station_metadata
from rapidmesh.types import ScanPose

# A survey-magnitude pose: the case SPATIAL-CONTRACT §3 rule 6 exists for, and
# the magnitude at which the old bare-ratio rule always chose "rewrite".
SURVEY_TRANSLATION = (515_000.0, 5_750_000.0, 120.0)


def _rotation(yaw: float = 0.4, pitch: float = 0.15) -> np.ndarray:
    """A genuine non-identity rotation — not a near-identity nudge."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    out: np.ndarray = rz @ ry
    return out


def _lattice_local(
    rows: int, cols: int, *, jitter_deg: float, fov: float, seed: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-major scanner-local samples on a regular lattice.

    Returns (row, col, xyz, rng, azimuth, elevation). `jitter_deg` perturbs each
    sample's azimuth *within* its column, which is what a real rotating head
    does and what makes `as_local` a physical quantity rather than 10^-16.
    """
    rng_gen = np.random.default_rng(seed)
    az_step = fov / cols
    el_step = 0.6 / rows

    col_idx = np.repeat(np.arange(cols, dtype=np.int64), rows)
    row_idx = np.tile(np.arange(rows, dtype=np.int64), cols)  # column-major

    az = -fov / 2.0 + col_idx * az_step
    if jitter_deg:
        az = az + rng_gen.normal(0.0, math.radians(jitter_deg), size=az.size)
    el = -0.3 + row_idx * el_step

    r = np.full(az.size, radius, dtype=np.float64)
    ce = np.cos(el)
    xyz = np.stack([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)], axis=1)
    return row_idx, col_idx, xyz, r, az, el


def _write_e57(
    path: Path,
    *,
    translation: tuple[float, float, float],
    rotation: np.ndarray | None = None,
    jitter_deg: float = 0.03,
    fov: float = 2 * math.pi,
    rows: int = 24,
    cols: int = 96,
    radius: float = 7.5,
    store_world: bool = False,
) -> tuple[str, float]:
    """Write a structured E57 and return its path and the true azimuth step.

    `store_world=True` reproduces the non-conforming exporter of
    SPATIAL-CONTRACT §3 rule 6: project coordinates stored beside a pose, which
    is the one case a rewrite is genuinely correct.
    """
    import pye57

    rot = np.eye(3) if rotation is None else rotation
    pose = ScanPose(translation=np.asarray(translation, np.float64), rotation=rot)
    row_idx, col_idx, xyz, r, az, el = _lattice_local(
        rows, cols, jitter_deg=jitter_deg, fov=fov, seed=5, radius=radius
    )

    data: dict[str, Any] = {
        "rowIndex": row_idx,
        "columnIndex": col_idx,
        "cartesianInvalidState": np.zeros(row_idx.size, np.int64),
    }
    # **pye57's `write_scan_raw` cannot emit spherical fields** — it writes a
    # fixed cartesian set and silently drops `sphericalRange`/`Azimuth`/
    # `Elevation`. So every file fixture here is cartesian, and the spherical
    # short-circuit is covered by the unit tests on `decide_frame` plus the real
    # posed spherical station in Package B. This limitation is stated in the
    # WP-F1 report rather than hidden behind a fixture that looks spherical.
    stored = pose.local_to_world(xyz) if store_world else xyz
    data["cartesianX"] = stored[:, 0]
    data["cartesianY"] = stored[:, 1]
    data["cartesianZ"] = stored[:, 2]

    # pye57 stores the pose the way E57 does — as a quaternion — so the matrix
    # is converted here rather than assumed. Deriving the fixture's own
    # `ScanPose` from the same quaternion keeps the written file and the
    # expected geometry exactly consistent.
    from pyquaternion import Quaternion

    quaternion = Quaternion(matrix=rot, atol=1e-8, rtol=1e-8)
    writer = pye57.E57(str(path), mode="w")
    writer.write_scan_raw(
        data, name="fixture", translation=np.asarray(translation, np.float64),
        rotation=quaternion.elements,
    )
    del writer
    return str(path), fov / cols


# ---------------------------------------------------------------------------
# F1 — the defect itself
# ---------------------------------------------------------------------------


def test_the_streamed_path_does_not_rewrite_a_posed_station(tmp_path: Path) -> None:
    """**The F1 regression test, at file level.**

    A conforming station — local coordinates, survey-magnitude pose, realistic
    column jitter — driven through the streamed metadata path that F1 lived in.
    Before the fix this took the rewrite branch and put the station 5.8 x 10^6 m
    from where it belonged; the plausibility guard now refuses that outright.
    """
    path, _ = _write_e57(
        tmp_path / "posed.e57", translation=SURVEY_TRANSLATION, rotation=_rotation(),
    )
    metadata, policy = e57_station_metadata(path)
    assert metadata.frame_path == FRAME_LEFT_AS_LOCAL
    assert policy.rewrite_to_local is False


def test_both_paths_record_the_same_frame_on_a_posed_station(tmp_path: Path) -> None:
    """§3.7 item 1: one decision, shared, asserted rather than argued.

    This is the assertion whose absence allowed the two implementations to
    diverge for two days.
    """
    path, _ = _write_e57(
        tmp_path / "agree.e57", translation=SURVEY_TRANSLATION, rotation=_rotation(),
    )
    _scan, in_memory = read_scan_with_frame_path(path)
    metadata, _policy = e57_station_metadata(path)
    assert in_memory == metadata.frame_path == FRAME_LEFT_AS_LOCAL


def test_the_streamed_station_stays_within_scanner_range(tmp_path: Path) -> None:
    """The consequence, measured rather than inferred.

    Under F1 every band carried ranges above 10^5 m. The scene is ~7.5 m.
    """
    from rapidmesh.streaming_e57 import e57_band_to_scan, e57_chunks

    path, _ = _write_e57(
        tmp_path / "ranges.e57",         translation=SURVEY_TRANSLATION, rotation=_rotation(), radius=7.5,
    )
    metadata, policy = e57_station_metadata(path)

    worst = 0.0
    for chunk in e57_chunks(path):
        scan = e57_band_to_scan(chunk, metadata, policy)
        if len(scan):
            worst = max(worst, float(np.max(scan.rng)))
    assert worst < 20.0, f"scene is ~7.5 m; got {worst:,.1f} m"


def test_a_cartesian_local_station_with_a_survey_pose_is_not_rewritten(
    tmp_path: Path
) -> None:
    """The latent in-memory case, review §3.4.

    A conforming cartesian export — local coordinates beside a georeferenced
    pose — with realistic column jitter. The old rule rewrote it and produced
    5.8 x 10^6 m coordinates for a 7.7 m scene; the reference dataset never
    exercised this because it is spherical, but a client's Faro or Trimble
    cartesian export would.
    """
    path, _ = _write_e57(
        tmp_path / "cart_local.e57",         translation=SURVEY_TRANSLATION, rotation=_rotation(), jitter_deg=0.05,
    )
    scan, frame_path = read_scan_with_frame_path(path)
    assert frame_path == FRAME_LEFT_AS_LOCAL
    assert float(np.max(np.abs(scan.xyz))) < 20.0

    metadata, policy = e57_station_metadata(path)
    assert metadata.frame_path == FRAME_LEFT_AS_LOCAL
    assert policy.rewrite_to_local is False


@pytest.mark.parametrize("jitter_deg", [0.0, 0.01, 0.05])
def test_jitter_does_not_change_the_verdict_at_survey_magnitude(
    tmp_path: Path, jitter_deg: float
) -> None:
    """§3.4's table, as a gate.

    The old rule flipped from left-as-local to rewritten between 0 and 0.01
    degrees of jitter — the verdict depended on instrument noise. It must not.
    """
    path, _ = _write_e57(
        tmp_path / f"jit{jitter_deg}.e57",         translation=SURVEY_TRANSLATION, rotation=_rotation(), jitter_deg=jitter_deg,
    )
    _scan, frame_path = read_scan_with_frame_path(path)
    assert frame_path == FRAME_LEFT_AS_LOCAL


def test_a_genuine_world_stored_export_is_still_rewritten(tmp_path: Path) -> None:
    """The fix must not blind the detector it hardens.

    A non-conforming exporter storing project coordinates beside a *modest*
    pose: both candidate frames are physically plausible, so the spread margin
    decides, and it decides to rewrite — which is correct and is the whole
    reason `_resolve_frame` exists.
    """
    path, _ = _write_e57(
        tmp_path / "world_stored.e57",         translation=(60.0, -40.0, 5.0), rotation=_rotation(), jitter_deg=0.0,
        store_world=True,
    )
    _scan, frame_path = read_scan_with_frame_path(path)
    assert frame_path == FRAME_REWRITTEN_TO_LOCAL

    metadata, policy = e57_station_metadata(path)
    assert metadata.frame_path == FRAME_REWRITTEN_TO_LOCAL
    assert policy.rewrite_to_local is True


def test_an_identity_pose_is_still_short_circuited(tmp_path: Path) -> None:
    """The ordinary synthetic case must not regress."""
    path, _ = _write_e57(
        tmp_path / "identity.e57", translation=(0.0, 0.0, 0.0),
    )
    _scan, frame_path = read_scan_with_frame_path(path)
    metadata, policy = e57_station_metadata(path)
    assert frame_path == metadata.frame_path == FRAME_IDENTITY_POSE
    assert policy.rewrite_to_local is False


# ---------------------------------------------------------------------------
# The decision rule, in isolation
# ---------------------------------------------------------------------------


def _decide(**kwargs: Any) -> Any:
    base: dict[str, Any] = dict(
        has_spherical=False, has_row_column=True, posed=True, sample_count=1000,
        as_local=1e-3, as_world=1e-16, max_range_local=10.0, max_range_world=10.0,
        az_step=0.01,
    )
    base.update(kwargs)
    return decide_frame(**base)


def test_plausibility_outranks_the_spread_margin() -> None:
    """Owner ruling (a): physical plausibility FIRST.

    This is the exact F1 numeric signature — a spread ratio of 10^-13 that says
    "rewrite" overwhelmingly, and a rewritten frame 5.8 x 10^6 m from the
    scanner. The margin must not get a vote.
    """
    evidence = _decide(
        as_local=4.1e-3, as_world=5.8e-15, max_range_local=10.0,
        max_range_world=5.8e6,
    )
    assert evidence.path == FRAME_LEFT_AS_LOCAL
    assert evidence.rewrite is False


def test_an_implausible_stored_frame_is_rewritten_when_the_other_is_sane() -> None:
    evidence = _decide(max_range_local=5.8e6, max_range_world=10.0)
    assert evidence.path == FRAME_REWRITTEN_TO_LOCAL


def test_both_implausible_is_undecided_and_never_a_rewrite() -> None:
    evidence = _decide(max_range_local=5.8e6, max_range_world=9.9e6)
    assert evidence.undecided
    assert evidence.rewrite is False


def test_a_weak_margin_fails_closed_with_the_reason_recorded() -> None:
    """SPATIAL-CONTRACT §3 rule 7: fail closed, and record which path was taken."""
    evidence = _decide(as_local=0.5, as_world=0.4)
    assert evidence.path == FRAME_UNDECIDED
    assert evidence.undecided
    assert evidence.rewrite is False


def test_the_alignment_bar_scales_with_the_lattice_not_with_the_ratio() -> None:
    """Owner ruling (c). A ratio alone is gameable; an absolute bar is not."""
    assert column_alignment_ceiling(0.1) > column_alignment_ceiling(0.001)
    # Decisive ratio, but the rewritten candidate is nowhere near column-aligned
    # for this lattice, so it must not win on the ratio alone.
    evidence = _decide(as_local=1.0, as_world=0.05, az_step=1e-4)
    assert evidence.rewrite is False


def test_spherical_short_circuits_before_anything_else() -> None:
    evidence = _decide(has_spherical=True, max_range_local=1e9, max_range_world=1e9)
    assert evidence.path == FRAME_SPHERICAL_IS_LOCAL


def test_the_evidence_carries_both_spreads_and_both_ranges() -> None:
    """§3.7 item 2(c): record both spreads, so a campaign row can be audited."""
    evidence = _decide(as_local=4.1e-3, as_world=5.8e-15, max_range_world=5.8e6)
    assert evidence.as_local == pytest.approx(4.1e-3)
    assert evidence.as_world == pytest.approx(5.8e-15)
    assert evidence.max_range_world == pytest.approx(5.8e6)
    assert "max_range_world" in evidence.describe()


# ---------------------------------------------------------------------------
# A2 — the ingest guard
# ---------------------------------------------------------------------------


def test_assert_plausible_range_raises_by_name() -> None:
    with pytest.raises(ImplausibleRange, match="exceeds the plausibility"):
        assert_plausible_range(np.array([1.0, 5.8e6]), where="test")


def test_assert_plausible_range_passes_ordinary_scanner_ranges() -> None:
    assert_plausible_range(np.array([0.5, 60.0, 120.0]), where="test")
    assert_plausible_range(np.empty(0), where="test")


def test_the_limit_is_a_recorded_setting_not_a_magic_number() -> None:
    assert MAX_PLAUSIBLE_RANGE_M == 10_000.0
    assert_plausible_range(np.array([50_000.0]), where="test", limit=100_000.0)


def test_a_band_with_impossible_ranges_is_refused(tmp_path: Path) -> None:
    """A2: a wrong station-level decision fails at ingest, not in the mesh.

    Forces the rewrite policy on a station that must not be rewritten — exactly
    the state F1 put every real station into — and requires the band builder to
    refuse it.
    """
    from rapidmesh.streaming_e57 import FramePolicy, e57_band_to_scan, e57_chunks

    path, _ = _write_e57(
        tmp_path / "guard.e57",         translation=SURVEY_TRANSLATION, rotation=_rotation(),
    )
    metadata, _policy = e57_station_metadata(path)
    wrong = FramePolicy(rewrite_to_local=True, shift_colour=False)
    chunk = next(iter(e57_chunks(path)))
    with pytest.raises(ImplausibleRange, match="e57_band_to_scan"):
        e57_band_to_scan(chunk, metadata, wrong)


# ---------------------------------------------------------------------------
# F2 — the azimuth step estimator
# ---------------------------------------------------------------------------


def test_the_step_is_recovered_from_a_column_major_stream() -> None:
    """**The F2 regression test.**

    Consecutive samples in a column-major stream share a *column*, so the old
    same-row-pair test found zero usable pairs on every real file and silently
    returned its default. Sorting by (row, col) first makes the estimator
    independent of storage order.
    """
    rows, cols, fov = 20, 120, 2 * math.pi
    row_idx, col_idx, _xyz, _r, az, _el = _lattice_local(
        rows, cols, jitter_deg=0.0, fov=fov, seed=3, radius=8.0
    )
    step = _circular_step(az, row_idx, col_idx, default=99.0)
    assert step != 99.0, "estimator fell back to its default on a column-major stream"
    assert step == pytest.approx(fov / cols, rel=1e-6)


def test_a_partial_field_of_view_does_not_report_a_full_sweep() -> None:
    """The failure the default hides: a 180 degree scan measured as 360.

    With the wrong step, `az_step * cols` equals 2*pi on every lattice and
    `columns_wrap` joins the two ends of the scan across the room.
    """
    rows, cols, fov = 20, 120, math.pi
    row_idx, col_idx, xyz, r, az, _el = _lattice_local(
        rows, cols, jitter_deg=0.0, fov=fov, seed=4, radius=8.0
    )
    step = _circular_step(az, row_idx, col_idx, default=2 * math.pi / cols)
    assert step == pytest.approx(fov / cols, rel=1e-6)
    assert abs(step * cols) == pytest.approx(math.pi, rel=1e-6)
    assert abs(step * cols) < 2 * math.pi - 1e-6, "would wrap a half-sweep"


def test_a_partial_fov_column_major_scan_does_not_wrap(tmp_path: Path) -> None:
    """A3, end to end: the lattice a half-sweep file produces must not wrap."""
    from rapidmesh.filters import columns_wrap

    path, true_step = _write_e57(
        tmp_path / "halfsweep.e57", translation=(0.0, 0.0, 0.0),
        jitter_deg=0.0, fov=math.pi, rows=20, cols=120,
    )
    scan, _frame = read_scan_with_frame_path(path)
    assert scan.lattice.az_step == pytest.approx(true_step, rel=1e-3)
    assert columns_wrap(scan) is False


def test_a_full_sweep_still_wraps(tmp_path: Path) -> None:
    """The complement: the fix must not stop a genuine 360 scan from wrapping."""
    from rapidmesh.filters import columns_wrap

    path, true_step = _write_e57(
        tmp_path / "fullsweep.e57", translation=(0.0, 0.0, 0.0),
        jitter_deg=0.0, fov=2 * math.pi, rows=20, cols=120,
    )
    scan, _frame = read_scan_with_frame_path(path)
    assert scan.lattice.az_step == pytest.approx(true_step, rel=1e-3)
    assert columns_wrap(scan) is True


def test_the_estimator_is_independent_of_storage_order() -> None:
    """Same samples, shuffled: same step. That is the property F2 lacked."""
    rows, cols, fov = 16, 90, 2 * math.pi
    row_idx, col_idx, xyz, r, az, _el = _lattice_local(
        rows, cols, jitter_deg=0.0, fov=fov, seed=9, radius=6.0
    )
    valid = np.ones(az.size, bool)
    column_major = _angular_steps_from_indices(xyz, r, row_idx, col_idx, valid, rows, cols)

    order = np.lexsort((col_idx, row_idx))  # row-major
    row_major = _angular_steps_from_indices(
        xyz[order], r[order], row_idx[order], col_idx[order], valid, rows, cols
    )
    assert column_major[0] == pytest.approx(row_major[0], rel=1e-9)
    assert column_major[0] == pytest.approx(fov / cols, rel=1e-6)
