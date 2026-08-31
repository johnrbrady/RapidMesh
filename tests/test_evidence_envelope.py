"""The exported evidence envelope (PLAN.md §5 item 11 / CLAUDE.md §9 item 1).

`PHASE1-DETERMINISM-SPEC.md` §2 lists what must be equal for two runs to be
compared bit for bit. Item 1 blocks *every* real-data fidelity claim on the
envelope carrying those conditions, so the tests here are about completeness
and honesty rather than about geometry:

* every field SPEC §2 names is present, and the required set is asserted
  against `evidence.REQUIRED_METADATA_FIELDS` rather than a list retyped here;
* the whole report survives `json.dumps` — an envelope that cannot be written
  out is not an exported envelope;
* the resident and streamed paths produce the *same* envelope apart from the
  three fields `EXCLUDED_FROM_COMPARISON` names, each excluded for a stated
  reason (SPEC §5(i));
* nothing is fabricated: `commit_sha` is a real 40-hex hash or the string
  `unknown`, `frame_path` is `unrecorded` for a scan that never went through
  `_resolve_frame`, and `peak_rss_bytes` is `None` unless measured;
* `qa_workers=-1` is refused on both paths (SPEC §3). "All processors this
  machine happens to have" is the opposite of a recorded condition.
"""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.e57_reader import (
    FRAME_IDENTITY_POSE,
    FRAME_NO_ROW_COLUMN,
    FRAME_REWRITTEN_TO_LOCAL,
    FRAME_SPHERICAL_IS_LOCAL,
    FRAME_TOO_FEW_SAMPLES,
    frame_decision,
)
from rapidmesh.equivalence import compare_qa_metadata_t1
from rapidmesh.evidence import (
    BLAS_THREAD_VARIABLES,
    DEFAULT_QA_WORKERS,
    EVIDENCE_ENVELOPE_VERSION,
    EXCLUDED_FROM_COMPARISON,
    FRAME_PATH_UNRECORDED,
    REQUIRED_METADATA_FIELDS,
    EnvironmentEvidence,
    EvidencePolicyError,
    commit_sha,
    pinned_thread_environment,
    resolve_qa_workers,
    working_tree_state,
)
from rapidmesh.pipeline import mesh_station, mesh_station_streamed
from rapidmesh.types import QAReportMetadata, ScanPose

# A synthetic station is not client data and has no file to digest, so the
# envelope's `source_sha256` is supplied by the caller. A fixed, obviously
# synthetic value keeps it from being mistaken for a real file's digest.
FIXTURE_DIGEST = "5f" * 32

ENVELOPE_ROWS = 12
ENVELOPE_COLS = 48


@pytest.fixture
def envelope_scan() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True),
        rows=ENVELOPE_ROWS,
        cols=ENVELOPE_COLS,
        dropout=0.0,
        range_noise=0.0,
        seed=17,
        station_id="envelope",
    )


def _resident(scan: synthetic.SyntheticScan, **kwargs: object) -> QAReportMetadata:
    result = mesh_station(scan.scan, measure=True, **kwargs)  # type: ignore[arg-type]
    return result.evidence_report(FIXTURE_DIGEST).metadata


def _streamed(scan: synthetic.SyntheticScan, **kwargs: object) -> QAReportMetadata:
    result = mesh_station_streamed(
        scan.scan, band_rows=4, chunk_points=200, halo=3, measure=True,
        **kwargs,  # type: ignore[arg-type]
    )
    return result.evidence_report(FIXTURE_DIGEST).metadata


# ---------------------------------------------------------------------------
# completeness — SPEC §2
# ---------------------------------------------------------------------------


def test_required_fields_match_the_metadata_dataclass() -> None:
    """The required set is the dataclass, not a list that can drift from it.

    A field added to the envelope and not to `REQUIRED_METADATA_FIELDS` would
    otherwise be untested, and one removed from the dataclass would leave a
    requirement naming something that no longer exists.
    """
    declared = {f.name for f in dataclasses.fields(QAReportMetadata)}
    assert set(REQUIRED_METADATA_FIELDS) == declared


def test_resident_envelope_carries_every_required_field(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    metadata = _resident(envelope_scan)
    for name in REQUIRED_METADATA_FIELDS:
        assert hasattr(metadata, name), name

    assert metadata.envelope_version == EVIDENCE_ENVELOPE_VERSION
    assert metadata.source_sha256 == FIXTURE_DIGEST
    assert metadata.rapidmesh_version
    assert dict(metadata.platform).keys() >= {"system", "machine", "python"}
    assert dict(metadata.libraries).keys() == {"numpy", "scipy", "pye57"}
    assert dict(metadata.versions).keys() >= {
        "envelope", "component_area", "canonical_order", "reverse_qa",
    }
    assert dict(metadata.seeds) == {"qa_seed": "0"}
    assert metadata.processing_seconds > 0.0


def test_settings_carry_the_streaming_and_qa_axes(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """SPEC §2's settings row, including the axes the kickoff names.

    `band_rows`, `chunk_points` and `halo` are the streamed run's own
    configuration and live in `streaming`; `qa_window_rows`, `qa_workers` and
    the sample counts are settings of both paths.
    """
    settings = dict(_streamed(envelope_scan).settings)
    assert settings.keys() >= {
        "max_incidence_deg", "noise_floor", "min_component_area", "despeckle",
        "measure", "measure_samples", "qa_window_rows", "qa_workers",
    }
    assert settings["qa_workers"] == str(DEFAULT_QA_WORKERS)

    streaming = dict(_streamed(envelope_scan).streaming)
    assert streaming.keys() >= {"band_rows", "chunk_points", "halo"}
    assert streaming["band_rows"] == "4"
    assert streaming["chunk_points"] == "200"
    assert streaming["halo"] == "3"


def test_streamed_envelope_carries_the_pass_diagnostics(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """The component-area ratio diagnostics ISLANDS §4.1 requires.

    The ratio is recorded with `repr` rather than rounded: the exactness
    condition is a comparison against 2**29, and a value written back at
    reduced precision cannot be re-checked against it.
    """
    streaming = dict(_streamed(envelope_scan).streaming)
    assert streaming.keys() >= {
        "band_count", "segment_bytes", "component_count",
        "triangles_before_cull", "triangles_after_cull",
        "component_area_max_ratio", "component_area_fallback_components",
    }
    ratio = float(streaming["component_area_max_ratio"])
    assert ratio >= 0.0
    assert repr(ratio) == streaming["component_area_max_ratio"]


def test_reverse_qa_evidence_is_exported(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    reverse = dict(_resident(envelope_scan).reverse_qa)
    assert reverse.keys() >= {
        "metric_version", "canonical_order_version", "seed", "max_samples",
        "samples_selected", "samples_measured", "samples_unmatched",
        "qa_window_rows", "positive_area_triangles", "windows_used",
    }
    assert reverse["metric_version"] == dict(_resident(envelope_scan).versions)[
        "reverse_qa"
    ]


def test_exclusions_are_the_full_disposition_ledger(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    assert set(_resident(envelope_scan).exclusions) == {
        "no-return", "despeckled", "carved", "island-culled", "otherwise-excluded",
    }


# ---------------------------------------------------------------------------
# it is an *exported* envelope
# ---------------------------------------------------------------------------


def test_report_round_trips_through_json(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """An envelope that cannot be serialised has not been exported.

    Round-tripped rather than merely dumped: `json.dumps` succeeding on a
    structure that decodes to something else would still fail the purpose.
    """
    result = mesh_station(envelope_scan.scan, measure=True)
    report = result.evidence_report(FIXTURE_DIGEST, peak_rss_bytes=123_456_789)
    payload = report.to_dict()
    decoded = json.loads(json.dumps(payload))

    # Stable under a second pass: the first decode turns the tuples into
    # lists, and nothing moves after that. A structure that kept changing
    # shape each time it was written out would not be a usable record.
    assert json.loads(json.dumps(decoded)) == decoded
    metadata = decoded["metadata"]
    for name in REQUIRED_METADATA_FIELDS:
        assert name in metadata, name
    assert metadata["peak_rss_bytes"] == 123_456_789
    assert isinstance(metadata["platform"], dict)
    assert isinstance(metadata["versions"], dict)
    assert decoded["lattice_source"]


def test_peak_rss_is_none_unless_measured(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """Absent rather than zero. A zero would read as a measured figure."""
    result = mesh_station(envelope_scan.scan, measure=True)
    assert result.evidence_report(FIXTURE_DIGEST).metadata.peak_rss_bytes is None


# ---------------------------------------------------------------------------
# cross-path equality — SPEC §5(i)
# ---------------------------------------------------------------------------


def test_resident_and_streamed_envelopes_agree_except_the_named_exclusions(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    resident = _resident(envelope_scan)
    streamed = _streamed(envelope_scan)

    compare_qa_metadata_t1(resident, streamed)

    for name in EXCLUDED_FROM_COMPARISON:
        assert hasattr(resident, name), name
    # And the exclusions are real: the streaming axes genuinely differ, which
    # is why comparing them would be comparing a run with itself.
    assert resident.streaming == ()
    assert streamed.streaming != ()


def test_a_field_neither_compared_nor_excluded_fails_the_comparison(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """Sensitivity: a silent gap in the comparison is an error, not a pass.

    Simulated by removing a field name from both lists for the duration of the
    check — the same state a future edit would create by adding a field to the
    envelope and to neither list.
    """
    from rapidmesh import equivalence

    metadata = _resident(envelope_scan)
    original = equivalence._QA_METADATA_COMPARED
    try:
        equivalence._QA_METADATA_COMPARED = tuple(
            name for name in original if name != "frame_path"
        )
        with pytest.raises(equivalence.EquivalenceMismatch, match="frame_path"):
            compare_qa_metadata_t1(metadata, metadata)
    finally:
        equivalence._QA_METADATA_COMPARED = original


def test_a_changed_envelope_field_is_caught(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """Sensitivity: the comparison is not vacuously green.

    Every newly compared field is perturbed in turn; each must be detected.
    """
    metadata = _resident(envelope_scan)
    for name, value in (
        ("commit_sha", "0" * 40),
        ("working_tree", "no-such-state"),
        ("envelope_version", "evidence-envelope-v0"),
        ("frame_path", FRAME_REWRITTEN_TO_LOCAL),
        ("platform", (("system", "PDP-11"),)),
        ("libraries", (("numpy", "0.0.0"),)),
        ("seeds", (("qa_seed", "99"),)),
        ("threads", (("qa_workers_effective", "8"),)),
        ("versions", (("reverse_qa", "reverse-qa-v0"),)),
        ("reverse_qa", (("seed", "99"),)),
    ):
        altered = dataclasses.replace(metadata, **{name: value})
        # A perturbation that happens to equal the real value would make this
        # pass for the wrong reason, so it is an error in its own right.
        assert getattr(altered, name) != getattr(metadata, name), name
        with pytest.raises(Exception, match=name):
            compare_qa_metadata_t1(metadata, altered)


# ---------------------------------------------------------------------------
# thread policy — SPEC §3
# ---------------------------------------------------------------------------


def test_minus_one_workers_is_refused() -> None:
    with pytest.raises(EvidencePolicyError, match="all processors"):
        resolve_qa_workers(-1)
    for bad in (0, -2):
        with pytest.raises(EvidencePolicyError):
            resolve_qa_workers(bad)
    assert resolve_qa_workers(1) == 1
    assert resolve_qa_workers(4) == 4


def test_minus_one_workers_is_refused_on_both_pipeline_paths(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    with pytest.raises(EvidencePolicyError):
        mesh_station(envelope_scan.scan, measure=True, qa_workers=-1)
    with pytest.raises(EvidencePolicyError):
        mesh_station_streamed(
            envelope_scan.scan, band_rows=4, chunk_points=200, halo=3,
            measure=True, qa_workers=-1,
        )


def test_streamed_path_refuses_a_count_it_cannot_honour(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """Reported unsupported, naming why — CLAUDE.md §4 rule 6.

    The streamed path's QA call sites live in `pass_b.py`, which this package
    does not change. Running at 1 while recording 2 would put a false
    reproduction condition in the envelope.
    """
    with pytest.raises(EvidencePolicyError, match="pass_b"):
        mesh_station_streamed(
            envelope_scan.scan, band_rows=4, chunk_points=200, halo=3,
            measure=True, qa_workers=2,
        )


def test_thread_environment_is_returned_not_applied() -> None:
    """SPEC §3: these must be set before NumPy imports, so a running process
    can only hand them to a child."""
    env = pinned_thread_environment(1)
    assert set(env) == set(BLAS_THREAD_VARIABLES)
    assert set(env.values()) == {"1"}

    evidence = EnvironmentEvidence.collect(qa_workers=1)
    threads = dict(evidence.threads)
    assert threads["qa_workers_requested"] == "1"
    assert threads["qa_workers_effective"] == "1"
    # Unset is recorded as `unset`, which is a different fact from `1`.
    for name in BLAS_THREAD_VARIABLES:
        assert threads[name.lower()]


def test_qa_workers_is_recorded_in_settings_and_threads(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    metadata = _resident(envelope_scan, qa_workers=2)
    assert dict(metadata.settings)["qa_workers"] == "2"
    assert dict(metadata.threads)["qa_workers_effective"] == "2"


# ---------------------------------------------------------------------------
# nothing fabricated
# ---------------------------------------------------------------------------


def test_commit_sha_is_a_real_hash_or_the_word_unknown() -> None:
    sha = commit_sha()
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha), sha


def test_working_tree_state_qualifies_the_commit_sha() -> None:
    """A hash from a modified tree names code that is not the code that ran.

    So the pair is asserted together: whatever `commit_sha` says, the envelope
    also says whether the tree it came from was clean, and neither is filled
    in with a plausible-looking default.
    """
    state = working_tree_state()
    assert state in {"clean", "modified", "unknown"}

    environment = EnvironmentEvidence.collect()
    assert environment.working_tree == state
    if environment.commit_sha == "unknown":
        assert state == "unknown"


def test_library_versions_are_strings_not_module_reprs() -> None:
    """pye57 exposes `__version__` as a *module*; reading it naively records
    the repr of a module object as the version."""
    for name, value in EnvironmentEvidence.collect().libraries:
        assert isinstance(value, str), name
        assert "module" not in value, (name, value)
        assert value == "absent" or re.match(r"^[0-9]", value), (name, value)


def test_frame_path_is_unrecorded_for_a_scan_that_never_resolved_a_frame(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    """A synthetic fixture does not go through `_resolve_frame` at all, so any
    branch name here would be a fiction."""
    assert _resident(envelope_scan).frame_path == FRAME_PATH_UNRECORDED
    assert _streamed(envelope_scan).frame_path == FRAME_PATH_UNRECORDED


def test_frame_path_is_recorded_when_a_frame_decision_was_taken(
    envelope_scan: synthetic.SyntheticScan,
) -> None:
    metadata = _resident(envelope_scan, frame_path=FRAME_IDENTITY_POSE)
    assert metadata.frame_path == FRAME_IDENTITY_POSE


# ---------------------------------------------------------------------------
# the frame branch itself
# ---------------------------------------------------------------------------


def _rowcol_raw(xyz: np.ndarray, cols: np.ndarray) -> dict[str, object]:
    return {
        "cartesianX": xyz[:, 0].copy(),
        "cartesianY": xyz[:, 1].copy(),
        "cartesianZ": xyz[:, 2].copy(),
        "rowIndex": np.zeros(len(cols), np.int64),
        "columnIndex": cols,
    }


def _posed(offset: float = 100.0) -> ScanPose:
    return ScanPose(
        translation=np.array([offset, 0.0, 0.0], np.float64),
        rotation=np.eye(3, dtype=np.float64),
    )


def test_frame_decision_names_each_branch() -> None:
    """Every branch `_resolve_frame` can take has a name it can be recorded by.

    Without this the envelope's `frame_path` could only ever say `unrecorded`
    for the branches nothing exercises.
    """
    identity = ScanPose(
        translation=np.zeros(3, np.float64), rotation=np.eye(3, dtype=np.float64)
    )
    valid = np.ones(128, bool)

    spherical = {
        "sphericalRange": np.ones(128), "sphericalAzimuth": np.zeros(128),
        "sphericalElevation": np.zeros(128),
    }
    assert frame_decision(spherical, _posed(), valid) == FRAME_SPHERICAL_IS_LOCAL

    xyz = np.zeros((128, 3), np.float64)
    xyz[:, 0] = 1.0
    cols = np.arange(128, dtype=np.int64) // 4
    assert frame_decision({"cartesianX": xyz[:, 0]}, _posed(), valid) == (
        FRAME_NO_ROW_COLUMN
    )
    assert frame_decision(_rowcol_raw(xyz, cols), identity, valid) == (
        FRAME_IDENTITY_POSE
    )
    assert frame_decision(
        _rowcol_raw(xyz[:8], cols[:8]), _posed(), np.ones(8, bool)
    ) == FRAME_TOO_FEW_SAMPLES


def test_frame_decision_matches_what_resolve_frame_did() -> None:
    """The recorded branch is the branch taken, not a parallel re-derivation.

    A station whose points are stored in world coordinates: local azimuth
    within a column is tight, world azimuth is not, so `_resolve_frame`
    rewrites and the decision must say so.
    """
    from rapidmesh.e57_reader import _resolve_frame

    pose = _posed(50.0)
    n = 256
    cols = np.repeat(np.arange(n // 8, dtype=np.int64), 8)
    azimuth = cols.astype(np.float64) * (2.0 * np.pi / (n // 8))
    elevation = np.tile(np.linspace(-0.3, 0.3, 8), n // 8)
    local = np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=1,
    ) * 5.0
    world = pose.local_to_world(local)

    raw = _rowcol_raw(world, cols)
    valid = np.ones(n, bool)
    decision = frame_decision(raw, pose, valid)
    assert decision == FRAME_REWRITTEN_TO_LOCAL

    from rapidmesh.e57_reader import _positions

    raw_xyz, raw_rng = _positions(raw)
    xyz, rng = _resolve_frame(raw_xyz, raw_rng, raw, pose, valid)
    # Rewritten means the returned coordinates are the local ones.
    assert np.allclose(xyz, local, atol=1e-9)
    assert np.allclose(rng, np.linalg.norm(local, axis=1), atol=1e-9)
