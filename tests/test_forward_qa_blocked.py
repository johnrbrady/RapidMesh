"""Blocked forward-QA query — WP-1.10, a Phase 1 correction.

`REPORTS/2026-08-31-WP-3.0-report.md` §1.5 measured `qa.distances` building its
entire query batch at once, at **2,955 bytes per QA sample** — about 886 MB at
the default 300,000 samples, and independent of station size, because the
ragged gather is sized by queries times incident triangles rather than by the
mesh.

The fix is to split the query loop. Whether that is allowed to be a fix at all
rests on one claim, and this module exists to stop that claim being an
argument:

> Splitting the queries into batches cannot change any distance. Each query's
> candidate triangle set is the same triangles either way,
> `_point_triangle_distance` is elementwise, and the reduction is a minimum,
> which does not depend on how the queries were grouped.

So the equality asserted here is **bitwise**, on the raw float64 array, at four
block sizes, and `np.allclose` is never used — a change too small for a
tolerance is exactly the change `PHASE1-DETERMINISM-SPEC.md` §2 forbids.

The memory gate is a separate measurement in its own child process, because a
peak is a high-water mark and two workloads in one process share one.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.memory import measure_in_child
from rapidmesh.pipeline import mesh_station
from rapidmesh.qa import QA_QUERY_BLOCK, deviation_report, distances

# The gate from the WP-1.10 kickoff. Measured on this fixture with the mesh
# loaded rather than rebuilt: one batch 1,159,700,480 B, blocked at the default
# 385,892,352 B. The bar sits between them with room for machine variation.
GATE_ROWS, GATE_COLS = 500, 2000
GATE_QA_SAMPLES = 300_000
GATE_PEAK_BYTES = 560_000_000

# Block sizes the WP-3.0 scratch replica proved identical, re-asserted here
# against the real implementation. A block larger than the query count is
# covered separately: it must take the single-batch path and still agree.
BLOCK_SIZES = (100_000, 50_000, 25_000, 10_000)


@pytest.fixture(scope="module")
def meshed() -> tuple[object, np.ndarray]:
    """One station, meshed once, reused by every case in this module."""
    scan = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=60, cols=240,
        dropout=0.0, range_noise=0.002, seed=17, station_id="blocked",
    ).scan
    result = mesh_station(scan, measure=False)
    offsets = result.mesh.vertices
    return result.mesh, offsets


# ---------------------------------------------------------------------------
# the claim: blocking does not move the answer
# ---------------------------------------------------------------------------


def test_blocked_distances_are_bitwise_identical_to_one_batch(
    meshed: tuple[object, np.ndarray],
) -> None:
    mesh, points = meshed
    reference = distances(mesh, points, block=points.shape[0])  # type: ignore[arg-type]
    assert reference.size == points.shape[0]
    assert np.isfinite(reference).all()

    for block in BLOCK_SIZES:
        got = distances(mesh, points, block=block)  # type: ignore[arg-type]
        assert np.array_equal(got, reference), block


def test_blocking_holds_when_the_query_set_is_subsampled(
    meshed: tuple[object, np.ndarray],
) -> None:
    """`max_samples` draws the subsample **before** blocking, so the two must
    still agree — the seed picks the same queries either way."""
    mesh, points = meshed
    cap = points.shape[0] // 3
    reference = distances(
        mesh, points, max_samples=cap, seed=5, block=points.shape[0]  # type: ignore[arg-type]
    )
    assert reference.size == cap
    for block in (7_000, 1_000, 97):
        got = distances(mesh, points, max_samples=cap, seed=5, block=block)  # type: ignore[arg-type]
        assert np.array_equal(got, reference), block


def test_a_block_larger_than_the_query_count_is_one_batch(
    meshed: tuple[object, np.ndarray],
) -> None:
    mesh, points = meshed
    reference = distances(mesh, points, block=points.shape[0])  # type: ignore[arg-type]
    huge = distances(mesh, points, block=points.shape[0] * 4)  # type: ignore[arg-type]
    assert np.array_equal(huge, reference)


def test_report_statistics_are_unchanged_by_block_size(
    meshed: tuple[object, np.ndarray],
) -> None:
    """The exported figures, not only the raw array.

    Compared field for field with `==`, not `pytest.approx`: an RMS that moved
    in its last bit would still be a changed acceptance number.
    """
    import dataclasses

    mesh, points = meshed
    reference = deviation_report(mesh, points, block=points.shape[0])  # type: ignore[arg-type]
    for block in BLOCK_SIZES:
        got = deviation_report(mesh, points, block=block)  # type: ignore[arg-type]
        assert dataclasses.asdict(got) == dataclasses.asdict(reference), block


def test_k_one_also_blocks_identically(meshed: tuple[object, np.ndarray]) -> None:
    """`k=1` takes the `reshape(-1, 1)` branch rather than `atleast_2d`."""
    mesh, points = meshed
    reference = distances(mesh, points, k=1, block=points.shape[0])  # type: ignore[arg-type]
    for block in (25_000, 1_000):
        assert np.array_equal(
            distances(mesh, points, k=1, block=block), reference  # type: ignore[arg-type]
        ), block


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", [0, -1, -50_000])
def test_a_non_positive_block_is_refused(
    meshed: tuple[object, np.ndarray], block: int
) -> None:
    """Zero would loop forever; a negative one would silently return an
    uninitialised array, which is worse than either."""
    mesh, points = meshed
    with pytest.raises(ValueError, match="positive query count"):
        distances(mesh, points, block=block)  # type: ignore[arg-type]


def test_every_query_is_written(meshed: tuple[object, np.ndarray]) -> None:
    """Block sizes that do not divide the query count evenly.

    Checked two ways, because one of them is not reliable on its own: an
    unwritten slot holds infinity by construction, which `isfinite` catches —
    but if that initialisation is ever changed back to `np.empty`, the leftover
    memory can look like a perfectly ordinary distance. The bitwise comparison
    against the single-batch result is what actually pins it, and it is the
    check that caught this exact fault when it was injected deliberately.
    """
    mesh, points = meshed
    reference = distances(mesh, points, block=points.shape[0])  # type: ignore[arg-type]
    for block in (9_999, 10_001, 3, 1):
        got = distances(mesh, points, block=block)  # type: ignore[arg-type]
        assert got.size == points.shape[0]
        assert np.isfinite(got).all(), block
        assert np.array_equal(got, reference), block


# ---------------------------------------------------------------------------
# the measured gate — WP-1.10's reason for existing
# ---------------------------------------------------------------------------
#
# Workloads are module level and take JSON-serialisable arguments so
# `measure_in_child` can name one in a spec, exactly as
# `tools/measure_peak_memory.py` does. The mesh is built **once** into an npz
# and every measured child loads it: meshing is not the term under test, and
# charging it to each row would bury the one that is.


def prepare_gate_mesh(path: str, rows: int, cols: int, seed: int = 7) -> dict[str, Any]:
    """Mesh one station and save what `distances` reads. Not a measured row."""
    import numpy as np

    from rapidmesh import synthetic
    from rapidmesh.pipeline import mesh_station

    scan = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=rows, cols=cols,
        dropout=0.01, range_noise=0.002, seed=seed, station_id="qa-gate",
    ).scan
    mesh = mesh_station(scan, measure=False).mesh
    np.savez(
        path, vertices=mesh.vertices, triangles=mesh.triangles,
        origin=mesh.origin, translation=mesh.source_pose.translation,
        rotation=mesh.source_pose.rotation,
    )
    return {
        "vertices": int(mesh.vertex_count), "triangles": int(mesh.triangle_count),
    }


def _load_gate_mesh(path: str) -> tuple[Any, Any]:
    import numpy as np

    from rapidmesh.types import MeshData, ScanPose

    data = np.load(path)
    mesh = MeshData(
        origin=data["origin"], vertices=data["vertices"],
        triangles=data["triangles"],
        source_pose=ScanPose(
            translation=data["translation"], rotation=data["rotation"]
        ),
        normals=None, rgb=None, source_sample_id=None,
    )
    return mesh, data["vertices"]


def forward_qa(path: str, block: int, max_samples: int = 300_000) -> dict[str, Any]:
    """`distances` alone, at one block size. The measured row."""
    mesh, points = _load_gate_mesh(path)
    from rapidmesh.qa import distances as measure

    d = measure(mesh, points, max_samples=max_samples, block=block)
    return {
        "queries": int(d.size), "block": int(block),
        "triangles": int(mesh.triangle_count), "mean": float(d.mean()),
    }


def forward_qa_resident(
    path: str, block: int, max_samples: int = 300_000
) -> dict[str, Any]:
    """The pre-WP-1.10 shape: the whole station resident, one query batch."""
    mesh, points = _load_gate_mesh(path)
    from rapidmesh.qa_reference import resident_distances

    d = resident_distances(mesh, points, max_samples=max_samples, block=block)
    return {
        "queries": int(d.size), "block": int(block),
        "triangles": int(mesh.triangle_count), "mean": float(d.mean()),
    }


@pytest.fixture(scope="module")
def gate_mesh(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("qa-gate") / "mesh.npz")
    made = measure_in_child(
        "test_forward_qa_blocked", "prepare_gate_mesh",
        {"path": path, "rows": GATE_ROWS, "cols": GATE_COLS},
        label="prepare gate mesh (not a measured row)",
        sys_path=[str(Path(__file__).resolve().parent)], timeout=1800.0,
    )
    assert made.detail["triangles"] > 1_500_000, made.describe()
    return path


def test_the_blocked_query_meets_the_wp_1_10_peak_gate(gate_mesh: str) -> None:
    """The package's headline claim, measured with the gate instrument.

    Both rows in their own fresh process, because a peak only rises and two
    workloads in one process share one.
    """
    here = [str(Path(__file__).resolve().parent)]
    blocked = measure_in_child(
        "test_forward_qa_blocked", "forward_qa",
        {"path": gate_mesh, "block": QA_QUERY_BLOCK,
         "max_samples": GATE_QA_SAMPLES},
        label=f"forward QA, block={QA_QUERY_BLOCK}", sys_path=here, timeout=1800.0,
    )
    assert blocked.detail["queries"] == GATE_QA_SAMPLES
    assert blocked.peak_rss_bytes <= GATE_PEAK_BYTES, blocked.describe()


def test_the_gate_would_reject_the_unblocked_query(gate_mesh: str) -> None:
    """A gate that nothing fails is not a gate.

    The single-batch path is what WP-1.10 replaced; it must fail the bar the
    blocked path passes, on the same fixture and the same instrument. Without
    this row the gate above could be passing because the fixture is small.

    **Re-pointed by WP-3.1, and the reason matters.** This row used to run
    `qa.distances` with `block = max_samples`, because `block` was the only
    thing standing between the ragged gather and the whole query set. It is not
    any more: `qa_stream` bounds that expansion with its own cap, so
    `qa.distances` at one query batch now peaks at 233,439,232 B and passes a
    bar it is supposed to fail. Leaving the row pointed at `block` would have
    left a sensitivity break that no longer breaks — the exact failure mode this
    test exists to prevent, one level up. The pre-fix shape now has a name of its
    own, `qa_reference.resident_distances`, and that is what is measured here.
    """
    here = [str(Path(__file__).resolve().parent)]
    unblocked = measure_in_child(
        "test_forward_qa_blocked", "forward_qa_resident",
        {"path": gate_mesh, "block": GATE_QA_SAMPLES,
         "max_samples": GATE_QA_SAMPLES},
        label="forward QA, resident + one batch", sys_path=here, timeout=1800.0,
    )
    assert unblocked.peak_rss_bytes > GATE_PEAK_BYTES, unblocked.describe()


# ---------------------------------------------------------------------------
# the setting is recorded
# ---------------------------------------------------------------------------


def test_the_block_size_is_a_recorded_setting() -> None:
    """`PHASE1-DETERMINISM-SPEC.md` §2: a run cannot be reproduced from
    conditions that were not written down, and this one is now a condition."""
    scan = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=20, cols=80,
        dropout=0.0, range_noise=0.0, seed=3, station_id="settings",
    ).scan
    metadata = mesh_station(scan, measure=True).evidence_report("0f" * 32).metadata
    assert dict(metadata.settings)["qa_query_block"] == str(QA_QUERY_BLOCK)


def test_the_default_block_is_the_measured_one() -> None:
    """25,000 is not arbitrary, and it is not the first number tried.

    Measured on the gate fixture with the mesh loaded rather than rebuilt:
    432,226,304 B at 50,000 against 385,892,352 B at both 25,000 and 10,000. So
    25,000 is the largest batch that reaches the floor. WP-3.0 read the floor as
    50,000 because it measured with the mesh built in the same process, which
    put a higher high-water mark above the transient and hid it.

    A change here changes a published measurement and should fail until the
    measurement is redone.
    """
    assert QA_QUERY_BLOCK == 25_000
