"""
QA's station-scaling working set, bounded — WP-3.1.

`REPORTS/2026-08-31-WP-3.0-report.md` §4 measured forward QA holding about
3.06 GB of structure at the reference station that is not the answer: a float64
vertex copy, an int64 triangle copy and a CSR vertex→triangle map. Measured on
this repository's own ladder at HEAD `b46aad6`, the marginal over the mesh
prefix was 138 MB at 356,462 samples, 255 MB at 990,045 and **651 MB** at
2,534,315 — 185 then 256 B per sample, a slope that is not even self-consistent
because several structures are stacked inside it.

Two things have to be true at once, and as in `test_pass_b_bound.py` they pull
against each other:

* the bounded implementation must produce **exactly** what the resident one
  produced — bit for bit, on the raw float64 distance array; and
* its peak must stop growing with the station.

So the tests come in pairs. `rapidmesh.qa_reference.resident_distances` is the
WP-1.10 implementation, kept in the tree unchanged for precisely this purpose,
in the same way `triangulate.component_area_v1` was kept for Pass B. Equality is
asserted with `np.array_equal` and never `np.allclose`: a difference too small
for a tolerance is exactly the difference `PHASE1-DETERMINISM-SPEC.md` §2
forbids, and a tolerance wide enough to absorb it would be comparable to the
quantity being measured.

The one place the two implementations are *allowed* to differ is the identity of
a tied neighbour — `cKDTree` breaks an exact distance tie by traversal order,
the streamed search by lowest global id. That exposure is measured here and
reported as a number rather than assumed away.

The memory gate is measured with `PeakWorkingSetSize` in fresh child processes,
against a mesh **loaded from disk**. WP-1.10 learned why: with the mesh built in
the same process the pipeline's own high-water mark sits above the QA transient
and hides it.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.memory import measure_in_child
from rapidmesh.pipeline import mesh_station
from rapidmesh.qa import distances
from rapidmesh.qa_neighbours import (
    _box_distance,
    nearest_vertices,
    tied_neighbour_count,
)
from rapidmesh.qa_reference import resident_distances
from rapidmesh.qa_stream import (
    QA_PAIR_BLOCK,
    QA_TRIANGLE_BLOCK_BYTES,
    QA_VERTEX_BLOCK_BYTES,
    ResidentMesh,
    surface_distances,
)
from rapidmesh.reverse_qa import mesh_to_source_report_v2, run_reverse_qa
from rapidmesh.reverse_qa_select import cumulative_total, selection_targets
from rapidmesh.types import MeshData

TOOLS = str(pathlib.Path(__file__).resolve().parent.parent / "tools")

# The WP-3.1 kickoff gate.
FORWARD_MARGINAL_GATE_BYTES = 64_000_000
FORWARD_SLOPE_GATE_BYTES = 8.0

# The WP ladder. Cells minus a 1% dropout, so 356k / 990k / 2.53M samples.
LADDER = ((300, 1200), (500, 2000), (800, 3200))
GATE_QA_SAMPLES = 300_000

# `mesh_to_source_report_v2` on the module fixture, run against a checkout of
# HEAD `b46aad6` with every WP-3.1 file removed from the tree — the WP-1.10
# baseline, written down so "unchanged" is a comparison and not a recollection.
# `repr`-exact float64 literals; a rounded copy would make the assertion a
# tolerance in disguise.
#
# The figures are large because the fixture is coarse — a 60 x 240 lattice puts
# lattice steps of order a metre on the far wall, so a triangle's interior point
# is genuinely far from any vertex. They are a fingerprint of this computation,
# not a quality claim about anything.
REVERSE_BASELINE = {
    "sampled_points": 27003,
    "rms": 0.07118714908046828,
    "mean": 0.05990109550866801,
    "p95": 0.13725047379139999,
    "p99_9": 0.24903526050548352,
    "maximum": 0.28943760700878696,
    "within_2mm": 0.0008517572121616117,
    "within_5mm": 0.006814057697292894,
}
REVERSE_BASELINE_TRIANGLES = 27003


@pytest.fixture(scope="module")
def meshed() -> tuple[MeshData, dict[str, Any]]:
    """One station, meshed once, with three deliberately different query sets.

    The mesh's own vertices are the set both QA directions actually use, and on
    a fixture without decimation every one of them is a vertex of its own
    triangles, so every distance is exactly zero. That is a real property of the
    metric and a weak test of an equality: an implementation that returned zero
    unconditionally would pass it. The jittered and analytic-truth sets exist so
    the comparison has non-degenerate numbers to disagree about.
    """
    fixture = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=60, cols=240,
        dropout=0.0, range_noise=0.002, seed=17, station_id="qabound",
    )
    mesh = mesh_station(fixture.scan, measure=False).mesh
    rng = np.random.default_rng(3)
    truth = fixture.scan.pose.rotate_local(
        fixture.true_range[:, None] * fixture.direction
    ).astype(np.float32)
    queries = {
        "mesh-vertices": mesh.vertices,
        "jittered": (
            mesh.vertices + rng.normal(0.0, 0.01, mesh.vertices.shape)
        ).astype(np.float32),
        "analytic-truth": truth,
    }
    return mesh, queries


# ---------------------------------------------------------------------------
# the claim: bounding the geometry did not move any number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ("mesh-vertices", "jittered", "analytic-truth"))
def test_streamed_distances_match_the_resident_reference(
    meshed: tuple[MeshData, dict[str, Any]], kind: str
) -> None:
    mesh, queries = meshed
    points = queries[kind]
    reference = resident_distances(mesh, points)
    assert reference.size == points.shape[0]
    assert np.isfinite(reference).all()
    assert np.array_equal(distances(mesh, points), reference)


def test_the_non_degenerate_query_sets_actually_measure_something(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """Guard on the guard. If every query set collapsed to zero the equality
    above would hold for an implementation that measured nothing at all."""
    mesh, queries = meshed
    assert resident_distances(mesh, queries["mesh-vertices"]).max() == 0.0
    for kind in ("jittered", "analytic-truth"):
        assert resident_distances(mesh, queries[kind]).max() > 1e-3, kind


@pytest.mark.parametrize("vertex_bytes", (QA_VERTEX_BLOCK_BYTES, 200_000, 24_000))
@pytest.mark.parametrize("triangle_bytes", (QA_TRIANGLE_BLOCK_BYTES, 40_000))
def test_identity_holds_at_every_block_size(
    meshed: tuple[MeshData, dict[str, Any]], vertex_bytes: int, triangle_bytes: int
) -> None:
    """Block boundaries are not allowed to be visible in the output.

    The small caps are the point: at 24,000 bytes the vertex set is cut into
    fifteen blocks, so the neighbour merge and the bounding-box prune both do
    real work rather than degenerating into the single-tree case.
    """
    mesh, queries = meshed
    points = queries["jittered"]
    reference = resident_distances(mesh, points)
    source = ResidentMesh(
        mesh, vertex_block_bytes=vertex_bytes, triangle_block_bytes=triangle_bytes
    )
    got = surface_distances(
        source, points.astype(np.float64), k=2, workers=1, block=25_000
    )
    assert np.array_equal(got, reference)


@pytest.mark.parametrize("pair_block", (QA_PAIR_BLOCK, 5_000, 97))
def test_identity_holds_at_every_expansion_cap(
    meshed: tuple[MeshData, dict[str, Any]], pair_block: int
) -> None:
    mesh, queries = meshed
    points = queries["analytic-truth"]
    reference = resident_distances(mesh, points)
    got = surface_distances(
        ResidentMesh(mesh), points.astype(np.float64), k=2, workers=1,
        block=25_000, pair_block=pair_block,
    )
    assert np.array_equal(got, reference)


def test_the_comparison_would_notice_a_single_ulp(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """Sensitivity break. An equality assertion that cannot fail proves nothing.

    One vertex is moved by one float32 step — far below any tolerance anyone
    would write — and the comparison must reject it.
    """
    mesh, queries = meshed
    points = queries["jittered"]
    reference = resident_distances(mesh, points)

    moved = mesh.vertices.copy()
    moved[0, 0] = np.nextafter(moved[0, 0], np.float32(1e30))
    assert moved[0, 0] != mesh.vertices[0, 0]
    nudged = MeshData(
        origin=mesh.origin, vertices=moved, triangles=mesh.triangles,
        source_pose=mesh.source_pose, source_sample_id=mesh.source_sample_id,
    )
    assert not np.array_equal(distances(nudged, points), reference)


# ---------------------------------------------------------------------------
# the candidate set is unchanged — the hard exclusion in the WP-3.1 kickoff
# ---------------------------------------------------------------------------


def test_the_neighbour_search_is_still_global(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """The two nearest vertices over the whole mesh, not over a block.

    Compared against one `cKDTree` over the entire station — the definition the
    metric is written against. Distances must agree exactly; the *indices* may
    differ only where the distance is tied, and that count is asserted to be the
    only disagreement rather than waved at.
    """
    from scipy.spatial import cKDTree

    mesh, queries = meshed
    points = queries["jittered"].astype(np.float64)
    whole = cKDTree(mesh.vertices.astype(np.float64))
    want_d, want_i = whole.query(points, k=2, workers=1)

    source = ResidentMesh(mesh, vertex_block_bytes=24_000)
    got_d, got_i = nearest_vertices(source, points, k=2, workers=1, block=25_000)

    assert np.array_equal(got_d, np.asarray(want_d, np.float64))
    differ = np.nonzero(got_i != np.asarray(want_i, np.int64))
    tied = np.asarray(want_d, np.float64)[differ] == got_d[differ]
    assert bool(np.all(tied)), "a neighbour differed at an untied distance"
    assert tied_neighbour_count(got_d) == 0, (
        "this fixture is expected to have no rank-2 distance ties; if it grows "
        "some, the identity assertions above are the thing that must still hold"
    )


def test_the_box_distance_is_a_lower_bound() -> None:
    """The prune is only exact if the box distance never over-states.

    Checked against the true minimum over the block, on random boxes, because
    an over-stating bound would silently discard a genuinely nearer vertex and
    the result would still look plausible.
    """
    rng = np.random.default_rng(11)
    for _ in range(20):
        block = rng.normal(0.0, 1.0, (200, 3))
        queries = rng.normal(0.0, 3.0, (500, 3))
        bound = _box_distance(queries, block)
        true_min = np.sqrt(
            ((queries[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)
        ).min(axis=1)
        assert np.all(bound <= true_min + 1e-12)


def test_the_prune_is_exercised_and_still_exact(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """A block cut small enough that most queries can skip most blocks.

    Without the prune this is the same answer more slowly; with a *non-strict*
    prune it would be a different answer, so the assertion is the equality and
    the small block is what makes the branch run at all.
    """
    mesh, queries = meshed
    points = queries["jittered"]
    source = ResidentMesh(mesh, vertex_block_bytes=24_000)
    assert sum(1 for _ in source.vertex_blocks()) > 5
    got = surface_distances(
        source, points.astype(np.float64), k=2, workers=1, block=25_000
    )
    assert np.array_equal(got, resident_distances(mesh, points))


# ---------------------------------------------------------------------------
# reverse QA — the cumulative, streamed
# ---------------------------------------------------------------------------


def _reverse_runs(mesh: MeshData, work: pathlib.Path) -> Any:
    from rapidmesh.qa_stream import as_source
    from rapidmesh.reverse_qa import _build_reverse_runs

    return _build_reverse_runs(as_source(mesh), work)


def test_the_streamed_cumulative_is_the_resident_one(
    meshed: tuple[MeshData, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """`cumsum` is a strict left-to-right recurrence; the carry must be inside
    it, not added to it. The wrong form differs by a rounding step, which is
    exactly the size of error that moves a systematic selection by one triangle
    and cannot be seen in a summary statistic."""
    from rapidmesh.pass_b_merge import merge_runs

    mesh, _ = meshed
    runs = _reverse_runs(mesh, tmp_path)
    resident = np.cumsum(
        np.concatenate([block["area"] for block in merge_runs(runs)])
    )
    assert resident.size == runs.record_count
    assert cumulative_total(runs) == float(resident[-1])


def test_the_streamed_selection_picks_the_resident_records(
    meshed: tuple[MeshData, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """The resident rule, spelled out here so the streamed one is compared
    against the algorithm and not only against its own output."""
    from rapidmesh.pass_b_merge import merge_runs
    from rapidmesh.reverse_qa_select import select_and_gather

    mesh, _ = meshed
    runs = _reverse_runs(mesh, tmp_path)
    merged = np.concatenate(list(merge_runs(runs)))
    cumulative = np.cumsum(merged["area"])
    count = runs.record_count

    n = min(300_000, max(count, 10_000))
    targets = selection_targets(float(cumulative[-1]), n)
    picked = np.clip(
        np.searchsorted(cumulative, targets, side="right"), 0, count - 1
    )
    want = merged[picked]

    _, triples, vertex_index = select_and_gather(runs, targets)
    assert np.array_equal(
        triples, np.stack([want["s0"], want["s1"], want["s2"]], axis=1)
    )
    assert np.array_equal(
        vertex_index,
        np.stack([want["i0"], want["i1"], want["i2"]], axis=1).astype(np.int64),
    )


def test_the_reverse_report_is_unchanged_from_the_wp_1_10_baseline(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """Against numbers captured before the edit, as exact float64 literals."""
    mesh, _ = meshed
    report, evidence = mesh_to_source_report_v2(
        mesh, mesh.vertices, source_rows=_rows_for(mesh), max_samples=300_000
    )
    for name, value in REVERSE_BASELINE.items():
        assert getattr(report, name) == value, name
    assert evidence.samples_measured + evidence.samples_unmatched == (
        evidence.samples_selected
    )


def test_the_reverse_baseline_would_catch_a_moved_selection(
    meshed: tuple[MeshData, dict[str, Any]]
) -> None:
    """Sensitivity break, so the pinned literals are testing something.

    Two independent nudges, because they enter the figure by different routes.
    Moving the seed changes only the interior point each selected triangle
    contributes; moving one vertex by a single float32 step changes that
    triangle's area, and therefore the cumulative, and therefore *which*
    triangles the systematic selection lands on.

    Narrowing the row window is deliberately **not** used here. It does not move
    this fixture's figure at all — every sample's nearest retained observation is
    a corner of its own triangle, so the window has nothing to add — and a
    sensitivity break that happens to be insensitive is worse than none.
    """
    mesh, _ = meshed
    rows = _rows_for(mesh)
    reseeded, _ = mesh_to_source_report_v2(
        mesh, mesh.vertices, source_rows=rows, max_samples=300_000, seed=1
    )
    assert reseeded.rms != REVERSE_BASELINE["rms"]

    moved = mesh.vertices.copy()
    moved[0, 0] = np.nextafter(moved[0, 0], np.float32(1e30))
    assert moved[0, 0] != mesh.vertices[0, 0]
    nudged = MeshData(
        origin=mesh.origin, vertices=moved, triangles=mesh.triangles,
        source_pose=mesh.source_pose, source_sample_id=mesh.source_sample_id,
    )
    shifted, _ = mesh_to_source_report_v2(
        nudged, moved, source_rows=rows, max_samples=300_000
    )
    assert shifted.rms != REVERSE_BASELINE["rms"]


def _rows_for(mesh: MeshData) -> Any:
    """Lattice rows for the mesh's own vertices, recovered from the sample ids.

    The fixture is generated at `cols = 240` with no dropout, so sample id
    integer-divided by the column count is the row. Doing it here keeps the
    baseline literals reproducible from the mesh alone.
    """
    ids = np.asarray(mesh.source_sample_id, np.int64)
    return (ids // 240).astype(np.int32)


# ---------------------------------------------------------------------------
# the memory gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ladder(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Three stations, meshed once each and saved, plus their measured rows.

    Every row is one child process, so a peak is that workload's own high-water
    mark and not one inherited from the row before it.
    """
    root = tmp_path_factory.mktemp("qa-ladder")
    rows: list[dict[str, Any]] = []
    for lattice_rows, cols in LADDER:
        scan_path = str(root / f"scan-{lattice_rows}x{cols}.npz")
        mesh_path = str(root / f"mesh-{lattice_rows}x{cols}.npz")
        made = measure_in_child(
            "measure_peak_memory", "prepare_fixture",
            {"path": scan_path, "rows": lattice_rows, "cols": cols},
            label="fixture", sys_path=[TOOLS],
        )
        measure_in_child(
            "measure_peak_memory", "prepare_mesh",
            {"fixture": scan_path, "path": mesh_path},
            label="mesh", sys_path=[TOOLS],
        )
        prefix = measure_in_child(
            "measure_peak_memory", "mesh_only", {"path": mesh_path},
            label="mesh only", sys_path=[TOOLS],
        )
        forward = measure_in_child(
            "measure_peak_memory", "mesh_forward_qa",
            {"path": mesh_path, "samples": GATE_QA_SAMPLES},
            label="forward qa", sys_path=[TOOLS],
        )
        resident = measure_in_child(
            "measure_peak_memory", "mesh_forward_qa_resident",
            {"path": mesh_path, "samples": GATE_QA_SAMPLES},
            label="forward qa, resident", sys_path=[TOOLS],
        )
        assert forward.detail["rms"] == resident.detail["rms"], (
            "the two implementations disagreed on the figure being measured"
        )
        rows.append({
            "samples": int(made.detail["samples"]),
            "prefix": prefix.peak_rss_bytes,
            "forward": forward.peak_rss_bytes,
            "marginal": forward.peak_rss_bytes - prefix.peak_rss_bytes,
            "resident_marginal": resident.peak_rss_bytes - prefix.peak_rss_bytes,
            "seconds": forward.seconds,
            "resident_seconds": resident.seconds,
        })
    return rows


def test_forward_qa_marginal_stays_under_the_gate(
    ladder: list[dict[str, Any]]
) -> None:
    """`peak(mesh + forward QA) - peak(mesh alone)`, both in fresh children."""
    over = [row for row in ladder if row["marginal"] > FORWARD_MARGINAL_GATE_BYTES]
    assert not over, "\n".join(
        f"{row['samples']:,} samples: marginal {row['marginal']:,} B "
        f"> {FORWARD_MARGINAL_GATE_BYTES:,} B"
        for row in over
    )


def test_forward_qa_marginal_does_not_grow_with_the_station(
    ladder: list[dict[str, Any]]
) -> None:
    """Every consecutive slope, not only the end points.

    DEC-014 requires at least three sizes and a statement of which slopes
    agreed; two points can be joined by a line whatever they are, so a
    two-point slope is not evidence of anything.
    """
    assert len(ladder) >= 3
    slopes = [
        (later["marginal"] - earlier["marginal"])
        / (later["samples"] - earlier["samples"])
        for earlier, later in zip(ladder[:-1], ladder[1:], strict=True)
    ]
    assert all(slope <= FORWARD_SLOPE_GATE_BYTES for slope in slopes), (
        f"slopes {[round(s, 3) for s in slopes]} B/sample exceed "
        f"{FORWARD_SLOPE_GATE_BYTES} B/sample"
    )


def test_the_ladder_actually_spans_a_range(ladder: list[dict[str, Any]]) -> None:
    """A flat slope across three nearly identical sizes would prove nothing."""
    assert ladder[-1]["samples"] >= 6 * ladder[0]["samples"]


def test_the_gate_rejects_the_implementation_it_replaced(
    ladder: list[dict[str, Any]]
) -> None:
    """A gate that nothing fails is not a gate.

    `qa_reference.resident_distances` is measured on the same fixtures, in the
    same instrument, against the same prefix — and must fail the bar the bounded
    path passes. Its slope must also be plainly non-flat, because a bound that
    the old shape also satisfied would mean the ladder was too small rather than
    that anything had been bounded.
    """
    worst = max(row["resident_marginal"] for row in ladder)
    assert worst > FORWARD_MARGINAL_GATE_BYTES, (
        f"the resident implementation peaked at only {worst:,} B over the "
        "prefix; the ladder is not large enough to be evidence"
    )
    slope = (
        ladder[-1]["resident_marginal"] - ladder[0]["resident_marginal"]
    ) / (ladder[-1]["samples"] - ladder[0]["samples"])
    assert slope > FORWARD_SLOPE_GATE_BYTES


def test_reverse_qa_selection_is_bounded(
    meshed: tuple[MeshData, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """The reverse selection holds one merge block, never the station.

    Asserted structurally rather than by peak: the arrays the pass returns are
    sized by the sample cap, and the run set is read through `merge_runs`, whose
    own caps `test_pass_b_bound.py` already pins.
    """
    mesh, _ = meshed
    cap = 512
    values, evidence = run_reverse_qa(
        mesh, mesh.vertices, source_rows=_rows_for(mesh), max_samples=cap,
        work_dir=tmp_path,
    )
    assert evidence.positive_area_triangles > 10 * cap
    assert evidence.samples_selected == cap
    assert values.size <= cap
