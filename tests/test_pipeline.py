"""
Pipeline tests, written as budget checks rather than golden values.

Every assertion here is a number from `00-PRODUCT-DEFINITION.md` §4. That is
deliberate: a test that pins the current output ("1,929,059 triangles") fails
every time the algorithm improves and teaches nobody anything. A test that
pins the *commitment* fails only when a commitment is broken, which is the
only time anyone should be interrupted.

The fixture is a million samples per station, which takes the suite to about a
minute. That is deliberate too: the accuracy budgets are resolution-dependent
(see the table below), and a fixture small enough to be instant would be
scoring the pipeline in a regime the product never operates in.
`tools/bench_synthetic.py` is the same scene at whatever resolution you want,
for when a threshold is actually being tuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidmesh import qa, synthetic
from rapidmesh.filters import (
    carve_movers,
    columns_wrap,
    isolation_despeckle,
    restore_parallax_carve,
)
from rapidmesh.grid import CoarseRangeGrid, ScanGrid, select
from rapidmesh.pipeline import mesh_station
from rapidmesh.triangulate import cull_islands, triangulate
from rapidmesh.types import LatticeSource

# 500 x 2000 is 0.18 degrees per sample. Chosen because it is roughly the
# angular resolution Cairn's fixed 2048 x 1024 grid produces, so the budgets
# below are being met at a resolution Cairn can be compared against directly —
# and because accuracy is genuinely resolution-dependent, which is worth
# stating rather than hiding behind a fast fixture.
#
# Measured on this scene (tools/bench_synthetic.py sweeps it):
#
#   0.300 deg   rms 1.76 mm   p99.9 19.49 mm   recall 96.8%   fp 0.304%
#   0.225 deg   rms 1.46 mm   p99.9 13.97 mm   recall 96.6%   fp 0.190%
#   0.180 deg   rms 1.26 mm   p99.9  4.83 mm   recall 97.0%   fp 0.065%   <- here
#   0.129 deg   rms 1.16 mm   p99.9  4.49 mm   recall 97.0%   fp 0.100%
#   0.090 deg   rms 1.00 mm   p99.9  3.55 mm   recall 97.0%   fp 0.015%
#
# A Trimble X7 at high resolution samples at about 0.017 degrees, so real data
# sits well to the right of this table. The budgets in
# 00-PRODUCT-DEFINITION.md §4 are stated for scans at or near native
# resolution; they are NOT claimed for a coarse preview scan, and this table is
# the evidence for where the line sits.
ROWS, COLS = 500, 2000


@pytest.fixture(scope="module")
def station_a() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(scanner=(0.35, -0.2, 0.0), mover=True),
        rows=ROWS, cols=COLS, station_id="A", seed=7,
    )


@pytest.fixture(scope="module")
def neighbours() -> list[CoarseRangeGrid]:
    b = synthetic.generate(
        synthetic.RoomScene(scanner=(-2.6, 1.7, 0.0), mover=False),
        rows=ROWS, cols=COLS, station_id="B", seed=11,
    )
    c = synthetic.generate(
        synthetic.RoomScene(scanner=(2.7, 1.4, 0.0), mover=False),
        rows=ROWS, cols=COLS, station_id="C", seed=13,
    )
    return [CoarseRangeGrid.build(b.scan), CoarseRangeGrid.build(c.scan)]


@pytest.fixture(scope="module")
def meshed(station_a: synthetic.SyntheticScan):  # type: ignore[no-untyped-def]
    """The pipeline run once, with no neighbours. Module-scoped because
    meshing a million samples per test would make the suite tedious enough
    that people stop running it."""
    return mesh_station(station_a.scan, measure=True)


# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------


def test_lattice_is_native_resolution(station_a: synthetic.SyntheticScan) -> None:
    """The whole premise: no fixed grid anywhere.

    Cairn bins to 2048x1024 regardless of what the scanner sampled. If a
    constant like that ever reappears here, this fails.
    """
    lat = station_a.scan.lattice
    assert (lat.rows, lat.cols) == (ROWS, COLS)
    assert lat.source is LatticeSource.SYNTHETIC


def test_row_major_ordering_is_maintained(station_a: synthetic.SyntheticScan) -> None:
    """ScanGrid's CSR index is only valid on sorted input and does not re-sort."""
    s = station_a.scan
    key = s.row.astype(np.int64) * s.lattice.cols + s.col
    assert np.all(np.diff(key) >= 0)


def test_dense_band_matches_csr_index(station_a: synthetic.SyntheticScan) -> None:
    """Every populated cell of a band must name a sample that claims exactly
    that row and column. This is the invariant the whole banding scheme rests
    on — an off-by-one here silently shears the mesh by one lattice row."""
    grid = ScanGrid.build(station_a.scan)
    r0 = 10
    band = grid.dense_rows(r0, r0 + 3)
    rr, cc = np.nonzero(band >= 0)
    idx = band[band >= 0]
    assert np.array_equal(grid.scan.row[idx], (rr + r0).astype(np.int32))
    assert np.array_equal(grid.scan.col[idx], cc.astype(np.int32))


def test_full_sweep_wraps_partial_does_not(station_a: synthetic.SyntheticScan) -> None:
    """Column wrap must follow the actual field of view.

    Treating a partial-FOV scan as a full sweep joins its two ends into a ring
    and lays a sheet of triangles across the middle of the room.
    """
    from dataclasses import replace

    assert columns_wrap(station_a.scan)
    narrow = replace(
        station_a.scan,
        lattice=replace(station_a.scan.lattice, az_step=station_a.scan.lattice.az_step / 4),
    )
    assert not columns_wrap(narrow)


# --------------------------------------------------------------------------
# accuracy budgets — 00-PRODUCT-DEFINITION.md §4
# --------------------------------------------------------------------------


def test_accuracy_against_analytic_truth(station_a, meshed) -> None:  # type: ignore[no-untyped-def]
    """RMS <= 2 mm and p99.9 <= 8 mm against the real surface, not the points.

    This is the number the product definition commits to, and it is measured
    against analytic geometry with the instrument noise removed — so it cannot
    be inflated by fitting the noise, which is what measuring against the input
    points would reward.

    Static surfaces only: where a mover blocked the view, the truth is a
    surface that was never observed, and leaving a hole there is correct
    behaviour rather than an error.
    """
    static = ~station_a.is_mover
    truth = (station_a.direction[static] * station_a.true_range[static][:, None]).astype(np.float32)
    rep = qa.deviation_report(meshed.mesh, truth, max_samples=150_000)

    assert rep.rms <= 0.002, f"RMS {rep.rms * 1000:.2f} mm exceeds the 2 mm budget"
    assert rep.p99_9 <= 0.008, f"p99.9 {rep.p99_9 * 1000:.2f} mm exceeds the 8 mm budget"


def test_undecimated_mesh_passes_through_its_own_points(meshed) -> None:  # type: ignore[no-untyped-def]
    """Fidelity to the retained points is sub-millimetre before decimation.

    Undecimated, nearly every mesh vertex IS a source point, so this is close
    to zero by construction; the residual comes from samples that survived
    filtering but landed in no triangle. It becomes the meaningful budget check
    once decimation lands, and until then it is a cheap guard against a
    coordinate-frame mistake — mixing scanner-local offsets with project
    coordinates would show up here as hundreds of kilometres, not millimetres.
    """
    assert meshed.deviation is not None
    assert meshed.deviation.rms < 0.001, meshed.deviation.summary()
    assert meshed.deviation.within_2mm > 0.999


# --------------------------------------------------------------------------
# detail preservation — the risk the brief names
# --------------------------------------------------------------------------


def _survival(syn: synthetic.SyntheticScan, in_mesh: np.ndarray, surf: int) -> float:
    sel = (syn.surface == surf) & ~syn.is_mover
    n = int(sel.sum())
    return float((sel & in_mesh).sum()) / n if n else 1.0


def _run(syn: synthetic.SyntheticScan, others: list[CoarseRangeGrid], **kw: object) -> np.ndarray:
    """Pipeline that reports which ORIGINAL samples reached the mesh."""
    grid = ScanGrid.build(syn.scan)
    keep = isolation_despeckle(grid)
    stage = select(grid.scan, keep)
    if others:
        drop = ~carve_movers(stage, others, min_votes=int(kw.get("min_votes", 1)))
        if kw.get("restore", True):
            drop = restore_parallax_carve(ScanGrid.build(stage), drop)
        keep[np.nonzero(keep)[0][drop]] = False
        stage = select(stage, ~drop)
    g = ScanGrid.build(stage)
    tris = cull_islands(g.scan.xyz, triangulate(g))
    used = np.zeros(len(stage), bool)
    if len(tris):
        used[tris.ravel()] = True
    in_mesh = np.zeros(len(syn.scan), bool)
    in_mesh[np.nonzero(keep)[0]] = used
    return in_mesh


def test_thin_detail_survives_filtering(
    station_a: synthetic.SyntheticScan, neighbours: list[CoarseRangeGrid]
) -> None:
    """A 60 mm handrail and a 300 mm column must not be filtered away.

    This is the brief's stated worry ("avoid stripping out real fine detail")
    turned into a number. 98 % rather than 100 % so ordinary sampling variation
    at the silhouette does not make the suite flaky.
    """
    in_mesh = _run(station_a, neighbours, min_votes=1, restore=True)
    assert _survival(station_a, in_mesh, synthetic.SURF_RAIL) >= 0.98
    assert _survival(station_a, in_mesh, synthetic.SURF_COLUMN) >= 0.98
    assert _survival(station_a, in_mesh, synthetic.SURF_BEYOND_DOOR) >= 0.98


def test_parallax_restore_beats_plain_one_vote_carving(
    station_a: synthetic.SyntheticScan, neighbours: list[CoarseRangeGrid]
) -> None:
    """The reason `restore` defaults to on.

    One-vote carving alone nicks the handrail, because a thin object standing
    off a wall is genuinely invisible from some angles and one station
    legitimately reports the wall behind it. Restoration recovers those without
    rebuilding the mover.
    """
    plain = _run(station_a, neighbours, min_votes=1, restore=False)
    restored = _run(station_a, neighbours, min_votes=1, restore=True)
    assert _survival(station_a, restored, synthetic.SURF_RAIL) >= _survival(
        station_a, plain, synthetic.SURF_RAIL
    )
    # and it must not resurrect the mover it just removed
    assert qa.score_mover_filter(station_a.is_mover, restored)["recall"] >= 0.90


def test_mover_removal_meets_budget(
    station_a: synthetic.SyntheticScan, neighbours: list[CoarseRangeGrid]
) -> None:
    """>= 95 % recall, <= 0.1 % false positives."""
    in_mesh = _run(station_a, neighbours, min_votes=1, restore=True)
    score = qa.score_mover_filter(station_a.is_mover, in_mesh)
    assert score["recall"] >= 0.95, f"recall {score['recall']:.3f}"
    assert score["false_positive"] <= 0.001, f"false positives {score['false_positive']:.5f}"


def test_no_carving_means_no_mover_removal(station_a: synthetic.SyntheticScan) -> None:
    """A single isolated scan cannot have its movers removed, and must not
    pretend to. See the `filters.py` docstring: nothing here claims to do
    single-scan mover detection, so a lone station keeps its ghost."""
    in_mesh = _run(station_a, [])
    assert qa.score_mover_filter(station_a.is_mover, in_mesh)["recall"] < 0.05


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_doorway_is_not_bridged(meshed) -> None:  # type: ignore[no-untyped-def]
    """No triangle may span the 4 m depth jump through the doorway.

    The signature artefact of grid triangulation. A bridged doorway produces a
    triangle with an edge far longer than the local sampling allows, so the
    test is simply that no such edge exists.
    """
    v, t = meshed.mesh.vertices, meshed.mesh.triangles.astype(np.int64)
    longest = np.maximum.reduce([
        np.linalg.norm(v[t[:, 1]] - v[t[:, 0]], axis=1),
        np.linalg.norm(v[t[:, 2]] - v[t[:, 1]], axis=1),
        np.linalg.norm(v[t[:, 0]] - v[t[:, 2]], axis=1),
    ])
    assert longest.max() < 1.0, f"longest edge {longest.max():.2f} m — a discontinuity was bridged"


def test_normals_are_present_and_face_the_scanner(meshed) -> None:  # type: ignore[no-untyped-def]
    """Normals are data, not baked lighting (00-PRODUCT-DEFINITION.md §7).

    Per-station meshing makes orientation unambiguous: every sample was seen
    from the origin, so every normal must have a non-negative dot product with
    the direction back to it.
    """
    n = meshed.mesh.normals
    assert n is not None
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)
    to_scanner = -meshed.mesh.vertices
    facing = np.einsum("ij,ij->i", n, to_scanner)
    assert float((facing >= 0).mean()) > 0.999


def test_colour_is_not_modified(meshed) -> None:  # type: ignore[no-untyped-def]
    """Cairn multiplies a lambert term into vertex RGB and throws the normals
    away, which contaminates survey colour irreversibly. Every colour in the
    output must still be one of the palette values the fixture assigned."""
    assert meshed.mesh.rgb is not None
    unique = {tuple(c) for c in np.unique(meshed.mesh.rgb, axis=0)}
    palette = {(200, 196, 188), (120, 130, 150), (170, 160, 150), (90, 90, 95), (200, 80, 70), (0, 0, 0)}
    assert unique <= palette, f"unexpected colours after meshing: {unique - palette}"


def test_origin_is_float64_and_vertices_are_offsets(meshed) -> None:  # type: ignore[no-untyped-def]
    """Millimetres at MGA coordinates depend on this and nothing else."""
    assert meshed.mesh.origin.dtype == np.float64
    assert meshed.mesh.vertices.dtype == np.float32
    assert np.abs(meshed.mesh.vertices).max() < 100.0  # offsets, not world coordinates


def test_empty_scan_does_not_crash() -> None:
    """Degenerate input is a normal occurrence (a scan aborted after two
    columns) and must produce an empty mesh, not an exception."""
    syn = synthetic.generate(rows=4, cols=8, dropout=1.0, seed=1)
    if len(syn.scan) == 0:
        res = mesh_station(syn.scan, measure=False)
        assert res.mesh.triangle_count == 0
