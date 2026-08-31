"""
Streaming component labelling, two-pass finalisation and the segment store —
PLAN.md §5 item 8, falsifiers from `PHASE1-ISLANDS-FINALISATION.md` §7.

Three claims are under test, and each has a test that can fail:

    A  Pass A's live state is bounded by the occupied width of one lattice row,
       because a component absent from the frontier can never grow again.
    B  While the §4.1 exactness condition holds, streamed area accumulation is
       bitwise equal to `cull_islands`' under any order — and the condition is
       checked, not assumed.
    C  Per-band culling breaks band-size invariance; the two-pass design does
       not.

Claim C is the one with a counter-example rather than an assertion: the same
station is culled per-band and two-pass, and the per-band answer is shown
deleting geometry the two-pass answer keeps.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.carvegrid import StationRef
from rapidmesh.grid import CoarseRangeGrid
from rapidmesh.islands import ComponentTable, UnionFind, pass_a_sweep
from rapidmesh.pipeline import carve_grids_streamed, mesh_station, mesh_station_streamed
from rapidmesh.segments_io import (
    SegmentError,
    read_component_table,
    read_disp_segment,
    read_pos_segment,
    read_tri_segment,
)
from rapidmesh.triangulate import (
    EXACTNESS_RATIO_LIMIT,
    MIN_COMPONENT_TRIANGLES,
    component_area_v1,
    cull_islands,
)

WITNESS_ROWS, WITNESS_COLS = 16, 360


@pytest.fixture(scope="module")
def witness() -> synthetic.HaloWitnessFixture:
    return synthetic.generate_halo_witness()


@pytest.fixture(scope="module")
def neighbours(witness: synthetic.HaloWitnessFixture) -> list[CoarseRangeGrid]:
    """Two neighbour carve grids for the witness, built by the item-7 path.

    The witness ships without neighbours, so carve and restore never ran on it
    (recorded as finding 3 of the WP-1.2 report). Generating the same scene
    from two other setups — mover removed, because a neighbour that sees the
    mover cannot carve it — gives the fixture the cross-station evidence the
    filter chain needs, without changing `synthetic.py`.
    """
    def other(station_id: str, scanner: tuple[float, float, float], seed: int) -> Any:
        scene = dataclasses.replace(witness.scene, scanner=scanner, mover=False)
        return synthetic.generate(
            scene, rows=WITNESS_ROWS, cols=WITNESS_COLS, dropout=0.0,
            range_noise=0.0, seed=seed, station_id=station_id,
        ).scan

    scans = [
        witness.scan,
        other("W-B", (-2.6, 1.7, 0.0), 31),
        other("W-C", (2.7, 1.4, 0.0), 37),
    ]
    return carve_grids_streamed(
        [StationRef.from_scan(s) for s in scans],
        exclude=witness.scan.station_id,
        nearest=2,
        chunk_points=50_000,
    )


@pytest.fixture(scope="module")
def reference(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid]
) -> Any:
    return mesh_station(witness.scan, others=neighbours, measure=False)


def _streamed(
    witness: synthetic.HaloWitnessFixture,
    neighbours: list[CoarseRangeGrid],
    *,
    band_rows: int,
    chunk_points: int = 80,
    halo: int = 3,
    work_dir: Path | None = None,
) -> Any:
    return mesh_station_streamed(
        witness.scan,
        band_rows=band_rows,
        chunk_points=chunk_points,
        halo=halo,
        others=neighbours,
        measure=False,
        work_dir=None if work_dir is None else str(work_dir),
    )


# ---------------------------------------------------------------------------
# vertex connectivity — the rule that decides which components exist at all
# ---------------------------------------------------------------------------


def _scipy_labels(tris: np.ndarray, n_verts: int) -> np.ndarray:
    """`cull_islands`' own labelling, extracted (`triangulate.py:218-223`)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = coo_matrix(
        (np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n_verts, n_verts)
    )
    _, label = connected_components(adj, directed=False)
    return np.asarray(label)


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_union_find_partitions_exactly_as_cull_islands_does(seed: int) -> None:
    """Equivalence to `cull_islands` depends on this and on nothing else.

    Random triangle soups, so vertex-only joins occur by construction. The two
    labellings are compared as *partitions*, not as label values: SciPy's
    numbering is its own business, only the grouping has to agree.
    """
    rng = np.random.default_rng(seed)
    n_verts = 40
    tris = rng.integers(0, n_verts, size=(60, 3), dtype=np.int64)
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])]

    uf = UnionFind(n_verts)
    for a, b, c in tris:
        uf.union(uf.union(int(a), int(b)), int(c))
    mine = np.array([uf.find(i) for i in range(n_verts)])
    theirs = _scipy_labels(tris, n_verts)

    touched = np.unique(tris.ravel())
    pairs_mine = mine[touched][:, None] == mine[touched][None, :]
    pairs_theirs = theirs[touched][:, None] == theirs[touched][None, :]
    assert np.array_equal(pairs_mine, pairs_theirs)


def test_face_adjacency_would_split_a_vertex_only_join() -> None:
    """The counter-example that makes the vertex rule load-bearing.

    Two triangles meeting at exactly one vertex are one component under
    `cull_islands` and two under face adjacency, so a face-adjacency streamed
    implementation would cull differently — silently, and only at vertex-only
    joins.
    """
    tris = np.array([[0, 1, 2], [2, 3, 4]], np.int64)
    uf = UnionFind(5)
    for a, b, c in tris:
        uf.union(uf.union(int(a), int(b)), int(c))
    assert len({uf.find(i) for i in range(5)}) == 1

    shared_edges = sum(
        1
        for i in range(3)
        for j in range(3)
        if len({int(tris[0][i]), int(tris[0][(i + 1) % 3])}
               & {int(tris[1][j]), int(tris[1][(j + 1) % 3])}) == 2
    )
    assert shared_edges == 0          # face adjacency sees two components
    assert np.array_equal(_scipy_labels(tris, 5)[[0, 4]], [0, 0])


def test_union_find_keeps_the_lowest_id_so_ids_are_not_reborn() -> None:
    """§3.2: a continuing component adopts the lowest incoming id.

    Not cosmetic — the §3.6 component-count bound, and therefore the size of
    the verdict table, depends on ids not being reallocated per band.
    """
    table = ComponentTable()
    a, b, c = table.birth(), table.birth(), table.birth()
    table.add_triangles(a, 3)
    table.add_triangles(b, 5)
    table.add_triangles(c, 7)
    assert table.union(b, a) == a
    assert table.find(b) == a
    assert table.roots_and_counts() == {a: 8, c: 7}


def test_min_triangles_default_has_not_drifted_from_cull_islands() -> None:
    """The streamed path restates the threshold; this is the tie."""
    default = inspect.signature(cull_islands).parameters["min_triangles"].default
    assert default == MIN_COMPONENT_TRIANGLES


# ---------------------------------------------------------------------------
# claim B — component-area-v1
# ---------------------------------------------------------------------------


def _areas_from_cull_islands(verts: np.ndarray, tris: np.ndarray) -> tuple[Any, Any]:
    label = _scipy_labels(tris, len(verts))
    area = 0.5 * np.linalg.norm(
        np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
        axis=1,
    )
    comp = label[tris[:, 0]]
    ncomp = int(label.max()) + 1
    totals = np.bincount(comp, weights=area, minlength=ncomp)
    counts = np.bincount(comp, minlength=ncomp)
    live = counts > 0
    return totals[live], counts[live]


def test_component_areas_are_bitwise_equal_to_cull_islands(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """Bitwise, not `allclose`. The difference lands on a delete decision."""
    verts, tris, roots, cells = _streamed_triangles(witness, neighbours, tmp_path, 3)
    mine = component_area_v1(verts, tris, roots, cells)
    theirs_area, theirs_count = _areas_from_cull_islands(verts, tris)

    assert np.array_equal(
        np.sort(mine.areas).view(np.uint64), np.sort(theirs_area).view(np.uint64)
    )
    assert np.array_equal(np.sort(mine.counts), np.sort(theirs_count))
    assert mine.max_ratio < EXACTNESS_RATIO_LIMIT
    assert mine.fallback_components == 0


def test_component_areas_do_not_move_under_a_permutation(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """§7 claim B: while the exactness condition holds, order cannot matter."""
    verts, tris, roots, cells = _streamed_triangles(witness, neighbours, tmp_path, 4)
    base = component_area_v1(verts, tris, roots, cells)
    order = np.random.default_rng(5).permutation(tris.shape[0])
    shuffled = component_area_v1(verts, tris[order], roots[order], cells)
    assert np.array_equal(base.root_ids, shuffled.root_ids)
    assert np.array_equal(base.areas.view(np.uint64), shuffled.areas.view(np.uint64))


def test_the_exactness_guard_fires_and_fsum_becomes_authoritative() -> None:
    """A tolerance that never rejects is not a tolerance.

    Forced past the 2**29 ratio, ordinary float64 accumulation becomes
    order-dependent and `component_area_v1` must switch to the `math.fsum`
    value for that component — and must say that it did.
    """
    import math

    # One 1e7 m^2 triangle and 400 of 5e-10 m^2. Each small area is below half
    # an ULP of the running total, so adding it changes nothing; their sum is
    # about 27 ULPs, so accumulating them first does. Ratio 2e16, well past the
    # 2**29 limit.
    verts = np.zeros((5, 3), np.float32)
    verts[1] = (6324.5, 0.0, 0.0)
    verts[2] = (0.0, 3162.3, 0.0)
    verts[3] = (1.0e-3, 0.0, 0.0)
    verts[4] = (0.0, 1.0e-6, 0.0)
    tris = np.vstack([
        np.array([[0, 1, 2]], np.int64),
        np.repeat(np.array([[0, 3, 4]], np.int64), 400, axis=0),
    ])
    roots = np.zeros(tris.shape[0], np.int64)
    cells = np.arange(5, dtype=np.int64)

    result = component_area_v1(verts, tris, roots, cells)
    assert result.max_ratio >= EXACTNESS_RATIO_LIMIT
    assert result.fallback_components == 1

    area = _areas(verts, tris)
    forward = float(np.bincount(roots, weights=area)[0])
    backward = float(np.bincount(roots, weights=area[::-1])[0])
    assert forward != backward              # the condition, violated, does bite
    assert result.areas[0] == math.fsum(area.tolist())
    assert result.areas[0] != forward


def _areas(verts: np.ndarray, tris: np.ndarray) -> Any:
    return 0.5 * np.linalg.norm(
        np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
        axis=1,
    )


# ---------------------------------------------------------------------------
# claim A — bounded Pass A state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band_rows", (2, 3, 4, 8))
def test_live_components_never_exceed_the_occupied_frontier(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], band_rows: int
) -> None:
    """§7 claim A, as an observable.

    A component absent from the frontier row can never grow again, so the live
    table is bounded by the occupied columns of one row. If connectivity ever
    crossed a band boundary anywhere other than the frontier, this is where it
    would show.
    """
    result = _streamed(witness, neighbours, band_rows=band_rows)
    diag = result.diagnostics
    assert diag is not None
    assert diag.max_frontier_occupied <= WITNESS_COLS
    assert diag.max_live_components <= diag.max_frontier_occupied or diag.band_count == 1
    assert diag.band_count == -(-WITNESS_ROWS // band_rows)


def test_pass_a_triangle_counts_equal_the_pass_b_recount(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """The frontier and retirement rules, checked against an independent count.

    Pass A accumulates counts incrementally as bands merge; Pass B counts the
    triangles it actually reads. A frontier that missed a merge would leave two
    components where there is one, and the two counts would disagree.
    """
    pass_a = pass_a_sweep(
        witness.scan, work_dir=tmp_path, others=neighbours, band_rows=3, halo=3
    )
    recount: dict[int, int] = {}
    alias = {r.component_id: r.root for r in read_component_table(pass_a.component_table_path)}
    for seg in pass_a.segments:
        _, provisional = read_tri_segment(seg.tri_path)
        for pid in provisional:
            root = alias[int(pid)]
            recount[root] = recount.get(root, 0) + 1
    assert recount == {k: v for k, v in pass_a.table.roots_and_counts().items() if v}


# ---------------------------------------------------------------------------
# claim C — band-size invariance, and the per-band counter-example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band_rows", (3, 4, 6, 12, 256))
def test_band_rows_invariance_on_a_carving_fixture(
    witness: synthetic.HaloWitnessFixture,
    neighbours: list[CoarseRangeGrid],
    reference: Any,
    band_rows: int,
) -> None:
    """Gate 1's requirement: output independent of band size.

    Carving is live here — the fixture has two neighbour grids — so this
    exercises despeckle, carve and restore as well as triangulation and the
    component frontier.
    """
    from rapidmesh.equivalence import compare_mesh_results

    streamed = _streamed(witness, neighbours, band_rows=band_rows)
    assert compare_mesh_results(reference, streamed, source_sha256="f" * 64).ok
    assert streamed.stats.dropped_island == reference.stats.dropped_island
    assert streamed.stats.dropped_island > 0        # not a vacuous comparison


@pytest.mark.parametrize("chunk_points", (37, 80, 5_000))
def test_chunk_points_does_not_move_the_output(
    witness: synthetic.HaloWitnessFixture,
    neighbours: list[CoarseRangeGrid],
    reference: Any,
    chunk_points: int,
) -> None:
    """Recorded honestly: `chunk_points` is not yet a live axis on this path
    (it belongs to `iter_row_bands`), so this shows the output does not depend
    on it rather than that it exercises it."""
    from rapidmesh.equivalence import compare_mesh_results

    streamed = _streamed(
        witness, neighbours, band_rows=4, chunk_points=chunk_points
    )
    assert compare_mesh_results(reference, streamed, source_sha256="f" * 64).ok
    assert streamed.diagnostics is not None
    assert streamed.diagnostics.chunk_points == chunk_points


def test_per_band_culling_would_delete_geometry_the_two_pass_design_keeps(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """§5's rejection of per-band culling, as a measured counter-example.

    A component straddling a band boundary is judged on a fragment of its area
    and fails `min_area` in every band despite clearing it in total. This runs
    both answers on the same station and asserts the per-band one is strictly
    worse — if it ever were not, §5's argument would need re-examining rather
    than the code.
    """
    band_rows = 3
    verts, tris, roots, cells = _streamed_triangles(
        witness, neighbours, tmp_path, band_rows
    )
    two_pass = component_area_v1(verts, tris, roots, cells)
    at = np.searchsorted(two_pass.root_ids, roots)
    kept_two_pass = int(
        np.count_nonzero(
            (two_pass.areas[at] >= 0.005)
            & (two_pass.counts[at] >= MIN_COMPONENT_TRIANGLES)
        )
    )

    kept_per_band = 0
    cols = witness.scan.lattice.cols
    top_row = cells[tris[:, 0]] // cols
    for start in range(0, WITNESS_ROWS, band_rows):
        in_band = (top_row >= start) & (top_row < start + band_rows)
        if not np.any(in_band):
            continue
        local = tris[in_band]
        labels = _scipy_labels(local, len(verts))[local[:, 0]].astype(np.int64)
        fragment = component_area_v1(verts, local, labels, cells)
        where = np.searchsorted(fragment.root_ids, labels)
        kept_per_band += int(
            np.count_nonzero(
                (fragment.areas[where] >= 0.005)
                & (fragment.counts[where] >= MIN_COMPONENT_TRIANGLES)
            )
        )

    assert kept_per_band < kept_two_pass, (kept_per_band, kept_two_pass)


def _streamed_triangles(
    witness: synthetic.HaloWitnessFixture,
    neighbours: list[CoarseRangeGrid],
    tmp_path: Path,
    band_rows: int,
) -> tuple[Any, Any, Any, Any]:
    """Run Pass A and return `(verts, tris, roots, cells)` as Pass B sees them."""
    from rapidmesh.segments_io import retained_scan_from_segments

    work = tmp_path / f"pa{band_rows}"
    pass_a = pass_a_sweep(
        witness.scan, work_dir=work, others=neighbours, band_rows=band_rows, halo=3
    )
    retained = retained_scan_from_segments(witness.scan, pass_a.segments)
    cells = retained.row.astype(np.int64) * witness.scan.lattice.cols + retained.col
    parts = [read_tri_segment(s.tri_path) for s in pass_a.segments]
    tri_cells = np.concatenate([c for c, _ in parts])
    provisional = np.concatenate([g for _, g in parts])
    alias = {r.component_id: r.root for r in read_component_table(pass_a.component_table_path)}
    roots = np.array([alias[int(p)] for p in provisional], np.int64)
    return retained.xyz, np.searchsorted(cells, tri_cells), roots, cells


# ---------------------------------------------------------------------------
# segment store
# ---------------------------------------------------------------------------


def test_segments_round_trip_and_own_each_record_once(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """Disjoint by construction: concatenating cores reproduces the retained set
    exactly once, in row-major order, with no de-duplication step anywhere."""
    pass_a = pass_a_sweep(
        witness.scan, work_dir=tmp_path, others=neighbours, band_rows=3, halo=3
    )
    cells = np.concatenate(
        [read_pos_segment(s.pos_path)["cell"] for s in pass_a.segments]
    )
    assert cells.size == len(set(cells.tolist()))
    assert bool(np.all(np.diff(cells) > 0))

    disp = [read_disp_segment(s.disp_path) for s in pass_a.segments]
    assert sum(d["retained"] for d in disp) == cells.size
    assert sum(d["dropped_despeckle"] for d in disp) == pass_a.dropped_despeckle
    assert sum(d["dropped_mover_carve"] for d in disp) == pass_a.dropped_mover_carve
    for seg, d in zip(pass_a.segments, disp, strict=True):
        assert (d["core_row_start"], d["core_row_stop"]) == (
            seg.core_row_start, seg.core_row_stop
        )


def test_a_corrupt_segment_payload_fails_closed(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """A short or altered read would delete survey geometry and report success."""
    pass_a = pass_a_sweep(
        witness.scan, work_dir=tmp_path, others=neighbours, band_rows=8, halo=3
    )
    path = pass_a.segments[0].tri_path
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(SegmentError, match="digest"):
        read_tri_segment(path)


def test_a_truncated_segment_fails_closed(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    pass_a = pass_a_sweep(
        witness.scan, work_dir=tmp_path, others=neighbours, band_rows=8, halo=3
    )
    path = pass_a.segments[0].pos_path
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(SegmentError, match="truncated"):
        read_pos_segment(path)


def test_no_temporary_files_survive_a_pass(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """Temp-then-replace, §5.1: nothing half-written is left addressable."""
    pass_a_sweep(witness.scan, work_dir=tmp_path, others=neighbours, band_rows=4, halo=3)
    assert not list(tmp_path.rglob("*.tmp"))


def test_pass_b_writes_a_finalised_component_table(
    witness: synthetic.HaloWitnessFixture, neighbours: list[CoarseRangeGrid], tmp_path: Path
) -> None:
    """Pass A leaves the verdict undecided; Pass B decides it and says so."""
    _streamed(witness, neighbours, band_rows=4, work_dir=tmp_path)

    provisional = read_component_table(tmp_path / "component.rmtable")
    final = read_component_table(tmp_path / "component.rmcomp")
    assert provisional and len(final) == len(provisional)
    assert all(r.verdict == 0 for r in provisional)
    assert all(r.area == 0.0 for r in provisional)
    assert {r.verdict for r in final} <= {1, 2}
    assert any(r.verdict == 2 for r in final)       # something was culled
    assert max(r.area for r in final) > 0.0
    assert not any(r.area_fallback for r in final)


def test_the_ledger_matches_the_in_memory_one_field_for_field(
    witness: synthetic.HaloWitnessFixture,
    neighbours: list[CoarseRangeGrid],
    reference: Any,
) -> None:
    """Both deferred dispositions (§4.3) are written once, in Pass B."""
    streamed = _streamed(witness, neighbours, band_rows=3)
    assert dataclasses.asdict(streamed.stats) == dataclasses.asdict(reference.stats)
    streamed.stats.require_balanced()
    assert streamed.diagnostics is not None
    assert streamed.diagnostics.retained_unmeshed > 0
