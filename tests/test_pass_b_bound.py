"""
Pass B's working-set bound — ITEM-009, DEC-009 step 1.

Two things have to be true at once and they pull against each other:

* the finalisation must produce **exactly** what the resident implementation
  produced — same mesh, same ledger, same component areas, bit for bit; and
* it must stop holding the station's triangles while doing it.

So the tests come in pairs. `test_streamed_component_areas_match_the_reference`
pins the first against `triangulate.component_area_v1`, which is deliberately
left in the tree as the reference. `test_the_merge_layer_peak_does_not_grow_
with_the_input` pins the second by measuring the merge layer with
`PeakWorkingSetSize`, not `tracemalloc`.

The exactness condition of `PHASE1-ISLANDS-FINALISATION.md` §4.1 is what makes
both possible at once, so it has its own tests: it is checked rather than
assumed, and a component that violates it escalates to `math.fsum` instead of
being quietly accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.islands import pass_a_sweep
from rapidmesh.memory import measure_in_child
from rapidmesh.pass_b import (
    BOUNDARY_SAFETY,
    EXACTNESS_RATIO_LIMIT,
    StreamedAreaAccumulator,
    build_triangle_runs,
    exact_component_areas,
    stream_component_areas,
    stream_kept_triangles,
)
from rapidmesh.pass_b_merge import (
    MERGE_BUFFER_BYTES,
    OUTPUT_BLOCK_BYTES,
    RUN_BUFFER_BYTES,
    merge_runs,
    write_runs,
)
from rapidmesh.segments_io import read_component_table
from rapidmesh.triangulate import component_area_v1

TOOLS = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools")


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=80, cols=320,
        dropout=0.01, range_noise=0.002, seed=7, station_id="passb",
    )


def _pass_a(station: synthetic.SyntheticScan, tmp_path, band_rows: int = 16):  # type: ignore[no-untyped-def]
    return pass_a_sweep(station.scan, work_dir=tmp_path, band_rows=band_rows, halo=3)


def _alias(pass_a) -> np.ndarray:  # type: ignore[no-untyped-def]
    records = read_component_table(pass_a.component_table_path)
    alias = np.full(len(records) + 1, -1, np.int64)
    for record in records:
        alias[record.component_id] = record.root
    return alias


def _retained(station: synthetic.SyntheticScan, pass_a):  # type: ignore[no-untyped-def]
    from rapidmesh.retained_io import retained_scan_from_segments

    retained = retained_scan_from_segments(station.scan, pass_a.segments)
    cells = (
        retained.row.astype(np.int64) * station.scan.lattice.cols + retained.col
    )
    return retained, cells


# ---------------------------------------------------------------------------
# the generic merge layer
# ---------------------------------------------------------------------------


def _records(rng: np.random.Generator, count: int) -> np.ndarray:
    dtype = np.dtype([("a", "<i8"), ("b", "<i8")])
    out = np.empty(count, dtype)
    out["a"] = rng.integers(0, 50, count)
    out["b"] = rng.integers(0, 1_000_000, count)
    return out


@pytest.mark.parametrize("blocks", (1, 3, 11))
def test_the_merge_reproduces_a_full_sort(blocks: int, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A k-way merge is only useful if it is a sort. Compared against one.

    The run buffer is large and the merge buffer small **on purpose**: each run
    is then far bigger than the slice a reader holds, so the merge refills many
    times and the frontier actually has work to do. With runs that fit in one
    buffer the frontier is unreachable and a broken one would still sort.
    """
    rng = np.random.default_rng(blocks)
    pieces = [_records(rng, 25_000) for _ in range(blocks)]
    runs = write_runs(pieces, tmp_path, "t", ("a", "b"), buffer_bytes=800_000)
    assert len(runs.paths) == blocks
    merged = np.concatenate(list(merge_runs(runs, merge_bytes=4_096, output_bytes=4_096)))

    whole = np.concatenate(pieces)
    expected = whole[np.lexsort((whole["b"], whole["a"]))]
    assert merged.shape[0] == expected.shape[0] == runs.record_count
    assert np.array_equal(merged["a"], expected["a"])
    assert np.array_equal(merged["b"], expected["b"])


def test_the_merge_output_is_ordered_across_block_boundaries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The frontier argument, checked at the seams rather than in aggregate.

    Concatenating blocks hides an ordering break inside one block; this asserts
    the sequence is non-decreasing across every emitted block as well.
    """
    rng = np.random.default_rng(5)
    runs = write_runs(
        [_records(rng, 25_000) for _ in range(7)], tmp_path, "t", ("a", "b"),
        buffer_bytes=800_000,
    )
    assert len(runs.paths) == 7          # runs far larger than a reader's slice
    previous = (-1, -1)
    seen = 0
    for block in merge_runs(runs, merge_bytes=4_096, output_bytes=4_096):
        for a, b in zip(block["a"], block["b"], strict=True):
            assert (int(a), int(b)) >= previous
            previous = (int(a), int(b))
            seen += 1
    assert seen == runs.record_count


def test_an_empty_run_set_merges_to_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runs = write_runs([], tmp_path, "t", ("a", "b"))
    assert runs.record_count == 0
    assert list(merge_runs(runs)) == []


def test_the_caps_are_byte_caps_not_record_counts() -> None:
    """A record count would silently change meaning when a record gains a
    field; the B10/B13 rows are stated in bytes for that reason."""
    assert RUN_BUFFER_BYTES == 8_000_000
    assert MERGE_BUFFER_BYTES == 32_000_000
    assert OUTPUT_BLOCK_BYTES == 8_000_000


# ---------------------------------------------------------------------------
# equivalence with the resident reference
# ---------------------------------------------------------------------------


def test_streamed_component_areas_match_the_reference(station, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bitwise against `component_area_v1`, which stays in the tree for this.

    Not "close": a component area decides whether survey geometry is deleted,
    and §4.1's exactness condition means there is no rounding to forgive.
    """
    pass_a = _pass_a(station, tmp_path)
    retained, cells = _retained(station, pass_a)
    alias = _alias(pass_a)
    runs, _mesh_cells = build_triangle_runs(pass_a.segments, alias, tmp_path)
    streamed = stream_component_areas(runs).as_area_result(set())

    from rapidmesh.segments_io import read_triangles_and_final_roots

    tri_cells, roots = read_triangles_and_final_roots(
        pass_a.segments, pass_a.component_table_path
    )
    tris = np.searchsorted(cells, tri_cells).astype(np.int64)
    reference = component_area_v1(retained.xyz, tris, roots, cells)

    assert np.array_equal(streamed.root_ids, reference.root_ids)
    assert np.array_equal(streamed.counts, reference.counts)
    assert np.array_equal(
        streamed.areas.view(np.uint64), reference.areas.view(np.uint64)
    ), "component areas must match bit for bit, not approximately"
    assert streamed.areas.size > 1                     # not a vacuous comparison
    assert reference.max_ratio < EXACTNESS_RATIO_LIMIT


def test_the_streamed_cull_selects_the_same_triangles(station, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same survivors, same membership masks, as a multiset of vertex triples."""
    from rapidmesh.segments_io import read_triangles_and_final_roots
    from rapidmesh.triangulate import MIN_COMPONENT_TRIANGLES, used_vertices

    pass_a = _pass_a(station, tmp_path)
    retained, cells = _retained(station, pass_a)
    alias = _alias(pass_a)
    runs, _mesh_cells = build_triangle_runs(pass_a.segments, alias, tmp_path)
    area = stream_component_areas(runs).as_area_result(set())

    # A threshold chosen to be *discriminating on this fixture*: at the 0.005
    # default nothing here is culled, and a cull comparison that culls nothing
    # would pass whatever the code did. Both paths use the same threshold.
    threshold = float(np.median(area.areas)) + 1e-9
    survives = (area.areas >= threshold) & (area.counts >= MIN_COMPONENT_TRIANGLES)
    keep_root = np.zeros(alias.size, dtype=bool)
    keep_root[area.root_ids] = survives
    kept, before, final = stream_kept_triangles(
        runs, cells, keep_root, int(area.counts[survives].sum()), len(retained)
    )

    tri_cells, roots = read_triangles_and_final_roots(
        pass_a.segments, pass_a.component_table_path
    )
    tris = np.searchsorted(cells, tri_cells).astype(np.int64)
    reference_keep = keep_root[roots]
    assert bool((~survives).any()), "the threshold must cull something"

    def canonical(triples: np.ndarray) -> np.ndarray:
        pick = np.argmin(triples, axis=1)
        rows = np.arange(triples.shape[0])
        out = np.stack([triples[rows, (pick + k) % 3] for k in range(3)], axis=1)
        return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]

    assert np.array_equal(canonical(kept), canonical(tris[reference_keep]))
    assert np.array_equal(before, used_vertices(len(retained), tris))
    assert np.array_equal(final, used_vertices(len(retained), tris[reference_keep]))
    assert int(before.sum()) > int(final.sum())        # something was culled


def test_the_original_winding_survives_the_canonical_merge(station, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The defect this package found, kept caught.

    The merge sorts by the canonical rotation of each triple. Emitting that
    rotation instead of the stored winding is invisible to the T2 triangle
    comparison — rotation preserves orientation — and still moves every
    `reverse-qa-v2` barycentric interior point, because the interior point is
    built from the stored corner order. So the rotation is a sort key only.
    """
    from rapidmesh.segments_io import read_tri_segment

    pass_a = _pass_a(station, tmp_path)
    retained, cells = _retained(station, pass_a)
    alias = _alias(pass_a)
    runs, _mesh_cells = build_triangle_runs(pass_a.segments, alias, tmp_path)
    keep_root = np.ones(alias.size, dtype=bool)
    total = sum(read_tri_segment(s.tri_path)[0].shape[0] for s in pass_a.segments)
    kept, _before, _final = stream_kept_triangles(
        runs, cells, keep_root, total, len(retained)
    )

    emitted = {tuple(int(v) for v in row) for row in cells[kept]}
    stored = {
        tuple(int(v) for v in row)
        for segment in pass_a.segments
        for row in read_tri_segment(segment.tri_path)[0]
    }
    assert emitted == stored, "winding must be the triangulator's, not the sort key's"


# ---------------------------------------------------------------------------
# the exactness condition, and what happens when it fails
# ---------------------------------------------------------------------------


def test_block_accumulation_is_exact_under_the_condition() -> None:
    """Why the streamed reduction is allowed to reassociate at all.

    While the ratio holds, every float64 addition of a float32 area is exact,
    so block-wise accumulation and one whole-array sum are the same bits. The
    test feeds the same values in three different block shapes.
    """
    rng = np.random.default_rng(11)
    areas = (rng.random(4_000).astype(np.float32) + 0.5).astype(np.float64)
    roots = np.zeros(areas.size, np.int64)

    totals = []
    for block in (4_000, 997, 13):
        accumulator = StreamedAreaAccumulator.empty()
        for start in range(0, areas.size, block):
            accumulator.add_block(
                roots[start : start + block], areas[start : start + block]
            )
        totals.append(accumulator.totals[0])
    assert totals[0] == totals[1] == totals[2]
    assert accumulator.ratios()[0] < EXACTNESS_RATIO_LIMIT


def test_a_component_past_the_limit_is_flagged_for_the_exact_sum() -> None:
    """A guard that never fires is not a guard.

    One 1e7 m2 triangle and four hundred of 5e-10 m2 puts the ratio at 2e16.
    The accumulator must refuse to vouch for its own streamed total there.
    """
    areas = np.concatenate(
        [np.array([1.0e7], np.float64), np.full(400, 5.0e-10, np.float64)]
    )
    accumulator = StreamedAreaAccumulator.empty()
    accumulator.add_block(np.zeros(areas.size, np.int64), areas)
    assert accumulator.ratios()[0] >= EXACTNESS_RATIO_LIMIT
    assert accumulator.needs_exact() == {0}


def test_the_escalation_trigger_is_conservative() -> None:
    """A component merely *near* the limit escalates too, because above it the
    streamed total is itself rounded and would be classifying itself."""
    accumulator = StreamedAreaAccumulator.empty()
    just_under = EXACTNESS_RATIO_LIMIT * BOUNDARY_SAFETY * 1.001
    accumulator.totals[7] = just_under
    accumulator.counts[7] = 2
    accumulator.smallest[7] = 1.0
    assert accumulator.ratios()[7] < EXACTNESS_RATIO_LIMIT
    assert 7 in accumulator.needs_exact()


def test_the_exact_pass_reads_only_the_flagged_components(station, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`math.fsum` over one component's own run, not over the station."""
    import math

    pass_a = _pass_a(station, tmp_path)
    retained, cells = _retained(station, pass_a)
    alias = _alias(pass_a)
    runs, _mesh_cells = build_triangle_runs(pass_a.segments, alias, tmp_path)
    accumulator = stream_component_areas(runs)

    assert accumulator.needs_exact() == set()          # nothing flagged naturally
    root = next(iter(accumulator.totals))
    exact = exact_component_areas(runs, {root})
    assert set(exact) == {root}
    assert exact[root] == accumulator.totals[root], (
        "under the condition the exact sum and the streamed sum are one number"
    )
    assert not math.isnan(exact[root])


# ---------------------------------------------------------------------------
# the bound itself, measured with the OS instrument
# ---------------------------------------------------------------------------


def test_the_merge_layer_peak_does_not_grow_with_the_input() -> None:
    """The claim ITEM-009 rests on, measured with the OS instrument.

    Eight times the records through the same merge, in fresh child processes,
    reading `PeakWorkingSetSize` — not `tracemalloc`, which cannot see past
    CPython's allocator. The input blocks are generated lazily at a fixed size,
    so the only thing that grows is the number of runs; if the merge held its
    input the peak would grow with it.

    This measures the **merge layer**, deliberately, and not Pass B as a whole.
    Pass B still contains `build_mesh`, the retained scan and the mesh itself,
    all of which are O(station) by construction — WP-1.7's report gives those
    figures and says plainly what remains for WP-1.8 and Phase 3.
    """
    caps = {"merge_bytes": 4_000_000, "output_bytes": 1_000_000}
    small = measure_in_child(
        "measure_peak_memory", "merge_only", {"records": 400_000, **caps},
        label="merge 400k", sys_path=[TOOLS],
    )
    large = measure_in_child(
        "measure_peak_memory", "merge_only", {"records": 3_200_000, **caps},
        label="merge 3.2M", sys_path=[TOOLS],
    )
    assert large.detail["records"] == 3_200_000
    assert large.detail["runs"] == 8 * small.detail["runs"]
    assert large.detail["largest_block_bytes"] <= caps["output_bytes"]

    # Eight times the input, both runs at the same caps. The merge's own peak
    # must not follow the input; the bar allows for run bookkeeping growing
    # with the number of runs and is still far below the 8x an unbounded
    # implementation would show.
    growth = large.working_set_delta_bytes / max(small.working_set_delta_bytes, 1)
    assert growth < 1.5, (small.describe(), large.describe(), growth)
