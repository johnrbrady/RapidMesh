"""
Pass B — bounded finalisation. ITEM-009, DEC-009 step 1.

`PHASE1-ISLANDS-FINALISATION.md` §4 splits the station's finalisation from its
sweep: Pass A decides component *membership* while streaming, Pass B evaluates
`component-area-v1` over completed components, culls, finalises the ledger,
meshes and measures. WP-1.5 then measured what Pass B actually cost — 45% of
the streamed pipeline's peak at 158k samples, rising to 76% at 990k — because
it held the whole station's triangles several times over at once.

**The insight that makes streaming safe.** Component area is a *reduction*, and
§4.1's exactness condition is exactly a licence to reassociate it. While

    component total area / smallest positive triangle area  <  2**29

every float64 addition of a float32 area is exact, so a streamed block-wise
accumulation, `np.bincount` over the whole station, and `math.fsum` all yield
the **identical** float64 value. That is why the resident sort can go without a
tolerance: under the condition there is nothing to lose, and the condition is
checked rather than assumed. Above it, `math.fsum` over the component's own
bounded run is authoritative for both paths, exactly as §4.1 requires.

Two merged passes replace it. The first reduces area per component; the second
emits the survivors into an array preallocated from the first pass's counts, so
the station's triangles are never held twice. What lives between them is one
entry per *component*, not one per triangle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pass_b_merge import RunSet, merge_runs, write_runs
from .tiles import DEFAULT_TILE_SIZE_M
from .types import FilterStats, StructuredScan

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .islands import PassAResult
    from .pipeline import MeshResult

    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

# `PHASE1-ISLANDS-FINALISATION.md` §4.1.
EXACTNESS_RATIO_LIMIT = 2**29

# Components whose ratio lands within this factor of the limit are escalated to
# the exact accumulator even though the streamed total says they are safe. The
# streamed total is itself rounded above the limit, so a component sitting on
# the boundary could otherwise be classified by a number that is already wrong.
BOUNDARY_SAFETY = 0.99


@dataclass
class StreamedAreaAccumulator:
    """Per-component area, count and smallest positive area, from a stream.

    Holds one entry per component, never one per triangle. §4.1's exactness
    condition is what licenses the block-wise accumulation: while it holds,
    this and `np.bincount` over the whole station are the same float64 value,
    not merely close.
    """

    totals: dict[int, float]
    counts: dict[int, int]
    smallest: dict[int, float]

    @classmethod
    def empty(cls) -> StreamedAreaAccumulator:
        return cls(totals={}, counts={}, smallest={})

    def add_block(self, roots: I64, areas: F64) -> None:
        """One merged block. `roots` is sorted, so components are contiguous.

        `areas` must already be float64 holding the float32 area values — the
        summands the reference uses, accumulated in the width the reference
        accumulates them in. Summing a float32 array would round in float32 and
        is not the same computation.
        """
        import numpy as np

        if roots.size == 0:
            return
        edges = np.flatnonzero(np.diff(roots)) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [roots.size]))
        for start, stop in zip(starts, stops, strict=True):
            root = int(roots[start])
            chunk = areas[start:stop]
            self.totals[root] = self.totals.get(root, 0.0) + float(chunk.sum())
            self.counts[root] = self.counts.get(root, 0) + int(chunk.size)
            positive = chunk[chunk > 0.0]
            if positive.size:
                low = float(positive.min())
                previous = self.smallest.get(root)
                self.smallest[root] = low if previous is None else min(previous, low)

    def ratios(self) -> dict[int, float]:
        return {
            root: (total / self.smallest[root] if self.smallest.get(root) else 0.0)
            for root, total in self.totals.items()
        }

    def as_area_result(self, fallback_roots: set[int]) -> Any:
        """The same shape `component_area_v1` returns, so Pass B's downstream —
        the finalised component table and the diagnostics — is unchanged."""
        import numpy as np

        from .triangulate import AreaResult

        roots = np.fromiter(sorted(self.totals), np.int64, len(self.totals))
        ratios = self.ratios()
        return AreaResult(
            root_ids=roots,
            areas=np.array([self.totals[int(r)] for r in roots], np.float64),
            counts=np.array([self.counts[int(r)] for r in roots], np.int64),
            smallest_positive=np.array(
                [self.smallest.get(int(r), 0.0) for r in roots], np.float64
            ),
            fallback=np.array([int(r) in fallback_roots for r in roots], bool),
            max_ratio=max(ratios.values()) if ratios else 0.0,
            fallback_components=len(fallback_roots),
        )

    def needs_exact(self) -> set[int]:
        """Components whose streamed total cannot be trusted to be exact.

        The trigger is deliberately conservative: a component whose ratio is
        merely *near* the limit is escalated too, because above the limit the
        streamed total is itself rounded and would be classifying itself.
        """
        limit = EXACTNESS_RATIO_LIMIT * BOUNDARY_SAFETY
        return {root for root, ratio in self.ratios().items() if ratio >= limit}


TRIANGLE_RUN_FIELDS = [
    ("root", "<i8"), ("c0", "<i8"), ("c1", "<i8"), ("c2", "<i8"), ("rot", "i1"),
]
TRIANGLE_RUN_KEY = ("root", "c0", "c1", "c2")


def build_triangle_runs(segments: Sequence[Any], alias: I64, work_dir: Path) -> RunSet:
    """One sorted run per band segment, keyed `(root, canonical cell triple)`.

    The key is the stable lattice cell id, never a band-local array index —
    `PHASE1-HALO-CALCULUS.md` §7 — and the triple is canonicalised by rotation
    only, never reversal, so a reversed winding stays a different triangle.
    A band's segment is already bounded, so a run is too.

    `rot` records how far the triple was rotated to reach canonical form, so
    the **original winding can be restored on emission**. That matters more
    than it looks: cyclic rotation preserves orientation and is therefore
    invisible to the T2 triangle comparison, but `reverse-qa-v2` builds its
    barycentric interior point from the stored corner order, so emitting a
    rotated triple would silently move every reverse-QA sample. Measured, not
    reasoned about — see the WP-1.7 report.
    """
    import numpy as np

    from .segments_io import read_tri_segment

    def blocks() -> Iterator[Any]:
        for segment in segments:
            cells, provisional = read_tri_segment(segment.tri_path)
            if cells.shape[0] == 0:
                continue
            pick = np.argmin(cells, axis=1)
            rows = np.arange(cells.shape[0])
            record = np.empty(cells.shape[0], dtype=np.dtype(TRIANGLE_RUN_FIELDS))
            record["root"] = alias[provisional]
            record["rot"] = pick.astype(np.int8)
            for offset, name in enumerate(("c0", "c1", "c2")):
                record[name] = cells[rows, (pick + offset) % 3]
            yield record

    return write_runs(blocks(), work_dir, "tri", TRIANGLE_RUN_KEY)


def _block_indices(block: Any, cells: I64) -> I64:
    """Canonical cell triples back to positions in the retained scan.

    `searchsorted` is exact because the concatenated `pos` cell ids are
    strictly increasing — Pass B asserts that before calling this.

    The stored rotation is always undone, so the returned triple carries the
    winding `_band_triangles` produced rather than the canonical one the merge
    sorted by. Area *looks* rotation-invariant — it is, in real arithmetic —
    but `cross(B-A, C-A)` and `cross(C-B, A-B)` are different float32
    expressions and round differently. Measured: rotating the triple moved one
    component's area by one ULP against the reference. So the canonical form is
    the sort key, and never the geometry.
    """
    import numpy as np

    triple = np.stack([block["c0"], block["c1"], block["c2"]], axis=1)
    indices = np.searchsorted(cells, triple).astype(np.int64)
    rot = block["rot"].astype(np.int64)
    rows = np.arange(indices.shape[0])
    out: I64 = np.stack(
        [indices[rows, (j - rot) % 3] for j in range(3)], axis=1
    )
    return out


def _block_areas(indices: I64, xyz: Any) -> F64:
    """`cull_islands`' own float32 expression, widened for accumulation only."""
    import numpy as np

    corners = xyz[indices]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    widened: F64 = area.astype(np.float64)
    return widened


def stream_component_areas(
    runs: RunSet, cells: I64, xyz: Any
) -> StreamedAreaAccumulator:
    """Phase one: accumulate per-component area without holding the station.

    One block of triangles is resident at a time. What survives the pass is one
    entry per component, not one per triangle — the reduction the whole
    external merge exists to make affordable.
    """
    accumulator = StreamedAreaAccumulator.empty()
    for block in merge_runs(runs):
        indices = _block_indices(block, cells)
        accumulator.add_block(block["root"], _block_areas(indices, xyz))
    return accumulator


def exact_component_areas(
    runs: RunSet, cells: I64, xyz: Any, roots: set[int]
) -> dict[int, float]:
    """`math.fsum` for components the streamed total cannot vouch for.

    Runs only for components flagged by `needs_exact`, and buffers only those
    components' areas — so the cost is paid by the geometry that earns it
    rather than by every station. Zero components have been flagged on any
    fixture measured so far.
    """
    import math

    import numpy as np

    if not roots:
        return {}
    collected: dict[int, list[float]] = {root: [] for root in roots}
    wanted = np.fromiter(sorted(roots), dtype=np.int64, count=len(roots))
    for block in merge_runs(runs):
        hit = np.isin(block["root"], wanted)
        if not bool(hit.any()):
            continue
        chosen = block[hit]
        areas = _block_areas(_block_indices(chosen, cells), xyz)
        for root, value in zip(chosen["root"], areas, strict=True):
            collected[int(root)].append(float(value))
    return {root: math.fsum(values) for root, values in collected.items()}


def stream_kept_triangles(
    runs: RunSet, cells: I64, keep_root: Any, kept_total: int, vertices: int
) -> tuple[I64, Any, Any]:
    """Phase two: the surviving triangles, and both vertex-membership masks.

    `keep_root` is a bool array indexed by component id, not a mapping: the
    lookup happens once per block rather than once per triangle.

    The kept array is preallocated from the phase-one counts, so the station's
    triangles are never held twice — no list of blocks, no concatenate. The
    pre-cull mask is accumulated here too, because `dropped_island` needs it
    and materialising the pre-cull triangle array to get it is exactly the
    allocation this pass removes.
    """
    import numpy as np

    kept = np.empty((kept_total, 3), np.int64)
    before = np.zeros(vertices, dtype=bool)
    final = np.zeros(vertices, dtype=bool)
    at = 0
    for block in merge_runs(runs):
        indices = _block_indices(block, cells)
        before[indices.ravel()] = True
        survives = keep_root[block["root"]]
        if not bool(survives.any()):
            continue
        chosen = indices[survives]
        kept[at : at + chosen.shape[0]] = chosen
        final[chosen.ravel()] = True
        at += chosen.shape[0]
    if at != kept_total:
        raise ValueError(f"kept triangle count disagrees: {at} != {kept_total}")
    return kept, before, final


def pass_b_finalise(
    scan: StructuredScan, pass_a: PassAResult, *, work_dir: Path,
    min_component_area: float, measure: bool, measure_samples: int,
    qa_window_rows: int, timings: dict[str, float], streaming: tuple[int, int, int],
    settings: dict[str, float | int | bool | str],
    out_dir: Path | None = None, tile_size: float = DEFAULT_TILE_SIZE_M,
    qa_workers: int = 1, qa_seed: int = 0,
) -> MeshResult:
    """Re-read the segments, cull on completed components, mesh and measure.

    The ordering is not incidental. `dropped_island` needs the completed
    component table, and retained-but-unmeshed needs certainty that *no* band
    produced a triangle containing the sample — one in band *k*'s halo may be a
    vertex of a triangle band *k+1* owns, so deciding either at band *k* would
    mis-attribute it (§4.3). Both are written once, here. QA follows, against
    final dispositions only: measuring provisional geometry is the defect
    `FINDING-002` records.

    **Two output shapes, and the difference is the WP-3.2 memory row.** With an
    `out_dir`, the surviving triangles are filed into an immutable spatial
    generation one tile at a time and no whole-station `MeshData` is ever
    constructed (ADR-006 Decision 2). Without one there is nowhere for the
    generation to live that outlives this call, so the old shape runs instead:
    `build_mesh` over the station, exactly as before. That path is kept
    deliberately — it is the reference the tiled one is compared against, the way
    `qa_reference` is for forward QA — but it is **not** the shape the memory gate
    is measured on, and a caller that wants the gate has to say where the output
    goes.
    """
    import numpy as np

    from .obs_store import write_store
    from .pipeline import MeshResult, StreamedDiagnostics
    from .qa import deviation_report
    from .reverse_qa import mesh_to_source_report_v2
    from .segments_io import (
        read_component_table,
        retained_scan_from_segments,
        write_finalised_component_table,
    )
    from .tile_build import build_tiles
    from .triangulate import MIN_COMPONENT_TRIANGLES, build_mesh

    retained = retained_scan_from_segments(scan, pass_a.segments)
    cells = retained.row.astype(np.int64) * scan.lattice.cols + retained.col
    if cells.size and not bool(np.all(np.diff(cells) > 0)):
        raise ValueError("pos segments are not in strictly increasing cell order")

    # ITEM-009. The station's triangles are never resident: they are sorted
    # per band into bounded runs, then merged twice — once to reduce area per
    # component, once to emit the survivors. What lives between the passes is
    # one entry per *component*, not one per triangle.
    records = read_component_table(pass_a.component_table_path)
    alias = np.full(len(records) + 1, -1, np.int64)
    for record in records:
        alias[record.component_id] = record.root
    runs = build_triangle_runs(pass_a.segments, alias, work_dir)

    accumulator = stream_component_areas(runs, cells, retained.xyz)
    flagged = accumulator.needs_exact()
    if flagged:
        accumulator.totals.update(
            exact_component_areas(runs, cells, retained.xyz, flagged)
        )
    area = accumulator.as_area_result(flagged)

    survives = (
        np.ones(area.root_ids.size, dtype=bool)
        if min_component_area <= 0
        else (area.areas >= min_component_area)
        & (area.counts >= MIN_COMPONENT_TRIANGLES)
    )
    keep_root = np.zeros(alias.size, dtype=bool)
    keep_root[area.root_ids] = survives
    kept_total = int(area.counts[survives].sum())

    triangles_before_cull = runs.record_count
    write_finalised_component_table(
        work_dir, pass_a.component_table_path, area,
        min_component_area=min_component_area, min_triangles=MIN_COMPONENT_TRIANGLES,
    )

    t0 = time.perf_counter()
    mesh: Any = None
    tiles = None
    built: Any = None
    qa_source: Any = None
    observation_bytes = 0
    observation_count = 0
    if out_dir is None:
        kept_tris, before, final = stream_kept_triangles(
            runs, cells, keep_root, kept_total, len(retained)
        )
        # Released before the mesh is assembled: the cell index and the run
        # buffers have no reader left, and `build_mesh` is the largest single
        # allocation in this pass. Holding them across it is how the old shape
        # reached its peak.
        del cells
        mesh = build_mesh(retained, kept_tris)
        triangles_after_cull = int(kept_tris.shape[0])
        del kept_tris
        qa_source = mesh
    else:
        built = build_tiles(
            retained, runs, cells, keep_root,
            work_dir=out_dir, tile_size=tile_size,
        )
        del cells
        before, final = built.before, built.final
        triangles_after_cull = built.triangles_written
        tiles = built.store
        # WP-1.G1b B2. Written here because this is the one moment the pipeline
        # holds the retained set and knows which of it survived the cull, and
        # because the generation directory is already open. Costs one sequential
        # write and no second read of the source.
        observation_bytes = write_store(
            out_dir / "generations" / "00000000" / "obs" / "observations.rmobs",
            retained, final, retained.pose,
        )
        observation_count = int(final.sum())
        # WP-3.3: both QA directions read the written generation. `TileStore` is
        # a `qa_stream.GeometrySource`, so neither direction has to know whether
        # it was handed a resident mesh or a tile store.
        qa_source = tiles
    timings["assemble"] = time.perf_counter() - t0

    deviation = None
    reverse = None
    reverse_evidence = None
    triangle_count = (
        triangles_after_cull if mesh is None else mesh.triangle_count
    )
    if measure and triangle_count and qa_source is not None:
        t0 = time.perf_counter()
        offsets = retained.pose.rotate_local(retained.xyz[final]).astype(np.float32)
        deviation = deviation_report(
            qa_source, offsets, max_samples=measure_samples,
            seed=qa_seed, workers=qa_workers,
        )
        # §4.4: both directions run here, after culling and the ledger, against
        # final dispositions only.
        reverse, reverse_evidence = mesh_to_source_report_v2(
            qa_source, offsets, source_rows=retained.row[final],
            qa_window_rows=qa_window_rows, max_samples=measure_samples,
            seed=qa_seed, work_dir=work_dir,
        )
        timings["measure"] = time.perf_counter() - t0

    source_count = scan.source_sample_count
    stats = FilterStats(
        input_points=(
            source_count
            if source_count is not None
            else len(scan) + scan.dropped_no_return + scan.dropped_other
        ),
        retained=int(final.sum()),
        dropped_no_return=scan.dropped_no_return,
        dropped_despeckle=pass_a.dropped_despeckle,
        dropped_mover_carve=pass_a.dropped_mover_carve,
        dropped_island=int((before & ~final).sum()),
        dropped_other=scan.dropped_other + int((~before).sum()),
        restored_from_carve=pass_a.restored_from_carve,
    )
    stats.require_balanced()

    band_rows, chunk_points, halo = streaming
    diagnostics = StreamedDiagnostics(
        band_rows=band_rows, chunk_points=chunk_points, halo=halo,
        band_count=pass_a.band_count,
        max_live_components=pass_a.max_live_components,
        max_frontier_occupied=pass_a.max_frontier_occupied,
        component_count=int(area.root_ids.size),
        triangles_before_cull=triangles_before_cull,
        triangles_after_cull=triangles_after_cull,
        retained_unmeshed=int((~before).sum()),
        component_area_version=area.version,
        max_area_ratio=area.max_ratio,
        area_fallback_components=area.fallback_components,
        segment_bytes=sum(
            p.stat().st_size
            for seg in pass_a.segments
            for p in (seg.tri_path, seg.pos_path, seg.disp_path)
        ),
        tiles=None if built is None else built.describe(),
        tile_bytes=0 if built is None else built.tile_bytes,
        observation_bytes=observation_bytes,
        observation_count=observation_count,
    )
    return MeshResult(
        mesh=mesh, stats=stats, lattice=scan.lattice, deviation=deviation,
        mesh_to_source=reverse, timings=timings, settings=settings,
        station_id=scan.station_id, diagnostics=diagnostics,
        reverse_qa_evidence=reverse_evidence, tiles=tiles,
    )
