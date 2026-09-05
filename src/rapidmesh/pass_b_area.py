"""
Pass B's component-area reduction — split from `pass_b.py` for DEC-010.

Nothing here changed when it moved, and `pass_b.py` re-exports every name this
module defines that had a caller, so `from rapidmesh.pass_b import
StreamedAreaAccumulator` and `from .pass_b import _block_indices` keep working.
The split exists because Round 4c added the mesh-vertex-store gather to
`pass_b.py` and the file would otherwise cross 500 lines.

**The insight that makes streaming safe.** Component area is a *reduction*, and
`PHASE1-ISLANDS-FINALISATION.md` §4.1's exactness condition is exactly a licence
to reassociate it. While

    component total area / smallest positive triangle area  <  2**29

every float64 addition of a float32 area is exact, so a streamed block-wise
accumulation, `np.bincount` over the whole station, and `math.fsum` all yield the
**identical** float64 value. That is why the resident sort can go without a
tolerance: under the condition there is nothing to lose, and the condition is
checked rather than assumed. Above it, `math.fsum` over the component's own
bounded run is authoritative for both paths, exactly as §4.1 requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pass_b_merge import RunSet, merge_runs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import numpy.typing as npt

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
        # WP-12a. `counts` and `fallback` were built inline; they are named here
        # only so `fallback_triangle_count` can be their inner product. No value
        # changes, and nothing about the accumulation is touched.
        counts = np.array([self.counts[int(r)] for r in roots], np.int64)
        fallback = np.array([int(r) in fallback_roots for r in roots], bool)
        return AreaResult(
            root_ids=roots,
            areas=np.array([self.totals[int(r)] for r in roots], np.float64),
            counts=counts,
            smallest_positive=np.array(
                [self.smallest.get(int(r), 0.0) for r in roots], np.float64
            ),
            fallback=fallback,
            max_ratio=max(ratios.values()) if ratios else 0.0,
            fallback_components=len(fallback_roots),
            fallback_triangle_count=int(counts[fallback].sum()),
        )

    def needs_exact(self) -> set[int]:
        """Components whose streamed total cannot be trusted to be exact.

        The trigger is deliberately conservative: a component whose ratio is
        merely *near* the limit is escalated too, because above the limit the
        streamed total is itself rounded and would be classifying itself.
        """
        limit = EXACTNESS_RATIO_LIMIT * BOUNDARY_SAFETY
        return {root for root, ratio in self.ratios().items() if ratio >= limit}


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


def _flagged_counts(runs: RunSet, wanted: I64) -> dict[int, int]:
    """Triangles per flagged component, from the run keys alone.

    `StreamedAreaAccumulator.counts` already holds these numbers exactly, and
    `pass_b_finalise` is holding that accumulator when it calls the second pass
    — but it does not pass them and `pass_b.py` is not this package's to change
    (WP-12b scope). Recomputing them costs one pass over the run records with
    **no geometry**: no `searchsorted` into `cells`, no vertex gather, no cross
    product, none of the three things that make the second pass expensive. The
    alternative the brief rules out is guessing a length and growing.
    """
    import numpy as np

    counts = {int(root): 0 for root in wanted}
    for block in merge_runs(runs):
        block_roots = block["root"]
        hit = np.isin(block_roots, wanted)
        if not bool(hit.any()):
            continue
        values, seen = np.unique(block_roots[hit], return_counts=True)
        for value, count in zip(values, seen, strict=True):
            counts[int(value)] += int(count)
    return counts


def exact_component_areas(
    runs: RunSet, cells: I64, xyz: Any, roots: set[int],
    counts: Mapping[int, int] | None = None,
) -> dict[int, float]:
    """`math.fsum` for components the streamed total cannot vouch for.

    Runs only for components flagged by `needs_exact`, and buffers only those
    components' areas — so the cost is paid by the geometry that earns it
    rather than by every station.

    **WP-12b — the buffer is a pre-sized float64 array, not a list of CPython
    floats.** Same summands, in the same merge order, through the same
    `math.fsum`: a storage change, not an arithmetic one. A list costs 32 B a
    triangle — a 24-byte float object plus an 8-byte slot — against 8 B in the
    array. On ordinal 20's one flagged component that is 819,161,152 B against
    204,790,288 B (WP-12a §5.4). `counts` supplies the sizes when a caller has
    them; without it `_flagged_counts` recovers the same integers.

    **This path does run on real data.** Every synthetic fixture in the
    repository still takes the empty early return below, which is why this
    docstring used to say no component had ever been flagged. That sentence was
    true of fixtures and false of the campaign: WP-10mb measured
    `fallback_components = 1` on ordinal 20 and WP-12a sized that one component
    at 25,598,786 triangles, 94% of the station.
    """
    import math

    import numpy as np

    if not roots:
        return {}
    wanted = np.fromiter(sorted(roots), dtype=np.int64, count=len(roots))
    sizes = _flagged_counts(runs, wanted) if counts is None else counts
    buffers = {int(r): np.empty(int(sizes[int(r)]), np.float64) for r in wanted}
    filled = {int(r): 0 for r in wanted}
    for block in merge_runs(runs):
        hit = np.isin(block["root"], wanted)
        if not bool(hit.any()):
            continue
        chosen = block[hit]
        areas = _block_areas(_block_indices(chosen, cells), xyz)
        block_roots = chosen["root"]
        # One slice assignment per root per block, so no CPython object is made
        # per triangle. `merge_runs` is sorted by `root` first, so in practice
        # this loop runs once; selecting by mask keeps the arrival order either
        # way, and arrival order is what `math.fsum` is being given.
        for root in np.unique(block_roots):
            key = int(root)
            take = areas[block_roots == root]
            at = filled[key]
            buffers[key][at : at + take.shape[0]] = take
            filled[key] = at + take.shape[0]
    for key, buffer in buffers.items():
        if filled[key] != buffer.shape[0]:
            raise ValueError(
                f"component {key} buffered {filled[key]} areas, "
                f"{buffer.shape[0]} were counted"
            )
    return {root: math.fsum(buffer) for root, buffer in buffers.items()}
