"""
Rebuilding a scan from `pos` segments — split from `segments_io.py` for DEC-010.

Two shapes, and Pass B uses only the second.

`retained_scan_from_segments` rebuilds every filter-retained sample. That is what
Pass B did until Round 4c, and at reference scale it *was* the memory gate: Class F
(8926 x 3495, 31,196,370 points) held 1,797,627,904 B against a 512,000,000 B
budget, and the excess tracked `pos` bytes rather than the 4.8 MB mesh produced.

`retained_scan_for_cells` rebuilds only the samples a triangle names. Nothing
downstream of the merge reads any other sample: area accumulation indexes triangle
corners, `build_tiles` projects and spools only survivors, `obs_store.build_records`
masks every field by the post-cull `keep`, and both QA directions run against
`final`. The remainder is retained-but-unmeshed, and it is a *count* — measured at
95.14% of the retained set on a real small-class station, carried at roughly 43
bytes each so that its absence could be counted.

The compact set is a **subsequence in ascending cell order**, not a re-ordering, so
relative vertex order is preserved and `build_mesh`'s compaction walks the same
vertices in the same sequence (`PHASE1-DETERMINISM-SPEC.md` §5(a), §5(c)). The
T1/T2/T3 equivalence tiers are what prove that, not this paragraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .segments_io import SegmentError, read_pos_segment

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from .segments_io import BandSegment
    from .types import StructuredScan

    I64 = npt.NDArray[np.int64]


def retained_scan_from_segments(
    scan: StructuredScan, segments: Sequence[BandSegment]
) -> StructuredScan:
    """Rebuild the retained set from the `pos` segments, in band order.

    Band order **is** row-major order, because cores tile the lattice in
    increasing row order and each core's records are already row-major. That is
    not a convenience: it is the order `build_mesh`'s vertex compaction walks
    (`triangulate.py:260`), so it is what makes a streamed vertex array
    identical to the in-memory one rather than a permutation of it
    (`PHASE1-DETERMINISM-SPEC.md` §5(a), §5(c)).

    Scan-level attributes come from `scan` because they are properties of the
    station, not of any band. `intensity` is not carried by the `pos` record and
    is therefore dropped; nothing downstream of meshing reads it, and inventing
    a value would be worse than its absence.

    **Pass B does not call this.** Real stations write nearly every filter-
    retained sample to `pos`, while only triangle vertices are ever indexed —
    Round 4c measured that gap at ~99% of `pos` on a structured station. Pass B
    loads through `retained_scan_for_cells` instead. This full rebuild remains
    for tests and diagnostics that deliberately want the whole `pos` set.
    """

    total = sum(seg.sample_count for seg in segments)
    if not total:
        return _empty_retained(scan)
    return _fill_retained_from_pos(scan, segments, total, cells_needed=None)


def retained_scan_for_cells(
    scan: StructuredScan,
    segments: Sequence[BandSegment],
    cells_needed: I64,
) -> StructuredScan:
    """Rebuild only the `pos` samples whose lattice cell is in `cells_needed`.

    `cells_needed` must be sorted and unique. Pass B passes the unique cells
    named by any triangle: that is the mesh vertex store, and on real stations
    it is two orders of magnitude smaller than the full `pos` set (Round 4c).
    Samples present in `pos` but absent from `cells_needed` are retained-but-
    unmeshed; Pass B counts them from the segment totals rather than loading
    them. Missing cells — a triangle naming a sample no band wrote — fail
    closed.
    """
    import numpy as np

    needed = np.asarray(cells_needed, dtype=np.int64)
    if needed.size == 0:
        return _empty_retained(scan)
    if needed.size > 1 and not bool(np.all(needed[1:] > needed[:-1])):
        raise SegmentError("cells_needed must be strictly increasing")
    return _fill_retained_from_pos(scan, segments, int(needed.size), cells_needed=needed)


def _empty_retained(scan: StructuredScan) -> StructuredScan:
    import numpy as np

    from .types import StructuredScan as _Scan

    empty = np.empty(0, np.int32)
    return _Scan(
        row=empty, col=empty.copy(), xyz=np.empty((0, 3), np.float32),
        rng=np.empty(0, np.float32), pose=scan.pose, lattice=scan.lattice,
        rgb=None if scan.rgb is None else np.empty((0, 3), np.uint8),
        intensity=None, station_id=scan.station_id,
        sample_id=None if scan.sample_id is None else np.empty(0, np.int64),
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
    )


def _fill_retained_from_pos(
    scan: StructuredScan,
    segments: Sequence[BandSegment],
    capacity: int,
    *,
    cells_needed: I64 | None,
) -> StructuredScan:
    """Preallocate `capacity` samples and fill from `pos` segments in band order.

    When `cells_needed` is set, only matching rows are copied and every needed
    cell must appear exactly once across the segments. When it is `None`, every
    `pos` row is copied and `capacity` must equal the declared sample total.
    """
    import numpy as np

    from .types import StructuredScan as _Scan

    cols = scan.lattice.cols
    cells = np.empty(capacity, np.int64)
    xyz = np.empty((capacity, 3), np.float32)
    rng = np.empty(capacity, np.float32)
    sample_id = None if scan.sample_id is None else np.empty(capacity, np.int64)
    rgb = None if scan.rgb is None else np.empty((capacity, 3), np.uint8)
    at = 0
    for seg in segments:
        block = read_pos_segment(seg.pos_path)
        n = int(block["cell"].size)
        if not n:
            continue
        # Either a whole-block slice or a boolean mask, depending on the mode;
        # the annotation is what lets the two share one indexing expression below.
        take: Any
        if cells_needed is None:
            take = np.s_[:n]
            count = n
        else:
            idx = np.searchsorted(cells_needed, block["cell"])
            in_range = idx < cells_needed.size
            hit = np.zeros(n, dtype=bool)
            hit[in_range] = cells_needed[idx[in_range]] == block["cell"][in_range]
            take = hit
            count = int(hit.sum())
            if not count:
                continue
        if at + count > capacity:
            raise SegmentError("pos filter overflowed its preallocated capacity")
        cells[at : at + count] = block["cell"][take]
        xyz[at : at + count] = block["xyz"][take]
        rng[at : at + count] = block["rng"][take]
        if sample_id is not None:
            sample_id[at : at + count] = block["sample_id"][take]
        if rgb is not None:
            if block["rgb"] is None:
                raise SegmentError("pos segment has no colour but the scan does")
            rgb[at : at + count] = block["rgb"][take]
        at += count
    if at != capacity:
        if cells_needed is None:
            raise SegmentError(
                "pos segments do not carry the sample counts they declare"
            )
        raise SegmentError(
            "a triangle names a lattice cell no pos segment carries"
        )
    return _Scan(
        row=(cells // cols).astype(np.int32), col=(cells % cols).astype(np.int32),
        xyz=xyz, rng=rng, pose=scan.pose, lattice=scan.lattice, rgb=rgb,
        intensity=None, station_id=scan.station_id, sample_id=sample_id,
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return, dropped_other=scan.dropped_other,
    )
