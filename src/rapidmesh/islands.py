"""
Streaming component labelling and the two-pass island structure —
PLAN.md §5 item 8, designed in `PHASE1-ISLANDS-FINALISATION.md`.

`cull_islands` (`triangulate.py:193`) is the only stage in the chain with no
bounded read-halo: it builds an edge list over *every* triangle in the station
and runs `connected_components` globally. On the reference station that edge
list alone is about 1.175 GB of int64 before SciPy's COO→CSR conversion — more
than twice the whole 512,000,000-byte working budget, for one intermediate.

A component can span the full height of the lattice, so no halo bounds it and
no band can decide it. The answer is not a bigger halo; it is to separate
*membership*, which streams, from *area and verdict*, which do not.

    Pass A   per band: label locally, merge across a one-row frontier,
             write provisional segments, retire what can no longer grow
    Pass B   re-read the segments, evaluate component-area-v1 over completed
             components, cull, finalise the ledger, mesh and measure —
             `pass_b.py`, bounded by `pass_b_merge.py` since ITEM-009

Three properties this has to get right, each of which fails silently:

**Connectivity is over vertices, not faces.** `cull_islands` labels vertices of
a graph built from triangle edges (`triangulate.py:218-223`), so two triangles
meeting at a single vertex are one component. Face-adjacency union-find — the
more natural streaming shape — would produce finer components at vertex-only
joins and cull differently, so every triangle unions all three of its vertices.

**A continuing component must not be reborn.** A local root touching incoming
frontier ids adopts the lowest and aliases the rest; only a root with no
incoming id draws from the monotone counter (§3.2). Allocating a fresh id per
band gives the same answer but loses the §3.6 component-count bound, which is
what makes the verdict table affordable.

**Area is not accumulated in Pass A.** Partial sums combined at merge time
would make the total depend on merge history, and that lands directly on a
keep-or-delete decision about survey geometry. Pass A records membership and
integer triangle counts only; `component_area_v1` (`triangulate.py`) evaluates
areas once, in Pass B, over completed components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .filters import BandFilterResult, columns_wrap
from .grid import ScanGrid, concat, select
from .segments_io import (
    BandSegment,
    ComponentRecord,
    write_band_segments,
    write_component_table,
)
from .streaming import StationMetadata
from .triangulate import band_triangle_indices
from .types import StructuredScan

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .grid import CoarseRangeGrid


    F32 = npt.NDArray[np.float32]
    I64 = npt.NDArray[np.int64]


class UnionFind:
    """Union by smallest id, with path compression.

    By *id* rather than by rank on purpose: §3.2 requires a continuing component
    to keep the lowest of the ids it touches, so the surviving root is the
    smallest one and not whichever tree happened to be taller.
    """

    __slots__ = ("_parent",)

    def __init__(self, size: int = 0) -> None:
        self._parent: list[int] = list(range(size))

    def add(self) -> int:
        self._parent.append(len(self._parent))
        return len(self._parent) - 1

    def __len__(self) -> int:
        return len(self._parent)

    def find(self, a: int) -> int:
        parent = self._parent
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:      # path compression, iterative
            parent[a], a = root, parent[a]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        low, high = (ra, rb) if ra < rb else (rb, ra)
        self._parent[high] = low
        return low


@dataclass
class ComponentTable:
    """Global component ids, their aliases, triangle counts and retirement.

    `retired` is not an optimisation record — it is the assertion that a
    component's membership is provably final, which bounds the live table to
    the occupied width of one lattice row (§3.4).
    """

    ids: UnionFind = field(default_factory=UnionFind)
    triangle_count: dict[int, int] = field(default_factory=dict)
    retired: set[int] = field(default_factory=set)
    max_live: int = 0

    def birth(self) -> int:
        new = self.ids.add()
        self.triangle_count[new] = 0
        return new

    def find(self, component: int) -> int:
        return self.ids.find(component)

    def union(self, a: int, b: int) -> int:
        ra, rb = self.ids.find(a), self.ids.find(b)
        if ra == rb:
            return ra
        root = self.ids.union(ra, rb)
        other = rb if root == ra else ra
        self.triangle_count[root] = self.triangle_count.get(root, 0) + self.triangle_count.pop(
            other, 0
        )
        self.retired.discard(other)
        return root

    def add_triangles(self, component: int, count: int) -> None:
        root = self.ids.find(component)
        self.triangle_count[root] = self.triangle_count.get(root, 0) + count

    def retire_all_except(self, keep: set[int]) -> int:
        """Retire every live root outside `keep`. Returns the live count after."""
        live = {r for r in self.triangle_count if r not in self.retired}
        self.retired |= live - keep
        remaining = len(live & keep)
        self.max_live = max(self.max_live, remaining)
        return remaining

    def roots_and_counts(self) -> dict[int, int]:
        """Canonical root -> final triangle count, aliases resolved."""
        out: dict[int, int] = {}
        for component, count in self.triangle_count.items():
            root = self.ids.find(component)
            out[root] = out.get(root, 0) + count
        return out

    def records(self) -> list[ComponentRecord]:
        """One row per id ever allocated, so Pass B can resolve every alias."""
        counts = self.roots_and_counts()
        return [
            ComponentRecord(
                component_id=cid,
                root=self.ids.find(cid),
                triangle_count=counts.get(cid, 0) if self.ids.find(cid) == cid else 0,
                retired=self.ids.find(cid) in self.retired,
            )
            for cid in range(len(self.ids))
        ]


@dataclass
class PassAResult:
    """What Pass A hands to Pass B, plus the §7 claim-A instrumentation."""

    segments: list[BandSegment]
    table: ComponentTable
    component_table_path: Path
    band_count: int
    max_live_components: int
    max_frontier_occupied: int
    dropped_despeckle: int
    dropped_mover_carve: int
    restored_from_carve: int


def pass_a_sweep(
    scan: StructuredScan,
    *,
    work_dir: Path,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    band_rows: int = 256,
    halo: int = 3,
    chunk_points: int = 250_000,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_quality: float = 0.015,
) -> PassAResult:
    """One forward sweep over a resident scan.

    Kept for callers that already hold a `StructuredScan` — fixtures, and the
    equivalence harness, which needs the same station down both paths. It is a
    thin wrapper: the scan is re-emitted as a chunk stream and the sweep itself
    is `pass_a_sweep_bands`, so there is one implementation and not two.

    Holding the scan is the caller's choice here and costs what it always did.
    `pipeline.mesh_station_from_chunks` is the form where the input is never
    resident.
    """
    from .streaming import StationMetadata, scan_band_stream

    return pass_a_sweep_bands(
        StationMetadata.from_scan(scan),
        scan_band_stream(
            scan, chunk_points=chunk_points, others=others, despeckle=despeckle,
            band_rows=band_rows, halo=halo,
        ),
        work_dir=work_dir,
        max_incidence_deg=max_incidence_deg,
        noise_floor=noise_floor,
        min_quality=min_quality,
    )


def pass_a_sweep_bands(
    metadata: StationMetadata,
    bands: Iterable[BandFilterResult],
    *,
    work_dir: Path,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_quality: float = 0.015,
) -> PassAResult:
    """One forward sweep: triangulate each filtered core, label, merge, retire.

    Takes a **stream** of already-filtered bands and the station's metadata, not
    a resident scan: DEC-009 step 2. Nothing here is sized by the station — the
    lattice and pose are fixed, the frontier is one row wide, and a band's
    points are released when its segment is written.

    The one-row lookahead is worth reading twice. A band's core `[b0, b1)` owns
    the quads whose *top* row is in the core (`PHASE1-HALO-CALCULUS.md` §7), and
    those quads reach down to row `b1`, which the **next** band's core owns. So
    each band is triangulated only once its successor has been filtered, and
    borrows exactly that one row of retained samples. Borrowing is not owning:
    the row's own sample records are written by the band whose core contains it,
    so nothing is written twice and no de-duplication step exists to hide an
    ownership bug.
    """
    import numpy as np

    handle = metadata.handle_scan()
    cols = metadata.lattice.cols
    rows = metadata.lattice.rows
    # Scan-level, from the lattice the metadata carries. Recomputing it from a
    # band would judge a band of a full sweep as a partial-FOV scan and drop the
    # wrap seam (`PHASE1-HALO-CALCULUS.md` §3).
    wrap = columns_wrap(handle)
    step = max(abs(metadata.lattice.az_step), abs(metadata.lattice.el_step))
    tan_limit = math.tan(math.radians(min(max_incidence_deg, 89.5)))

    table = ComponentTable()
    frontier = np.full(cols, -1, np.int64)
    frontier_row = -1
    segments: list[BandSegment] = []
    counts = [0, 0, 0]
    band_count = 0
    max_frontier = 0

    def emit(band: BandFilterResult, following: BandFilterResult | None) -> None:
        nonlocal frontier, frontier_row, band_count, max_frontier

        b0, b1 = band.plan.core_row_start, band.plan.core_row_stop
        parts = [band.retained]
        if following is not None and following.plan.core_row_start == b1:
            borrowed = following.retained
            on_row = borrowed.row == b1
            if bool(np.any(on_row)):
                parts.append(select(borrowed, on_row))
        combined = concat(parts) if len(parts) > 1 else parts[0]

        tris = np.empty((0, 3), np.int64)
        cells = np.empty(0, np.int64)
        local = combined
        if len(combined):
            band_grid = ScanGrid.build(combined)
            # `ScanGrid.build` may re-sort, so every index below is against
            # `band_grid.scan` and never against `combined`.
            local = band_grid.scan
            cells = local.row.astype(np.int64) * cols + local.col
            tris = band_triangle_indices(
                band_grid, b0, min(b1 + 1, rows), wrap=wrap, step=step,
                tan_limit=tan_limit, noise_floor=noise_floor, min_quality=min_quality,
            )

        comp_of_triangle, new_frontier = _label_band(
            table, tris, cells, local=local,
            frontier=frontier, frontier_row=frontier_row, core_start=b0,
            core_stop=b1, cols=cols, rows=rows,
        )

        occupied = int(np.count_nonzero(new_frontier >= 0))
        max_frontier = max(max_frontier, occupied)
        live = {table.find(int(g)) for g in np.unique(new_frontier[new_frontier >= 0])}
        table.retire_all_except(live)

        segments.append(
            write_band_segments(
                work_dir, core_row_start=b0, core_row_stop=b1, cols=cols,
                tri_cells=cells[tris] if tris.shape[0] else np.empty((0, 3), np.int64),
                tri_components=comp_of_triangle, owned=band.retained,
                dropped_despeckle=band.dropped_despeckle,
                dropped_mover_carve=band.dropped_mover_carve,
                restored_from_carve=band.restored_from_carve,
            )
        )
        frontier, frontier_row = new_frontier, b1
        band_count += 1

    pending: BandFilterResult | None = None
    for result in bands:
        counts[0] += result.dropped_despeckle
        counts[1] += result.dropped_mover_carve
        counts[2] += result.restored_from_carve
        if pending is not None:
            emit(pending, result)
        pending = result
    if pending is not None:
        emit(pending, None)

    path = write_component_table(work_dir, table.records())
    return PassAResult(
        segments=segments,
        table=table,
        component_table_path=path,
        band_count=band_count,
        max_live_components=table.max_live,
        max_frontier_occupied=max_frontier,
        dropped_despeckle=counts[0],
        dropped_mover_carve=counts[1],
        restored_from_carve=counts[2],
    )


def _label_band(
    table: ComponentTable, tris: I64, cells: I64, *, local: StructuredScan,
    frontier: I64, frontier_row: int, core_start: int, core_stop: int,
    cols: int, rows: int,
) -> tuple[I64, I64]:
    """Band-local vertex union-find, frontier adoption, and the next frontier."""
    import numpy as np

    new_frontier = np.full(cols, -1, np.int64)
    if tris.shape[0] == 0:
        return np.empty(0, np.int64), new_frontier

    local_rows, local_cols = local.row, local.col
    n_local = cells.shape[0]
    uf = UnionFind(n_local)
    for a, b, c in tris:
        root = uf.union(int(a), int(b))
        uf.union(root, int(c))
    local_root = np.fromiter(
        (uf.find(i) for i in range(n_local)), dtype=np.int64, count=n_local
    )

    in_triangle = np.zeros(n_local, dtype=bool)
    in_triangle[tris.ravel()] = True

    # Which incoming global ids each local root inherits. Only the frontier row
    # can carry any: every path to an earlier band goes through it (§3.3).
    incoming: dict[int, set[int]] = {}
    if frontier_row == core_start:
        on_frontier = (local_rows == core_start) & in_triangle
        for root, col in zip(
            local_root[on_frontier], local_cols[on_frontier], strict=True
        ):
            prior = int(frontier[int(col)])
            if prior >= 0:
                incoming.setdefault(int(root), set()).add(table.find(prior))

    root_to_global = np.full(n_local, -1, np.int64)
    for root in np.unique(local_root[in_triangle]):
        ids = sorted(incoming.get(int(root), ()))
        if ids:
            canonical = ids[0]
            for other in ids[1:]:
                canonical = table.union(canonical, other)
        else:
            canonical = table.birth()
        root_to_global[int(root)] = canonical

    comp_of_triangle: I64 = root_to_global[local_root[tris[:, 0]]]
    for canonical, count in zip(
        *np.unique(comp_of_triangle, return_counts=True), strict=True
    ):
        table.add_triangles(int(canonical), int(count))

    if core_stop < rows:
        carry = (local_rows == core_stop) & in_triangle
        if bool(np.any(carry)):
            new_frontier[local_cols[carry]] = np.array(
                [table.find(int(g)) for g in root_to_global[local_root[carry]]],
                dtype=np.int64,
            )
    return comp_of_triangle, new_frontier
