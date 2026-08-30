"""
Lattice indexing — how a native-resolution scan is addressed without being
materialised densely.

The problem this module exists to solve
---------------------------------------
A Trimble X7 at high resolution samples roughly 20000 x 10000. A dense int32
cell -> sample index over that lattice is 800 MB, and a dense float32 range
grid is another 800 MB. Cairn sidesteps this by binning to 2048 x 1024 up
front, which is exactly the decimation-first mistake RapidMesh exists to undo.

So we never materialise the whole lattice. Two structures instead:

`ScanGrid` — a CSR-style row index over samples already sorted in row-major
    lattice order. `row_start[r]` is where row r begins. Any stage that only
    needs local neighbourhoods (triangulation needs two adjacent rows,
    despeckle needs three) asks for a **band** and gets a small dense array.
    Peak memory is O(band_rows x cols), not O(rows x cols).

`CoarseRangeGrid` — a deliberately low-resolution nearest-range grid in
    world-aligned spherical coordinates about the station origin, used only
    for cross-station occlusion carving. Line-of-sight testing does not need
    native resolution: a 2048 x 1024 f32 grid is 8 MB and is plenty to answer
    "did another station see straight through this location". Making this
    coarse on purpose is what keeps carving affordable across a whole site.

Both are pure indexing. Neither filters, triangulates or decides anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import ScanPose, StructuredScan

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I32 = npt.NDArray[np.int32]

# Carving grid budget, in cells. 16 M cells is 64 MB as float32, so two or
# three neighbour grids sit comfortably in memory while still resolving about
# 0.06 degrees on a full sphere.
#
# A budget rather than a fixed 2048 x 1024 because carving accuracy tracks
# angular resolution in BOTH directions, and both failure modes are silent:
# a grid finer than the scan is full of empty bins and stops carving at all,
# while a grid much coarser than the scan mixes near and far surfaces into one
# bin and carves real geometry off silhouette edges. Measured on the synthetic
# fixture, dropping from a matched grid to a 2048-wide cap on a 2800-column
# scan roughly doubled the false-positive rate.
CARVE_MAX_CELLS = 16_000_000


def sort_row_major(scan: StructuredScan) -> StructuredScan:
    """Return `scan` with every per-sample array in row-major lattice order.

    `ScanGrid` requires this ordering and will not re-sort, because at native
    resolution an argsort of 100M elements is not something to do implicitly.
    Readers are expected to call this once; it is a cheap no-op when the
    source already emitted rows in order (E57 usually does).
    """
    import numpy as np

    key = scan.row.astype(np.int64) * scan.lattice.cols + scan.col
    if bool(np.all(np.diff(key) >= 0)):
        return scan
    order = np.argsort(key, kind="stable")
    return StructuredScan(
        row=scan.row[order],
        col=scan.col[order],
        xyz=scan.xyz[order],
        rng=scan.rng[order],
        pose=scan.pose,
        lattice=scan.lattice,
        rgb=None if scan.rgb is None else scan.rgb[order],
        intensity=None if scan.intensity is None else scan.intensity[order],
        station_id=scan.station_id,
        sample_id=None if scan.sample_id is None else scan.sample_id[order],
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
    )


def select(scan: StructuredScan, keep: npt.NDArray[np.bool_]) -> StructuredScan:
    """Subset every per-sample array by a boolean mask, preserving order.

    Used by each filter stage. Kept here rather than on the dataclass so
    `types.py` stays free of numpy at runtime.
    """
    return StructuredScan(
        row=scan.row[keep],
        col=scan.col[keep],
        xyz=scan.xyz[keep],
        rng=scan.rng[keep],
        pose=scan.pose,
        lattice=scan.lattice,
        rgb=None if scan.rgb is None else scan.rgb[keep],
        intensity=None if scan.intensity is None else scan.intensity[keep],
        station_id=scan.station_id,
        sample_id=None if scan.sample_id is None else scan.sample_id[keep],
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
    )


def select_rows(grid: ScanGrid, row_start: int, row_stop: int) -> StructuredScan:
    """Contiguous lattice-row subset of a row-major scan, as its own scan.

    Uses the CSR row pointer, so the cost is the size of the band and not the
    size of the scan — which is the whole point of asking for a band.

    **Absolute lattice rows are preserved.** The returned scan keeps the
    parent's `lattice`, so `row`/`col` still address the full lattice and a
    `ScanGrid` built over the result answers `dense_rows` in absolute row
    coordinates. Ownership therefore stays keyed on `(row, col)` rather than on
    a position in a band-local array, which is what
    `PHASE1-HALO-CALCULUS.md` §7 requires.

    Per-sample arrays are **views** into `grid.scan`. Nothing in the band-local
    filter chain writes through them (`select` copies, and the filter kernels
    allocate their own masks), so no copy is made here.
    """
    r0 = max(0, min(row_start, grid.rows))
    r1 = max(r0, min(row_stop, grid.rows))
    scan = grid.scan
    s, e = int(grid.row_start[r0]), int(grid.row_start[r1])
    return StructuredScan(
        row=scan.row[s:e],
        col=scan.col[s:e],
        xyz=scan.xyz[s:e],
        rng=scan.rng[s:e],
        pose=scan.pose,
        lattice=scan.lattice,
        rgb=None if scan.rgb is None else scan.rgb[s:e],
        intensity=None if scan.intensity is None else scan.intensity[s:e],
        station_id=scan.station_id,
        sample_id=None if scan.sample_id is None else scan.sample_id[s:e],
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
    )


def concat(parts: Sequence[StructuredScan]) -> StructuredScan:
    """Join band-local scans back into one, preserving order.

    The band pipeline emits one retained scan per band core; the cores tile the
    lattice in increasing row order, so concatenating them in band order
    reproduces row-major order without a re-sort.

    Scan-level attributes are taken from the first part and every other part
    must agree, because a disagreement means two different stations were
    joined — a silent corruption otherwise.
    """
    import numpy as np

    if not parts:
        raise ValueError("concat requires at least one scan")
    head = parts[0]
    for other in parts[1:]:
        if other.lattice != head.lattice or other.station_id != head.station_id:
            raise ValueError("cannot concatenate scans from different stations")
        if (other.rgb is None) != (head.rgb is None):
            raise ValueError("cannot concatenate scans with mixed rgb presence")
        if (other.intensity is None) != (head.intensity is None):
            raise ValueError("cannot concatenate scans with mixed intensity presence")
        if (other.sample_id is None) != (head.sample_id is None):
            raise ValueError("cannot concatenate scans with mixed sample_id presence")

    def join(name: str) -> Any:
        arrays = [getattr(part, name) for part in parts]
        return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]

    return StructuredScan(
        row=join("row"),
        col=join("col"),
        xyz=join("xyz"),
        rng=join("rng"),
        pose=head.pose,
        lattice=head.lattice,
        rgb=None if head.rgb is None else join("rgb"),
        intensity=None if head.intensity is None else join("intensity"),
        station_id=head.station_id,
        sample_id=None if head.sample_id is None else join("sample_id"),
        source_sample_count=head.source_sample_count,
        dropped_no_return=head.dropped_no_return,
        dropped_other=head.dropped_other,
    )


@dataclass
class ScanGrid:
    """Band-addressable index over a row-major-sorted `StructuredScan`."""

    scan: StructuredScan
    row_start: npt.NDArray[np.int64]   # (rows + 1,) CSR row pointer

    @classmethod
    def build(cls, scan: StructuredScan) -> ScanGrid:
        import numpy as np

        scan = sort_row_major(scan)
        rows = scan.lattice.rows
        # searchsorted over the (sorted) row array gives the CSR pointer in one
        # pass and without a Python loop over rows, which matters when `rows`
        # is 10000+.
        starts = np.searchsorted(scan.row, np.arange(rows + 1, dtype=np.int32), side="left")
        return cls(scan=scan, row_start=starts.astype(np.int64))

    @property
    def rows(self) -> int:
        return self.scan.lattice.rows

    @property
    def cols(self) -> int:
        return self.scan.lattice.cols

    def dense_rows(self, r0: int, r1: int) -> I32:
        """Dense `(r1 - r0, cols)` array of sample indices, -1 where empty.

        `r0`/`r1` are clamped to the lattice, so callers can ask for a padded
        band at the poles without special-casing the edges.
        """
        import numpy as np

        r0 = max(0, min(r0, self.rows))
        r1 = max(r0, min(r1, self.rows))
        out = np.full((r1 - r0, self.cols), -1, np.int32)
        if r1 == r0:
            return out
        s, e = int(self.row_start[r0]), int(self.row_start[r1])
        if e > s:
            out[self.scan.row[s:e] - r0, self.scan.col[s:e]] = np.arange(s, e, dtype=np.int32)
        return out

    def bands(self, band_rows: int = 512, overlap: int = 1) -> list[tuple[int, int]]:
        """Row ranges covering the lattice, each overlapping the next by
        `overlap` rows.

        Triangulation consumes a cell and its row-below neighbour, so a band
        must overlap its successor by one row or every band boundary becomes a
        one-row seam of missing triangles. That seam is subtle enough to ship
        by accident, which is why the overlap is a parameter here rather than
        an assumption at the call site.
        """
        if band_rows <= overlap:
            raise ValueError("band_rows must exceed overlap")
        out: list[tuple[int, int]] = []
        r = 0
        while r < self.rows - 1:
            end = min(r + band_rows, self.rows)
            out.append((r, end))
            if end >= self.rows:
                break
            r = end - overlap
        return out

    def estimated_dense_bytes(self, band_rows: int = 512) -> int:
        """Peak bytes for one dense band. Exposed so the CLI can warn before
        it allocates, rather than after the machine starts swapping."""
        return band_rows * self.cols * 4


@dataclass
class CoarseRangeGrid:
    """Nearest-return range per angular bin, world-aligned about `origin`.

    World-aligned rather than scanner-aligned on purpose: carving queries come
    from *other* stations, so every grid has to answer questions in a shared
    frame. Applying each scan's own rotation here means the query side never
    has to know about it.
    """

    origin: F64                 # (3,) station origin, project coordinates
    rng: F32                    # (H, W) nearest range per bin, inf where empty
    width: int
    height: int

    @classmethod
    def build(
        cls,
        scan: StructuredScan,
        max_cells: int = CARVE_MAX_CELLS,
        fill_holes: bool = True,
    ) -> CoarseRangeGrid:
        """Range grid for `scan`, matched to the scan's own lattice.

        Resolution follows the source and is only reduced to stay inside
        `max_cells`. **Matching matters in both directions, and both mismatches
        fail silently:**

        * *Finer than the scan* — most bins receive no sample and read `inf`,
          so `seen_through` finds nothing behind anything and carving stops
          working while reporting success. A single low-resolution station
          would disable mover removal for all its neighbours, with no error
          anywhere.
        * *Much coarser than the scan* — one bin spans near and far surfaces
          across a silhouette, keeps the near one, and then reports the far
          surface as "seen through" for points that were genuinely there.
          That deletes real geometry off every depth edge.

        Reduction, when needed, preserves the scan's aspect ratio so angular
        cells stay roughly square.
        """
        import math


        rows, cols = max(scan.lattice.rows, 8), max(scan.lattice.cols, 16)
        if rows * cols > max_cells:
            k = math.sqrt(max_cells / (rows * cols))
            rows, cols = max(int(rows * k), 8), max(int(cols * k), 16)
        world = scan.pose.local_to_world(scan.xyz)
        grid = cls.from_world_points(world, scan.pose.translation, cols, rows)
        if fill_holes:
            grid.close_dropout()
        return grid

    @classmethod
    def from_world_points(
        cls,
        world: F64,
        origin: F64,
        width: int = 2048,
        height: int = 1024,
    ) -> CoarseRangeGrid:
        import numpy as np

        q = world - origin
        r = np.linalg.norm(q, axis=1)
        good = r > 1e-6
        q, r = q[good], r[good]
        u, v = _spherical_bin(q, r, width, height)
        flat = v.astype(np.int64) * width + u
        grid = np.full(height * width, np.inf, np.float32)
        # np.minimum.at is the correct reduction here: many samples land in one
        # coarse bin and we want the NEAREST, since the near surface is the one
        # that occludes. A plain fancy-index assignment would keep whichever
        # sample happened to be written last.
        np.minimum.at(grid, flat, r.astype(np.float32))
        return cls(origin=origin.astype(np.float64), rng=grid.reshape(height, width),
                   width=width, height=height)

    def close_dropout(self) -> None:
        """Fill empty bins with the NEAREST range among their neighbours.

        Even at matched resolution a scan has no-return cells (dark surfaces,
        grazing hits, sky), and an empty bin means "no evidence", so carving
        skips it and a mover gets a scatter of surviving speckle through it.

        The direction of the fill is the safety property. `seen_through` drops
        a point when the bin behind it reads *further* away, so filling with a
        value that is too far causes false carving — deleting real surface —
        while filling with a value that is too near only causes a missed carve.
        Taking the minimum of the finite neighbours therefore fails safe, which
        is the only acceptable direction for a filter that deletes survey data.
        """
        import numpy as np

        r = self.rng
        out = r.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(r, dr, axis=0), dc, axis=1)
                if dr > 0:
                    shifted[:dr, :] = np.inf
                elif dr < 0:
                    shifted[dr:, :] = np.inf
                out = np.minimum(out, np.where(np.isfinite(r), np.inf, shifted))
        self.rng = out

    def seen_through(
        self,
        world: F64,
        clear_margin: float = 0.4,
        relative_margin: float = 1.02,
    ) -> npt.NDArray[np.bool_]:
        """True for world points this station recorded a return *beyond*.

        If this station saw a solid surface further away along the same ray,
        the queried point was not solid when this station captured — it was a
        person, a car, or dust. Inherited from Cairn's mover block, which is
        the strongest idea in that file: a real door frame is never seen
        through, so this has a far lower false-positive rate than judging a
        single scan's range gradients.

        Both margins must be exceeded. The absolute one (`clear_margin`)
        ignores same-surface noise; the relative one keeps the test meaningful
        at 60 m, where 0.4 m is inside the noise floor.
        """
        import numpy as np

        q = world - self.origin
        r = np.linalg.norm(q, axis=1)
        u, v = _spherical_bin(q, np.maximum(r, 1e-9), self.width, self.height)
        behind = self.rng[v, u]
        out: npt.NDArray[np.bool_] = (
            np.isfinite(behind) & (behind > r + clear_margin) & (behind > r * relative_margin)
        )
        return out


def _spherical_bin(
    q: F64, r: F64, width: int, height: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Direction vectors -> (column, row) bin indices.

    Azimuth wraps with a modulo; elevation is clipped, because elevation does
    NOT wrap across the poles. Cairn's despeckle originally wrapped both and
    made points near one pole judge points near the other — worth not
    repeating.
    """
    import numpy as np

    az = np.arctan2(q[:, 1], q[:, 0])
    el = np.arcsin(np.clip(q[:, 2] / r, -1.0, 1.0))
    u = (((az + np.pi) / (2 * np.pi)) * width).astype(np.int64) % width
    v = np.clip((((el + np.pi / 2) / np.pi) * height).astype(np.int64), 0, height - 1)
    return u, v


def station_origin(pose: ScanPose) -> F64:
    """The f64 anchor for a station's mesh. Trivial, but named so the
    millimetre-precision rule in `00-PRODUCT-DEFINITION.md` §7 has one obvious
    place to be enforced."""
    return pose.translation
