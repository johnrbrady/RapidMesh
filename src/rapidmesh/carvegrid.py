"""
Streaming carve-grid construction, row-blocked hole closing and the cached grid
artefact — PLAN.md §5 item 7.

Why this module exists, in one sentence: `CoarseRangeGrid.build` transforms a
whole scan at once, and on the reference station that single call has an
immediate lower bound of 748,590,240 B, which fails the 512,000,000 B working
budget before any later temporary is counted
(`PHASE1-TILE-CONTRACT-V0.md` §6.2).

Three separate problems, three separate pieces here:

`build_grid_chunked`
    Bins the scan in fixed-capacity chunks, so the f64 transform buffers are
    O(chunk) rather than O(scan). It is required to be **bitwise identical** to
    `CoarseRangeGrid.build`, not merely close: a carve grid decides which
    survey points get deleted, and "approximately the same deletions" is not a
    property anyone can audit. Identity holds because every per-sample step is
    elementwise and the bin reduction is `minimum`, which selects an input
    value rather than computing a new one — so chunk boundaries cannot round.

`close_dropout_blocked`
    The second transient. The whole-grid `close_dropout` holds 5–7 grid-sized
    f32 arrays at once (305,574,192–421,983,408 B on the worst reference
    lattice, `PHASE1-TILE-CONTRACT-V0.md` §6.3). This reads an **immutable**
    source and writes a **distinct** output in row blocks with a one-row halo.
    An in-place sweep is forbidden and is not a style preference: it would read
    values filled earlier in the same pass, propagate a return beyond the
    one-cell neighbourhood the whole-grid operation defines, and carve real
    geometry that the reference implementation keeps.

`CarveGridKey` / `StationRef` / `build_or_load`
    The key that makes the build affordable once per station rather than once
    per neighbour-of-a-neighbour, and the one-at-a-time orchestration that
    loads a neighbour's points, builds its grid and releases them before the
    next. Invalidation is **by name only** (`PHASE1-TILE-CONTRACT-V0.md` §7):
    every component matches or it is a miss. Nothing is updated in place and no
    partial match is accepted, because a stale grid does not raise — it
    silently carves the wrong geometry from a neighbour. The bytes-on-disk half
    lives in `carvegrid_io.py`.

Working-memory budget, carve grids — the B1/B2/B3 rows of
`PHASE1-TILE-CONTRACT-V0.md` §6.1. Resident figures are arithmetic from the
constants; transients are **measured on a synthetic fixture** and scaled by
arithmetic. They are not real-data results, and they are not the Gate 1
measurement, which is PLAN.md §5 item 10 and uses `PeakWorkingSetSize`.

    resident, 1 grid at CARVE_MAX_CELLS       16,000,000 x 4 =  64,000,000 B
    resident, 2 grids at the cap                               128,000,000 B
    resident, 1 worst-reference grid   2387 x 6096 x 4 =        58,204,608 B
    resident, 2 worst-reference grids (the default count)      116,409,216 B

Measured with `tracemalloc` on a 973,181-sample synthetic scan binned to a
500 x 2000 grid (4,000,000 B payload), `tools`-free, in
`tests/test_streaming_carve_grid.py`:

    chunked build, transient above the grid   ~97 B per chunk point
        at chunk_points =  50,000                    4,856,888 B
        at chunk_points = 250,000                   24,252,304 B
    close_dropout_blocked, above the source    1.68 grid copies
    whole-grid close_dropout, above the source 5.00 grid copies

The per-chunk-point coefficient is what matters: the build transient is a
function of `chunk_points`, not of the scan, which is the property that makes
the 512,000,000 B budget reachable. Scaled to the worst reference grid, hole
closing costs the immutable source plus one distinct output plus roughly
8,300,000 B of block temporaries — about 124,700,000 B — against roughly
291,000,000 B for the whole-grid sweep at the same measured 5.00 copies.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .grid import CARVE_MAX_CELLS, CoarseRangeGrid, _spherical_bin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    import numpy.typing as npt

    from .types import LatticeInfo, ScanPose, StructuredScan

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]

# Bumped **by hand** whenever the binning rule (`_spherical_bin`), the hole
# fill or the resolution rule changes. It is part of the artefact name, so a
# bump invalidates every cached grid rather than reinterpreting old bytes.
GRID_VERSION = 0

# `PHASE1-TILE-CONTRACT-V0.md` §6.4 recommends 250,000 so chunk storage stays
# small relative to the band buffer.
DEFAULT_CHUNK_POINTS = 250_000
CLOSE_DROPOUT_BLOCK_ROWS = 64

# Domain separator for the params digest, fixed width. Printable on purpose:
# an artefact header is the first thing anyone will look at with a hex editor.
_PARAMS_MAGIC = b"RMCARVEPARMS"


class CarveGridArtefactError(ValueError):
    """A cached grid file is unusable and must not be trusted.

    Distinct from a cache miss on purpose. A missing file is normal; a file
    whose header, key or payload digest disagrees with its own name is either
    corrupt or renamed, and silently rebuilding would hide the fault.
    """


# ---------------------------------------------------------------------------
# resolution rule and cache key
# ---------------------------------------------------------------------------


def resolve_grid_dimensions(rows: int, cols: int, max_cells: int = CARVE_MAX_CELLS) -> tuple[int, int]:
    """`(height, width)` for a lattice, matching `CoarseRangeGrid.build`.

    Duplicated from `grid.py:317-320` rather than shared, so the legacy path
    stays an independent reference for the bitwise equivalence tests. The
    duplication is guarded by `test_resolution_rule_matches_legacy`, and by
    `GRID_VERSION`, which must be bumped if this rule ever changes.
    """
    import math

    height, width = max(rows, 8), max(cols, 16)
    if height * width > max_cells:
        k = math.sqrt(max_cells / (height * width))
        height, width = max(int(height * k), 8), max(int(width * k), 16)
    return height, width


def carve_params_digest(
    *,
    max_cells: int,
    fill_holes: bool,
    rotation: F64,
    translation: F64,
    lattice_rows: int,
    lattice_cols: int,
    frame_path: str = "",
    algorithm_version: int = GRID_VERSION,
) -> str:
    """SHA-256 over canonical fixed-width bytes for everything `build` bakes in.

    The effective transform chain is in the key because of a fact this
    repository does not settle: if a project can be re-registered without the
    source E57's bytes changing, `source_sha256` would be stable while the pose
    moved, and a cached grid would be silently wrong
    (`PHASE1-TILE-CONTRACT-V0.md` §7). Hashing the matrix makes the cache
    correct either way, without inferring where Cairn stores registration.

    `frame_path` is the `_resolve_frame` branch taken by the reader. It is not
    yet recorded anywhere (PLAN.md §5 item 11), so it defaults to empty; it is
    in the digest now so that recording it later changes keys rather than
    requiring a format change.
    """
    import hashlib

    import numpy as np

    frame = frame_path.encode("utf-8")
    body = struct.pack(
        "<12sIQBIII",
        _PARAMS_MAGIC,
        algorithm_version,
        max_cells,
        1 if fill_holes else 0,
        lattice_rows,
        lattice_cols,
        len(frame),
    )
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3).astype("<f8").tobytes()
    offset = np.asarray(translation, dtype=np.float64).reshape(3).astype("<f8").tobytes()
    return hashlib.sha256(body + matrix + offset + frame).hexdigest()


@dataclass(frozen=True)
class CarveGridKey:
    """Every component of the artefact name, per `PHASE1-TILE-CONTRACT-V0.md` §7.

    `source_identity` is the source file's SHA-256 for a real scan. Synthetic
    fixtures have no file, so an opaque identity string is allowed and the
    in-memory store accepts it; only `DirectoryGridStore` requires a real
    64-character digest, because that is what goes in the filename.
    """

    source_identity: str
    scan_index: int
    grid_version: int
    cells: int
    params_digest: str

    @property
    def is_file_nameable(self) -> bool:
        return _is_sha256_hex(self.source_identity) and _is_sha256_hex(self.params_digest)

    def filename(self) -> str:
        if not self.is_file_nameable:
            raise CarveGridArtefactError(
                "a file-backed carve grid needs a sha256 source digest and params digest; "
                "synthetic identities are in-memory only"
            )
        if self.scan_index < 0 or self.grid_version < 0 or self.cells <= 0:
            raise CarveGridArtefactError("carve grid key fields must be non-negative")
        return (
            f"{self.source_identity}-{self.scan_index}"
            f"-v{self.grid_version}-{self.cells}-{self.params_digest}.rmgrid"
        )


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class GridStore(Protocol):
    """Load/store by exact key. Stored grids are treated as immutable."""

    def load(self, key: CarveGridKey) -> CoarseRangeGrid | None: ...

    def store(self, key: CarveGridKey, grid: CoarseRangeGrid) -> None: ...


# ---------------------------------------------------------------------------
# chunked build
# ---------------------------------------------------------------------------


def build_grid_chunked(
    scan: StructuredScan,
    *,
    max_cells: int = CARVE_MAX_CELLS,
    fill_holes: bool = True,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    block_rows: int = CLOSE_DROPOUT_BLOCK_ROWS,
) -> CoarseRangeGrid:
    """`CoarseRangeGrid.build` without the whole-scan f64 materialisation.

    Bitwise identical to the legacy path, and required to stay so. Chunking
    cannot change the result because every per-sample step is elementwise and
    the reduction is `np.minimum.at`, which selects one of its inputs.
    """
    import numpy as np

    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    height, width = resolve_grid_dimensions(scan.lattice.rows, scan.lattice.cols, max_cells)
    origin = np.asarray(scan.pose.translation, dtype=np.float64)
    flat = np.full(height * width, np.inf, np.float32)

    n = int(scan.xyz.shape[0])
    for start in range(0, n, chunk_points):
        stop = min(start + chunk_points, n)
        _accumulate_chunk(flat, scan.pose, scan.xyz[start:stop], origin, width, height)

    grid = CoarseRangeGrid(
        origin=origin.astype(np.float64),
        rng=flat.reshape(height, width),
        width=width,
        height=height,
    )
    if fill_holes:
        grid.rng = close_dropout_blocked(grid.rng, block_rows=block_rows)
    return grid


def _accumulate_chunk(
    flat: F32, pose: ScanPose, local: F32, origin: F64, width: int, height: int
) -> None:
    """One chunk of `from_world_points`, accumulated into a shared flat grid.

    Mirrors `grid.py:337-348` exactly, on a slice. The only whole-scan-sized
    array in the build is the grid itself.
    """
    import numpy as np

    world = pose.local_to_world(local)
    q = world - origin
    r = np.linalg.norm(q, axis=1)
    good = r > 1e-6
    q, r = q[good], r[good]
    if r.size == 0:
        return
    u, v = _spherical_bin(q, r, width, height)
    bins = v.astype(np.int64) * width + u
    np.minimum.at(flat, bins, r.astype(np.float32))


def close_dropout_blocked(rng: F32, *, block_rows: int = CLOSE_DROPOUT_BLOCK_ROWS) -> F32:
    """Row-blocked `close_dropout`: immutable source in, distinct output out.

    Bitwise equal to `CoarseRangeGrid.close_dropout` and proven so by test.
    Rows do not wrap and columns do — azimuth is periodic, elevation is not —
    which is why the row halo is explicit padding while the column shift is
    still a `roll`.
    """
    import numpy as np

    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    source = np.asarray(rng)
    if source.ndim != 2:
        raise ValueError("a carve grid must be two-dimensional")
    height, width = int(source.shape[0]), int(source.shape[1])
    out: F32 = np.empty_like(source)
    if height == 0 or width == 0:
        return out

    # One padded scratch buffer, reused by every block: block_rows + 2 rows so
    # the one-row halo above and below is always addressable, including at the
    # lattice edges where it reads `inf` — exactly what the whole-grid version
    # does when it blanks the wrapped rows after each roll.
    pad = np.empty((min(block_rows, height) + 2, width), dtype=source.dtype)
    for first in range(0, height, block_rows):
        last = min(first + block_rows, height)
        span = last - first
        view = pad[: span + 2]
        view[0] = np.inf if first == 0 else source[first - 1]
        view[1 : span + 1] = source[first:last]
        view[span + 1] = np.inf if last == height else source[last]

        block = view[1 : span + 1]
        empty = ~np.isfinite(block)
        filled = block.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbour = view[1 - dr : 1 - dr + span]
                shifted = np.roll(neighbour, dc, axis=1) if dc else neighbour
                filled = np.minimum(filled, np.where(empty, shifted, np.inf))
        out[first:last] = filled
    return out


# ---------------------------------------------------------------------------
# neighbour selection and one-at-a-time orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationRef:
    """A neighbour's identity, pose and lattice — but **not** its points.

    This is the type that makes the streamed schedule possible
    (`PHASE1-TILE-CONTRACT-V0.md` §6.2): headers, poses, digests and lattice
    metadata are cheap and can all be held at once; point arrays are loaded one
    station at a time and released before the next. Because the cache key is
    computable from this alone, a cache hit never calls `load` at all.
    """

    station_id: str
    pose: ScanPose
    lattice: LatticeInfo
    load: Callable[[], StructuredScan]
    source_identity: str = ""
    scan_index: int = 0
    frame_path: str = ""

    @property
    def origin(self) -> F64:
        return self.pose.translation

    @classmethod
    def from_scan(cls, scan: StructuredScan, *, source_identity: str = "",
                  scan_index: int = 0, frame_path: str = "") -> StationRef:
        """Wrap an already-resident scan. The scan stays alive via the closure,
        so this form does not bound memory — it exists so the in-memory callers
        keep working unchanged."""
        return cls(
            station_id=scan.station_id,
            pose=scan.pose,
            lattice=scan.lattice,
            load=lambda: scan,
            source_identity=source_identity or f"station:{scan.station_id}",
            scan_index=scan_index,
            frame_path=frame_path,
        )

    def grid_key(self, *, max_cells: int = CARVE_MAX_CELLS, fill_holes: bool = True) -> CarveGridKey:
        height, width = resolve_grid_dimensions(self.lattice.rows, self.lattice.cols, max_cells)
        return CarveGridKey(
            source_identity=self.source_identity,
            scan_index=self.scan_index,
            grid_version=GRID_VERSION,
            cells=height * width,
            params_digest=carve_params_digest(
                max_cells=max_cells,
                fill_holes=fill_holes,
                rotation=self.pose.rotation,
                translation=self.pose.translation,
                lattice_rows=self.lattice.rows,
                lattice_cols=self.lattice.cols,
                frame_path=self.frame_path,
            ),
        )


def select_neighbours(refs: Sequence[StationRef], exclude: str, nearest: int = 2) -> list[StationRef]:
    """The `nearest` refs to `exclude`, ranked by origin distance only.

    Selection reads nothing but station origins, which is the property that
    lets the caller hold every neighbour's metadata without holding any
    neighbour's points.
    """
    import numpy as np

    target = next((r for r in refs if r.station_id == exclude), None)
    if target is None:
        return []
    o = np.asarray(target.origin, dtype=np.float64)
    ranked = sorted(
        (r for r in refs if r.station_id != exclude),
        key=lambda r: float(np.linalg.norm(np.asarray(r.origin, dtype=np.float64) - o)),
    )
    return ranked[: max(nearest, 0)]


def build_or_load(
    ref: StationRef,
    *,
    store: GridStore | None = None,
    max_cells: int = CARVE_MAX_CELLS,
    fill_holes: bool = True,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    block_rows: int = CLOSE_DROPOUT_BLOCK_ROWS,
) -> CoarseRangeGrid:
    """One grid: cache hit, or load-build-release-store.

    The loaded scan is bound to a local and dropped before this returns, so a
    caller looping over neighbours never holds two stations' points at once.
    """
    key = ref.grid_key(max_cells=max_cells, fill_holes=fill_holes)
    if store is not None:
        cached = store.load(key)
        if cached is not None:
            return cached

    scan = ref.load()
    grid = build_grid_chunked(
        scan,
        max_cells=max_cells,
        fill_holes=fill_holes,
        chunk_points=chunk_points,
        block_rows=block_rows,
    )
    del scan

    if store is not None:
        store.store(key, grid)
    return grid
