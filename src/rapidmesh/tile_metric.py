"""
The metric partition's ownership — pre-DEC-021, kept for the tests that pin it.

ADR-006 Decision 2a cut the intermediate into squares of floor and WP-A replaced
that with lattice windows, because a metric cell does not bound how many samples
fall in it (DEC-021, and the amendment inside 2a). This module is what remains
of the old rule: project every mesh vertex, cover the result with a grid of a
chosen size, and assign each vertex to the cell it lands in.

It is **not production**. `tile_build.build_tiles` reaches it only for
`partition="metric"`, which `test_tiles.py` uses to hold the pre-WP-A behaviour
and the three golden files use to keep their pre-change digests comparable. It
needs the station-wide positions WP-B removed from the production path, which is
the other reason it lives behind its own door rather than inside the builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tiles import TileGrid

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .types import ScanPose

    F32 = npt.NDArray[np.float32]
    I64 = npt.NDArray[np.int64]

# Vertices projected per block while the grid bounds and the ownership map are
# built. 262,144 float32 positions is 3.1 MB, the same order as the QA caps.
PROJECT_BLOCK = 262_144


def _grid_for(
    pose: ScanPose, xyz: F32, keep_idx: I64, tile_size: float
) -> TileGrid:
    """Bounds of the projected vertices, one block at a time."""
    import numpy as np

    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for start in range(0, keep_idx.size, PROJECT_BLOCK):
        piece = _project(pose, xyz, keep_idx[start : start + PROJECT_BLOCK])
        low = np.minimum(low, piece.min(axis=0))
        high = np.maximum(high, piece.max(axis=0))
    if not np.all(np.isfinite(low)):
        low = np.zeros(3)
        high = np.zeros(3)
    return TileGrid.covering(low, high, tile_size)


def _assign_tiles(
    pose: ScanPose, xyz: F32, keep_idx: I64, grid: TileGrid
) -> Any:
    """Owning tile per mesh vertex, as int32. Four bytes a vertex, and the only
    station-wide array this package adds."""
    import numpy as np

    out = np.empty(keep_idx.size, np.int32)
    for start in range(0, keep_idx.size, PROJECT_BLOCK):
        stop = min(start + PROJECT_BLOCK, keep_idx.size)
        out[start:stop] = grid.index_of(
            _project(pose, xyz, keep_idx[start:stop])
        ).astype(np.int32)
    return out


def _project(pose: ScanPose, xyz: F32, index: I64) -> F32:
    """Scanner-local offsets to project-axis float32, exactly as `build_mesh`.

    The rotation is applied once and only then narrowed, per
    `SPATIAL-CONTRACT.md`; doing it on a subset is the same arithmetic row by
    row, so a tile's stored position is bitwise what the resident mesh held.
    """
    import numpy as np

    out: F32 = pose.rotate_local(xyz[index]).astype(np.float32)
    return out
