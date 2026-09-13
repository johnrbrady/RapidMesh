"""How the decimated output is cut into tiles a viewer loads — WP-3.6.

`decimate_tiles.decimate_generation` decimates one **intermediate** tile at a
time and writes one output file per intermediate tile, so today the output
partition *is* the intermediate partition: DEC-021's 512x512 lattice window. That
is a reasonable default and it is not a measured one, which is what
`docs/adr/ADR-006` "Phase 3b tile-size benchmark" exists to fix.

This module makes the output partition a **parameter**. It changes nothing about
the decimator, the metric, the stop rule or the kernel — a patch still goes to
`decimate.decimate_patch` with a locked boundary, exactly as before. What it
changes is *which geometry constitutes a patch*.

Why the partition is the thing that matters, and not merely how files are named
----------------------------------------------------------------------------
ADR-006 Decision 2a: a tile is decimated with its boundary **locked**, so both
sides of a seam agree exactly and no crack can open. The cost is stated as
plainly as the benefit — "locked boundaries never simplify", so a fine partition
leaves a lattice of full-resolution seams running through the surface and eats
the size budget decimation exists to serve.

So output tile size is not a packaging choice made after the fact. It decides
how much of the surface is refused permission to simplify, which decides the
output size, the decimation ratio and the peak working memory all at once. That
is the three-way tension ADR-006 asks to be measured rather than chosen.

The two partitions this measures, and why both
-----------------------------------------------
**Lattice** (`kind="lattice"`). An output tile is `R x C` lattice cells. With
`R = C = 512` it is one DEC-021 window and reproduces `decimate_generation` byte
for byte — which `tests/test_output_tiles.py` asserts, because a new partition
path whose default case did not reproduce the recorded baseline would invalidate
every figure measured against it. Larger tiles group whole windows; smaller
tiles split a window by lattice **row**.

*Rows and not columns, and the reason is a limit rather than a preference.*
`tile_io.TileJoin` carries a per-vertex lattice `row` and a compacted
`global_vertex_index` that is **not** `row * cols + col` — checked, not assumed —
so the column of a vertex is not recoverable from a written generation. Row
splitting is therefore the only sub-window division available without rebuilding
the intermediate at a smaller window, which this package's brief fixes at
512x512. Sub-512 tiles are measured in one axis and that is said plainly rather
than presented as a square.

**Metric** (`kind="metric"`). An output tile is a `cell_m` square of ground, by
the horizontal position of a vertex. DEC-021 removed the metric partition from
the **intermediate** on measured evidence — sample density on a range image goes
as 1/r^2, so a fixed square of floor holds an unbounded number of samples, and
ADR-006 records 5 tiles on a close station against 364 on a long one at the same
4 m. **That finding is about the full-resolution intermediate and this module
does not assume it carries over**, because an error-bounded sweep flattens
density and `CLAUDE.md` §11 requires a cause to be isolated at the stage the
claim is about. It is offered here so the question can be measured on decimated
output instead of inherited.

Determinism
-----------
ADR-006 requires ownership that does not depend on processing order. A vertex
belongs to the output tile its **own** lattice cell or metric cell falls in; a
triangle belongs to the tile owning its lowest global vertex id. Groups are
assembled from intermediate tiles in ascending tile id and vertices are
compacted owned-first in ascending global id, so the same generation and the
same partition produce the same patches, in the same order, with the same
indices.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .output_assemble import (
    OutputPatch,
    designated_corner,
    finish_patch,
    triangle_bucket,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .tiles import TileStore

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]

#: The most vertices one output patch may hold before it is refused rather than
#: attempted. It is a guard, not an operating parameter: the benchmark's whole
#: point is to push tile size up until memory becomes the binding cost, and a
#: run that dies of an allocation failure reports nothing, where a refusal
#: reports the size that would have been needed. Raise it only with the host's
#: free memory in hand.
DEFAULT_MAX_PATCH_VERTICES = 4_000_000


@dataclass(frozen=True)
class OutputPartition:
    """How the decimated output is cut. One recordable object, as settings are.

    `kind="lattice"` takes `rows` x `cols` lattice cells per output tile;
    `kind="metric"` takes a `cell_m` square of ground. Exactly one shape is used
    and the other is ignored, so a record always says which.
    """

    kind: str = "lattice"
    rows: int = 512
    cols: int = 512
    cell_m: float = 4.0

    def __post_init__(self) -> None:
        if self.kind not in ("lattice", "metric"):
            raise ValueError(f"unknown partition kind {self.kind!r}")
        if self.kind == "lattice":
            if self.rows <= 0 or self.cols <= 0:
                raise ValueError("lattice partition needs positive rows and cols")
        elif not (self.cell_m > 0.0):
            raise ValueError("metric partition needs a positive cell_m")

    def describe(self) -> dict[str, Any]:
        if self.kind == "lattice":
            return {"kind": "lattice", "rows": self.rows, "cols": self.cols,
                    "cells_per_tile": self.rows * self.cols}
        return {"kind": "metric", "cell_m": float(self.cell_m)}

    @property
    def label(self) -> str:
        if self.kind == "lattice":
            return f"lattice {self.rows}x{self.cols}"
        return f"metric {self.cell_m:g} m"


def lattice_grid(store: TileStore) -> dict[str, Any]:
    """The generation's lattice grid block, refusing a non-lattice generation.

    A metric-partitioned intermediate has no window and no lattice extent, so
    every rule in this module would be reading fields that are not there. It is
    refused loudly rather than producing an empty plan.
    """
    grid = dict(store.manifest.get("grid") or {})
    if grid.get("kind") != "lattice":
        raise ValueError(
            f"output tiling needs a lattice-partitioned generation, got "
            f"{grid.get('kind')!r}"
        )
    return grid


def tile_window(grid: dict[str, Any], tile_id: int) -> tuple[int, int]:
    """`(first lattice row, first lattice column)` of one intermediate tile.

    Window ids are row-major over `across` columns, which DEC-021's amendment
    states is the total order the ownership rule requires.
    """
    across = int(grid["across"])
    return (
        (int(tile_id) // across) * int(grid["window_rows"]),
        (int(tile_id) % across) * int(grid["window_cols"]),
    )


def plan(store: TileStore, partition: OutputPartition) -> dict[tuple[int, int], list[int]]:
    """Which intermediate tiles each output tile needs, before any is read.

    Returned for the **lattice** kind only, and it is what makes the size guard
    cheap: an output tile's vertex count can be bounded from the manifest
    without opening a single tile, so a patch too large to hold is refused
    before it is assembled rather than during.

    A row-split output tile draws from one intermediate tile; a grouped one
    draws from several. Both are expressed the same way.
    """
    grid = lattice_grid(store)
    out: dict[tuple[int, int], list[int]] = {}
    for tile_id in store.tile_ids:
        row0, col0 = tile_window(grid, tile_id)
        # A window spans `window_rows` lattice rows, which may straddle several
        # output tiles when the output tile is shorter than the window.
        span = range(row0, row0 + int(grid["window_rows"]), 1)
        keys = {(r // partition.rows, col0 // partition.cols) for r in span}
        for key in sorted(keys):
            out.setdefault(key, []).append(int(tile_id))
    return {key: sorted(set(ids)) for key, ids in sorted(out.items())}


def patches(
    store: TileStore,
    partition: OutputPartition,
    *,
    max_patch_vertices: int = DEFAULT_MAX_PATCH_VERTICES,
) -> Iterator[OutputPatch | tuple[tuple[int, int], int]]:
    """Every output patch of a generation, in ascending key order.

    Yields an `OutputPatch`, or a `(key, vertices_needed)` pair for a patch
    refused by `max_patch_vertices`. The caller records the refusal — the size
    that could not be held is the measurement, and swallowing it would turn a
    result into a gap.
    """
    if partition.kind == "metric":
        yield from _metric_patches(store, partition, max_patch_vertices)
        return
    yield from _lattice_patches(store, partition, max_patch_vertices)


def _lattice_patches(
    store: TileStore, partition: OutputPartition, ceiling: int
) -> Iterator[OutputPatch | tuple[tuple[int, int], int]]:
    """Group whole windows, or split one window by lattice row."""

    grid = lattice_grid(store)
    sizes = {int(e["tile_id"]): int(e["vertex_count"]) for e in store.manifest["tiles"]}
    for key, tile_ids in plan(store, partition).items():
        # The bound is the sum of the source tiles' vertex counts. It over-counts
        # a shared halo vertex, so it is an upper bound on what the patch will
        # hold — which is the direction a guard has to err in.
        bound = sum(sizes[t] for t in tile_ids)
        if bound > ceiling:
            yield key, bound
            continue
        patch = _assemble(store, grid, key, tile_ids, partition)
        if patch.triangle_count:
            yield patch


def _assemble(
    store: TileStore,
    grid: dict[str, Any],
    key: tuple[int, int],
    tile_ids: list[int],
    partition: OutputPartition,
) -> OutputPatch:
    """Merge the named intermediate tiles into one patch, owned-first.

    Vertices are keyed by **global id**, so a vertex held by two source tiles —
    one owning it, one carrying it as halo — becomes one vertex here, and the
    seam between those two tiles disappears from the output. That disappearance
    *is* the benefit a larger tile buys, and it is why grouping is not a renaming
    of files.

    **Triangle ownership is inherited, never re-derived.** Pass B already
    assigned each triangle to exactly one intermediate tile, and those
    assignments partition the surface exactly; a grouped output tile is simply
    the union of its members'. Re-deriving ownership from the geometry looked
    reasonable and was wrong — a "lowest global id" rule silently dropped 274 of
    tile 0's 256,786 triangles on the test fixture, because the rule disagreed
    with Pass B's and no tile then claimed them.
    """
    import numpy as np

    split = partition.rows < int(grid["window_rows"])
    gid_parts: list[Any] = []
    pos_parts: list[Any] = []
    own_parts: list[Any] = []
    tri_parts: list[Any] = []
    for tile_id in tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        gids = np.asarray(join.global_vertex_index, np.int64)
        owned = np.asarray(join.owned, bool)
        local = np.asarray(payload.triangles, np.int64)
        if split:
            row = np.asarray(join.row, np.int64)
            bucket = row // partition.rows
            owned = owned & (bucket == key[0])
            local = local[triangle_bucket(local, bucket, np.asarray(join.owned, bool))
                          == key[0]]
        gid_parts.append(gids)
        pos_parts.append(np.asarray(payload.positions, np.float32))
        own_parts.append(owned)
        tri_parts.append(gids[local])
        del payload, join

    gid = np.concatenate(gid_parts)
    pos = np.concatenate(pos_parts)
    own = np.concatenate(own_parts)
    tri_gid = (
        np.concatenate(tri_parts) if tri_parts else np.empty((0, 3), np.int64)
    )
    return finish_patch(key, gid, pos, own, tri_gid, tuple(tile_ids))


def metric_cell(positions: Any, cell_m: float) -> Any:
    """Which `cell_m` square of ground each vertex sits in, as `(i, j)` packed.

    Horizontal only. A range image's density problem is a *floor area* problem —
    DEC-021's 1/r^2 argument is about how many samples land on a fixed square of
    ground — so adding the vertical axis would measure a different thing and
    would flatter the answer by splitting a wall into storeys.
    """
    import numpy as np

    xy = np.asarray(positions, np.float64)[:, :2]
    return np.floor(xy / float(cell_m)).astype(np.int64)


def _metric_patches(
    store: TileStore, partition: OutputPartition, ceiling: int
) -> Iterator[OutputPatch | tuple[tuple[int, int], int]]:
    """One patch per occupied ground cell, assembled from whatever tiles reach it.

    Two passes. The first reads each intermediate tile once and records, per
    cell, which tiles contribute to it and how many owned vertices it will hold;
    the second assembles a cell at a time. Nothing full-resolution and
    station-sized is held between them except the per-cell tallies, which are
    bounded by the number of occupied cells.
    """
    import numpy as np

    reach: dict[tuple[int, int], set[int]] = {}
    size: dict[tuple[int, int], int] = {}
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        owned = np.asarray(join.owned, bool)
        if bool(owned.any()):
            cell = metric_cell(np.asarray(payload.positions)[owned], partition.cell_m)
            keys, freq = np.unique(cell, axis=0, return_counts=True)
            for (i, j), n in zip(keys.tolist(), freq.tolist(), strict=True):
                key = (int(i), int(j))
                reach.setdefault(key, set()).add(int(tile_id))
                size[key] = size.get(key, 0) + int(n)
        del payload, join

    sizes = {int(e["tile_id"]): int(e["vertex_count"]) for e in store.manifest["tiles"]}
    for key in sorted(reach):
        tile_ids = sorted(reach[key])
        bound = sum(sizes[t] for t in tile_ids)
        if bound > ceiling:
            yield key, bound
            continue
        patch = _assemble_metric(store, key, tile_ids, partition)
        if patch.triangle_count:
            yield patch


def _assemble_metric(
    store: TileStore, key: tuple[int, int], tile_ids: list[int],
    partition: OutputPartition,
) -> OutputPatch:
    """`_assemble`, with ownership decided by ground cell instead of by window.

    A triangle goes to the cell of its lowest **owned** corner, falling back to
    its lowest corner — the same exclusive-within-one-payload rule
    `triangle_bucket` applies to row bands, for the same reason: Pass B's
    assignment is the one that partitions the surface, and this only subdivides
    it.
    """
    import numpy as np

    gid_parts: list[Any] = []
    pos_parts: list[Any] = []
    own_parts: list[Any] = []
    tri_parts: list[Any] = []
    for tile_id in tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        gids = np.asarray(join.global_vertex_index, np.int64)
        position = np.asarray(payload.positions, np.float32)
        cell = metric_cell(position, partition.cell_m)
        here = (cell[:, 0] == key[0]) & (cell[:, 1] == key[1])
        owned = np.asarray(join.owned, bool)
        local = np.asarray(payload.triangles, np.int64)
        chosen = designated_corner(local, owned, gids)
        local = local[here[chosen]]
        gid_parts.append(gids)
        pos_parts.append(position)
        own_parts.append(owned & here)
        tri_parts.append(gids[local])
        del payload, join

    gid = np.concatenate(gid_parts)
    pos = np.concatenate(pos_parts)
    own = np.concatenate(own_parts)
    tri_gid = (
        np.concatenate(tri_parts) if tri_parts else np.empty((0, 3), np.int64)
    )
    return finish_patch(key, gid, pos, own, tri_gid, tuple(tile_ids))
