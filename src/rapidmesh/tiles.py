"""
The spatial tile grid and the store that reads a written generation — WP-3.2.

`PHASE1-TILE-CONTRACT-V0.md` §2 separates write order from view order: the sweep
writes band-aligned segments (elevation annuli), and Pass B merges them into a
new immutable generation of compact spatial tiles. This module owns the *spatial*
half — where the tile boundaries are, which tile owns what, and how a written
generation is read back.

Tile size is a setting, not a constant
--------------------------------------
ADR-006 Decision 2a is explicit that tile size is a **measured parameter**, with
a three-way tension behind it: smaller tiles lower peak working memory but lock
more boundary vertices and stream more requests; larger tiles decimate better but
raise the peak. Phase 3b benchmarks it on the reference station. So `TileGrid`
takes an edge length and the manifest records it, and nothing downstream may
assume a particular value — `tests/test_tiles.py` runs the tile-independent
outputs at two sizes and requires them equal.

Deterministic ownership, and why it is a reproducibility rule
-------------------------------------------------------------
ADR-006 Decision 2a again: *"A shared vertex or triangle belongs to exactly one
tile, by a rule that does not depend on processing order or thread scheduling —
for example lowest tile index by (x, y, z) of the tile origin."* Here:

* a **vertex** belongs to the tile containing its own position;
* a **triangle** belongs to the **lowest tile index** among its three corners'
  tiles.

Tile ids are laid out x-major, so "lowest tile index" *is* "lowest by (x, y, z)".
Both rules are functions of position alone — no order, no thread, no arrival
time — which is what makes two runs produce identical tiles rather than merely
equivalent ones.

Reading a generation
--------------------
`TileStore` is also the `qa_stream.GeometrySource` WP-3.3 measures against, so
QA can read a written generation without a resident mesh. Its vertex stream
yields each vertex **once**, from the tile that owns it, and its triangle stream
yields each triangle once, from the tile that emitted it. Neither can double
count, because ownership is exclusive by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .tile_io import read_tile, read_tile_join

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .qa_stream import IdentifiedTriangleBlock, TriangleBlock, VertexBlock

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

# Default tile edge in metres. A default, not a decision: ADR-006 2a makes the
# value a Phase 3b measurement and the manifest records whatever was used.
DEFAULT_TILE_SIZE_M = 4.0

# Guard on the grid's cell count, checked before a grid is built. A station whose
# bounds and tile size imply more cells than this has a bad tile size, and a
# silent 10^9-cell grid would be a memory fault dressed as geometry.
MAX_TILE_CELLS = 1 << 22

#: DEC-021's intermediate partition: a tile is W x H lattice cells.
#: 512 x 512 caps a window at 262,144 owned vertices, which is ~116 MB at the
#: 442 B/vertex WP-10m measured. A setting, not a constant — the manifest
#: records it, and `PHASE1-DETERMINISM-SPEC.md` §3 requires that of anything
#: that can move a figure.
DEFAULT_WINDOW_ROWS = 512
DEFAULT_WINDOW_COLS = 512

MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.json"
#: v1 (WP-A, DEC-021): the grid block gained `kind` and, for lattice
#: generations, the window and lattice dimensions. A v0 store cannot be read by
#: this code and is not meant to be — the partition it describes is not the one
#: Pass B now produces.
TILE_CONTRACT = "rapidmesh-tile-v1"


@dataclass(frozen=True)
class TileGrid:
    """A uniform spatial partition of one station, in project-axis offsets.

    `origin` is the grid's own anchor — the minimum corner of the covered box —
    and is **not** the station's float64 origin. Positions handed here are
    already offsets from that; `SPATIAL-CONTRACT.md` §2.4 forbids a tile
    introducing a second transform, and this one does not: it only decides which
    box a point is in.
    """

    origin: F64                 # (3,)
    size: float
    shape: tuple[int, int, int]

    @classmethod
    def covering(
        cls, low: F64, high: F64, size: float = DEFAULT_TILE_SIZE_M
    ) -> TileGrid:
        """The smallest grid of `size` cells that covers `[low, high]`."""
        import numpy as np

        if not (size > 0.0):
            raise ValueError(f"tile size must be positive, got {size}")
        lo = np.asarray(low, np.float64)
        hi = np.asarray(high, np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("tile grid bounds must be three-dimensional")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("tile grid bounds must be finite")
        span = np.maximum(hi - lo, 0.0)
        counts = np.maximum(np.ceil(span / size).astype(np.int64), 1)
        cells = int(counts.prod())
        if cells > MAX_TILE_CELLS:
            raise ValueError(
                f"tile size {size} m implies {cells:,} cells over a "
                f"{span.tolist()} m station; choose a larger tile"
            )
        return cls(origin=lo, size=float(size), shape=(int(counts[0]), int(counts[1]), int(counts[2])))

    @property
    def count(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def index_of(self, positions: F32 | F64) -> I64:
        """Linear tile id per position, x-major so lower id means lower (x,y,z)."""
        import numpy as np

        p = np.asarray(positions, np.float64)
        cell = np.floor((p - self.origin) / self.size).astype(np.int64)
        for axis in range(3):
            np.clip(cell[:, axis], 0, self.shape[axis] - 1, out=cell[:, axis])
        out: I64 = (
            cell[:, 0] * self.shape[1] + cell[:, 1]
        ) * self.shape[2] + cell[:, 2]
        return out

    def bounds_of(self, tile_id: int) -> F64:
        """`(6,)` min xyz then max xyz of one cell, in the same offsets."""
        import numpy as np

        z = tile_id % self.shape[2]
        y = (tile_id // self.shape[2]) % self.shape[1]
        x = tile_id // (self.shape[2] * self.shape[1])
        low = self.origin + np.array([x, y, z], np.float64) * self.size
        out: F64 = np.concatenate((low, low + self.size))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "metric",
            "origin": [float(v) for v in self.origin],
            "size_m": self.size,
            "shape": list(self.shape),
            "cells": self.count,
        }


@dataclass(frozen=True)
class LatticeGrid:
    """A partition of one station into windows of `W x H` **lattice cells**.

    DEC-021. The metric grid above cuts the floor; this cuts the range image
    the scanner actually produced, and that is the difference between a bound
    and a hope.

    **Why the metric grid could not bound anything.** Sample density on a
    terrestrial range image goes as 1/r². A 4 m cell two metres from the
    scanner subtends a large part of the lattice and holds most of a station;
    the same cell at forty metres holds a handful of samples. WP-10m through
    WP-11m.t measured exactly that: ordinal 20 packs its surface into **5**
    tiles whose largest holds 9,341,715 vertices, while ordinal 3 — comparable
    point count, longer range — spreads the same work over 364. Shrinking the
    metre count does not fix it; it moves the pathology to a different range
    and multiplies tiles on the far stations (advisory candidate B).

    **Why this one bounds by construction.** A vertex belongs to the window
    containing its own lattice cell, so a window owns at most `W x H` vertices
    whatever the geometry does. A triangle joins cells that are adjacent on the
    lattice, so every corner a window does not own lies in its 1-cell ring and
    the resident set is at most `(W+2) x (H+2)`. Both are arithmetic on the cell
    id — no lookup, no sort, and nothing to measure before trusting.

    The ring needs no code of its own: `tile_build._spool_block` already files a
    triangle under each distinct corner window, which *is* the ring, and it
    crosses the azimuth seam wherever the filters already joined across it.
    `wrap` is recorded because a reader should not have to infer it, not because
    the arithmetic needs it.
    """

    lattice_rows: int
    lattice_cols: int
    window_rows: int = DEFAULT_WINDOW_ROWS
    window_cols: int = DEFAULT_WINDOW_COLS
    wrap: bool = False

    def __post_init__(self) -> None:
        for name in ("lattice_rows", "lattice_cols", "window_rows", "window_cols"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.windows > MAX_TILE_CELLS:
            raise ValueError(
                f"a {self.window_rows}x{self.window_cols} window over a "
                f"{self.lattice_rows}x{self.lattice_cols} lattice implies "
                f"{self.windows:,} windows; choose a larger window"
            )

    @property
    def across(self) -> int:
        """Windows across the lattice, so a window id is row-major."""
        return -(-int(self.lattice_cols) // int(self.window_cols))

    @property
    def down(self) -> int:
        return -(-int(self.lattice_rows) // int(self.window_rows))

    @property
    def windows(self) -> int:
        return self.across * self.down

    @property
    def count(self) -> int:
        return self.windows

    @property
    def owned_cap(self) -> int:
        """The most vertices one window can own. The point of the whole change."""
        return int(self.window_rows) * int(self.window_cols)

    @property
    def resident_cap(self) -> int:
        """Owned plus the 1-cell ring: the most one window can be resident for."""
        return (int(self.window_rows) + 2) * (int(self.window_cols) + 2)

    def index_of(self, cell_ids: I64) -> I64:
        """Window id per **lattice cell id** (`row * lattice_cols + col`).

        Takes cell ids, not positions — the metric grid's `index_of` takes
        positions and the two are not interchangeable. Row-major, so a lower id
        is a lower (row-window, col-window) exactly as ADR-006 2a's ownership
        rule requires of *some* total order on tiles.
        """
        import numpy as np

        ids = np.asarray(cell_ids, np.int64)
        row, col = np.divmod(ids, int(self.lattice_cols))
        out: I64 = (row // int(self.window_rows)) * self.across + (
            col // int(self.window_cols)
        )
        return out

    def bounds_of(self, tile_id: int) -> F64:
        """A degenerate box: a lattice window has no metric extent of its own.

        `tile_assemble._bounds` uses this only when a tile has no positions, and
        a tile with no positions is never written, so this is the shape of an
        answer rather than an answer. The written `bounds` are always the tile's
        actual projected extent, which is what a viewer culls against.
        """
        import numpy as np

        out: F64 = np.zeros(6, np.float64)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "lattice",
            "window_rows": int(self.window_rows),
            "window_cols": int(self.window_cols),
            "lattice_rows": int(self.lattice_rows),
            "lattice_cols": int(self.lattice_cols),
            "across": self.across,
            "down": self.down,
            "cells": self.windows,
            "columns_wrap": bool(self.wrap),
            "owned_cap": self.owned_cap,
            "resident_cap": self.resident_cap,
        }


#: Either intermediate partition. A union of two concrete grids rather than a
#: Protocol, because `index_of` does not mean the same thing on both — the
#: metric grid takes projected positions and the lattice grid takes cell ids.
#: A caller has to know which one it is holding, and a Protocol would hide that.
Partition = TileGrid | LatticeGrid


@dataclass(frozen=True)
class TileStore:
    """A published generation on disk, read one tile at a time.

    Doubles as `qa_stream.GeometrySource` so WP-3.3's QA can measure a written
    generation. The counts are read from the manifest rather than by opening
    every tile, so asking a store how big it is does not cost a full scan.
    """

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, root: Path) -> TileStore:
        """Open the generation `current.json` points at, verifying nothing else.

        Digests are checked when a tile is actually read — `tile_io` refuses a
        payload whose digest does not match its header — so opening a store is
        cheap and reading one is safe.
        """
        pointer = json.loads((root / CURRENT_NAME).read_text(encoding="utf-8"))
        generation = root / "generations" / str(pointer["generation"])
        manifest = json.loads((generation / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("contract") != TILE_CONTRACT:
            raise ValueError(f"unknown tile contract {manifest.get('contract')!r}")
        return cls(root=generation, manifest=manifest)

    @property
    def tile_ids(self) -> list[int]:
        return [int(entry["tile_id"]) for entry in self.manifest["tiles"]]

    @property
    def vertex_count(self) -> int:
        """Owned vertices across the generation — each vertex exactly once."""
        return int(self.manifest["owned_vertex_count"])

    @property
    def triangle_count(self) -> int:
        return int(self.manifest["triangle_count"])

    @property
    def tile_size(self) -> float:
        return float(self.manifest["grid"]["size_m"])

    @property
    def has_rgb(self) -> bool:
        """Whether the source scan carried colour. See the manifest comment."""
        return bool(self.manifest["has_rgb"])

    @property
    def origin(self) -> F64:
        """The station's float64 anchor, from the manifest rather than a tile.

        A generation with no tiles still has an origin, and taking it from the
        first tile would silently return zeros for that case.
        """
        import numpy as np

        out: F64 = np.asarray(self.manifest["origin"], np.float64)
        return out

    def tile_path(self, tile_id: int) -> Path:
        return self.root / "tile" / f"{tile_id:012d}.rmtile"

    def join_path(self, tile_id: int) -> Path:
        return self.root / "tile" / f"{tile_id:012d}.rmtjoin"

    def payload(self, tile_id: int) -> Any:
        return read_tile(self.tile_path(tile_id))

    def join(self, tile_id: int) -> Any:
        return read_tile_join(self.join_path(tile_id))

    # -- GeometrySource ------------------------------------------------------

    def vertex_blocks(self) -> Iterator[VertexBlock]:
        """Every owned vertex once, tile by tile, with its global index.

        A tile used to *be* the block. WP-13b makes it the unit of **reading** —
        which is all a tile file ever was — and hands QA slices of it that
        honour `QA_VERTEX_BLOCK_BYTES`, exactly as `ResidentMesh` always has.
        Same vertices, same order, more blocks: nothing downstream can tell the
        difference except by its own peak.

        ADR-006 2a still makes tile size the measured parameter that bounds the
        *output*. WP-13a is why it no longer also bounds QA's resident set —
        ordinal 20 packs its surface into 5 tiles, so "one tile is the block"
        made a single QA block 2.74 GB.
        """
        import numpy as np

        from .qa_stream import VertexBlock, vertex_block_step

        step = vertex_block_step()
        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            owned = payload.owned_count
            if owned == 0:
                continue
            join = self.join(tile_id)
            ids = np.asarray(join.global_vertex_index[:owned], np.int64)
            for lo in range(0, owned, step):
                hi = min(lo + step, owned)
                yield VertexBlock(
                    ids=ids[lo:hi],
                    xyz=payload.positions[lo:hi].astype(np.float64),
                )

    def triangle_blocks(self) -> Iterator[TriangleBlock]:
        """Every emitted triangle once, with global corner ids and positions.

        The corners come from the tile's own vertex array, which is why a tile
        stores positions for boundary vertices it does not own: a triangle must
        be resolvable inside the tile that emitted it, with no cross-tile join.

        Sliced to `QA_TRIANGLE_BLOCK_BYTES` — see `vertex_blocks`.
        """
        import numpy as np

        from .qa_stream import TriangleBlock, triangle_block_step

        step = triangle_block_step()
        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            count = payload.triangle_count
            if count == 0:
                continue
            join = self.join(tile_id)
            globals_ = np.asarray(join.global_vertex_index, np.int64)
            for lo in range(0, count, step):
                # Slice, *then* widen. Widening the tile's triangles or corners
                # whole and slicing the result would still build the arrays this
                # package exists to remove: 443,786,352 B of int64 indices and
                # 1,331,359,056 B of float64 corners on ordinal 20's largest
                # tile (WP-13a §2.2).
                local = payload.triangles[lo : lo + step].astype(np.int64)
                yield TriangleBlock(
                    ids=globals_[local],
                    corners=payload.positions[local].astype(np.float64),
                )

    def identified_triangle_blocks(self) -> Iterator[IdentifiedTriangleBlock]:
        """The same triangles, with each corner's `source_sample_id` beside it.

        The ids come from the tile's own server-only join file, so a reader that
        was handed only the client-eligible payload cannot produce this block at
        all — DEC-004's boundary is the file split, not a runtime check.

        Sliced to `QA_TRIANGLE_BLOCK_BYTES` — see `vertex_blocks`.
        """
        import numpy as np

        from .qa_stream import IdentifiedTriangleBlock, triangle_block_step

        step = triangle_block_step()
        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            count = payload.triangle_count
            if count == 0:
                continue
            join = self.join(tile_id)
            samples = np.asarray(join.source_sample_id, np.int64)
            globals_ = np.asarray(join.global_vertex_index, np.int64)
            for lo in range(0, count, step):
                local = payload.triangles[lo : lo + step].astype(np.int64)
                yield IdentifiedTriangleBlock(
                    sample_ids=samples[local],
                    ids=globals_[local],
                    corners=payload.positions[local].astype(np.float64),
                )

    # -- reconstitution, for the equivalence harness only --------------------

    def station_vertex_order(self) -> tuple[I64, I64]:
        """Global vertex index and source sample id for every owned vertex.

        Returned in ascending global-index order, which is `build_mesh`'s own
        compaction order. Used by the T2 adapter and by nothing in the pipeline:
        it materialises a station-wide array, which is the thing the tiling
        exists to avoid.
        """
        import numpy as np

        indices: list[Any] = []
        samples: list[Any] = []
        for tile_id in self.tile_ids:
            join = self.join(tile_id)
            owned = np.asarray(join.owned, bool)
            indices.append(join.global_vertex_index[owned])
            samples.append(join.source_sample_id[owned])
        if not indices:
            empty = np.empty(0, np.int64)
            return empty, empty.copy()
        index = np.concatenate(indices)
        sample = np.concatenate(samples)
        order = np.argsort(index, kind="stable")
        return index[order], sample[order]
