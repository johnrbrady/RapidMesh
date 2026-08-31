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

MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.json"
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
            "origin": [float(v) for v in self.origin],
            "size_m": self.size,
            "shape": list(self.shape),
            "cells": self.count,
        }


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

        One tile is the block. That is the whole point of the tiling: tile size
        is the parameter that bounds this, which is why ADR-006 2a makes it a
        measured one rather than a constant.
        """
        import numpy as np

        from .qa_stream import VertexBlock

        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            owned = payload.owned_count
            if owned == 0:
                continue
            join = self.join(tile_id)
            yield VertexBlock(
                ids=np.asarray(join.global_vertex_index[:owned], np.int64),
                xyz=payload.positions[:owned].astype(np.float64),
            )

    def triangle_blocks(self) -> Iterator[TriangleBlock]:
        """Every emitted triangle once, with global corner ids and positions.

        The corners come from the tile's own vertex array, which is why a tile
        stores positions for boundary vertices it does not own: a triangle must
        be resolvable inside the tile that emitted it, with no cross-tile join.
        """
        import numpy as np

        from .qa_stream import TriangleBlock

        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            if payload.triangle_count == 0:
                continue
            join = self.join(tile_id)
            local = payload.triangles.astype(np.int64)
            yield TriangleBlock(
                ids=np.asarray(join.global_vertex_index, np.int64)[local],
                corners=payload.positions[local].astype(np.float64),
            )

    def identified_triangle_blocks(self) -> Iterator[IdentifiedTriangleBlock]:
        """The same triangles, with each corner's `source_sample_id` beside it.

        The ids come from the tile's own server-only join file, so a reader that
        was handed only the client-eligible payload cannot produce this block at
        all — DEC-004's boundary is the file split, not a runtime check.
        """
        import numpy as np

        from .qa_stream import IdentifiedTriangleBlock

        for tile_id in self.tile_ids:
            payload = self.payload(tile_id)
            if payload.triangle_count == 0:
                continue
            join = self.join(tile_id)
            local = payload.triangles.astype(np.int64)
            yield IdentifiedTriangleBlock(
                sample_ids=np.asarray(join.source_sample_id, np.int64)[local],
                ids=np.asarray(join.global_vertex_index, np.int64)[local],
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
