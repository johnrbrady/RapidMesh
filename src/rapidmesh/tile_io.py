"""
Final spatial tiles on disk — `PHASE1-TILE-CONTRACT-V0.md` §3 and §5.1, WP-3.2.

A `.rmtile` is one compact region of the finished surface, written once and never
rewritten. `segments_io` holds the *provisional* band-aligned segments Pass A
writes; this holds the immutable spatial generation Pass B publishes. §2 is the
reason there are two: a band is an elevation annulus and a tile is a compact
region, and neither ordering can be the other without the station going resident.

Two files per tile, and the split is the DEC-004 boundary
--------------------------------------------------------
    <tile>.rmtile     positions, normals, optional colour, local triangles
    <tile>.rmtjoin    server-only: source sample id, global vertex index,
                      lattice row, per vertex

Triangles index **tile-local** vertex slots, `0…V-1`. Nothing in `.rmtile`
carries a stable sample id, a lattice cell, a component id or a source digest,
so the redaction boundary is structural rather than a serialiser's good manners.
Phase 1's generation is server-only in its entirety and emits no client manifest
(§3); the split exists so that when a decimated client tier does become valid,
the thing it would ship already excludes the evidence join.

Owned vertices come first, and the normals array is shorter than the positions
array
-------------------------------------------------------------------------------
A vertex belongs to exactly one tile — ADR-006 Decision 2a's deterministic
ownership — but a triangle owned by tile T can have a corner owned by tile U, so
T stores that corner's *position* to stay geometrically self-contained. It does
**not** store that corner's normal, and the format makes that impossible rather
than merely discouraged: `normals` has `owned_count` rows, not `vertex_count`.

The reason is arithmetic, not tidiness. `triangulate.build_mesh` accumulates an
area-weighted normal over every triangle incident to a vertex. T holds every
triangle incident to a vertex **it owns** — a triangle incident to that vertex has
it as a corner, and the spool sends a triangle to each of its corners' tiles — so
T's sum for an owned vertex is the complete one. For a corner owned by U, T sees
only the triangles that happen to touch T, and a partial sum written into a
normal slot would look exactly like a finished normal. Storing a shorter array
means a reader that wants a boundary vertex's normal has to go and ask the tile
that owns it.

Layout
------
    header      magic, contract, kind, flags, header bytes, record count,
                payload bytes, SHA-256 of the payload
    fixed       tile_id:u64, origin:f64[3], bounds:f64[6],
                vertex_count:u32, owned_count:u32, triangle_count:u32, pad:u32
    arrays      positions:f32[V,3]
                normals:f32[O,3]
                rgb:u8[V,3]            (only when FLAG_HAS_RGB)
                triangles:u32[T,3]

Writes are temp-then-`os.replace` in the destination directory, exactly as
`segments_io` does it: atomic within a volume on Windows and POSIX, and
`os.rename` over an existing file raises on Windows (§5.1). The payload digest is
in the header, so a truncated tile fails closed instead of being read short.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I32 = npt.NDArray[np.int32]
    I64 = npt.NDArray[np.int64]
    U8 = npt.NDArray[np.uint8]
    U32 = npt.NDArray[np.uint32]

TILE_MAGIC = b"RMTILE_0"
TILE_CONTRACT_VERSION = 0

KIND_TILE = 1
KIND_TILE_JOIN = 2

FLAG_HAS_RGB = 1

# magic, contract, kind, flags, header bytes, record count, payload bytes, digest
_HEADER = struct.Struct("<8sIHHIQQ32s")
# tile id, origin[3], bounds[6], vertex count, owned count, triangle count, pad
_FIXED = struct.Struct("<Q3d6dIIII")

_JOIN_FIELDS = [
    ("source_sample_id", "<i8"), ("global_vertex_index", "<i8"),
    ("row", "<i4"), ("owned", "u1"), ("pad", "u1", (3,)),
]

# Checked before any allocation sized from a header, as `segments_io.MAX_RECORDS`
# is. A tile with more vertices than this is a corrupt length field, not a tile.
MAX_TILE_VERTICES = 1 << 32


class TileError(ValueError):
    """A tile file is unusable and must not be trusted.

    Raised rather than repaired: a short read that silently yields fewer
    triangles would delete survey geometry and report success.
    """


@dataclass(frozen=True)
class TilePayload:
    """One tile's geometry, as it sits on disk and in memory.

    `origin` is the station's float64 anchor and is the *same* for every tile —
    `SPATIAL-CONTRACT.md` §2.4 and `MeshData`. A tile never introduces a second
    rotation, translation or scale; `bounds` describe where the tile sits, they
    do not move it.
    """

    tile_id: int
    origin: F64                 # (3,)
    bounds: F64                 # (6,) min xyz then max xyz, project-axis offsets
    positions: F32              # (V,3)
    normals: F32                # (O,3), owned vertices only
    triangles: U32              # (T,3) local indices
    rgb: U8 | None = None       # (V,3)

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def owned_count(self) -> int:
        return int(self.normals.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class TileJoin:
    """The server-only per-vertex evidence join, in tile vertex-slot order."""

    source_sample_id: I64       # (V,)
    global_vertex_index: I64    # (V,)
    row: I32                    # (V,)
    owned: Any                  # (V,) bool


def write_tile(path: Path, payload: TilePayload) -> Path:
    """Fixed block then arrays, temp-then-replace, digest in the header."""
    import numpy as np

    if payload.owned_count > payload.vertex_count:
        raise TileError(
            f"tile {payload.tile_id} claims {payload.owned_count} owned vertices "
            f"of {payload.vertex_count}"
        )
    if payload.triangle_count and int(payload.triangles.max()) >= payload.vertex_count:
        raise TileError(
            f"tile {payload.tile_id} has a triangle referencing a vertex slot "
            "outside its own vertex array"
        )
    origin = np.asarray(payload.origin, np.float64)
    bounds = np.asarray(payload.bounds, np.float64)
    flags = FLAG_HAS_RGB if payload.rgb is not None else 0
    body = [
        _FIXED.pack(
            payload.tile_id, *origin.tolist(), *bounds.tolist(),
            payload.vertex_count, payload.owned_count, payload.triangle_count, 0,
        ),
        np.ascontiguousarray(payload.positions, np.float32).tobytes(),
        np.ascontiguousarray(payload.normals, np.float32).tobytes(),
    ]
    if payload.rgb is not None:
        body.append(np.ascontiguousarray(payload.rgb, np.uint8).tobytes())
    body.append(np.ascontiguousarray(payload.triangles, np.uint32).tobytes())
    return _write(path, KIND_TILE, payload.vertex_count, b"".join(body), flags)


def read_tile(path: Path) -> TilePayload:
    """Validate the header, verify the digest, rebuild the arrays."""
    import numpy as np

    payload, flags, _ = _read(path, KIND_TILE)
    if len(payload) < _FIXED.size:
        raise TileError(f"tile is truncated before its fixed block: {path.name}")
    fields = _FIXED.unpack(payload[: _FIXED.size])
    tile_id = int(fields[0])
    origin = np.array(fields[1:4], np.float64)
    bounds = np.array(fields[4:10], np.float64)
    vertices, owned, triangles = (int(v) for v in fields[10:13])
    if vertices > MAX_TILE_VERTICES or owned > vertices:
        raise TileError(f"tile header is not self-consistent: {path.name}")

    at = _FIXED.size
    at, positions = _take(payload, at, path, np.float32, (vertices, 3))
    at, normals = _take(payload, at, path, np.float32, (owned, 3))
    colour = None
    if flags & FLAG_HAS_RGB:
        at, colour = _take(payload, at, path, np.uint8, (vertices, 3))
    at, faces = _take(payload, at, path, np.uint32, (triangles, 3))
    if at != len(payload):
        raise TileError(f"tile has {len(payload) - at} unread trailing bytes: {path.name}")
    return TilePayload(
        tile_id=tile_id, origin=origin, bounds=bounds, positions=positions,
        normals=normals, triangles=faces, rgb=colour,
    )


def write_tile_join(path: Path, join: TileJoin) -> Path:
    """The server-only evidence join for one tile, in vertex-slot order."""
    import numpy as np

    count = int(join.source_sample_id.shape[0])
    records = np.zeros(count, dtype=np.dtype(_JOIN_FIELDS))
    records["source_sample_id"] = join.source_sample_id
    records["global_vertex_index"] = join.global_vertex_index
    records["row"] = join.row
    records["owned"] = np.asarray(join.owned, bool).astype(np.uint8)
    return _write(path, KIND_TILE_JOIN, count, records.tobytes(), 0)


def read_tile_join(path: Path) -> TileJoin:
    import numpy as np

    payload, _, count = _read(path, KIND_TILE_JOIN)
    dtype = np.dtype(_JOIN_FIELDS)
    if len(payload) != count * dtype.itemsize:
        raise TileError(f"tile join payload is not {count} records: {path.name}")
    records = np.frombuffer(payload, dtype=dtype, count=count)
    return TileJoin(
        source_sample_id=records["source_sample_id"].copy(),
        global_vertex_index=records["global_vertex_index"].copy(),
        row=records["row"].copy(),
        owned=records["owned"].astype(bool),
    )


def tile_digest(path: Path) -> str:
    """The payload digest recorded in the file's own header, as hex.

    Read from the header rather than recomputed, so a manifest entry and the
    integrity check a reader performs are the same number by construction.
    """
    with open(path, "rb") as handle:
        raw = handle.read(_HEADER.size)
    if len(raw) < _HEADER.size:
        raise TileError(f"tile is truncated: {path.name}")
    return bytes(_HEADER.unpack(raw)[7]).hex()


# ---------------------------------------------------------------------------


def _write(path: Path, kind: int, count: int, payload: bytes, flags: int) -> Path:
    import hashlib
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    header = _HEADER.pack(
        TILE_MAGIC, TILE_CONTRACT_VERSION, kind, flags, _HEADER.size,
        count, len(payload), hashlib.sha256(payload).digest(),
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _read(path: Path, kind: int) -> tuple[bytes, int, int]:
    import hashlib

    raw = path.read_bytes()
    if len(raw) < _HEADER.size:
        raise TileError(f"tile file is truncated: {path.name}")
    magic, contract, got, flags, header_bytes, count, payload_bytes, digest = (
        _HEADER.unpack(raw[: _HEADER.size])
    )
    if magic != TILE_MAGIC:
        raise TileError(f"not a tile file: {path.name}")
    if contract > TILE_CONTRACT_VERSION:
        raise TileError(f"tile contract {contract} is newer than this reader")
    if got != kind or header_bytes != _HEADER.size:
        raise TileError(f"tile file is not kind {kind}: {path.name}")
    if count > MAX_TILE_VERTICES:
        raise TileError(f"tile record count {count} is beyond the guard")
    payload = raw[_HEADER.size :]
    if len(payload) != payload_bytes:
        raise TileError(
            f"tile payload is {len(payload)} bytes, header says {payload_bytes}: "
            f"{path.name}"
        )
    if hashlib.sha256(payload).digest() != digest:
        raise TileError(f"tile payload digest does not match its header: {path.name}")
    return payload, int(flags), int(count)


def _take(
    payload: bytes, at: int, path: Path, dtype: Any, shape: tuple[int, int]
) -> tuple[int, Any]:
    """One array out of the payload, with its length checked before it is read."""
    import numpy as np

    kind = np.dtype(dtype)
    span = shape[0] * shape[1] * kind.itemsize
    if at + span > len(payload):
        raise TileError(f"tile payload is short of a {shape} {kind} array: {path.name}")
    block = np.frombuffer(payload, dtype=kind, count=shape[0] * shape[1], offset=at)
    return at + span, block.reshape(shape).copy()
