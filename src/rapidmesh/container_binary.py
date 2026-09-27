"""DEC-005 candidate 2 — a bespoke binary container for this payload — WP-4.1.

Deliberately unnamed. DEC-005's final sentence retires the assumption that the
container is called "RMX", and naming a candidate before it is chosen is how a
benchmark becomes a formality. The magic bytes say what it is and nothing about
what it would be called if adopted.

The shape, and the one thing it is designed around
----------------------------------------------------
A fixed 96-byte header, a fixed-stride table of contents, then the tile blocks::

    header   magic, contract, audience, codec, lod, budget, counts, digest
    toc      one 96-byte entry per tile: extent, plane bound, three byte ranges
    blocks   positions, indices, and — QC tiers only — source identity

Two properties fall out of that layout and they are the candidate's whole case.

**The audience is one byte at a fixed offset.** A reader learns who a tier is
for from 11 bytes, and that number does not move when the tier has sixty tiles
instead of six. The standard candidate must parse its entire JSON chunk to
answer the same question, and that chunk grows with the tile count.

**Every block starts on a 4-byte boundary, and that is not housekeeping.** A
browser can only build a `Float32Array` over a buffer at an aligned offset; an
unaligned block forces a copy of every tile before anything can be drawn with
it. The header is padded to 96 bytes and each block to 4 for exactly that
reason. glTF mandates the same alignment, and a bespoke format has to
rediscover it — which is one concrete item on the standing-cost side of the
ledger rather than an abstract one.

**The table of contents is fixed-stride.** Tile *i*'s byte ranges are at a
computed offset, so a client that wants one tile issues one ranged read for the
entry and one for the block, with no parse in between (ITEM-028).

The identity fields exist, and that is the point
-------------------------------------------------
Every table entry reserves an offset/length pair for per-vertex source identity,
because the **QC tier legitimately carries it** — DEC-004 keeps that tier
server-side rather than pretending the data does not exist. On a client tier the
pair is zero *and the bytes are absent*, which is a stronger statement than a
format with nowhere to put them: the audit can show the field had a place to go
and still is not there.

Codec — the same lossless filter, owned by neither candidate
--------------------------------------------------------------
`CODEC_FILTERED` is `container.byte_shuffle` plus row deltas on indices, then
DEFLATE. It is meshopt-*class* in the sense that matters for bytes — separate
the byte planes so a generic compressor sees structure — and it is not the
meshopt bitstream and is not claimed to be.

It is implemented here rather than in the glTF candidate because glTF has no
standard way to carry it, which is exactly what `EXT_meshopt_compression`
exists for. **The gain it produces is a codec gain and not a container gain**,
and the report is required to say so: scoring it as an advantage of this
candidate would be measuring the absence of an extension the other candidate was
never given.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import zlib
from typing import TYPE_CHECKING, Any

from .container import (
    AUDIENCE_CODE,
    CODE_AUDIENCE,
    CONTAINER_CONTRACT,
    DEFECTS_NONE,
    ByteLedger,
    ContainerError,
    Defects,
    TierAudience,
    TierDescriptor,
    TierHeader,
    TileEntry,
    TileGeometry,
    check_tier,
)
from .container_codec import byte_shuffle, byte_unshuffle, delta_rows, undelta_rows

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

CONTAINER_MAGIC = b"RMCONTNR"
CONTAINER_VERSION = 0

CODEC_STORED = 0
CODEC_FILTERED = 1

#: magic, contract, audience, codec, lod level, lod count, flags,
#: station ordinal, tile count, error budget, toc offset, toc bytes,
#: total bytes, toc digest, and six reserved bytes that exist only to make the
#: header a multiple of 4 so every block after it can be viewed without a copy.
_HEADER = struct.Struct("<8sHBBHHHIIdQQQ32s6x")
#: tile id, vertices, triangles, plane bound, then three (offset, stored, raw)
#: triples: positions, indices, source identity.
_TOC = struct.Struct("<QIId9Q")

#: The audience byte's offset, stated as a constant because a test asserts it and
#: because the candidate's case rests on it being a constant.
AUDIENCE_OFFSET = 10


def _write_pad(handle: Any, at: int) -> int:
    """Zero bytes up to the next 4-byte boundary. Returns how many were written."""
    pad = (4 - at % 4) % 4
    if pad:
        handle.write(b"\x00" * pad)
    return pad


def _encode(array: Any, codec: int, *, indices: bool) -> tuple[bytes, int]:
    """One array to bytes under the chosen codec. Returns (stored, raw length)."""
    import numpy as np

    block = np.ascontiguousarray(array)
    raw = block.nbytes
    if codec == CODEC_STORED:
        return block.tobytes(), raw
    source = delta_rows(block) if indices else block
    return zlib.compress(byte_shuffle(source), 6), raw


def _decode(blob: bytes, codec: int, dtype: Any, shape: tuple[int, ...], *, indices: bool) -> Any:
    import numpy as np

    if codec == CODEC_STORED:
        count = 1
        for dim in shape:
            count *= int(dim)
        return np.frombuffer(blob, np.dtype(dtype), count=count).reshape(shape)
    block = byte_unshuffle(zlib.decompress(blob), dtype, shape)
    return undelta_rows(block) if indices else block


def write_binary_tier(
    path: Path,
    tier: TierDescriptor,
    tiles: Iterable[TileGeometry],
    *,
    codec: int = CODEC_STORED,
    defects: Defects = DEFECTS_NONE,
) -> ByteLedger:
    """Write one tier as a bespoke binary container. Returns its byte ledger.

    `tiles` is consumed one at a time and the blocks are streamed to a scratch
    file, so writing costs one tile and not one tier. The scratch pass is the
    same cost the standard candidate pays and for the same reason: the table of
    contents precedes the payload and its length is unknown until the last tile.

    `defects` is for sensitivity tests only; nothing in the pipeline sets a field
    on it. Each flag breaks exactly one capability rule (standing rule 8).
    """
    if codec not in (CODEC_STORED, CODEC_FILTERED):
        raise ContainerError(f"unknown codec {codec}")
    check_tier(tier, [], defects)

    rows: list[tuple[Any, ...]] = []
    at = 0
    position_bytes = 0
    index_bytes = 0
    identity_bytes = 0
    scratch = path.with_name(path.name + ".blockpart")

    try:
        with scratch.open("wb") as blocks:
            for tile in tiles:
                check_tier(tier, [tile], defects)
                position_blob, position_raw = _encode(
                    tile.positions.astype("float32", copy=False), codec, indices=False
                )
                index_blob, index_raw = _encode(
                    tile.triangles.astype("uint32", copy=False), codec, indices=True
                )
                at += _write_pad(blocks, at)
                position_at = at
                blocks.write(position_blob)
                at += len(position_blob)
                at += _write_pad(blocks, at)
                index_at = at
                blocks.write(index_blob)
                at += len(index_blob)
                position_bytes += len(position_blob)
                index_bytes += len(index_blob)

                identity_at = identity_len = identity_raw = 0
                write_identity = tile.source_vertex_id is not None and (
                    tier.audience is TierAudience.QC or defects.leak_source_identity
                )
                if write_identity and tile.source_vertex_id is not None:
                    identity_blob, identity_raw = _encode(
                        tile.source_vertex_id.astype("int64", copy=False), codec, indices=False
                    )
                    at += _write_pad(blocks, at)
                    identity_at = at
                    blocks.write(identity_blob)
                    at += len(identity_blob)
                    identity_len = len(identity_blob)
                    identity_bytes += identity_len

                rows.append(
                    (
                        int(tile.tile_id),
                        tile.vertex_count,
                        tile.triangle_count,
                        -1.0 if defects.omit_bounds else float(tile.plane_deviation_bound_m),
                        position_at,
                        len(position_blob),
                        position_raw,
                        index_at,
                        len(index_blob),
                        index_raw,
                        identity_at,
                        identity_len,
                        identity_raw,
                    )
                )

        # Block offsets were recorded relative to the payload; the table of
        # contents sits in front of it and its length is only known now.
        toc_bytes = _TOC.size * len(rows)
        base = _HEADER.size + toc_bytes
        toc = b"".join(
            _TOC.pack(
                row[0], row[1], row[2], row[3],
                row[4] + base, row[5], row[6],
                row[7] + base, row[8], row[9],
                (row[10] + base) if row[11] else 0, row[11], row[12],
            )
            for row in rows
        )
        header = _HEADER.pack(
            CONTAINER_MAGIC,
            CONTAINER_VERSION,
            0 if defects.omit_audience else AUDIENCE_CODE[tier.audience],
            codec,
            tier.lod_level,
            tier.lod_count,
            0,
            int(tier.station_ordinal),
            len(rows),
            float(tier.error_budget_m),
            _HEADER.size,
            toc_bytes,
            base + at,
            hashlib.sha256(toc).digest(),
        )
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(toc)
            with scratch.open("rb") as source:
                shutil.copyfileobj(source, handle, 1 << 22)
    finally:
        scratch.unlink(missing_ok=True)

    sections: list[tuple[str, int]] = [
        ("header", _HEADER.size),
        ("toc", toc_bytes),
        ("positions", position_bytes),
        ("indices", index_bytes),
        ("padding", at - position_bytes - index_bytes - identity_bytes),
    ]
    if identity_bytes:
        sections.append(("source-identity", identity_bytes))
    return ByteLedger(tuple(sections), base + at)


def _header(path: Path) -> tuple[Any, ...]:
    with path.open("rb") as handle:
        blob = handle.read(_HEADER.size)
    if len(blob) != _HEADER.size:
        raise ContainerError(f"{path.name} is too short to be a container")
    fields = _HEADER.unpack(blob)
    if fields[0] != CONTAINER_MAGIC or fields[1] != CONTAINER_VERSION:
        raise ContainerError(f"{path.name} is not a v{CONTAINER_VERSION} container")
    return fields


def read_binary_header(path: Path) -> TierHeader:
    """Who this tier is for, from a fixed-size prefix and nothing else.

    The file's name is not consulted, and neither is the table of contents: the
    audience is a single byte at `AUDIENCE_OFFSET`, so the answer costs the same
    on a sixty-tile tier as on a one-tile tier.
    """
    fields = _header(path)
    code = int(fields[2])
    if code not in CODE_AUDIENCE:
        raise ContainerError(
            f"{path.name} carries audience code {code}: the tier is unattributed and "
            "DEC-004 cannot be enforced on it"
        )
    return TierHeader(
        contract=CONTAINER_CONTRACT,
        audience=CODE_AUDIENCE[code],
        lod_level=int(fields[4]),
        lod_count=int(fields[5]),
        error_budget_m=float(fields[9]),
        tile_count=int(fields[8]),
        header_bytes=_HEADER.size,
        total_bytes=int(fields[12]),
    )


def _toc(path: Path) -> list[tuple[Any, ...]]:
    fields = _header(path)
    toc_at, toc_bytes, count = int(fields[10]), int(fields[11]), int(fields[8])
    if toc_bytes != _TOC.size * count:
        raise ContainerError(f"{path.name}: table of contents length disagrees with tile count")
    with path.open("rb") as handle:
        handle.seek(toc_at)
        blob = handle.read(toc_bytes)
    if len(blob) != toc_bytes or hashlib.sha256(blob).digest() != fields[13]:
        raise ContainerError(f"{path.name}: table of contents is short or corrupt")
    return [_TOC.unpack_from(blob, i * _TOC.size) for i in range(count)]


def read_binary_entries(path: Path) -> list[TileEntry]:
    """Every tile's extent and plane bound, from the header and table only."""
    out: list[TileEntry] = []
    for row in _toc(path):
        if not float(row[3]) >= 0.0:
            raise ContainerError(
                f"{path.name}: tile {int(row[0])} carries no error bound, so its LOD "
                "cannot be judged against the budget it claims"
            )
        start = int(row[4])
        end = int(row[7]) + int(row[8])
        if int(row[10]):
            end = max(end, int(row[10]) + int(row[11]))
        out.append(
            TileEntry(
                tile_id=int(row[0]),
                vertex_count=int(row[1]),
                triangle_count=int(row[2]),
                plane_deviation_bound_m=float(row[3]),
                byte_offset=start,
                byte_length=end - start,
            )
        )
    return out


def binary_vertex_attributes(path: Path) -> tuple[str, ...]:
    """Which per-vertex arrays this container actually stores bytes for."""
    names = {"POSITION"}
    for row in _toc(path):
        if int(row[11]):
            names.add("_SOURCE_VERTEX_ID")
    return tuple(sorted(names))


def read_binary_tile(path: Path, index: int) -> TileGeometry:
    """One tile, read by byte range. The rest of the tier is never touched."""
    fields = _header(path)
    codec = int(fields[3])
    row = _toc(path)[index]
    vertices, triangles = int(row[1]), int(row[2])
    with path.open("rb") as handle:
        handle.seek(int(row[4]))
        position_blob = handle.read(int(row[5]))
        handle.seek(int(row[7]))
        index_blob = handle.read(int(row[8]))
        identity_blob = b""
        if int(row[11]):
            handle.seek(int(row[10]))
            identity_blob = handle.read(int(row[11]))
    positions = _decode(position_blob, codec, "float32", (vertices, 3), indices=False)
    faces = _decode(index_blob, codec, "uint32", (triangles, 3), indices=True)
    identity = (
        _decode(identity_blob, codec, "int64", (vertices,), indices=False)
        if identity_blob
        else None
    )
    return TileGeometry(
        tile_id=int(row[0]),
        positions=positions,
        triangles=faces,
        plane_deviation_bound_m=float(row[3]),
        source_vertex_id=identity,
    )


class BinaryTierReader:
    """A tier opened once, then read one tile at a time — what a client does.

    The counterpart of `container_gltf.GltfTierReader`, and it exists for the
    same reason: timing the one-shot `read_binary_tile` would charge each
    candidate a directory read per tile that no viewer pays, and the two
    candidates' directories cost very different amounts, so the artefact would
    not even cancel.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        fields = _header(path)
        self._codec = int(fields[3])
        self._rows = _toc(path)
        self._handle = path.open("rb")

    def __enter__(self) -> BinaryTierReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()

    @property
    def tile_count(self) -> int:
        return len(self._rows)

    @property
    def header_bytes(self) -> int:
        return _HEADER.size

    def tile(self, index: int) -> TileGeometry:
        row = self._rows[index]
        vertices, triangles = int(row[1]), int(row[2])
        self._handle.seek(int(row[4]))
        position_blob = self._handle.read(int(row[5]))
        self._handle.seek(int(row[7]))
        index_blob = self._handle.read(int(row[8]))
        identity = None
        if int(row[11]):
            self._handle.seek(int(row[10]))
            identity = _decode(
                self._handle.read(int(row[11])), self._codec, "int64", (vertices,), indices=False
            )
        return TileGeometry(
            tile_id=int(row[0]),
            positions=_decode(
                position_blob, self._codec, "float32", (vertices, 3), indices=False
            ),
            triangles=_decode(index_blob, self._codec, "uint32", (triangles, 3), indices=True),
            plane_deviation_bound_m=float(row[3]),
            source_vertex_id=identity,
        )


def binary_stream_offsets(path: Path) -> list[int]:
    """Where every compressed block starts, for the audit to look inside them.

    A container knows where its own streams are, so the DEC-004 audit does not
    have to scan a hundred megabytes for zlib headers to find them.
    """
    if int(_header(path)[3]) == CODEC_STORED:
        return []
    offsets: list[int] = []
    for row in _toc(path):
        offsets.append(int(row[4]))
        offsets.append(int(row[7]))
        if int(row[11]):
            offsets.append(int(row[10]))
    return offsets


def audience_byte(path: Path) -> int:
    """The one byte the audience lives in, read directly. 11 bytes of file."""
    with path.open("rb") as handle:
        blob = handle.read(AUDIENCE_OFFSET + 1)
    if len(blob) != AUDIENCE_OFFSET + 1:
        raise ContainerError(f"{path.name} is too short to carry an audience")
    return blob[AUDIENCE_OFFSET]
