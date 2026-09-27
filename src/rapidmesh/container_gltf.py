"""DEC-005 candidate 1 — the standard route: glTF 2.0, binary (GLB) — WP-4.1.

One GLB per **tier**, where a tier is one station at one LOD level. Every tile
is a separate mesh with its own primitive, its own two buffer views and its own
byte range, so a client fetches one tile with one ranged request and a reader
never has to hold the tier (ITEM-028). The tier is one *file* rather than sixty
because the capability gate is about a reader answering "who is this for?" from
one place, and sixty files is sixty chances to answer differently.

Why there is no compression extension here, stated plainly
-----------------------------------------------------------
DEC-005 names "glTF + a meshopt-class compression extension, decoders
self-hosted". `EXT_meshopt_compression` is not implemented in this module, and
the reason is not that it was hard to fit in:

* the extension is a specific bitstream, and an encoder for it is only worth
  anything if a *reference decoder* reads what it writes. There is no reference
  meshopt decoder in this environment to check against, and an encoder verified
  only against its own decoder is a private format wearing a standard's name;
* "decoders self-hosted" requires the decoder artefact — `meshopt_decoder.js` —
  to be in hand. It is not, and obtaining it is a third-party fetch and a
  dependency adoption, which is a decision and not an implementation detail.

So this candidate is written **stored**: core glTF 2.0 accessors, which every
conforming loader reads with no decoder to host at all, and transport
compression left to the one codec a browser already has natively. The report
carries what that omission costs and where it could flatter the other candidate.

Lossless, and why that is not a limitation here
------------------------------------------------
`KHR_mesh_quantization` would shrink this file considerably and is a ratified,
decoder-free extension. It is deliberately not used: it moves vertices, and a
size number that moves geometry is a fidelity claim in disguise. Both candidates
are lossless so that a size difference is a difference of container. What
quantisation would buy is a separate question for whoever writes the ADR.

Structural, in the glTF sense
------------------------------
`RM_tier` is listed in **`extensionsRequired`**, not merely `extensionsUsed`.
That is the whole of the DEC-004 teeth in this candidate: the glTF 2.0
specification requires a loader that does not understand a *required* extension
to refuse the asset. A tier is therefore unreadable by a client that has not
acknowledged which audience it is for.

The cost is real and is reported rather than buried: a stock loader will not
open these files. Listing the extension as merely *used* would restore that, and
would also let a stock loader open a QC tier and ignore the marking — which is
the exact failure DEC-004 exists to prevent.

Reading lives next door in `container_gltf_read.py`, split at the write/read
boundary the way `carvegrid` and `carvegrid_io` are (DEC-010).

Writing costs one tile, not one tier
-------------------------------------
`tiles` is consumed one at a time and the payload is streamed to a scratch file
beside the target. The scratch pass is a cost of **both** candidates and not of
this one: a GLB's JSON chunk and a bespoke container's table of contents are
each a variable-length directory that precedes the payload it describes, and
neither length is known until the last tile has been seen.
"""

from __future__ import annotations

import json
import shutil
import struct
from typing import TYPE_CHECKING, Any

from .container import (
    AUDIENCE_CODE,
    CONTAINER_CONTRACT,
    DEFECTS_NONE,
    ByteLedger,
    Defects,
    TierAudience,
    TierDescriptor,
    TileGeometry,
    check_tier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

GLB_MAGIC = 0x46546C67          # "glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A         # "JSON"
CHUNK_BIN = 0x004E4942          # "BIN\0"

#: The RapidMesh glTF extension. The manifest is RapidMesh's regardless of which
#: container wins (DEC-005's final sentence), so this name belongs to the
#: manifest question and not to the container question.
RM_EXTENSION = "RM_tier"

COMPONENT_FLOAT = 5126
COMPONENT_UNSIGNED_INT = 5125
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963
MODE_TRIANGLES = 4

_GLB_HEADER = struct.Struct("<III")
_CHUNK_HEADER = struct.Struct("<II")


def _pad4(length: int) -> int:
    return (4 - length % 4) % 4


class _Payload:
    """The BIN chunk under construction, 4-aligned, on disk rather than in memory."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.at = 0
        self.positions = 0
        self.indices = 0
        self.identity = 0

    def emit(self, blob: bytes, kind: str) -> int:
        pad = b"\x00" * _pad4(self.at)
        self._handle.write(pad)
        self.at += len(pad)
        start = self.at
        self._handle.write(blob)
        self.at += len(blob)
        setattr(self, kind, getattr(self, kind) + len(blob))
        return start

    def finish(self) -> int:
        padding = _pad4(self.at)
        self._handle.write(b"\x00" * padding)
        self.at += padding
        return self.at


def write_gltf_tier(
    path: Path,
    tier: TierDescriptor,
    tiles: Iterable[TileGeometry],
    *,
    defects: Defects = DEFECTS_NONE,
) -> ByteLedger:
    """Write one tier as a GLB. Returns the byte ledger, which must balance.

    `defects` is for sensitivity tests only; nothing in the pipeline sets a
    field on it. Each flag breaks exactly one capability rule so that a red
    check is evidence about that rule (standing rule 8).
    """
    import numpy as np

    check_tier(tier, [], defects)
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    scratch = path.with_name(path.name + ".binpart")
    try:
        with scratch.open("wb") as handle:
            payload = _Payload(handle)
            for tile in tiles:
                check_tier(tier, [tile], defects)
                _emit_tile(tile, payload, buffer_views, accessors, meshes, entries,
                           defects, tier.audience, np)
            payload_bytes = payload.finish()

        extension = _extension(tier, entries, len(meshes), defects)
        document: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": tier.generator},
            "extensionsUsed": [RM_EXTENSION],
            "extensionsRequired": [RM_EXTENSION],
            "extensions": {RM_EXTENSION: extension},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(meshes)))}],
            "nodes": [{"mesh": i} for i in range(len(meshes))],
            "meshes": meshes,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": payload_bytes}],
        }
        json_blob = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_blob += b" " * _pad4(len(json_blob))
        total = _GLB_HEADER.size + _CHUNK_HEADER.size * 2 + len(json_blob) + payload_bytes

        with path.open("wb") as out:
            out.write(_GLB_HEADER.pack(GLB_MAGIC, GLB_VERSION, total))
            out.write(_CHUNK_HEADER.pack(len(json_blob), CHUNK_JSON))
            out.write(json_blob)
            out.write(_CHUNK_HEADER.pack(payload_bytes, CHUNK_BIN))
            with scratch.open("rb") as source:
                shutil.copyfileobj(source, out, 1 << 22)
    finally:
        scratch.unlink(missing_ok=True)

    sections: list[tuple[str, int]] = [
        ("glb-header", _GLB_HEADER.size + _CHUNK_HEADER.size * 2),
        ("json", len(json_blob)),
        ("positions", payload.positions),
        ("indices", payload.indices),
        ("padding", payload_bytes - payload.positions - payload.indices - payload.identity),
    ]
    if payload.identity:
        sections.append(("source-identity", payload.identity))
    return ByteLedger(tuple(sections), total)


def _emit_tile(
    tile: TileGeometry,
    payload: _Payload,
    buffer_views: list[dict[str, Any]],
    accessors: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    defects: Defects,
    audience: TierAudience,
    np: Any,
) -> None:
    positions = np.ascontiguousarray(tile.positions, np.float32)
    indices = np.ascontiguousarray(tile.triangles, np.uint32).reshape(-1)

    position_view = len(buffer_views)
    blob = positions.tobytes()
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": payload.emit(blob, "positions"),
            "byteLength": len(blob),
            "target": TARGET_ARRAY_BUFFER,
        }
    )
    index_view = len(buffer_views)
    blob = indices.tobytes()
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": payload.emit(blob, "indices"),
            "byteLength": len(blob),
            "target": TARGET_ELEMENT_ARRAY_BUFFER,
        }
    )

    # POSITION min/max are required by the glTF 2.0 specification. They are the
    # tile's own bounds in the station frame and are written into the container
    # only — never into a log, a summary or a report.
    low = positions.min(axis=0) if positions.size else np.zeros(3, np.float32)
    high = positions.max(axis=0) if positions.size else np.zeros(3, np.float32)
    position_accessor = len(accessors)
    accessors.append(
        {
            "bufferView": position_view,
            "componentType": COMPONENT_FLOAT,
            "count": int(positions.shape[0]),
            "type": "VEC3",
            "min": [float(v) for v in low],
            "max": [float(v) for v in high],
        }
    )
    index_accessor = len(accessors)
    accessors.append(
        {
            "bufferView": index_view,
            "componentType": COMPONENT_UNSIGNED_INT,
            "count": int(indices.shape[0]),
            "type": "SCALAR",
        }
    )

    attributes: dict[str, int] = {"POSITION": position_accessor}
    # A QC tier legitimately carries source identity — DEC-004 keeps that tier
    # server-side rather than pretending the data does not exist. The defect
    # switch forces the same bytes onto a *client* tier, which is the mutation
    # the DEC-004 sensitivity test needs.
    #
    # Worth knowing before adopting this route: **glTF 2.0 has no 64-bit integer
    # component type at all.** The identity has to go in as paired uint32 under a
    # private `_`-prefixed attribute name, which no stock loader understands.
    # That is a real limitation of the standard route on the QC tier, and it is
    # precisely why the audit searches bytes rather than attribute names.
    if tile.source_vertex_id is not None and (
        audience is TierAudience.QC or defects.leak_source_identity
    ):
        ident = np.ascontiguousarray(tile.source_vertex_id, "<i8")
        blob = ident.tobytes()
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": payload.emit(blob, "identity"),
                "byteLength": len(blob),
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": COMPONENT_UNSIGNED_INT,
                "count": int(ident.shape[0]) * 2,
                "type": "SCALAR",
            }
        )
        attributes["_SOURCE_VERTEX_ID"] = len(accessors) - 1

    meshes.append(
        {
            "primitives": [
                {"attributes": attributes, "indices": index_accessor, "mode": MODE_TRIANGLES}
            ]
        }
    )
    entry: dict[str, Any] = {
        "tileId": int(tile.tile_id),
        "mesh": len(meshes) - 1,
        "vertexCount": tile.vertex_count,
        "triangleCount": tile.triangle_count,
    }
    if not defects.omit_bounds:
        entry["planeDeviationBoundMetres"] = float(tile.plane_deviation_bound_m)
    entries.append(entry)


def _extension(
    tier: TierDescriptor, entries: list[dict[str, Any]], tile_count: int, defects: Defects
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "contract": CONTAINER_CONTRACT,
        "lodLevel": tier.lod_level,
        "lodCount": tier.lod_count,
        "errorBudgetMetres": float(tier.error_budget_m),
        "stationOrdinal": int(tier.station_ordinal),
        "tileCount": tile_count,
        # ITEM-027: the bound these entries carry is over the planes of the
        # merged triangles, not over the surface. The key says so.
        "boundIsOverPlanesNotSurface": True,
        "tiles": entries,
    }
    if defects.omit_audience:
        return body
    # First keys in the object, so a reader meets the audience at the front of
    # the JSON chunk rather than after sixty tile entries.
    return {
        "audience": str(tier.audience),
        "audienceCode": AUDIENCE_CODE[tier.audience],
        **body,
    }
