"""Reading a glTF 2.0 tier — the other half of `container_gltf.py` — WP-4.1.

Split at the write/read boundary the way `carvegrid` and `carvegrid_io` are, and
for a reason beyond the line count: a service that only *writes* tiers has no
business importing a reader, and the audience question the capability gate turns
on is answered entirely on this side.

Two readers, and the difference between them is load-bearing for the benchmark.
`read_gltf_tile` opens the file, parses the JSON chunk and returns one tile — the
right shape for a one-off, and the wrong model of a viewer. `GltfTierReader`
opens once and then streams tiles, which is what a viewer does; timing the first
against a bespoke container's fixed-stride table would charge the standard route
a JSON parse per tile it would never pay in service.

Neither ever holds the tier (ITEM-028). Both read by byte range.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .container import (
    CLIENT_VERTEX_ATTRIBUTES,
    ContainerError,
    TierAudience,
    TierHeader,
    TileEntry,
    TileGeometry,
)
from .container_gltf import (
    _CHUNK_HEADER,
    _GLB_HEADER,
    CHUNK_JSON,
    GLB_MAGIC,
    GLB_VERSION,
    RM_EXTENSION,
)

if TYPE_CHECKING:
    from pathlib import Path


def _read_json(path: Path) -> tuple[dict[str, Any], int, int, int]:
    """The JSON chunk, and where the BIN chunk begins. Geometry is not touched."""
    with path.open("rb") as handle:
        head = handle.read(_GLB_HEADER.size)
        if len(head) != _GLB_HEADER.size:
            raise ContainerError(f"{path.name} is too short to be a GLB")
        magic, version, total = _GLB_HEADER.unpack(head)
        if magic != GLB_MAGIC or version != GLB_VERSION:
            raise ContainerError(f"{path.name} is not a glTF 2.0 binary container")
        chunk = handle.read(_CHUNK_HEADER.size)
        if len(chunk) != _CHUNK_HEADER.size:
            raise ContainerError(f"{path.name} carries no JSON chunk")
        json_len, json_type = _CHUNK_HEADER.unpack(chunk)
        if json_type != CHUNK_JSON:
            raise ContainerError(f"{path.name}: first chunk is not JSON")
        blob = handle.read(json_len)
        if len(blob) != json_len:
            raise ContainerError(f"{path.name}: JSON chunk is short")
        bin_at = _GLB_HEADER.size + _CHUNK_HEADER.size * 2 + json_len
    document: dict[str, Any] = json.loads(blob)
    return document, bin_at, bin_at, int(total)


def read_gltf_header(path: Path) -> TierHeader:
    """Who this tier is for, read from the front of the file and nowhere else.

    The file's *name* is not consulted. `header_bytes` reports exactly what was
    read: the GLB header, both chunk headers and the JSON chunk. The BIN chunk —
    which is all of the geometry and nearly all of the file — is never opened.
    """
    document, _, read, total = _read_json(path)
    extension = document.get("extensions", {}).get(RM_EXTENSION)
    if extension is None:
        raise ContainerError(f"{path.name} declares no {RM_EXTENSION} tier block")
    if RM_EXTENSION not in document.get("extensionsRequired", []):
        raise ContainerError(
            f"{path.name} lists {RM_EXTENSION} as optional: a loader may ignore the "
            "audience and DEC-004 would then rest on the loader's goodwill"
        )
    if "audience" not in extension:
        raise ContainerError(f"{path.name} carries no audience: the tier is unattributed")
    return TierHeader(
        contract=str(extension.get("contract", "")),
        audience=TierAudience(str(extension["audience"])),
        lod_level=int(extension["lodLevel"]),
        lod_count=int(extension["lodCount"]),
        error_budget_m=float(extension["errorBudgetMetres"]),
        tile_count=int(extension["tileCount"]),
        header_bytes=read,
        total_bytes=total,
    )


def read_gltf_entries(path: Path) -> list[TileEntry]:
    """Every tile's extent and its guaranteed plane bound, without any geometry."""
    document, bin_at, _, _ = _read_json(path)
    extension = document["extensions"][RM_EXTENSION]
    views = document["bufferViews"]
    accessors = document["accessors"]
    out: list[TileEntry] = []
    for entry in extension["tiles"]:
        if "planeDeviationBoundMetres" not in entry:
            raise ContainerError(
                f"{path.name}: tile {entry['tileId']} carries no error bound, so its "
                "LOD cannot be judged against the budget it claims"
            )
        primitive = document["meshes"][int(entry["mesh"])]["primitives"][0]
        position_view = views[accessors[primitive["attributes"]["POSITION"]]["bufferView"]]
        index_view = views[accessors[primitive["indices"]]["bufferView"]]
        start = bin_at + int(position_view["byteOffset"])
        end = bin_at + int(index_view["byteOffset"]) + int(index_view["byteLength"])
        out.append(
            TileEntry(
                tile_id=int(entry["tileId"]),
                vertex_count=int(entry["vertexCount"]),
                triangle_count=int(entry["triangleCount"]),
                plane_deviation_bound_m=float(entry["planeDeviationBoundMetres"]),
                byte_offset=start,
                byte_length=end - start,
            )
        )
    return out


def gltf_vertex_attributes(path: Path) -> tuple[str, ...]:
    """Every per-vertex attribute name any primitive declares, deduplicated."""
    document, _, _, _ = _read_json(path)
    names: set[str] = set()
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            names.update(primitive["attributes"])
    return tuple(sorted(names))


def read_gltf_tile(path: Path, index: int) -> TileGeometry:
    """One tile, read by byte range. The rest of the tier is never touched.

    This is the ITEM-028 property in code: the peak cost of reading a tier is
    one tile, whatever the tier weighs.
    """
    import numpy as np

    document, bin_at, _, _ = _read_json(path)
    entry = document["extensions"][RM_EXTENSION]["tiles"][index]
    primitive = document["meshes"][int(entry["mesh"])]["primitives"][0]
    accessors = document["accessors"]
    views = document["bufferViews"]
    position_view = views[accessors[primitive["attributes"]["POSITION"]]["bufferView"]]
    index_view = views[accessors[primitive["indices"]]["bufferView"]]

    with path.open("rb") as handle:
        handle.seek(bin_at + int(position_view["byteOffset"]))
        position_blob = handle.read(int(position_view["byteLength"]))
        handle.seek(bin_at + int(index_view["byteOffset"]))
        index_blob = handle.read(int(index_view["byteLength"]))

    return TileGeometry(
        tile_id=int(entry["tileId"]),
        positions=np.frombuffer(position_blob, np.float32).reshape(-1, 3),
        triangles=np.frombuffer(index_blob, np.uint32).reshape(-1, 3),
        plane_deviation_bound_m=float(entry.get("planeDeviationBoundMetres", -1.0)),
        source_vertex_id=None,
    )


class GltfTierReader:
    """A tier opened once, then read one tile at a time — what a client does.

    `read_gltf_tile` re-parses the JSON chunk on every call, which is correct for
    a single read and wrong as a model of a viewer: a viewer opens the tier,
    keeps the directory and then streams tiles. Timing the one-shot function
    would charge the standard route a JSON parse per tile it would never pay,
    and would report a property of the harness as a property of the format.

    Opening costs the directory only. The BIN chunk is still never held.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._document, self._bin_at, self._read, self._total = _read_json(path)
        self._entries = self._document["extensions"][RM_EXTENSION]["tiles"]
        self._handle = path.open("rb")

    def __enter__(self) -> GltfTierReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()

    @property
    def tile_count(self) -> int:
        return len(self._entries)

    @property
    def header_bytes(self) -> int:
        return self._read

    def tile(self, index: int) -> TileGeometry:
        import numpy as np

        entry = self._entries[index]
        primitive = self._document["meshes"][int(entry["mesh"])]["primitives"][0]
        accessors = self._document["accessors"]
        views = self._document["bufferViews"]
        position_view = views[accessors[primitive["attributes"]["POSITION"]]["bufferView"]]
        index_view = views[accessors[primitive["indices"]]["bufferView"]]
        self._handle.seek(self._bin_at + int(position_view["byteOffset"]))
        position_blob = self._handle.read(int(position_view["byteLength"]))
        self._handle.seek(self._bin_at + int(index_view["byteOffset"]))
        index_blob = self._handle.read(int(index_view["byteLength"]))
        return TileGeometry(
            tile_id=int(entry["tileId"]),
            positions=np.frombuffer(position_blob, np.float32).reshape(-1, 3),
            triangles=np.frombuffer(index_blob, np.uint32).reshape(-1, 3),
            plane_deviation_bound_m=float(entry.get("planeDeviationBoundMetres", -1.0)),
            source_vertex_id=None,
        )


def gltf_client_attributes_are_clean(path: Path) -> bool:
    """True when no primitive declares a per-vertex attribute a client may not see."""
    return not set(gltf_vertex_attributes(path)) - set(CLIENT_VERTEX_ATTRIBUTES)
