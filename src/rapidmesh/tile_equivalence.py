"""
Reassembling a whole-station mesh from a written tile generation — WP-3.2.

`compare_mesh_results` reads `MeshResult.mesh`, and the tiled Pass B deliberately
no longer has one: assembling the station's mesh in memory is precisely what
ADR-006 Decision 2 removes. `PHASE1-DETERMINISM-SPEC.md` §7 still has to be
checkable, so the harness reassembles the mesh here instead, from the tiles.

**This module is for the equivalence harness and for nothing else.** It
materialises exactly the array the pipeline refuses to; the pipeline must never
import it. It is affordable in a harness, which holds two whole runs anyway, and
it is the only way to compare a tiled run against a resident one on §7's own
terms instead of on weaker ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tiles import TileStore
    from .types import MeshData



class TileReassemblyError(ValueError):
    """A generation cannot be reassembled and must not be compared."""


def reconstitute_mesh(store: TileStore) -> MeshData:
    """A whole-station `MeshData` from a written tile generation — T2 adapter.

    WP-3.2. `compare_mesh_results` reads `MeshResult.mesh`, and the tiled path
    deliberately no longer has one: assembling the station's mesh in memory is
    the thing ADR-006 Decision 2 removes. So the harness reassembles it here
    instead, from the tiles on disk.

    **This function is for the harness and for nothing else.** It materialises
    exactly the array the pipeline refuses to, and the pipeline must never call
    it. It is affordable here because a harness holds two whole runs anyway, and
    it is the only way to compare a tiled run against a resident one on SPEC §7's
    own terms rather than on weaker ones.

    Reassembly is by **ownership**, not by concatenation. Every vertex is written
    by exactly one tile, so taking each tile's owned slots and sorting by global
    vertex index reproduces `build_mesh`'s compaction order exactly — not a
    permutation of it, which T1 would reject. Boundary duplicates are skipped:
    they carry the same position bits, and no normal at all.
    """
    import numpy as np

    from .types import MeshData

    positions: list[Any] = []
    normals: list[Any] = []
    colours: list[Any] = []
    samples: list[Any] = []
    indices: list[Any] = []
    faces: list[Any] = []
    origin = store.origin
    has_rgb = True

    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        owned = payload.owned_count
        if owned:
            positions.append(payload.positions[:owned])
            normals.append(payload.normals)
            samples.append(join.source_sample_id[:owned])
            indices.append(join.global_vertex_index[:owned])
            if payload.rgb is None:
                has_rgb = False
            else:
                colours.append(payload.rgb[:owned])
        if payload.triangle_count:
            local = payload.triangles.astype(np.int64)
            faces.append(np.asarray(join.global_vertex_index, np.int64)[local])

    if not indices:
        # A station whose every component was culled. `build_mesh` returns empty
        # arrays and a `None` normals field for that case, and colour follows the
        # *source*, so the store is asked rather than the (absent) tiles.
        return MeshData(
            origin=origin, vertices=np.empty((0, 3), np.float32),
            triangles=np.empty((0, 3), np.uint32),
            rgb=np.empty((0, 3), np.uint8) if store.has_rgb else None,
            source_sample_id=np.empty(0, np.int64),
        )
    index = np.concatenate(indices)
    order = np.argsort(index, kind="stable")
    if not np.array_equal(index[order], np.arange(index.size, dtype=index.dtype)):
        raise TileReassemblyError(
            "owned vertices do not form the contiguous global index range"
        )
    triangles = (
        np.concatenate(faces).astype(np.uint32)
        if faces
        else np.empty((0, 3), np.uint32)
    )
    return MeshData(
        origin=origin,
        vertices=np.concatenate(positions)[order],
        triangles=triangles,
        normals=np.concatenate(normals)[order],
        rgb=np.concatenate(colours)[order] if has_rgb and colours else None,
        source_sample_id=np.concatenate(samples)[order],
    )
