"""What a streamable container must carry, and what it must refuse — WP-4.1.

`decimate_tiles.py` writes a **prototype** generation: one `.npz` per tile plus
a manifest, and it says so in terms. This module is the contract a shipping
container has to meet instead, shared by both DEC-005 candidates so that the
benchmark compares containers and not two different ideas of what a tier is.

Nothing here chooses a container. `container_gltf.py` and `container_binary.py`
are the two candidates; this module holds what they must both satisfy.
`container_audit.py` holds the check that they did, and `container_codec.py` the
lossless byte filter either may use — split out under DEC-010 because a
contract, a search and an array transform are three responsibilities, and only
the first belongs to every caller.

The one requirement that is not about bytes — DEC-004
-----------------------------------------------------
DEC-004 says client audiences receive decimated, error-bounded LODs only, and
that per-vertex source identity **never leaves the server**. The prototype tile
carries `global_vertex_index` per vertex as int64 (ITEM-029). A container that
inherits that field has not implemented DEC-004, whatever else it does.

So `TileGeometry` holds source identity in a field that is explicitly optional
and explicitly server-side, and `check_tier` refuses to let a client tier be
written with it populated. The refusal is at the writer, not at the reader: a
byte that was never written cannot leak, and a reader-side filter would put the
field on the wire and then decline to look at it.

Structural, and what that word has to mean here
------------------------------------------------
"A reader must be able to tell a client tier from a QC tier without parsing
geometry and without trusting a filename" is a statement about **where the
answer lives**, not about whether the answer is present somewhere. Both
candidates therefore expose a header reader, which is required to answer from a
bounded prefix of the file and to report how many bytes it read. A test asserts
that prefix is small and that the file's *name* plays no part: the same bytes
renamed to anything at all still answer the same way.

The bound is over planes, not the surface — ITEM-027
-----------------------------------------------------
`plane_deviation_bound_m` is named for what Round 12 actually guarantees: the
distance from a surviving vertex to the **planes** of the triangles merged into
it. It is not a surface-to-surface bound, and at 3 of 30 measured levels a
sampled surface point moved further than it. The field carries the word `plane`
so that no reader can quote it as "the surface moved at most this".

What this module deliberately does not do
------------------------------------------
No quantisation, no lossy anything. Both candidates are lossless, so a size
difference between them is a difference of container and never of fidelity —
which is DEC-024's business and not this package's. A lossy mode would confound
the two in one number.

No tier is ever joined into a resident mesh here. ITEM-028 records that
measuring a 284 MB tier costs ~3.6 GB because `read_decimated` joins every tile;
a container reader that only works that way has failed the thing the product
exists to avoid, so every reader in this family reads **one tile at a time**.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    I64 = npt.NDArray[np.int64]
    U32 = npt.NDArray[np.uint32]


#: Bumped when the meaning of any field below changes, so a reader that predates
#: the change fails loudly rather than reading a field that has moved.
CONTAINER_CONTRACT = "rapidmesh-container-v0"

#: Per-vertex attribute names a **client** tier may carry. Anything outside this
#: set is refused at write time. It is a deliberately short allow-list rather
#: than a deny-list of known-bad names: a deny-list would pass the next field
#: somebody adds, which is exactly how `global_vertex_index` reached every tile.
CLIENT_VERTEX_ATTRIBUTES = frozenset({"POSITION"})


class ContainerError(ValueError):
    """A container is unusable, or was asked to write something it must not.

    Raised rather than repaired. A writer that quietly dropped a field it was
    handed would make the DEC-004 guarantee depend on the caller remembering to
    check the return value.
    """


class TierAudience(str, Enum):
    """Who a tier is for. DEC-004's distinction, as a value rather than a habit.

    `CLIENT` — decimated, error-bounded, no per-vertex source identity of any
    kind. `QC` — the surveyor/admin tier, which may carry source identity and
    never leaves the server.
    """

    CLIENT = "client"
    QC = "qc"

    def __str__(self) -> str:
        return self.value


#: The byte a bespoke header stores, and the value a glTF extension records.
#: Fixed here so the two candidates cannot drift apart on the one question the
#: capability gate turns on.
AUDIENCE_CODE: dict[TierAudience, int] = {TierAudience.CLIENT: 1, TierAudience.QC: 2}
CODE_AUDIENCE: dict[int, TierAudience] = {v: k for k, v in AUDIENCE_CODE.items()}


@dataclass(frozen=True)
class TileGeometry:
    """One decimated tile, as a container sees it.

    `positions` are station-local float32 in the station's own anchored frame —
    a container never introduces a translation, rotation or scale of its own
    (`SPATIAL-CONTRACT.md` §2.4).

    `source_vertex_id` is per-vertex source identity. It is `None` for a client
    tier and that is not a convention: `check_tier` refuses to write a client
    tier whose tiles populate it.
    """

    tile_id: int
    positions: F32                     # (V,3) float32
    triangles: U32                     # (T,3) uint32, tile-local
    plane_deviation_bound_m: float     # ITEM-027 — over planes, not the surface
    source_vertex_id: I64 | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class TierDescriptor:
    """What the tier is, independent of which container carries it.

    `station_ordinal` is an ordinal within the campaign set and never a
    filename, scan name or client identifier: the container is written from real
    data and a written identifier would travel with it.
    """

    audience: TierAudience
    lod_level: int
    lod_count: int
    error_budget_m: float
    station_ordinal: int
    generator: str = CONTAINER_CONTRACT


@dataclass(frozen=True)
class TierHeader:
    """What a reader learned from the front of the file, and how far it read.

    `header_bytes` is the load-bearing field. A container that answered the
    audience question only after decoding geometry would report a number close
    to the file size, and the capability test would fail on it.
    """

    contract: str
    audience: TierAudience
    lod_level: int
    lod_count: int
    error_budget_m: float
    tile_count: int
    header_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class TileEntry:
    """Where one tile sits, and what it guarantees, without reading the tile."""

    tile_id: int
    vertex_count: int
    triangle_count: int
    plane_deviation_bound_m: float
    byte_offset: int
    byte_length: int


@dataclass(frozen=True)
class ByteLedger:
    """Every byte of a written container, attributed to exactly one section.

    The point is not accounting for its own sake. `sections` summing to
    `total_bytes` is what makes "there is no per-vertex source identity in this
    file" checkable as an *exhaustive* statement rather than as a search that
    might have missed a place to look: if positions, indices and metadata
    account for every byte, there is nowhere else for an identity array to be.
    """

    sections: tuple[tuple[str, int], ...]
    total_bytes: int

    @property
    def balanced(self) -> bool:
        return sum(n for _, n in self.sections) == self.total_bytes

    def named(self, name: str) -> int:
        return sum(n for key, n in self.sections if key == name)


@dataclass(frozen=True)
class Defects:
    """Deliberate breakages, for sensitivity tests only.

    Nothing in the pipeline may construct this with a field set true. It exists
    because standing rule 8 requires each capability rule to be proven red
    **separately** — a writer broken in one way at a time, so that a green check
    is evidence about the rule it names rather than about the file as a whole.
    """

    leak_source_identity: bool = False   # rule: no per-vertex source identity
    omit_audience: bool = False          # rule: the tier distinction is structural
    omit_bounds: bool = False            # rule: per-tile error bounds are carried

    @property
    def any_set(self) -> bool:
        return self.leak_source_identity or self.omit_audience or self.omit_bounds


DEFECTS_NONE = Defects()


def check_tier(
    tier: TierDescriptor, tiles: Sequence[TileGeometry], defects: Defects = DEFECTS_NONE
) -> None:
    """Refuse, at the writer, anything DEC-004 or the LOD contract forbids.

    Raises `ContainerError`. Called by both candidates before a byte is written,
    so the guarantee is a property of the family and not of one implementation.
    """
    if tier.lod_count < 1 or not 0 <= tier.lod_level < tier.lod_count:
        raise ContainerError(
            f"lod_level {tier.lod_level} is not inside a chain of {tier.lod_count}"
        )
    if not tier.error_budget_m > 0.0:
        raise ContainerError("a tier carries the error budget it was built to, in metres")
    for tile in tiles:
        if tile.triangle_count and int(tile.triangles.max()) >= tile.vertex_count:
            raise ContainerError(f"tile {tile.tile_id} indexes past its own vertices")
        if not defects.omit_bounds and not tile.plane_deviation_bound_m >= 0.0:
            raise ContainerError(f"tile {tile.tile_id} carries no plane deviation bound")
        if (
            tier.audience is TierAudience.CLIENT
            and tile.source_vertex_id is not None
            and not defects.leak_source_identity
        ):
            raise ContainerError(
                f"tile {tile.tile_id}: a client tier carries no per-vertex source "
                "identity (DEC-004); strip it before writing, do not filter it on read"
            )


def strip_for_client(tile: TileGeometry) -> TileGeometry:
    """The same geometry with source identity removed rather than hidden.

    This is the whole of ITEM-029's fix from the container's side, and it is one
    line because the container never had a reason to carry the field — the
    prototype wrote it so a measurement harness could rejoin tiles.
    """
    if tile.source_vertex_id is None:
        return tile
    return TileGeometry(
        tile_id=tile.tile_id,
        positions=tile.positions,
        triangles=tile.triangles,
        plane_deviation_bound_m=tile.plane_deviation_bound_m,
        source_vertex_id=None,
    )
