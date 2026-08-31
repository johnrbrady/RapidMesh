"""
Observation store v0 — PLAN.md §5 item 12, WP-1.G1b Part B2.

Phase 5's surveyor/admin QC comparison needs the **retained observations** of a
station: the samples that survived filtering, with enough context to be compared
against a model without going back to the E57. `docs/adr/ADR-007` is explicit
that the comparison engine must never be fed a display mesh, so the evidence it
consumes has to be the observations themselves, persisted at the moment the
pipeline still knows which ones they are.

That moment is Pass B's finalisation. The retained set is already assembled
there to emit the mesh, and the tile generation is already being written, so the
store costs one more sequential write and **no second read of the source**.

Server-only, structurally
-------------------------
Every record carries `source_sample_id`, which DEC-004 says never leaves the
server. The store is therefore written **outside** the tile payload, in its own
file under the generation, exactly as `tile_io`'s `.rmtjoin` join is: a reader
handed the client-eligible `.rmtile` bytes cannot reach any of this. The
boundary is the file split, not a runtime check.

What v0 stores, and the redundancy in it
-----------------------------------------
Per retained observation: the stable sample id, its lattice cell, the spherical
triple the scanner actually measured, the project-axis position, and a
disposition byte.

Position is derivable from the spherical triple and the pose, and the spherical
angles are derivable from the lattice cell and `LatticeInfo`. Both are stored
anyway, and that is a deliberate trade rather than an oversight: the point of an
observation store is to be **self-contained evidence**, readable by a later
comparison stage that should not have to re-derive geometry from a pose it might
apply differently. ADR-006 Decision 1 excludes written output from the working
budget, so the cost is disk. `describe()` reports the byte count and the evidence
envelope records it, because a by-product nobody sizes is a by-product nobody
notices growing.

**v0 persists the retained set only.** Every record's disposition is `RETAINED`.
The excluded dispositions stay in `FilterStats`, which is the ledger that must
balance; this store's contract with it is a count, asserted in
`tests/test_obs_store.py`. Carrying the excluded samples too would be a
different and much larger artefact, and PLAN §5 item 12 does not ask for it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .types import ScanPose, StructuredScan

    F32 = npt.NDArray[np.float32]
    I32 = npt.NDArray[np.int32]
    I64 = npt.NDArray[np.int64]
    U8 = npt.NDArray[np.uint8]

OBS_MAGIC = b"RMOBS_V0"
OBS_CONTRACT_VERSION = 0

# Dispositions. v0 writes only RETAINED; the rest are reserved so a later
# version can widen the store without renumbering what is already on disk.
DISP_RETAINED = 1
DISP_DROPPED_DESPECKLE = 2
DISP_DROPPED_MOVER_CARVE = 3
DISP_DROPPED_ISLAND = 4
DISP_DROPPED_OTHER = 5

# magic, contract, flags, header bytes, record count, payload bytes, digest
_HEADER = struct.Struct("<8sIHHQQ32s")
# origin, then the lattice geometry needed to re-derive angles independently
_FIXED = struct.Struct("<3d4dII")

_FIELDS = [
    ("sample_id", "<i8"),
    ("row", "<i4"), ("col", "<i4"),
    ("range", "<f4"), ("azimuth", "<f4"), ("elevation", "<f4"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("disposition", "u1"), ("pad", "u1", (3,)),
]

# Guard checked before any allocation sized from a header.
MAX_OBSERVATIONS = 1 << 32


class ObservationStoreError(ValueError):
    """A store file is unusable and must not be trusted.

    Raised rather than repaired, for `segments_io`'s reason: a short read that
    silently yields fewer observations would drop survey evidence and report
    success.
    """


@dataclass(frozen=True)
class ObservationStore:
    """One station's retained observations, as written and as read back."""

    origin: Any                 # (3,) float64 station anchor
    az0: float
    el0: float
    az_step: float
    el_step: float
    rows: int
    cols: int
    records: Any                # structured array, `_FIELDS`

    @property
    def count(self) -> int:
        return int(self.records.shape[0])

    @property
    def record_bytes(self) -> int:
        return int(self.records.dtype.itemsize)

    def describe(self) -> str:
        return (
            f"observation-store-v0 {self.count:,} observations, "
            f"{self.count * self.record_bytes:,} B payload, "
            f"{self.record_bytes} B/record"
        )


def dtype() -> Any:
    import numpy as np

    return np.dtype(_FIELDS)


def build_records(
    retained: StructuredScan, keep: Any, pose: ScanPose
) -> Any:
    """The retained observations of one station, in lattice order.

    `keep` is the post-cull membership mask over `retained` — the same `final`
    mask Pass B uses for the ledger and for QA, so the store and the ledger
    cannot disagree about what "retained" means.

    Angles come from the lattice rather than from `arctan2` over the positions:
    the lattice is what the scanner actually sampled, and re-deriving angles
    from a narrowed float32 position would hand a later stage a slightly
    different direction than the one that was measured.
    """
    import numpy as np

    rows = np.asarray(retained.row, np.int64)[keep]
    cols = np.asarray(retained.col, np.int64)[keep]
    lattice = retained.lattice
    out = np.zeros(rows.shape[0], dtype=dtype())
    out["sample_id"] = (
        np.flatnonzero(keep).astype(np.int64)
        if retained.sample_id is None
        else np.asarray(retained.sample_id, np.int64)[keep]
    )
    out["row"] = rows.astype(np.int32)
    out["col"] = cols.astype(np.int32)
    out["range"] = np.asarray(retained.rng, np.float32)[keep]
    out["azimuth"] = (lattice.az0 + cols * lattice.az_step).astype(np.float32)
    out["elevation"] = (lattice.el0 + rows * lattice.el_step).astype(np.float32)
    project = pose.rotate_local(retained.xyz[keep]).astype(np.float32)
    for axis, name in enumerate(("x", "y", "z")):
        out[name] = project[:, axis]
    out["disposition"] = DISP_RETAINED
    return out


def write_store(
    path: Path, retained: StructuredScan, keep: Any, pose: ScanPose
) -> int:
    """Write the store and return its byte size on disk.

    Temp-then-`os.replace` inside the destination directory, as every other
    RapidMesh artefact is written: atomic within a volume on Windows and POSIX,
    and `os.rename` over an existing file raises on Windows.
    """
    import hashlib
    import os

    import numpy as np

    records = build_records(retained, keep, pose)
    lattice = retained.lattice
    payload = b"".join((
        _FIXED.pack(
            *np.asarray(pose.translation, np.float64).tolist(),
            float(lattice.az0), float(lattice.el0),
            float(lattice.az_step), float(lattice.el_step),
            int(lattice.rows), int(lattice.cols),
        ),
        np.ascontiguousarray(records).tobytes(),
    ))
    header = _HEADER.pack(
        OBS_MAGIC, OBS_CONTRACT_VERSION, 0, _HEADER.size,
        int(records.shape[0]), len(payload), hashlib.sha256(payload).digest(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return int(path.stat().st_size)


def read_store(path: Path) -> ObservationStore:
    """Validate the header, verify the digest, rebuild the records."""
    import hashlib

    import numpy as np

    raw = path.read_bytes()
    if len(raw) < _HEADER.size:
        raise ObservationStoreError(f"observation store is truncated: {path.name}")
    magic, contract, _flags, header_bytes, count, payload_bytes, digest = (
        _HEADER.unpack(raw[: _HEADER.size])
    )
    if magic != OBS_MAGIC:
        raise ObservationStoreError(f"not an observation store: {path.name}")
    if contract > OBS_CONTRACT_VERSION:
        raise ObservationStoreError(
            f"observation store contract {contract} is newer than this reader"
        )
    if header_bytes != _HEADER.size or count > MAX_OBSERVATIONS:
        raise ObservationStoreError(f"store header is not self-consistent: {path.name}")
    payload = raw[_HEADER.size :]
    if len(payload) != payload_bytes:
        raise ObservationStoreError(
            f"store payload is {len(payload)} bytes, header says {payload_bytes}"
        )
    if hashlib.sha256(payload).digest() != digest:
        raise ObservationStoreError(
            f"store payload digest does not match its header: {path.name}"
        )
    if len(payload) < _FIXED.size:
        raise ObservationStoreError("store is truncated before its fixed block")

    fields = _FIXED.unpack(payload[: _FIXED.size])
    kind = dtype()
    body = payload[_FIXED.size :]
    if len(body) != count * kind.itemsize:
        raise ObservationStoreError(
            f"store body is not {count} records: {path.name}"
        )
    return ObservationStore(
        origin=np.array(fields[0:3], np.float64),
        az0=float(fields[3]), el0=float(fields[4]),
        az_step=float(fields[5]), el_step=float(fields[6]),
        rows=int(fields[7]), cols=int(fields[8]),
        records=np.frombuffer(body, dtype=kind, count=count).copy(),
    )
