"""
Provisional band segments and the component table on disk —
`PHASE1-TILE-CONTRACT-V0.md` §3 (object model) and §5.1 (atomic write).

Pass A holds one band at a time and hands the rest to the filesystem. These are
the file kinds that makes that possible:

    <band>.rmseg.tri    core-owned triangles, as stable-id triples plus the
                        global component id the band assigned
    <band>.rmseg.pos    core-owned sample records — the positions, ranges,
                        colours and stable ids Pass B needs to mesh
    <band>.rmseg.disp   the band's filter dispositions, final at Pass A
    component.rmtable   completed membership and triangle counts (Pass A)
    component.rmcomp    the same, plus area, verdict and fallback (Pass B)

Two rules from the contract that this module exists to enforce
--------------------------------------------------------------
**Records key on stable ids, never on array positions.** A triangle names its
vertices by lattice cell id, `row * cols + col`. `clean` renumbers its arrays at
every stage (`grid.py:92`), so a band-local index means nothing in the next
band and nothing at all in Pass B.

**Segments are disjoint by construction, and no de-duplication step exists.**
A sample record is written by the band whose core contains its row; a triangle
record by the band whose core contains its quad's top row
(`PHASE1-HALO-CALCULUS.md` §7). Adding a de-duplication pass would hide an
ownership bug rather than fail on it, so there is none — the read side simply
concatenates in band order and expects the result to be exactly right.

Writes are temp-then-`os.replace` inside the destination directory: `os.replace`
is atomic within one volume on Windows and POSIX, and `os.rename` over an
existing file raises on Windows (§5.1). Every file carries a SHA-256 of its
payload in its header, so a truncated or half-written segment fails closed
rather than being read as a shorter one.

**Scope, stated plainly.** This is the minimum of §3 that item 8 needs. There
is no `run.json`, no `journal.jsonl`, no `rmstate` checkpoint and therefore no
resumability; `nrm`, `rgb` and `sid` are folded into the `pos` record instead of
being separate kinds. Those are §5.3's recovery spine and are not built here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from .types import StructuredScan

    I64 = npt.NDArray[np.int64]

SEGMENT_MAGIC = b"RMSEG_V0"
SEGMENT_CONTRACT_VERSION = 0

KIND_TRI = 1
KIND_POS = 2
KIND_DISP = 3
KIND_COMPONENT = 4

FLAG_HAS_RGB = 1
FLAG_FINALISED = 2

# magic, contract, kind, flags, header bytes, record count, payload bytes,
# payload digest.
_HEADER = struct.Struct("<8sIHHIQQ32s")

# Guard checked before any allocation sized from a header. 2**32 records is far
# above anything this code produces; it exists so a corrupt length field cannot
# ask for an arbitrary allocation.
MAX_RECORDS = 1 << 32

_TRI_FIELDS = [("cell0", "<i8"), ("cell1", "<i8"), ("cell2", "<i8"), ("component", "<i8")]
_POS_FIELDS = [
    ("cell", "<i8"), ("sample_id", "<i8"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rng", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"), ("pad", "u1"),
]
_DISP_FIELDS = [
    ("core_row_start", "<u4"), ("core_row_stop", "<u4"), ("retained", "<u8"),
    ("dropped_despeckle", "<u8"), ("dropped_mover_carve", "<u8"),
    ("restored_from_carve", "<u8"),
]
_COMPONENT_FIELDS = [
    ("component_id", "<i8"), ("root", "<i8"), ("triangle_count", "<u8"),
    ("area", "<f8"), ("smallest_positive_area", "<f4"),
    ("retired", "u1"), ("verdict", "u1"), ("area_fallback", "u1"), ("pad", "u1"),
]

_KIND_FIELDS = {
    KIND_TRI: _TRI_FIELDS,
    KIND_POS: _POS_FIELDS,
    KIND_DISP: _DISP_FIELDS,
    KIND_COMPONENT: _COMPONENT_FIELDS,
}


class SegmentError(ValueError):
    """A segment file is unusable and must not be trusted.

    Raised rather than repaired. A short read that silently yields fewer
    triangles would delete survey geometry and report success.
    """


@dataclass(frozen=True)
class BandSegment:
    """One band's provisional output, named by its core row range."""

    core_row_start: int
    core_row_stop: int
    tri_path: Path
    pos_path: Path
    disp_path: Path
    triangle_count: int
    sample_count: int


def _write_records(path: Path, kind: int, records: Any, flags: int = 0) -> Path:
    """Header + fixed-width payload, temp-then-replace, digest in the header."""
    import hashlib
    import os

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.ascontiguousarray(records).tobytes()
    header = _HEADER.pack(
        SEGMENT_MAGIC, SEGMENT_CONTRACT_VERSION, kind, flags, _HEADER.size,
        int(np.asarray(records).shape[0]), len(payload),
        hashlib.sha256(payload).digest(),
    )
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _read_records(path: Path, kind: int) -> tuple[Any, int]:
    """Validate the header, verify the digest, return the record array and flags."""
    import hashlib

    import numpy as np

    raw = path.read_bytes()
    if len(raw) < _HEADER.size:
        raise SegmentError(f"segment is truncated: {path.name}")
    magic, contract, got_kind, flags, header_bytes, count, payload_bytes, digest = (
        _HEADER.unpack(raw[: _HEADER.size])
    )
    if magic != SEGMENT_MAGIC:
        raise SegmentError(f"not a segment: {path.name}")
    if contract > SEGMENT_CONTRACT_VERSION:
        raise SegmentError(f"segment contract {contract} is newer than this reader")
    if got_kind != kind or header_bytes != _HEADER.size:
        raise SegmentError(f"segment header is malformed: {path.name}")
    if count > MAX_RECORDS:
        raise SegmentError(f"segment record count is out of range: {count}")

    dtype = np.dtype(_KIND_FIELDS[kind])
    if payload_bytes != count * dtype.itemsize:
        raise SegmentError(f"segment length fields disagree: {path.name}")
    payload = raw[_HEADER.size :]
    if len(payload) != payload_bytes:
        raise SegmentError(f"segment payload is truncated: {path.name}")
    if hashlib.sha256(payload).digest() != digest:
        raise SegmentError(f"segment payload digest does not match: {path.name}")
    return np.frombuffer(payload, dtype=dtype, count=int(count)), int(flags)


def write_band_segments(
    work_dir: Path,
    *,
    core_row_start: int,
    core_row_stop: int,
    tri_cells: I64,
    tri_components: I64,
    owned: StructuredScan,
    cols: int,
    dropped_despeckle: int,
    dropped_mover_carve: int,
    restored_from_carve: int,
) -> BandSegment:
    """Write one band's `tri`, `pos` and `disp` segments."""
    import numpy as np

    stem = f"{core_row_start:08d}-{core_row_stop:08d}"
    root = work_dir / "seg"

    tri = np.zeros(int(tri_cells.shape[0]), dtype=np.dtype(_TRI_FIELDS))
    if tri.size:
        tri["cell0"], tri["cell1"], tri["cell2"] = (
            tri_cells[:, 0], tri_cells[:, 1], tri_cells[:, 2]
        )
        tri["component"] = tri_components
    tri_path = _write_records(root / f"{stem}.rmseg.tri", KIND_TRI, tri)

    n = len(owned)
    pos = np.zeros(n, dtype=np.dtype(_POS_FIELDS))
    if n:
        pos["cell"] = owned.row.astype(np.int64) * cols + owned.col
        pos["sample_id"] = (
            np.arange(n, dtype=np.int64) if owned.sample_id is None else owned.sample_id
        )
        pos["x"], pos["y"], pos["z"] = owned.xyz[:, 0], owned.xyz[:, 1], owned.xyz[:, 2]
        pos["rng"] = owned.rng
        if owned.rgb is not None:
            pos["r"], pos["g"], pos["b"] = (
                owned.rgb[:, 0], owned.rgb[:, 1], owned.rgb[:, 2]
            )
    pos_path = _write_records(
        root / f"{stem}.rmseg.pos", KIND_POS, pos,
        flags=FLAG_HAS_RGB if owned.rgb is not None else 0,
    )

    disp = np.zeros(1, dtype=np.dtype(_DISP_FIELDS))
    disp["core_row_start"], disp["core_row_stop"] = core_row_start, core_row_stop
    disp["retained"] = n
    disp["dropped_despeckle"] = dropped_despeckle
    disp["dropped_mover_carve"] = dropped_mover_carve
    disp["restored_from_carve"] = restored_from_carve
    disp_path = _write_records(root / f"{stem}.rmseg.disp", KIND_DISP, disp)

    return BandSegment(
        core_row_start=core_row_start,
        core_row_stop=core_row_stop,
        tri_path=tri_path,
        pos_path=pos_path,
        disp_path=disp_path,
        triangle_count=int(tri.size),
        sample_count=n,
    )


def read_tri_segment(path: Path) -> tuple[I64, I64]:
    """`(T,3)` stable-id triples and the `(T,)` provisional component ids."""
    import numpy as np

    records, _ = _read_records(path, KIND_TRI)
    cells = np.stack(
        [records["cell0"], records["cell1"], records["cell2"]], axis=1
    ).astype(np.int64)
    return cells, records["component"].astype(np.int64)


def read_pos_segment(path: Path) -> dict[str, Any]:
    """One band's owned sample records, as plain arrays."""
    import numpy as np

    records, flags = _read_records(path, KIND_POS)
    out: dict[str, Any] = {
        "cell": records["cell"].astype(np.int64),
        "sample_id": records["sample_id"].astype(np.int64),
        "xyz": np.stack(
            [records["x"], records["y"], records["z"]], axis=1
        ).astype(np.float32),
        "rng": records["rng"].astype(np.float32),
        "rgb": None,
    }
    if flags & FLAG_HAS_RGB:
        out["rgb"] = np.stack(
            [records["r"], records["g"], records["b"]], axis=1
        ).astype(np.uint8)
    return out


def read_disp_segment(path: Path) -> dict[str, int]:
    """The band's filter dispositions, which are final at Pass A."""
    records, _ = _read_records(path, KIND_DISP)
    if records.shape[0] != 1:
        raise SegmentError(f"disp segment must carry exactly one record: {path.name}")
    row = records[0]
    return {name: int(row[name]) for name, _ in _DISP_FIELDS}


@dataclass(frozen=True)
class ComponentRecord:
    """One component. `verdict` is 0 undecided, 1 keep, 2 drop."""

    component_id: int
    root: int
    triangle_count: int
    retired: bool = False
    area: float = 0.0
    smallest_positive_area: float = 0.0
    verdict: int = 0
    area_fallback: bool = False


def write_component_table(
    work_dir: Path,
    records: Sequence[ComponentRecord],
    *,
    name: str = "component.rmtable",
    finalised: bool = False,
) -> Path:
    """Pass A's completed membership table, or Pass B's finalised one.

    **One row per component id ever allocated, not one per surviving root.**
    A `tri` segment carries the id the band assigned at write time, and a later
    band may have merged it into a lower id; without the alias rows Pass B
    could not resolve those triangles to their final component. Pass A leaves
    the area and verdict columns undecided, because deciding them needs the
    whole completed table (`PHASE1-ISLANDS-FINALISATION.md` §4).
    """
    import numpy as np

    table = np.zeros(len(records), dtype=np.dtype(_COMPONENT_FIELDS))
    for i, rec in enumerate(records):
        table[i] = (
            rec.component_id, rec.root, rec.triangle_count, rec.area,
            rec.smallest_positive_area, int(rec.retired), int(rec.verdict),
            int(rec.area_fallback), 0,
        )
    return _write_records(
        work_dir / name, KIND_COMPONENT, table,
        flags=FLAG_FINALISED if finalised else 0,
    )


def read_component_table(path: Path) -> list[ComponentRecord]:
    records, _ = _read_records(path, KIND_COMPONENT)
    return [
        ComponentRecord(
            component_id=int(row["component_id"]),
            root=int(row["root"]),
            triangle_count=int(row["triangle_count"]),
            retired=bool(row["retired"]),
            area=float(row["area"]),
            smallest_positive_area=float(row["smallest_positive_area"]),
            verdict=int(row["verdict"]),
            area_fallback=bool(row["area_fallback"]),
        )
        for row in records
    ]


def retained_scan_from_segments(
    scan: StructuredScan, segments: Sequence[BandSegment]
) -> StructuredScan:
    """Rebuild the retained set from the `pos` segments, in band order.

    Band order **is** row-major order, because cores tile the lattice in
    increasing row order and each core's records are already row-major. That is
    not a convenience: it is the order `build_mesh`'s vertex compaction walks
    (`triangulate.py:260`), so it is what makes a streamed vertex array
    identical to the in-memory one rather than a permutation of it
    (`PHASE1-DETERMINISM-SPEC.md` §5(a), §5(c)).

    Scan-level attributes come from `scan` because they are properties of the
    station, not of any band. `intensity` is not carried by the `pos` record and
    is therefore dropped; nothing downstream of meshing reads it, and inventing
    a value would be worse than its absence.
    """
    import numpy as np

    from .types import StructuredScan as _Scan

    cols = scan.lattice.cols
    total = sum(seg.sample_count for seg in segments)
    if not total:
        empty = np.empty(0, np.int32)
        return _Scan(
            row=empty, col=empty.copy(), xyz=np.empty((0, 3), np.float32),
            rng=np.empty(0, np.float32), pose=scan.pose, lattice=scan.lattice,
            rgb=None if scan.rgb is None else np.empty((0, 3), np.uint8),
            intensity=None, station_id=scan.station_id,
            sample_id=None if scan.sample_id is None else np.empty(0, np.int64),
            source_sample_count=scan.source_sample_count,
            dropped_no_return=scan.dropped_no_return,
            dropped_other=scan.dropped_other,
        )

    # Preallocate and fill band by band, for the same reason as
    # `read_triangles_and_final_roots`: never hold the parts and the join.
    cells = np.empty(total, np.int64)
    xyz = np.empty((total, 3), np.float32)
    rng = np.empty(total, np.float32)
    sample_id = None if scan.sample_id is None else np.empty(total, np.int64)
    rgb = None if scan.rgb is None else np.empty((total, 3), np.uint8)
    at = 0
    for seg in segments:
        block = read_pos_segment(seg.pos_path)
        n = int(block["cell"].size)
        if not n:
            continue
        cells[at : at + n] = block["cell"]
        xyz[at : at + n] = block["xyz"]
        rng[at : at + n] = block["rng"]
        if sample_id is not None:
            sample_id[at : at + n] = block["sample_id"]
        if rgb is not None:
            if block["rgb"] is None:
                raise SegmentError("pos segment has no colour but the scan does")
            rgb[at : at + n] = block["rgb"]
        at += n
    if at != total:
        raise SegmentError("pos segments do not carry the sample counts they declare")

    return _Scan(
        row=(cells // cols).astype(np.int32), col=(cells % cols).astype(np.int32),
        xyz=xyz, rng=rng, pose=scan.pose, lattice=scan.lattice, rgb=rgb,
        intensity=None, station_id=scan.station_id, sample_id=sample_id,
        source_sample_count=scan.source_sample_count,
        dropped_no_return=scan.dropped_no_return, dropped_other=scan.dropped_other,
    )


def write_finalised_component_table(
    work_dir: Path,
    provisional_table: Path,
    area: Any,
    *,
    min_component_area: float,
    min_triangles: int,
) -> Path:
    """Pass B's `component.rmcomp`: the Pass A rows plus area and verdict.

    Written as a separate file rather than by editing the Pass A table in
    place. A published generation is immutable (`PHASE1-TILE-CONTRACT-V0.md`
    §3), and an in-place update is the one edit that could leave a reader
    holding a table that is half Pass A and half Pass B.
    """
    lookup = {int(root): i for i, root in enumerate(area.root_ids)}
    finalised: list[ComponentRecord] = []
    for rec in read_component_table(provisional_table):
        i = lookup.get(rec.root)
        if i is None:
            finalised.append(rec)
            continue
        total = float(area.areas[i])
        kept = min_component_area <= 0 or (
            total >= min_component_area and int(area.counts[i]) >= min_triangles
        )
        finalised.append(
            ComponentRecord(
                component_id=rec.component_id,
                root=rec.root,
                triangle_count=int(area.counts[i]) if rec.component_id == rec.root else 0,
                retired=rec.retired,
                area=total,
                smallest_positive_area=float(area.smallest_positive[i]),
                verdict=1 if kept else 2,
                area_fallback=bool(area.fallback[i]),
            )
        )
    return write_component_table(
        work_dir, finalised, name="component.rmcomp", finalised=True
    )


def read_triangles_and_final_roots(
    segments: Sequence[BandSegment], component_table_path: Path
) -> tuple[I64, I64]:
    """Every band's triangles, with their component ids resolved to final roots.

    The alias step is the reason the Pass A table carries a row per id ever
    allocated rather than one per surviving root: a `tri` segment records the id
    its band assigned at write time, and a later band may since have merged that
    id into a lower one (`PHASE1-ISLANDS-FINALISATION.md` §3.2).
    """
    import numpy as np

    # Preallocated from the counts each segment already records, then filled
    # one band at a time. Collecting the bands into a list and concatenating
    # would hold every band's triangles *and* the joined array at once, which
    # is the transient Pass A exists to avoid reintroducing here.
    total = sum(seg.triangle_count for seg in segments)
    cells = np.empty((total, 3), np.int64)
    provisional = np.empty(total, np.int64)
    at = 0
    for seg in segments:
        band_cells, band_ids = read_tri_segment(seg.tri_path)
        n = band_cells.shape[0]
        cells[at : at + n] = band_cells
        provisional[at : at + n] = band_ids
        at += n
    if at != total:
        raise SegmentError("tri segments do not carry the triangle counts they declare")
    records = read_component_table(component_table_path)
    alias = np.full(len(records) + 1, -1, np.int64)
    for rec in records:
        alias[rec.component_id] = rec.root
    if provisional.size and bool(np.any(alias[provisional] < 0)):
        raise SegmentError("a triangle names a component id the table does not carry")
    return cells, (alias[provisional] if provisional.size else provisional)
