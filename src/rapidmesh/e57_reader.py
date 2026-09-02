"""
Structured E57 reader — the scanner's own sample lattice, not a reconstruction.

Why this module is the whole point
----------------------------------
A structured E57 stores, per point, the row and column the scanner sampled it
at. That is the exact lattice: no projection, no binning, no collisions. Cairn
never sees it, because Cairn converts E57 -> LAZ first and LAZ has nowhere to
put it; `backend/mesher.py` then re-derives an approximate lattice with
`arctan2`/`arcsin` and bins it to 2048 x 1024, discarding roughly 99 % of the
angular samples on a high-resolution Trimble X7 scan before it makes a single
triangle.

This reader takes the best lattice the file actually offers, in order:

1. **`rowIndex` + `columnIndex`** — exact. `LatticeSource.ROW_COL`.
2. **`sphericalAzimuth` + `sphericalElevation`** — a regular angular lattice
   recovered by detecting the sampling step. Very close to exact for the
   scanners in scope, which sample uniformly in angle.
3. **`cartesianX/Y/Z` only** — reprojection, i.e. what Cairn is stuck with.
   Supported so unstructured files still work, but flagged
   `LatticeSource.PROJECTED` so the deviation report can say so and downstream
   thresholds can loosen.

Everything degrades; nothing silently pretends. `probe()` reports which tier a
file will land in without reading a single point, so a site's scans can be
triaged before committing to hours of processing.

Memory-safe boundary
--------------------
`iter_raw_chunks` drives libE57's compressed-vector reader repeatedly into
fixed-capacity buffers. It is the bounded-memory ingestion primitive for the
streamed band pipeline. The convenience `read_scan` path still materialises a
whole scan and is therefore explicitly not the production path for large
stations.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import LatticeInfo, LatticeSource, ScanPose, StructuredScan

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]

# E57 field names, as they appear in the CompressedVector prototype.
_XYZ = ("cartesianX", "cartesianY", "cartesianZ")
_SPHERICAL = ("sphericalRange", "sphericalAzimuth", "sphericalElevation")
_ROWCOL = ("rowIndex", "columnIndex")
_RGB = ("colorRed", "colorGreen", "colorBlue")
_SUPPORTED_FIELDS = frozenset(
    _XYZ
    + _SPHERICAL
    + _ROWCOL
    + _RGB
    + (
        "intensity",
        "cartesianInvalidState",
        "sphericalInvalidState",
    )
)

# Points closer than this to the scanner origin are not returns. PDAL and
# several exporters write no-return cells as a point AT the origin, which on
# Galvin Road was more than half the "points" in the file. Inherited from
# Cairn's `r > 0.3` filter for exactly the same reason.
MIN_RANGE = 0.3


def deps_ok() -> bool:
    """True when E57 reading is available. Kept separate from `rapidmesh.deps`
    so a caller can check just this one capability."""
    try:
        import pye57  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class ScanProbe:
    """What a scan offers, without reading its points."""

    index: int
    name: str
    point_count: int
    fields: tuple[str, ...]
    source: LatticeSource
    has_rgb: bool
    has_intensity: bool

    def describe(self) -> str:
        return (
            f"[{self.index}] {self.name or '(unnamed)'}  {self.point_count:,} pts  "
            f"lattice={self.source.value}  rgb={'y' if self.has_rgb else 'n'}  "
            f"intensity={'y' if self.has_intensity else 'n'}"
        )


@dataclass(frozen=True)
class RawChunk:
    """One copied slice of an E57 compressed-vector stream.

    Arrays are owned by the chunk. libE57 reuses its destination buffers on
    the next read, so yielding views would silently mutate previously yielded
    chunks and make concurrent band assembly impossible.
    """

    offset: int
    count: int
    data: Mapping[str, Any]

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(value, "nbytes", 0)) for value in self.data.values())


@dataclass(frozen=True)
class RawRowBand:
    """A row-major raw band with explicit non-overlapping ownership.

    ``data`` includes up to ``halo`` neighbouring rows on both sides.
    Consumers emit results only for ``core_row_start <= row < core_row_stop``;
    that ownership rule makes the repeated halo evidence available without
    double-counting samples or triangles.
    """

    core_row_start: int
    core_row_stop: int
    data_row_start: int
    data_row_stop: int
    data: Mapping[str, Any]
    buffered_points_before_emit: int

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(value, "nbytes", 0)) for value in self.data.values())


def probe(path: str | Path) -> list[ScanProbe]:
    """Report every scan in an E57 and which lattice tier it will use.

    Read-only and cheap — it inspects the XML prototype, never the point data.
    Point this at a site's raw folder before processing to find out in seconds
    whether the export carried the structured grid, which is the single fact
    that decides how good the meshes can possibly be.
    """
    import pye57

    e57 = pye57.E57(str(path))
    out: list[ScanProbe] = []
    for i in range(e57.scan_count):
        header = e57.get_header(i)
        fields = tuple(_header_fields(header))
        out.append(
            ScanProbe(
                index=i,
                name=str(getattr(header, "name", "") or ""),
                point_count=int(getattr(header, "point_count", 0) or 0),
                fields=fields,
                source=_classify(fields),
                has_rgb=all(f in fields for f in _RGB),
                has_intensity="intensity" in fields,
            )
        )
    return out


def read_scan(
    path: str | Path,
    index: int = 0,
    max_points: int | None = None,
    station_id: str = "",
) -> StructuredScan:
    """Read one scan into a `StructuredScan` on its native lattice."""
    import pye57

    e57 = pye57.E57(str(path))
    if not 0 <= index < e57.scan_count:
        raise IndexError(f"scan {index} out of range (file has {e57.scan_count})")
    header = e57.get_header(index)
    raw: dict[str, Any] = e57.read_scan_raw(index)
    pose = _pose_from_header(header)
    sid = station_id or str(getattr(header, "name", "") or f"scan{index:03d}")
    scan, _frame_path = _build(raw, pose, sid, max_points)
    return scan


def read_scan_with_frame_path(
    path: str | Path,
    index: int = 0,
    max_points: int | None = None,
    station_id: str = "",
) -> tuple[StructuredScan, str]:
    """`read_scan`, plus which `_resolve_frame` branch it took.

    The branch is a reproduction condition — it decides whether the pipeline
    saw the exporter's coordinates or the recovered local ones — so PLAN.md §5
    item 11 requires it in the evidence envelope. It cannot ride on
    `StructuredScan`: `grid.sort_row_major`, `select` and `concat` rebuild that
    dataclass field by field, so a field added there is dropped by the first
    derived scan without anything raising. Returned beside the scan instead,
    where losing it takes an edit rather than an omission.

    `read_scan` stays exactly as it was for the callers that do not need it.
    """
    import pye57

    e57 = pye57.E57(str(path))
    if not 0 <= index < e57.scan_count:
        raise IndexError(f"scan {index} out of range (file has {e57.scan_count})")
    header = e57.get_header(index)
    raw: dict[str, Any] = e57.read_scan_raw(index)
    pose = _pose_from_header(header)
    sid = station_id or str(getattr(header, "name", "") or f"scan{index:03d}")
    return _build(raw, pose, sid, max_points)


def iter_raw_chunks(
    path: str | Path,
    index: int = 0,
    chunk_points: int = 1_000_000,
    fields: tuple[str, ...] | None = None,
) -> Iterator[RawChunk]:
    """Yield fixed-capacity raw point chunks without materialising the scan.

    This is deliberately a raw-stream API. Turning arbitrary compressed-vector
    chunks directly into separate lattices would estimate different angular
    steps per chunk and create seams. The next layer groups this stream into
    row bands with one-row overlap before filtering and triangulation.
    """
    import pye57

    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    e57 = pye57.E57(str(path))
    if not 0 <= index < e57.scan_count:
        raise IndexError(f"scan {index} out of range (file has {e57.scan_count})")
    header = e57.get_header(index)
    available = tuple(_header_fields(header))
    selected = (
        tuple(field for field in available if field in _SUPPORTED_FIELDS)
        if fields is None
        else fields
    )
    missing = sorted(set(selected) - set(available))
    unsupported = sorted(set(selected) - _SUPPORTED_FIELDS)
    if missing:
        raise ValueError(f"scan does not carry requested fields: {missing}")
    if unsupported:
        raise ValueError(f"unsupported E57 fields requested: {unsupported}")
    if not selected:
        raise ValueError("scan has no supported point fields")

    arrays, buffers = e57.make_buffers(selected, chunk_points)
    reader = header.points.reader(buffers)
    offset = 0
    try:
        while True:
            count = int(reader.read())
            if count <= 0:
                break
            # Copy only the filled prefix; the destination arrays are reused by
            # the next reader.read() call.
            data = {field: array[:count].copy() for field, array in arrays.items()}
            yield RawChunk(offset=offset, count=count, data=data)
            offset += count
    finally:
        reader.close()


def iter_row_bands(
    chunks: Iterator[RawChunk],
    *,
    row_min: int,
    row_stop: int,
    band_rows: int = 256,
    halo: int = 1,
) -> Iterator[RawRowBand]:
    """Group a row-major raw stream into bounded, overlapping row bands.

    This function is format-independent and performs no geometry conversion.
    It establishes the seam discipline used by streamed E57 processing:
    filtering may inspect both halo rows, triangulation may inspect the row
    below, and only the core owns output. Input that is not row-major is
    rejected rather than silently producing incomplete bands.
    """
    import numpy as np

    if band_rows <= 0:
        raise ValueError("band_rows must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if row_stop < row_min:
        raise ValueError("row_stop must not precede row_min")

    buffered: dict[str, Any] = {}
    core_start = row_min
    last_row: int | None = None

    def append(chunk: RawChunk) -> None:
        nonlocal last_row
        if "rowIndex" not in chunk.data:
            raise ValueError("row-band assembly requires rowIndex")
        row = np.asarray(chunk.data["rowIndex"])
        if row.size != chunk.count:
            raise ValueError("chunk count does not match rowIndex length")
        if row.size and (
            np.any(np.diff(row.astype(np.int64)) < 0)
            or (last_row is not None and int(row[0]) < last_row)
        ):
            raise ValueError("E57 point stream is not row-major")
        if row.size:
            last_row = int(row[-1])
        if buffered and set(buffered) != set(chunk.data):
            raise ValueError("raw chunk fields changed within the stream")
        for field, values in chunk.data.items():
            array = np.asarray(values)
            if array.shape[0] != chunk.count:
                raise ValueError(f"chunk count does not match {field} length")
            buffered[field] = (
                array.copy()
                if field not in buffered
                else np.concatenate((buffered[field], array))
            )

    def ready(eof: bool) -> bool:
        """Is every row this band needs *complete* in the buffer?

        Strictly greater, not `>=`. A row is only known to be finished once a
        sample from a later row has arrived, so `row[-1] >= need_row` fires when
        the last needed row has merely *started*. For an interior band that
        emits a partial halo row — survivable at halo 3, because the composed
        filter chain reaches only two rows and never reads it. For the **final**
        band `need_row` is its own core row, and emitting early truncated the
        core: the rest of that row arrived after `core_start` had advanced past
        it and was silently dropped. Measured on a 16 x 360 fixture at
        `chunk_points=80`: 250 of 360 samples in the last row lost, and the
        ledger stopped balancing. The end-of-stream pass below is what releases
        the final band now.
        """
        if not buffered or core_start >= row_stop:
            return False
        row = np.asarray(buffered["rowIndex"])
        core_stop = min(core_start + band_rows, row_stop)
        need_row = min(core_stop + halo, row_stop) - 1
        return eof or (row.size > 0 and int(row[-1]) > need_row)

    def take() -> RawRowBand | None:
        nonlocal buffered, core_start
        row = np.asarray(buffered["rowIndex"])
        core_stop = min(core_start + band_rows, row_stop)
        data_start = max(row_min, core_start - halo)
        data_stop = min(row_stop, core_stop + halo)
        chosen = (row >= data_start) & (row < data_stop)
        band_data = {field: np.asarray(values)[chosen].copy() for field, values in buffered.items()}

        next_keep_from = max(row_min, core_stop - halo)
        keep = row >= next_keep_from
        buffered = {field: np.asarray(values)[keep] for field, values in buffered.items()}
        old_start = core_start
        core_start = core_stop
        if not np.any(chosen):
            return None
        return RawRowBand(
            core_row_start=old_start,
            core_row_stop=core_stop,
            data_row_start=data_start,
            data_row_stop=data_stop,
            data=band_data,
            buffered_points_before_emit=int(row.size),
        )

    for chunk in chunks:
        append(chunk)
        while ready(eof=False):
            band = take()
            if band is not None:
                yield band
    while ready(eof=True):
        band = take()
        if band is not None:
            yield band


def _build(
    raw: dict[str, Any],
    pose: ScanPose,
    station_id: str,
    max_points: int | None,
) -> tuple[StructuredScan, str]:
    """Turn one scan's raw field dict into a `StructuredScan`.

    Split out from `read_scan` with no pye57 in its signature so the synthetic
    fixtures in `synthetic.py` exercise this exact code path — the lattice
    detection and validity handling below is where the bugs live, and it would
    be untestable without real client scans otherwise.
    """
    import numpy as np

    xyz_local, rng = _positions(raw)
    valid = _validity(raw, rng)
    frame_path = frame_decision(raw, pose, valid)
    xyz_local, rng = _resolve_frame(xyz_local, rng, raw, pose, valid)
    # After the frame is settled and before anything derives an angle from it.
    # Only the valid samples: no-return records legitimately carry sentinel
    # ranges and are excluded from the mesh anyway.
    assert_plausible_range(rng[valid], where=f"read_scan (frame: {frame_path})")
    row, col, lattice = _lattice(raw, xyz_local, rng, valid)

    source_sample_count = int(valid.size)
    dropped_no_return = int(np.count_nonzero(~valid))
    keep = valid
    dropped_other = 0
    if max_points is not None and int(keep.sum()) > max_points:
        stride_keep = _lattice_stride(row, col, lattice, int(keep.sum()), max_points)
        dropped_other = int(np.count_nonzero(valid & ~stride_keep))
        keep = keep & stride_keep

    row, col = row[keep].astype(np.int32), col[keep].astype(np.int32)
    xyz_local = xyz_local[keep].astype(np.float32)
    rng = rng[keep].astype(np.float32)

    rgb = _rgb(raw, keep)
    intensity = _intensity(raw, keep)

    from .grid import sort_row_major

    return sort_row_major(
        StructuredScan(
            row=row,
            col=col,
            xyz=xyz_local,
            rng=rng,
            pose=pose,
            lattice=lattice,
            rgb=rgb,
            intensity=intensity,
            station_id=station_id,
            sample_id=np.nonzero(keep)[0].astype(np.int64),
            source_sample_count=source_sample_count,
            dropped_no_return=dropped_no_return,
            dropped_other=dropped_other,
        )
    ), frame_path


# --------------------------------------------------------------------------
# positions and validity
# --------------------------------------------------------------------------


def _positions(raw: dict[str, Any]) -> tuple[Any, Any]:
    """Scanner-local offsets (N,3) f64 and range (N,) f64.

    Spherical is preferred over cartesian when both are present: the spherical
    triple is what the instrument measured, and the cartesian triple is the
    exporter's conversion of it. Using the measurement avoids inheriting
    someone else's rounding.
    """
    import numpy as np

    if all(f in raw for f in _SPHERICAL):
        r = np.asarray(raw["sphericalRange"], np.float64)
        az = np.asarray(raw["sphericalAzimuth"], np.float64)
        el = np.asarray(raw["sphericalElevation"], np.float64)
        ce = np.cos(el)
        xyz = np.empty((r.size, 3), np.float64)
        xyz[:, 0] = r * ce * np.cos(az)
        xyz[:, 1] = r * ce * np.sin(az)
        xyz[:, 2] = r * np.sin(el)
        return xyz, r
    if all(f in raw for f in _XYZ):
        xyz = np.stack(
            [np.asarray(raw[f], np.float64) for f in _XYZ], axis=1
        )
        return xyz, np.linalg.norm(xyz, axis=1)
    raise ValueError(
        "scan carries neither cartesian nor spherical coordinates — "
        f"fields present: {sorted(raw.keys())}"
    )


def _resolve_frame(xyz: Any, rng: Any, raw: dict[str, Any], pose: ScanPose, valid: Any) -> tuple[Any, Any]:
    """Detect and correct points written in world coordinates.

    The E57 standard puts a scan's cartesian points in the **scan-local** frame,
    with `pose` mapping local to world. Not every exporter obeys it: writing
    already-transformed project coordinates alongside a non-identity pose is a
    known interoperability wart, and it is catastrophic here — every angle in
    the pipeline is measured from the scanner origin, so a scan whose points are
    offset by the pose produces a wrong lattice, wrong incidence thresholds, and
    a mesh in the wrong place. Nothing raises. It just comes out subtly wrong.

    The discriminator is a physical property of a rotating-head scanner: it
    sweeps a **vertical plane** per head position, so within one lattice column
    every sample must share the same azimuth. Under the correct frame that
    spread is essentially zero; under a translated or rotated project frame it
    is large and obvious. Recovery applies the full inverse rigid pose.

    Only checkable when the file carries row/column indices, which is also the
    only case where the lattice is precise enough for the error to matter.
    Spherical coordinates are unambiguously local by definition and are left
    alone.
    """
    import numpy as np

    if frame_decision(raw, pose, valid) is not FRAME_REWRITTEN_TO_LOCAL:
        return xyz, rng
    out = pose.world_to_local(xyz).astype(np.float64)
    return out, np.linalg.norm(out, axis=1)


# The branch names `_resolve_frame` can take. Recorded in the evidence envelope
# because `SPATIAL-CONTRACT.md` §3 rule 7 turns on which one happened, and
# PLAN.md §5 item 11 closes that conflict by writing it down.
FRAME_SPHERICAL_IS_LOCAL = "spherical-is-local-by-definition"
FRAME_NO_ROW_COLUMN = "no-row-column-lattice"
FRAME_IDENTITY_POSE = "identity-pose-nothing-to-detect"
FRAME_TOO_FEW_SAMPLES = "too-few-valid-samples-to-discriminate"
FRAME_LEFT_AS_LOCAL = "left-as-local"
FRAME_REWRITTEN_TO_LOCAL = "rewritten-to-local"
FRAME_UNDECIDED = "frame-undecided-left-as-local"
FRAME_IMPLAUSIBLE = "both-candidates-implausible"

#: No terrestrial scanner measures ten kilometres. A candidate frame that puts
#: samples further away than this is not a frame, it is arithmetic on the wrong
#: numbers — the F1 failure produced ranges of 5.8 x 10^6 m. Configurable
#: because an airborne or mobile survey would want a different setting; the
#: value in force is recorded rather than assumed.
MAX_PLAUSIBLE_RANGE_M = 10_000.0

#: How many lattice angular steps of column spread still count as "aligned".
FRAME_ALIGNMENT_STEPS = 4.0

#: Floor under the alignment ceiling, so a degenerate az_step of 0 cannot make
#: the bar unreachable and force every station to `frame-undecided`.
FRAME_ALIGNMENT_FLOOR = 1e-12

#: How decisively the rewritten candidate must beat leaving it alone.
FRAME_DECISIVE_RATIO = 0.2

#: Below this many valid samples the spread statistic is not discriminating.
FRAME_MIN_SAMPLES = 64


class ImplausibleRange(ValueError):
    """Samples lie further from the scanner origin than any scanner can measure.

    Named rather than generic because the one thing this must never be is
    quiet. It is the guard that would have failed F1's first real run on
    31 August instead of letting eight reports be written on wrong geometry.
    """


@dataclass(frozen=True)
class FrameEvidence:
    """What the frame decision saw, not merely what it concluded.

    `SPATIAL-CONTRACT.md` §3 rule 7 requires the heuristic to record which path
    it took; recording only the verdict is what let F1 run for two days without
    anyone able to see *why* it fired. The spreads and both candidates' maximum
    ranges are carried so a campaign record can be audited after the fact.
    """

    path: str
    as_local: float = float("nan")
    as_world: float = float("nan")
    max_range_local: float = float("nan")
    max_range_world: float = float("nan")
    alignment_ceiling: float = float("nan")
    undecided: bool = False

    @property
    def rewrite(self) -> bool:
        return self.path == FRAME_REWRITTEN_TO_LOCAL

    def describe(self) -> str:
        return (
            f"{self.path} (as_local={self.as_local:.3g} as_world={self.as_world:.3g} "
            f"max_range_local={self.max_range_local:.4g} m "
            f"max_range_world={self.max_range_world:.4g} m)"
        )


def column_alignment_ceiling(az_step: float) -> float:
    """Largest circular variance still consistent with aligned lattice columns.

    **This is what makes the rule scale-aware.** A bare ratio between two
    spreads compares one artefact against another: subtracting a 10^6 m
    translation from a 10^1 m scene collapses every azimuth onto one value, so
    the wrong candidate scores ~10^-15 and wins by twelve orders of magnitude
    however wrong it is. An absolute bar, set by the lattice the scanner
    actually sampled, cannot be gamed that way.

    For a small angular spread sigma the circular variance ``1 - R`` is
    approximately ``sigma^2 / 2``. A column whose samples sit within a few
    lattice steps of one plane is aligned; anything wider is not.
    """
    sigma = FRAME_ALIGNMENT_STEPS * abs(az_step)
    return max(0.5 * sigma * sigma, FRAME_ALIGNMENT_FLOOR)


def decide_frame(
    *,
    has_spherical: bool,
    has_row_column: bool,
    posed: bool,
    sample_count: int,
    as_local: float,
    as_world: float,
    max_range_local: float,
    max_range_world: float,
    az_step: float,
    max_plausible_range_m: float = MAX_PLAUSIBLE_RANGE_M,
) -> FrameEvidence:
    """The one frame decision, shared by the in-memory and streamed paths.

    Pure: it takes scalars, so the streamed path can feed it a bounded
    subsample's statistics and the in-memory path can feed it the whole scan's,
    and both provably reach the same verdict. Having two implementations of
    this rule is what F1 was — the streamed copy omitted the spherical
    short-circuit and rewrote every real station into a frame 5.8 x 10^6 m from
    where it belonged.

    Order matters, and it is the owner's ruling of 2 September 2026:

    1. **Spherical is local by definition.** No detection, no exceptions.
    2. **Physical plausibility first.** A candidate frame whose maximum range
       exceeds `max_plausible_range_m` is rejected outright — no terrestrial
       scanner measures ten kilometres, so such a frame is not a frame. This
       alone would have stopped F1 on its first real run.
    3. **Only when both candidates are physically plausible** does the
       azimuth-spread margin get a say, and then it must clear an absolute bar
       set by the lattice angular step, not merely beat the other candidate.
    4. **Fail closed.** Undecided means left-as-local and said so, never a
       rewrite on weak evidence (`SPATIAL-CONTRACT.md` §3 rule 7).
    """
    if has_spherical:
        return FrameEvidence(FRAME_SPHERICAL_IS_LOCAL)
    if not has_row_column:
        return FrameEvidence(FRAME_NO_ROW_COLUMN)
    if not posed:
        return FrameEvidence(FRAME_IDENTITY_POSE)
    if sample_count < FRAME_MIN_SAMPLES:
        return FrameEvidence(FRAME_TOO_FEW_SAMPLES, undecided=True)

    ceiling = column_alignment_ceiling(az_step)

    def verdict(path: str, *, undecided: bool = False) -> FrameEvidence:
        return FrameEvidence(
            path=path,
            as_local=float(as_local),
            as_world=float(as_world),
            max_range_local=float(max_range_local),
            max_range_world=float(max_range_world),
            alignment_ceiling=ceiling,
            undecided=undecided,
        )

    local_plausible = max_range_local <= max_plausible_range_m
    world_plausible = max_range_world <= max_plausible_range_m

    if not local_plausible and not world_plausible:
        # Neither frame puts the samples within reach of a scanner. The file is
        # not describable by this rule; `assert_plausible_range` raises on the
        # data itself rather than letting a mesh be built somewhere impossible.
        return verdict(FRAME_IMPLAUSIBLE, undecided=True)
    if local_plausible and not world_plausible:
        return verdict(FRAME_LEFT_AS_LOCAL)
    if world_plausible and not local_plausible:
        return verdict(FRAME_REWRITTEN_TO_LOCAL)

    # Both plausible: the ordinary case, and the only one where the spread
    # comparison is meaningful. Require the rewritten candidate to be aligned
    # *in absolute terms* as well as decisively better than leaving it alone.
    if as_world <= ceiling and as_world < as_local * FRAME_DECISIVE_RATIO:
        return verdict(FRAME_REWRITTEN_TO_LOCAL)
    if as_local <= ceiling:
        return verdict(FRAME_LEFT_AS_LOCAL)
    return verdict(FRAME_UNDECIDED, undecided=True)


def frame_evidence(
    raw: dict[str, Any],
    pose: ScanPose,
    valid: Any,
    *,
    az_step: float | None = None,
    max_plausible_range_m: float = MAX_PLAUSIBLE_RANGE_M,
) -> FrameEvidence:
    """`decide_frame` fed from a whole raw scan — the in-memory path's entry."""
    import numpy as np

    has_spherical = all(f in raw for f in _SPHERICAL)
    has_row_column = all(f in raw for f in _ROWCOL)
    if has_spherical or not has_row_column:
        return decide_frame(
            has_spherical=has_spherical, has_row_column=has_row_column,
            posed=False, sample_count=0, as_local=float("nan"), as_world=float("nan"),
            max_range_local=float("nan"), max_range_world=float("nan"), az_step=1.0,
        )

    t = np.asarray(pose.translation, np.float64)
    rotation = np.asarray(pose.rotation, np.float64)
    posed = not (
        float(np.linalg.norm(t)) < 0.01
        and np.allclose(rotation, np.eye(3), rtol=1e-12, atol=1e-12)
    )

    col = np.asarray(raw["columnIndex"], np.int64)
    idx = np.nonzero(valid)[0]
    if idx.size > 100_000:
        idx = idx[:: idx.size // 100_000]
    if not posed or idx.size < FRAME_MIN_SAMPLES:
        return decide_frame(
            has_spherical=False, has_row_column=True, posed=posed,
            sample_count=int(idx.size), as_local=float("nan"), as_world=float("nan"),
            max_range_local=float("nan"), max_range_world=float("nan"), az_step=1.0,
        )

    xyz, _rng = _positions(raw)
    local_candidate = xyz[idx]
    world_candidate = pose.world_to_local(xyz)[idx]
    if az_step is None:
        az_step = 2 * math.pi / max(int(col.max()) + 1, 1)

    return decide_frame(
        has_spherical=False,
        has_row_column=True,
        posed=True,
        sample_count=int(idx.size),
        as_local=_column_azimuth_spread(local_candidate, col[idx]),
        as_world=_column_azimuth_spread(world_candidate, col[idx]),
        max_range_local=float(np.max(np.linalg.norm(local_candidate, axis=1))),
        max_range_world=float(np.max(np.linalg.norm(world_candidate, axis=1))),
        az_step=az_step,
        max_plausible_range_m=max_plausible_range_m,
    )


def frame_decision(raw: dict[str, Any], pose: ScanPose, valid: Any) -> str:
    """Which branch `_resolve_frame` takes, as a recordable name.

    Split out of the correction itself so the decision can be recorded without
    re-deriving it, and so the streamed path can take it **once** for the
    station instead of once per band — a band holds a fraction of each lattice
    column, and this test reads azimuth spread *within* columns.
    """
    return frame_evidence(raw, pose, valid).path


def assert_plausible_range(rng: Any, *, where: str, limit: float = MAX_PLAUSIBLE_RANGE_M) -> None:
    """Refuse samples no terrestrial scanner could have produced.

    The cheapest check in the codebase and the one whose absence cost the most:
    every band F1 corrupted carried ranges above 10^5 m, and nothing between the
    reader and the triangulator ever asked. Raising here converts a silently
    wrong mesh into a named failure at the point of ingest.
    """
    import numpy as np

    values = np.asarray(rng, dtype=np.float64)
    if values.size == 0:
        return
    worst = float(np.max(np.abs(values)))
    if worst > limit:
        raise ImplausibleRange(
            f"{where}: maximum range {worst:,.1f} m exceeds the plausibility "
            f"setting of {limit:,.1f} m. Either the frame decision is wrong or "
            "this is not terrestrial scanner data; refusing to mesh coordinates "
            "no scanner could have measured"
        )


def _column_azimuth_spread(xyz: Any, col: Any) -> float:
    """Mean circular spread of azimuth within lattice columns. Lower is better.

    Circular, not linear: azimuth near the +/-pi branch cut would otherwise show
    an enormous spread for samples that are in fact identical.
    """
    import numpy as np

    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    order = np.argsort(col, kind="stable")
    c, a = col[order], az[order]
    bounds = np.flatnonzero(np.diff(c)) + 1
    groups = np.split(np.arange(a.size), bounds)
    spreads = []
    for g in groups:
        if g.size < 8:
            continue
        s, cch = np.mean(np.sin(a[g])), np.mean(np.cos(a[g]))
        # 1 - R, the circular variance: 0 when perfectly aligned, 1 when uniform.
        spreads.append(1.0 - float(np.hypot(s, cch)))
    return float(np.mean(spreads)) if spreads else 1.0


def _validity(raw: dict[str, Any], rng: Any) -> Any:
    """Boolean mask of usable returns.

    Three separate reasons a cell is not a return, and all three have been seen
    in the same file: an explicit invalid-state flag, a zero/NaN range, and a
    point written at the scanner origin. The last one is not an error state in
    the spec, which is why it needs its own check.
    """
    import numpy as np

    ok = np.isfinite(rng) & (rng > MIN_RANGE)
    for flag in ("cartesianInvalidState", "sphericalInvalidState"):
        if flag in raw:
            ok &= np.asarray(raw[flag]).astype(np.int32) == 0
    return ok


def _rgb(raw: dict[str, Any], keep: Any) -> Any | None:
    import numpy as np

    if not all(f in raw for f in _RGB):
        return None
    chans = []
    for f in _RGB:
        c = np.asarray(raw[f])[keep]
        # E57 colour is integer with a declared maximum, commonly 255 but
        # legally 65535. Scale by the observed width rather than assuming 8-bit,
        # or a 16-bit export renders black.
        if c.dtype.kind in "iu" and int(c.max(initial=0)) > 255:
            c = c >> 8
        chans.append(np.clip(c, 0, 255).astype(np.uint8))
    return np.stack(chans, axis=1)


def _intensity(raw: dict[str, Any], keep: Any) -> Any | None:
    import numpy as np

    if "intensity" not in raw:
        return None
    v = np.asarray(raw["intensity"])[keep]
    if v.dtype.kind == "f":
        # Float intensity is normalised 0..1 in every export seen so far.
        # Promote to the full 16-bit range instead of Cairn's `>> 8` collapse
        # to 8 bits, which throws away half the dynamic range for no reason.
        v = np.clip(v, 0.0, 1.0) * 65535.0
    return np.clip(v, 0, 65535).astype(np.uint16)


# --------------------------------------------------------------------------
# lattice detection — the part that matters
# --------------------------------------------------------------------------


def _classify(fields: tuple[str, ...] | list[str]) -> LatticeSource:
    f = set(fields)
    if all(x in f for x in _ROWCOL):
        return LatticeSource.ROW_COL
    if "sphericalAzimuth" in f and "sphericalElevation" in f:
        return LatticeSource.SPHERICAL
    return LatticeSource.PROJECTED


def _lattice(
    raw: dict[str, Any], xyz: Any, rng: Any, valid: Any
) -> tuple[Any, Any, LatticeInfo]:
    """Row/column per sample plus the lattice description.

    Returns indices for **every** sample including invalid ones, so the caller
    can apply its own mask once. Lattice extents are computed from valid
    samples only — invalid cells frequently carry garbage indices.
    """
    import numpy as np

    source = _classify(tuple(raw.keys()))

    if source is LatticeSource.ROW_COL:
        row = np.asarray(raw["rowIndex"], np.int64)
        col = np.asarray(raw["columnIndex"], np.int64)
        rows = int(row[valid].max(initial=0)) + 1
        cols = int(col[valid].max(initial=0)) + 1
        az_step, el_step, az0, el0 = _angular_steps_from_indices(xyz, rng, row, col, valid, rows, cols)
        return row, col, LatticeInfo(rows, cols, az_step, el_step, az0, el0, source)

    if source is LatticeSource.SPHERICAL:
        az = np.asarray(raw["sphericalAzimuth"], np.float64)
        el = np.asarray(raw["sphericalElevation"], np.float64)
        return _lattice_from_angles(az, el, valid, LatticeSource.SPHERICAL)

    # Reprojection: exactly what Cairn does, kept only as a floor.
    with np.errstate(invalid="ignore", divide="ignore"):
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(rng, 1e-12), -1.0, 1.0))
    return _lattice_from_angles(az, el, valid, LatticeSource.PROJECTED)


def _lattice_from_angles(
    az: Any, el: Any, valid: Any, source: LatticeSource
) -> tuple[Any, Any, LatticeInfo]:
    """Build a regular lattice by detecting the angular sampling step.

    The step is the **median of the positive differences between consecutive
    distinct sorted angles**, taken over a subsample. Median, not mean, because
    a scan has occasional large gaps (no-return runs) that would drag a mean
    upward and collapse the lattice. This recovers the true step to within
    floating point on a uniformly-sampled instrument, which every scanner in
    scope is.
    """
    import numpy as np

    a, e = az[valid], el[valid]
    if a.size == 0:
        raise ValueError("scan has no valid returns")

    az_step = _detect_step(a, span=2 * math.pi)
    el_step = _detect_step(e, span=math.pi)

    az0, el0 = float(a.min()), float(e.min())
    cols = max(int(round((float(a.max()) - az0) / az_step)) + 1, 1)
    rows = max(int(round((float(e.max()) - el0) / el_step)) + 1, 1)

    # Reprojection cannot recover a true lattice; two real neighbours land in
    # one cell and one is lost. Halving the density (doubling the cell) keeps
    # collisions rare, which is the least-bad behaviour for a file that never
    # carried the grid. Honest, and flagged as PROJECTED in the report.
    if source is LatticeSource.PROJECTED:
        cols, rows = max(cols // 2, 1), max(rows // 2, 1)
        az_step *= 2.0
        el_step *= 2.0

    col = np.clip(np.round((az - az0) / az_step).astype(np.int64), 0, cols - 1)
    row = np.clip(np.round((el - el0) / el_step).astype(np.int64), 0, rows - 1)
    return row, col, LatticeInfo(rows, cols, az_step, el_step, az0, el0, source)


def _detect_step(values: Any, span: float, sample: int = 400_000) -> float:
    """Median spacing between consecutive distinct values."""
    import numpy as np

    v = values
    if v.size > sample:
        v = v[:: max(v.size // sample, 1)]
    v = np.unique(v)
    if v.size < 2:
        return span
    d = np.diff(v)
    d = d[d > 0]
    if d.size == 0:
        return span
    step = float(np.median(d))
    # A degenerate median (all-identical angles, or float noise below the real
    # step) would produce a lattice with billions of columns. Clamp to
    # something a physical instrument could plausibly sample.
    return max(step, span / 200_000.0)


def _angular_steps_from_indices(
    xyz: Any, rng: Any, row: Any, col: Any, valid: Any, rows: int, cols: int
) -> tuple[float, float, float, float]:
    """Angle-per-column and angle-per-row for a row/column lattice.

    A row/column lattice gives exact connectivity but says nothing about the
    angular *size* of a cell, and every triangulation threshold is expressed in
    angle. So the step has to be recovered from the geometry.

    **Azimuth cannot be least-squares fitted directly.** It has a branch cut at
    +/-pi, and `np.unwrap` does not help because the samples arrive in row-major
    order: azimuth sweeps through a full turn within each row and then resets at
    every row boundary, so unwrapping accumulates a spurious 2*pi per row and the
    fitted slope comes out wrong. That mistake is quiet — it produced a step
    within 1 % of correct on a test scan, which is close enough to look fine and
    far enough to matter at 60 m.

    Instead: take differences between samples that are **adjacent within the
    same row**, wrap each difference into (-pi, pi], divide by the column gap,
    and take the median. Differencing removes the branch cut entirely, the wrap
    handles the one pair per row that straddles it, and the median absorbs the
    no-return gaps.

    Elevation has no branch cut (it is bounded by +/-pi/2 and never wraps), so an
    ordinary least-squares fit against the row index is correct there.
    """
    import numpy as np

    idx = np.nonzero(valid)[0]
    if idx.size > 200_000:
        idx = idx[:: idx.size // 200_000]
    if idx.size < 8:
        return 2 * math.pi / max(cols, 1), math.pi / max(rows, 1), -math.pi, -math.pi / 2

    p, r = xyz[idx], np.maximum(rng[idx], 1e-12)
    az = np.arctan2(p[:, 1], p[:, 0])
    el = np.arcsin(np.clip(p[:, 2] / r, -1.0, 1.0))
    rr, cc = row[idx].astype(np.int64), col[idx].astype(np.int64)

    az_step = _circular_step(az, rr, cc, 2 * math.pi / max(cols, 1))
    el_step, el0 = _linfit(rr.astype(np.float64), el, math.pi / max(rows, 1), -math.pi / 2)

    # Intercept for azimuth, taken as a circular mean of the de-trended angle so
    # the branch cut cannot drag it to the middle of the range.
    resid = az - cc * az_step
    az0 = float(np.arctan2(np.mean(np.sin(resid)), np.mean(np.cos(resid))))
    return az_step, el_step, az0, el0


def _circular_step(az: Any, row: Any, col: Any, default: float) -> float:
    """Median azimuth increment per column, immune to the +/-pi branch cut.

    **Sorted by (row, col) before differencing — F2.** The pairs this needs are
    same-row neighbours, but the samples arrive in the file's order, and every
    authorised structured E57 is *column-major*: consecutive picks share a
    column, not a row, so same-row pairs essentially never occurred and this
    returned `default` on every real file. That default is right for a full
    360 degree sweep, which is why it went unnoticed; on a partial field of
    view it is wrong by the FOV ratio, and `columns_wrap` — which compares
    `az_step * cols` against 2*pi — then returns True on every lattice and
    joins the two ends of a 180 degree scan across the room.

    Sorting costs one `lexsort` and makes the estimator independent of the
    file's storage order, which is the property it needed all along.
    """
    import numpy as np

    order = np.lexsort((col, row))
    row, col, az = row[order], col[order], az[order]

    same_row = np.diff(row) == 0
    dcol = np.diff(col)
    # A pair spanning more than half the lattice cannot be unwrapped
    # unambiguously: the wrap below would fold a genuine multi-column gap into
    # the wrong branch, and subsampling makes such gaps common. The bound comes
    # from the column span in the data, not from `default` — `default` is a
    # fallback value, and using it as a scale estimate would make this filter
    # depend on the caller's choice of fallback.
    span = int(col[-1] - col[0]) + 1 if col.size else 1
    unambiguous = dcol < max(span // 2, 1)
    ok = same_row & (dcol > 0) & unambiguous
    if not np.any(ok):
        return default
    daz = np.diff(az)[ok]
    # Wrap into (-pi, pi]: exactly one pair per row crosses the cut, and left
    # unwrapped it would contribute a ~2*pi outlier.
    daz = (daz + math.pi) % (2 * math.pi) - math.pi
    step = float(np.median(daz / dcol[ok]))
    if not math.isfinite(step) or abs(step) < 1e-12:
        return default
    return step


def _linfit(x: Any, y: Any, step_default: float, intercept_default: float) -> tuple[float, float]:
    import numpy as np

    if x.size < 2 or float(x.max() - x.min()) < 1e-9:
        return step_default, intercept_default
    slope, intercept = np.polyfit(x, y, 1)
    if not math.isfinite(slope) or abs(slope) < 1e-12:
        return step_default, intercept_default
    return float(slope), float(intercept)


def _lattice_stride(
    row: Any, col: Any, lattice: LatticeInfo, n_valid: int, max_points: int
) -> Any:
    """Subsample by keeping every k-th lattice row and column.

    Striding the lattice, never sampling points randomly: random sampling
    destroys the neighbour relationships the entire engine is built on and
    would turn a structured scan into exactly the unstructured cloud we refuse
    to work from. Striding keeps a coarser but still *structured* lattice.
    """

    k = max(int(math.ceil(math.sqrt(n_valid / max(max_points, 1)))), 1)
    return (row % k == 0) & (col % k == 0)


# --------------------------------------------------------------------------
# pose
# --------------------------------------------------------------------------


def _pose_from_header(header: Any) -> ScanPose:
    """Scanner pose from a pye57 header, defensively.

    pye57 has exposed pose through several shapes across versions
    (`rotation_matrix`, a `rotation` quaternion, a nested `pose` dict). Probe
    for each rather than pinning a version: getting this wrong does not raise,
    it silently places the whole station in the wrong spot, which is the most
    expensive class of bug in this pipeline.
    """
    import numpy as np

    t = np.zeros(3, np.float64)
    for attr in ("translation",):
        v = getattr(header, attr, None)
        if v is not None:
            arr = np.asarray(_as_xyz(v), np.float64).reshape(-1)
            if arr.size == 3:
                t = arr
            break

    rot = getattr(header, "rotation_matrix", None)
    if rot is not None:
        R = np.asarray(rot, np.float64).reshape(3, 3)
        return ScanPose(translation=t, rotation=R)

    q = getattr(header, "rotation", None)
    if q is not None:
        arr = np.asarray(_as_quat(q), np.float64).reshape(-1)
        if arr.size == 4:
            return ScanPose(translation=t, rotation=quat_to_matrix(arr))

    return ScanPose(translation=t, rotation=np.eye(3, dtype=np.float64))


def _as_xyz(v: Any) -> Any:
    if isinstance(v, dict):
        return [v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
    return v


def _as_quat(v: Any) -> Any:
    """Normalise to [w, x, y, z]. E57 declares the scalar part as `w`, and a
    dict form is unambiguous; an array form is taken as w-first, which is what
    pye57 documents."""
    if isinstance(v, dict):
        return [v.get("w", 1.0), v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
    return v


def quat_to_matrix(q: Any) -> Any:
    """[w, x, y, z] -> 3x3 rotation matrix."""
    import numpy as np

    w, x, y, z = (float(c) for c in np.asarray(q, np.float64).reshape(4))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _header_fields(header: Any) -> list[str]:
    """Field names in a scan's point prototype, across pye57 versions."""
    for attr in ("point_fields", "scan_fields"):
        v = getattr(header, attr, None)
        if v:
            return [str(x) for x in v]
    proto = getattr(header, "points", None)
    if proto is not None:
        try:
            return [str(proto.get(i).elementName()) for i in range(proto.childCount())]
        except Exception:  # noqa: BLE001 — probing an optional API shape
            pass
    return []
