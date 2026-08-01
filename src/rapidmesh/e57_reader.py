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

Known limit (v0.1)
------------------
`pye57.read_scan_raw` materialises a whole scan's fields at once. A 100 M-point
scan with XYZ + RGB + intensity is roughly 2.4 GB in flight. A chunked reader
is planned; until then `read_scan` accepts `max_points` and subsamples on the
lattice (keeping whole rows/columns, never random points, so the lattice
structure survives) rather than dying.
"""

from __future__ import annotations

import math
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
    return _build(raw, pose, sid, max_points)


def _build(
    raw: dict[str, Any],
    pose: ScanPose,
    station_id: str,
    max_points: int | None,
) -> StructuredScan:
    """Turn one scan's raw field dict into a `StructuredScan`.

    Split out from `read_scan` with no pye57 in its signature so the synthetic
    fixtures in `synthetic.py` exercise this exact code path — the lattice
    detection and validity handling below is where the bugs live, and it would
    be untestable without real client scans otherwise.
    """
    import numpy as np

    xyz_local, rng = _positions(raw)
    valid = _validity(raw, rng)
    xyz_local, rng = _resolve_frame(xyz_local, rng, raw, pose, valid)
    row, col, lattice = _lattice(raw, xyz_local, rng, valid)

    keep = valid
    if max_points is not None and int(keep.sum()) > max_points:
        keep = keep & _lattice_stride(row, col, lattice, int(keep.sum()), max_points)

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
        )
    )


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
    already-transformed world coordinates alongside a non-identity pose is a
    known interoperability wart, and it is catastrophic here — every angle in
    the pipeline is measured from the scanner origin, so a scan whose points are
    offset by the pose produces a wrong lattice, wrong incidence thresholds, and
    a mesh in the wrong place. Nothing raises. It just comes out subtly wrong.

    The discriminator is a physical property of a rotating-head scanner: it
    sweeps a **vertical plane** per head position, so within one lattice column
    every sample must share the same azimuth. Under the correct frame that
    spread is essentially zero; under a shifted origin it is large and obvious.

    Only checkable when the file carries row/column indices, which is also the
    only case where the lattice is precise enough for the error to matter.
    Spherical coordinates are unambiguously local by definition and are left
    alone.
    """
    import numpy as np

    if all(f in raw for f in _SPHERICAL):
        return xyz, rng
    if not all(f in raw for f in _ROWCOL):
        return xyz, rng
    t = np.asarray(pose.translation, np.float64)
    if float(np.linalg.norm(t)) < 0.01:
        return xyz, rng

    col = np.asarray(raw["columnIndex"], np.int64)
    idx = np.nonzero(valid)[0]
    if idx.size > 100_000:
        idx = idx[:: idx.size // 100_000]
    if idx.size < 64:
        return xyz, rng

    as_local = _column_azimuth_spread(xyz[idx], col[idx])
    as_world = _column_azimuth_spread(xyz[idx] - t, col[idx])

    # Require a decisive margin. A borderline result means the test did not
    # discriminate — a partial-FOV scan, or a station whose pose translation is
    # small relative to the scene — and silently rewriting coordinates on weak
    # evidence would be worse than the problem.
    if as_world < as_local * 0.2:
        out = (xyz - t).astype(np.float64)
        return out, np.linalg.norm(out, axis=1)
    return xyz, rng


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
    """Median azimuth increment per column, immune to the +/-pi branch cut."""
    import numpy as np

    same_row = np.diff(row) == 0
    dcol = np.diff(col)
    ok = same_row & (dcol > 0)
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
