"""
Synthetic structured scans with analytic ground truth.

Why this exists before any real data is loaded
----------------------------------------------
"Survey-grade" is a number, and you cannot measure a number against a real
scan, because a real scan has no ground truth — the points *are* the only
truth available, so measuring a mesh against them tells you how well you fitted
the noise, not how accurate you are.

A synthetic scan solves that. The scene is defined analytically, so for every
ray we know the exact surface distance to the micron. That makes three
otherwise-unanswerable questions answerable:

1. **How accurate is the mesh, really?** Compare vertices to the analytic
   surface, not to the input points.
2. **Does the motion filter work?** The mover is scripted, so every sample it
   produced is labelled. Recall and false-positive rate are both exact.
3. **Does filtering eat real detail?** The scene deliberately contains the
   features the brief worries about — a doorway with a 4 m depth jump behind
   it, a thin handrail, and a shallow door reveal. If a threshold is stripping
   them, the number moves.

The mover is the important fixture. A terrestrial scanner sweeps azimuth over
time, so a moving object is sampled at a *different position in each column*.
Modelling the mover's position as a function of column index reproduces the
real smeared artefact exactly, rather than approximating it with random noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from .types import LatticeInfo, LatticeSource, ScanPose, StructuredScan

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]
    BOOL = npt.NDArray[np.bool_]

# Surface identifiers carried through the ray cast so a report can attribute
# error to a specific feature ("the handrail is 6 mm out") instead of quoting
# one number for the whole scene.
SURF_NONE = 0
SURF_ROOM = 1
SURF_BEYOND_DOOR = 2
SURF_COLUMN = 3
SURF_RAIL = 4
SURF_MOVER = 5


# Fields of `RoomScene` that are **not** lengths. `scaled` multiplies everything
# else, which puts the burden of correctness on this one list: a metre-valued
# field added later is scaled automatically, and a non-metric one added later
# must be declared here or it gets scaled and breaks loudly. The alternative —
# listing the fields to scale — fails the other way, silently leaving a new
# length unscaled and quietly ending the similarity the ladder depends on.
# `tests/test_extent_ladder.py` asserts the two sets cover every field.
NON_METRIC_SCENE_FIELDS = frozenset({"mover"})


@dataclass
class RoomScene:
    """An axis-aligned room, viewed from inside, with the awkward features.

    Dimensions in metres, scanner at the origin unless `scanner` says
    otherwise. Defaults describe a 8 x 6 x 3 m room with the scanner slightly
    off-centre, because a perfectly centred scanner produces a suspiciously
    symmetric lattice that hides indexing bugs.
    """

    half_x: float = 4.0
    half_y: float = 3.0
    floor_z: float = -1.6
    ceil_z: float = 1.4
    scanner: tuple[float, float, float] = (0.35, -0.2, 0.0)

    # Doorway in the +x wall: a depth discontinuity of `door_depth` metres
    # across a single lattice cell. This is the feature §1 of the brief calls
    # out — bridging it produces the classic stretched triangle.
    door_half_width: float = 0.45
    door_bottom: float = -1.6
    door_top: float = 0.45
    door_depth: float = 4.0

    # A 60 mm vertical handrail. Thin enough that an over-eager despeckle
    # threshold removes it, which is the failure the brief warns about.
    rail_centre: tuple[float, float] = (-2.4, 1.9)
    rail_radius: float = 0.03

    # A 300 mm structural column.
    column_centre: tuple[float, float] = (1.6, 1.4)
    column_radius: float = 0.15

    # Mover: a person-sized box crossing the room during the sweep. `mover_t`
    # runs 0..1 across the azimuth sweep, so the object is at a different place
    # in every column — the real artefact, not simulated noise.
    mover: bool = True
    mover_half: tuple[float, float, float] = (0.25, 0.2, 0.85)
    mover_from: tuple[float, float] = (-2.0, -2.2)
    mover_to: tuple[float, float] = (2.2, -1.4)
    mover_z: float = -0.7

    def scaled(self, factor: float) -> RoomScene:
        """A geometrically similar room, `factor` times the size — WP-1.G0.

        Every length scales, **the scanner position included**. That is the part
        worth stating: the scanner stands inside the room, so leaving it put
        while the walls move would change every incidence angle and the scene
        would stop being a similarity. Scaling it keeps the rays leaving at the
        same angles and hitting the same surfaces at scaled ranges, which is
        exactly what makes the extent ladder interpretable — the lattice hit
        pattern, the dropout draw and the sample count do not move, so extent is
        the only thing that varies.

        `range_noise` is deliberately not part of this. It is an instrument
        parameter, it belongs to `generate`, and a bigger room does not make a
        scanner noisier. The consequence is real and is reported rather than
        hidden: relative noise falls as the room grows, so the absolute
        thresholds downstream (`noise_floor`, `min_component_area`) see a
        different scene at each rung.
        """
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(
                f"extent scale must be a finite positive factor, got {factor!r}"
            )
        values: dict[str, Any] = {}
        for spec in fields(self):
            current = getattr(self, spec.name)
            if spec.name in NON_METRIC_SCENE_FIELDS:
                values[spec.name] = current
            elif isinstance(current, tuple):
                values[spec.name] = tuple(float(c) * factor for c in current)
            else:
                values[spec.name] = float(current) * factor
        return RoomScene(**values)


@dataclass
class SyntheticScan:
    """A generated scan plus everything needed to score a pipeline against it."""

    scan: StructuredScan
    true_range: F64
    """Exact analytic distance to the STATIC scene along each sample's ray.
    Where a sample hit the mover this is the distance to whatever the mover was
    hiding, so a correct pipeline drops the sample rather than reproducing it."""

    surface: npt.NDArray[np.int32]
    """Which surface each sample actually hit (SURF_* constants)."""

    is_mover: BOOL
    """Ground-truth label. Recall and false-positive rate are computed against
    this, so mover-filter tuning is a measurement rather than an argument."""

    direction: F64
    """(N,3) unit ray direction per sample, in scanner-local coordinates.
    Kept so accuracy can be measured along the ray, which is the axis a range
    error actually lives on."""

    scene: RoomScene = field(default_factory=RoomScene)

    @property
    def mover_count(self) -> int:
        import numpy as np

        return int(np.count_nonzero(self.is_mover))


def generate(
    scene: RoomScene | None = None,
    rows: int = 900,
    cols: int = 3600,
    el_min: float = -math.radians(60.0),
    el_max: float = math.radians(60.0),
    range_noise: float = 0.002,
    dropout: float = 0.01,
    seed: int = 7,
    station_id: str = "synthetic",
    pose: ScanPose | None = None,
) -> SyntheticScan:
    """Ray-cast a structured scan of `scene`.

    `range_noise` is the 1-sigma range error in metres; 2 mm is a realistic
    figure for a Trimble X7 at room distances. `dropout` is the fraction of
    cells with no return, which every real scan has and which the triangulator
    must survive without leaving square holes.

    Defaults are 900 x 3600 (0.1 deg) — fine enough to be a real test of the
    lattice code, small enough to run in a couple of seconds.
    """
    import numpy as np

    scene = scene or RoomScene()
    rng_gen = np.random.default_rng(seed)

    az_step = 2.0 * math.pi / cols
    el_step = (el_max - el_min) / max(rows - 1, 1)
    az = -math.pi + np.arange(cols, dtype=np.float64) * az_step
    el = el_min + np.arange(rows, dtype=np.float64) * el_step

    row_grid, col_grid = np.meshgrid(
        np.arange(rows, dtype=np.int64), np.arange(cols, dtype=np.int64), indexing="ij"
    )
    row_i, col_i = row_grid.ravel(), col_grid.ravel()

    A = az[col_i]
    E = el[row_i]
    ce = np.cos(E)
    d = np.stack([ce * np.cos(A), ce * np.sin(A), np.sin(E)], axis=1)

    origin = np.asarray(scene.scanner, np.float64)
    # Sweep time from the column index: column 0 is t=0, the last column is
    # t=1. This is what makes the mover smear the way a real one does.
    sweep_t = col_i.astype(np.float64) / max(cols - 1, 1)

    t_static, surf_static = _cast_static(scene, origin, d)
    t_mover = (  # noqa: SIM108 — the branch names the two cases; a ternary here reads worse
        _cast_mover(scene, origin, d, sweep_t)
        if scene.mover
        else np.full(d.shape[0], np.inf)
    )

    hit_mover = t_mover < t_static
    t = np.where(hit_mover, t_mover, t_static)
    surf = np.where(hit_mover, np.int32(SURF_MOVER), surf_static).astype(np.int32)

    valid = np.isfinite(t) & (t > 0.3)
    if dropout > 0:
        valid &= rng_gen.random(t.shape[0]) >= dropout

    measured = t + rng_gen.normal(0.0, range_noise, t.shape[0])

    row = row_i[valid].astype(np.int32)
    col = col_i[valid].astype(np.int32)
    dirs = d[valid]
    r = measured[valid]
    xyz = (dirs * r[:, None]).astype(np.float32)

    lattice = LatticeInfo(
        rows=rows,
        cols=cols,
        az_step=az_step,
        el_step=el_step,
        az0=-math.pi,
        el0=el_min,
        source=LatticeSource.SYNTHETIC,
    )
    pose = pose or ScanPose(
        translation=origin.copy(), rotation=np.eye(3, dtype=np.float64)
    )

    from .grid import sort_row_major

    scan = sort_row_major(
        StructuredScan(
            row=row,
            col=col,
            xyz=xyz,
            rng=r.astype(np.float32),
            pose=pose,
            lattice=lattice,
            rgb=_shade(surf[valid]),
            intensity=None,
            station_id=station_id,
            sample_id=np.nonzero(valid)[0].astype(np.int64),
            source_sample_count=int(valid.size),
            dropped_no_return=int(np.count_nonzero(~valid)),
        )
    )

    # sort_row_major may reorder; recompute the parallel truth arrays under the
    # same key so nothing silently desynchronises. Cheap, and a desynchronised
    # ground truth would invalidate every number the harness produces.
    key = row.astype(np.int64) * cols + col
    order = np.argsort(key, kind="stable")
    return SyntheticScan(
        scan=scan,
        true_range=t_static[valid][order],
        surface=surf[valid][order],
        is_mover=hit_mover[valid][order],
        direction=dirs[order],
        scene=scene,
    )


# --------------------------------------------------------------------------
# ray casting
# --------------------------------------------------------------------------


def _cast_static(scene: RoomScene, o: F64, d: F64) -> tuple[F64, npt.NDArray[np.int32]]:
    """Nearest static-surface hit per ray."""
    import numpy as np

    t_room, wall = _room_exit(scene, o, d)
    surf = np.where(t_room < np.inf, np.int32(SURF_ROOM), np.int32(SURF_NONE)).astype(np.int32)

    # Doorway: rays leaving through the +x wall inside the opening carry on and
    # land on a wall `door_depth` further out.
    hit = o + d * t_room[:, None]
    through = (
        (wall == _WALL_XMAX)
        & (np.abs(hit[:, 1]) <= scene.door_half_width)
        & (hit[:, 2] >= scene.door_bottom)
        & (hit[:, 2] <= scene.door_top)
    )
    if np.any(through):
        far_x = scene.half_x + scene.door_depth
        with np.errstate(divide="ignore", invalid="ignore"):
            t_far = (far_x - o[0]) / d[:, 0]
        use = through & np.isfinite(t_far) & (t_far > 0)
        t_room = np.where(use, t_far, t_room)
        surf = np.where(use, np.int32(SURF_BEYOND_DOOR), surf).astype(np.int32)

    t = t_room
    for centre, radius, sid in (
        (scene.column_centre, scene.column_radius, SURF_COLUMN),
        (scene.rail_centre, scene.rail_radius, SURF_RAIL),
    ):
        t_cyl = _cylinder(o, d, centre, radius, scene.floor_z, scene.ceil_z)
        closer = t_cyl < t
        t = np.where(closer, t_cyl, t)
        surf = np.where(closer, np.int32(sid), surf).astype(np.int32)

    return t, surf


_WALL_XMIN, _WALL_XMAX, _WALL_YMIN, _WALL_YMAX, _WALL_FLOOR, _WALL_CEIL = range(6)


def _room_exit(scene: RoomScene, o: F64, d: F64) -> tuple[F64, npt.NDArray[np.int32]]:
    """Distance at which a ray from inside leaves the box, and which face.

    Standard slab test, but taking the *smallest positive* per-axis exit rather
    than the usual entry/exit pair, because the origin is inside the volume.
    """
    import numpy as np

    lo = np.array([-scene.half_x, -scene.half_y, scene.floor_z], np.float64)
    hi = np.array([scene.half_x, scene.half_y, scene.ceil_z], np.float64)

    best = np.full(d.shape[0], np.inf)
    which = np.full(d.shape[0], _WALL_XMIN, np.int32)
    faces = (
        (0, lo[0], _WALL_XMIN),
        (0, hi[0], _WALL_XMAX),
        (1, lo[1], _WALL_YMIN),
        (1, hi[1], _WALL_YMAX),
        (2, lo[2], _WALL_FLOOR),
        (2, hi[2], _WALL_CEIL),
    )
    for axis, plane, tag in faces:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane - o[axis]) / d[:, axis]
        ok = np.isfinite(t) & (t > 1e-6)
        # Only a hit if the crossing point lies within the other two axes'
        # extents; without this a ray "exits" through the plane of a face it
        # never actually reaches.
        p = o + d * np.where(ok, t, 0.0)[:, None]
        for other in (0, 1, 2):
            if other == axis:
                continue
            ok &= (p[:, other] >= lo[other] - 1e-9) & (p[:, other] <= hi[other] + 1e-9)
        closer = ok & (t < best)
        best = np.where(closer, t, best)
        which = np.where(closer, np.int32(tag), which).astype(np.int32)
    return best, which


def _cylinder(
    o: F64, d: F64, centre: tuple[float, float], radius: float, z0: float, z1: float
) -> F64:
    """Nearest hit on a vertical finite cylinder, inf if missed."""
    import numpy as np

    ox, oy = o[0] - centre[0], o[1] - centre[1]
    dx, dy = d[:, 0], d[:, 1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    out = np.full(d.shape[0], np.inf)
    hit = (disc > 0) & (a > 1e-12)
    if not np.any(hit):
        return out
    sq = np.sqrt(np.where(hit, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - sq) / (2.0 * a)
        t1 = (-b + sq) / (2.0 * a)
    for t in (t0, t1):
        z = o[2] + d[:, 2] * t
        ok = hit & np.isfinite(t) & (t > 1e-6) & (z >= z0) & (z <= z1) & (t < out)
        out = np.where(ok, t, out)
    return out


def _cast_mover(scene: RoomScene, o: F64, d: F64, sweep_t: F64) -> F64:
    """Nearest hit on the moving box, whose centre depends on sweep time."""
    import numpy as np

    fx, fy = scene.mover_from
    tx, ty = scene.mover_to
    cx = fx + (tx - fx) * sweep_t
    cy = fy + (ty - fy) * sweep_t
    hx, hy, hz = scene.mover_half

    lo = np.stack([cx - hx, cy - hy, np.full_like(cx, scene.mover_z - hz)], axis=1)
    hi = np.stack([cx + hx, cy + hy, np.full_like(cx, scene.mover_z + hz)], axis=1)

    t_near = np.full(d.shape[0], -np.inf)
    t_far = np.full(d.shape[0], np.inf)
    for axis in (0, 1, 2):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[:, axis] - o[axis]) / d[:, axis]
            tb = (hi[:, axis] - o[axis]) / d[:, axis]
        t_lo = np.minimum(ta, tb)
        t_hi = np.maximum(ta, tb)
        parallel = np.abs(d[:, axis]) < 1e-12
        inside = (o[axis] >= lo[:, axis]) & (o[axis] <= hi[:, axis])
        t_near = np.where(parallel, np.where(inside, t_near, np.inf), np.maximum(t_near, t_lo))
        t_far = np.where(parallel, np.where(inside, t_far, -np.inf), np.minimum(t_far, t_hi))

    hit = (t_far >= t_near) & (t_far > 1e-6)
    t = np.where(t_near > 1e-6, t_near, t_far)
    return np.where(hit & np.isfinite(t) & (t > 1e-6), t, np.inf)


@dataclass(frozen=True)
class HaloWitnessFixture:
    """Purpose-built halo falsifier (PHASE1-HALO-CALCULUS.md §9).

    Places a mover and a thin restored feature (rail) across a documented
    band boundary so that the missing lower support row at ``halo=2`` is
    predicted to change a named keep/restore decision, while ``halo=3``
    and ``halo=4`` are predicted to match the in-memory reference.

    This package only *builds* the witness. The streamed halo matrix stays
    required-red until PLAN.md §5 item 6 connects band-local filtering.
    An ordinary fixture that matches at every halo is not a falsifier;
    this one is designed so that prediction can be checked when item 6
    exists.
    """

    scan: StructuredScan
    true_range: F64
    surface: npt.NDArray[np.int32]
    is_mover: BOOL
    direction: F64
    scene: RoomScene
    # Absolute lattice row that is the first row of band 1 when
    # ``intended_band_rows`` is used (core boundary between band 0 and 1).
    band_boundary_row: int
    intended_band_rows: int
    # Named decision the missing lower halo row at halo=2 is predicted to flip.
    predicted_decision_name: str
    halo_2_changes_decision: bool
    halo_3_and_4_match: bool

    @property
    def mover_count(self) -> int:
        import numpy as np

        return int(np.count_nonzero(self.is_mover))


def generate_halo_witness(
    *,
    rows: int = 16,
    cols: int = 360,
    intended_band_rows: int = 8,
    seed: int = 23,
    station_id: str = "halo-witness",
) -> HaloWitnessFixture:
    """Build the HALO §9 witness on a small synthetic lattice.

    Geometry choices (documented predictions, not measured results):

    * ``intended_band_rows=8`` on a 16-row lattice puts the sole band boundary
      at row 8. Band 0 owns rows ``[0, 8)``; band 1 owns ``[8, 16)``.
    * ``cols=360`` is the smallest full-sweep lattice that still returns the
      default 30 mm rail (coarser azimuth misses it entirely). The rail is the
      thin restored feature; the mover supplies carve/restore traffic across
      the same neighbourhood.
    * Restore support for a carved-then-restored sample at the last core row
      of band 0 reaches the triangulation down-row and the filter-chain halo
      below it; at ``halo=2`` that lowest support row is the one the
      composed-halo derivation says is missing.

    Predicted decision table (to be checked when item 6 exists):

    | halo | predicted vs in-memory |
    |------|------------------------|
    | 2    | differs at restore keep for the boundary-straddling rail sample |
    | 3, 4 | match |

    The named decision is ``restore_keep_at_band_boundary_rail``.
    """
    if intended_band_rows <= 0 or intended_band_rows >= rows:
        raise ValueError("intended_band_rows must be in 1 .. rows-1")
    boundary = intended_band_rows

    scene = RoomScene(
        mover=True,
        rail_centre=(-2.4, 1.9),
        rail_radius=0.03,
    )
    syn = generate(
        scene,
        rows=rows,
        cols=cols,
        dropout=0.0,
        range_noise=0.0,
        seed=seed,
        station_id=station_id,
    )
    return HaloWitnessFixture(
        scan=syn.scan,
        true_range=syn.true_range,
        surface=syn.surface,
        is_mover=syn.is_mover,
        direction=syn.direction,
        scene=syn.scene,
        band_boundary_row=boundary,
        intended_band_rows=intended_band_rows,
        predicted_decision_name="restore_keep_at_band_boundary_rail",
        halo_2_changes_decision=True,
        halo_3_and_4_match=True,
    )


def _shade(surf: npt.NDArray[np.int32]) -> npt.NDArray[np.uint8]:
    """Flat per-surface colour.

    Deliberately flat and unlit. Cairn multiplies a lambert term into vertex
    RGB and discards the normals; `00-PRODUCT-DEFINITION.md` §7 forbids that
    here, so even the test fixture keeps colour and lighting separate.
    """
    import numpy as np

    palette = np.array(
        [
            [0, 0, 0],          # SURF_NONE
            [200, 196, 188],    # SURF_ROOM
            [120, 130, 150],    # SURF_BEYOND_DOOR
            [170, 160, 150],    # SURF_COLUMN
            [90, 90, 95],       # SURF_RAIL
            [200, 80, 70],      # SURF_MOVER
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(surf, 0, len(palette) - 1)]
