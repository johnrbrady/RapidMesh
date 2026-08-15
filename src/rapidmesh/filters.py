"""
Cleanup — removing what is provably not surface, and nothing else.

The rule this module is written to
----------------------------------
**A filter must be able to say why a point is not surface.** Anything weaker
than that is a threshold, and thresholds eat handrails. The brief flags this
exact risk ("threshold tuning still needs work to avoid stripping out real fine
detail (door frames, railings) while catching genuine noise"), so every filter
here is either grounded in a physical argument or it is not here at all.

Two filters clear that bar. A third does not, and the reasoning is recorded
below because "we deliberately did not build this" is worth more than a
half-working version of it.

1. `isolation_despeckle` — **sound.** A real surface sample has neighbours on
   the surface. A flying point, a mixed pixel at an edge, or airborne dust does
   not. Judging *support* rather than deviation from a local median is the key
   difference from Cairn's despeckle: a handrail sample deviates enormously
   from the median (which is the wall 3 m behind it) but has excellent support
   from the rail samples above and below it, so it survives here and does not
   survive there.

2. `carve_movers` — **sound.** Inherited from Cairn's mover block, which is the
   best idea in that file. If another station recorded a return *beyond* a
   point's position along the same ray, nothing solid was there when that
   station captured. Solid surfaces block line of sight; a door frame is never
   seen through. This is geometry, not a threshold.

3. Single-scan mover detection — **not implemented, on purpose.** The brief
   proposes finding movers from one scan's range spikes ("close-far-close as
   the beam sweeps across"). The trouble is that a moving person and a static
   fence post produce range images that are not reliably separable: both are
   narrow foreground objects with clean discontinuities on every side, both are
   consistent down a column (a rotating-head scanner captures a whole vertical
   profile in a few milliseconds, so the mover is effectively frozen within one
   column), and both differ from their surroundings by the same kind of jump.
   Any threshold that removes the person also removes the post — which is the
   failure mode the brief is worried about, arrived at from the direction it
   did not expect. **Distinguishing them requires a second observation**, which
   is exactly what `carve_movers` uses. The honest position is that a single
   isolated scan cannot have its movers removed reliably, and pretending
   otherwise ships a filter that quietly deletes survey detail.

   What *can* be done single-scan, and is done, is removing points that are not
   plausibly on any surface at all — which is `isolation_despeckle`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .grid import CoarseRangeGrid, ScanGrid, select
from .types import FilterStats, StructuredScan

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    BOOL = npt.NDArray[np.bool_]
    F32 = npt.NDArray[np.float32]

# 8-neighbourhood on the lattice.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def columns_wrap(scan: StructuredScan) -> bool:
    """Does column `cols - 1` neighbour column `0`?

    True for a full 360 degree sweep, false for a partial-FOV scan. Getting
    this wrong joins the two ends of a 90 degree scan into a ring and produces
    a sheet of triangles across the middle of the room, which is obvious once
    seen and invisible until then.
    """
    import math

    span = abs(scan.lattice.az_step) * scan.lattice.cols
    return bool(abs(span - 2.0 * math.pi) < abs(scan.lattice.az_step) * 2.0)


def isolation_despeckle(
    grid: ScanGrid,
    tol_abs: float = 0.02,
    tol_rel: float = 0.006,
    min_support: int = 2,
    band_rows: int = 256,
) -> BOOL:
    """Keep-mask: drop samples with too few agreeing lattice neighbours.

    A neighbour "agrees" when its range is within `tol_abs + tol_rel * r` of
    this sample's. The relative term matters: 20 mm is a generous tolerance at
    3 m and a meaningless one at 60 m, where the beam footprint alone is wider
    than that.

    `min_support = 2` is deliberately low. One agreeing neighbour could be a
    second dust particle; two on an 8-neighbourhood means the sample sits on
    something with extent. Raising it to 4 or 5 starts removing genuine
    single-sample-wide features, which is the trade the brief warns about — so
    the default sits on the permissive side and the deviation report is where
    the consequences show up.

    Samples with fewer than `min_support` *available* neighbours (lattice
    edges, or an area so sparse there is nothing to judge against) are kept.
    Absence of evidence is not evidence here.
    """
    import numpy as np

    scan = grid.scan
    n = len(scan)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    wrap = columns_wrap(scan)
    rng = scan.rng

    for b0, b1 in _judged_bands(grid.rows, band_rows, halo=1):
        lo, hi = max(b0 - 1, 0), min(b1 + 1, grid.rows)
        idx = grid.dense_rows(lo, hi)
        if idx.size == 0:
            continue
        R = np.where(idx >= 0, rng[np.maximum(idx, 0)], np.float32(np.inf))
        tol = tol_abs + tol_rel * R

        support = np.zeros(R.shape, np.int8)
        available = np.zeros(R.shape, np.int8)
        for dr, dc in _NEIGHBOURS:
            Rn = _shift(R, dr, dc, wrap)
            has = np.isfinite(Rn)
            available += has
            # `Rn - R` is inf - inf = NaN wherever both cells are empty, and a
            # NaN comparison is False, so the result would be correct but the
            # warning is real noise. Subtract only where both exist.
            delta = np.abs(np.subtract(Rn, R, out=np.zeros_like(R), where=has))
            support += has & (delta <= tol)

        s0, s1 = b0 - lo, b1 - lo
        drop = (
            (idx[s0:s1] >= 0)
            & (available[s0:s1] >= min_support)
            & (support[s0:s1] < min_support)
        )
        keep[idx[s0:s1][drop]] = False

    return keep


def carve_movers(
    scan: StructuredScan,
    others: list[CoarseRangeGrid],
    clear_margin: float = 0.4,
    relative_margin: float = 1.02,
    min_votes: int = 1,
    chunk: int = 2_000_000,
) -> BOOL:
    """Keep-mask: drop samples that `min_votes` other stations saw beyond.

    `others` are coarse range grids from *neighbouring* stations — build them
    with `CoarseRangeGrid.build`.

    **`min_votes` is the important parameter and Cairn does not have one.**
    Cairn ORs its two nearest neighbours, so a single station disagreeing is
    enough to delete a point. That is right for a mover, which every station
    sees through, and wrong for a thin object standing off a wall: a 60 mm
    handrail 1 m in front of a wall is genuinely invisible from some angles, so
    one station legitimately reports the wall behind it and the rail gets
    carved away. The synthetic harness measures this — with one vote the
    handrail loses about 4 % of its samples.

    Requiring two independent stations to agree costs a little mover recall and
    buys back most of that, because parallax blind spots rarely coincide
    between two setups while a genuinely absent object is absent from all of
    them. Set to 1 when only one neighbour is available, in which case the
    trade is unavoidable and worth stating in the report.

    Chunked because the world-coordinate array is f64 and a native-resolution
    scan can carry 100 M samples: 2.4 GB in one allocation, versus 48 MB per
    chunk here.
    """
    import numpy as np

    n = len(scan)
    keep = np.ones(n, dtype=bool)
    if n == 0 or not others:
        return keep
    votes_needed = max(1, min(min_votes, len(others)))

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        world = scan.pose.local_to_world(scan.xyz[s:e])
        votes = np.zeros(e - s, dtype=np.int16)
        for other in others:
            votes += other.seen_through(world, clear_margin, relative_margin)
        keep[s:e] = votes < votes_needed

    return keep


def restore_parallax_carve(
    grid: ScanGrid,
    dropped: BOOL,
    tol_abs: float = 0.02,
    tol_rel: float = 0.006,
    min_support: int = 5,
    band_rows: int = 256,
) -> BOOL:
    """Un-drop carved samples whose lattice neighbourhood mostly survived.

    This is what makes one-vote carving safe, and it is the reason RapidMesh
    does not have to choose between mover recall and thin detail.

    The two failure modes look identical point-by-point and completely
    different in a neighbourhood:

    * A **mover** is carved as a solid blob. Its interior samples are
      surrounded by other carved samples, because every station saw through all
      of it.
    * A **thin object lost to parallax** is carved in scattered isolated spots
      inside an otherwise intact surface — the neighbouring rail samples were
      visible from the other station and survived.

    So: restore a carved sample when at least `min_support` of its eight
    neighbours both survived and agree with it in range. A mover's core cannot
    pass that test; a nicked handrail passes it easily.

    Applied **once**, not iterated. Iterating would let restoration creep
    inwards from a mover's boundary one ring per pass and eventually rebuild
    the whole object.

    Returns the revised drop mask.
    """
    import numpy as np

    scan = grid.scan
    out = dropped.copy()
    if not np.any(dropped):
        return out
    wrap = columns_wrap(scan)
    rng = scan.rng

    for b0, b1 in _judged_bands(grid.rows, band_rows, halo=1):
        lo, hi = max(b0 - 1, 0), min(b1 + 1, grid.rows)
        idx = grid.dense_rows(lo, hi)
        if idx.size == 0:
            continue
        present = idx >= 0
        safe = np.maximum(idx, 0)
        R = np.where(present, rng[safe], np.float32(np.inf))
        # A neighbour only counts as evidence if it survived the carve.
        alive = present & ~dropped[safe]
        Rlive = np.where(alive, R, np.float32(np.inf))
        tol = tol_abs + tol_rel * R

        support = np.zeros(R.shape, np.int8)
        for dr, dc in _NEIGHBOURS:
            Rn = _shift(Rlive, dr, dc, wrap)
            has = np.isfinite(Rn)
            delta = np.abs(np.subtract(Rn, R, out=np.zeros_like(R), where=has))
            support += has & (delta <= tol)

        s0, s1 = b0 - lo, b1 - lo
        sub = idx[s0:s1]
        restore = (sub >= 0) & dropped[np.maximum(sub, 0)] & (support[s0:s1] >= min_support)
        out[sub[restore]] = False

    return out


def clean(
    scan: StructuredScan,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    restore: bool = True,
    **kwargs: float | int,
) -> tuple[StructuredScan, FilterStats]:
    """Run the sound filters in order and report what each one removed.

    Order matters: despeckle first, so the flying points in front of a surface
    do not survive into carving and get counted there instead. The numbers are
    only useful for tuning if each stage's contribution is attributed to it.

    Defaults are one-vote carving followed by `restore_parallax_carve`. On the
    three-station synthetic fixture that combination scores 96.6 % mover recall
    with 100 % survival of both the 60 mm handrail and the 300 mm column —
    better recall than two-vote carving and better detail retention than
    one-vote carving alone. Run `tools/bench_synthetic.py` after changing any
    of these; the numbers move.
    """
    import numpy as np

    stats_kw: dict[str, int] = {}
    n0 = len(scan)
    current = scan

    if despeckle:
        grid = ScanGrid.build(current)
        keep = isolation_despeckle(
            grid,
            tol_abs=float(kwargs.get("tol_abs", 0.02)),
            tol_rel=float(kwargs.get("tol_rel", 0.006)),
            min_support=int(kwargs.get("min_support", 2)),
        )
        stats_kw["dropped_despeckle"] = int(np.count_nonzero(~keep))
        current = select(grid.scan, keep)

    if others:
        initial_drop = ~carve_movers(
            current,
            others,
            clear_margin=float(kwargs.get("clear_margin", 0.4)),
            relative_margin=float(kwargs.get("relative_margin", 1.02)),
            min_votes=int(kwargs.get("min_votes", 1)),
        )
        drop = initial_drop
        if restore and np.any(initial_drop):
            drop = restore_parallax_carve(
                ScanGrid.build(current),
                drop,
                min_support=int(kwargs.get("restore_support", 5)),
            )
            stats_kw["restored_from_carve"] = int(
                np.count_nonzero(initial_drop & ~drop)
            )
        stats_kw["dropped_mover_carve"] = int(np.count_nonzero(drop))
        current = select(current, ~drop)

    source_count = scan.source_sample_count
    input_points = (
        source_count
        if source_count is not None
        else n0 + scan.dropped_no_return + scan.dropped_other
    )
    return current, FilterStats(
        input_points=input_points,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
        **stats_kw,
    )


# --------------------------------------------------------------------------
# banding helpers
# --------------------------------------------------------------------------


def _judged_bands(rows: int, band_rows: int, halo: int) -> list[tuple[int, int]]:
    """Row ranges to *judge*, tiling the lattice with no overlap.

    Distinct from `ScanGrid.bands`, which produces overlapping ranges for
    triangulation. Here the halo is loaded but never judged, so bands must not
    overlap or a sample gets evaluated twice — harmless for a keep-mask, but it
    would double-count in the statistics, and the statistics are how thresholds
    get tuned.
    """
    step = max(band_rows, 1)
    return [(b, min(b + step, rows)) for b in range(0, rows, step)]


def _shift(A: F32, dr: int, dc: int, wrap: bool) -> F32:
    """Lattice-shifted copy of `A`, filled with +inf where nothing exists.

    Rows never wrap: the top and bottom of a scan are the zenith and nadir, not
    each other. Columns wrap only for a full 360 degree sweep. Cairn's original
    despeckle padded both axes and made points near one pole judge points near
    the other; the asymmetry here is deliberate.
    """
    import numpy as np

    out = A
    if dc:
        out = np.roll(out, dc, axis=1)
        if not wrap:
            if dc > 0:
                out[:, :dc] = np.inf
            else:
                out[:, dc:] = np.inf
    if dr:
        out = np.roll(out, dr, axis=0)
        if dr > 0:
            out[:dr, :] = np.inf
        else:
            out[dr:, :] = np.inf
    return out
