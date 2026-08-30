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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .grid import CoarseRangeGrid, ScanGrid, concat, select, select_rows
from .types import FilterStats, StructuredScan

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    import numpy.typing as npt

    BOOL = npt.NDArray[np.bool_]
    F32 = npt.NDArray[np.float32]

# 8-neighbourhood on the lattice.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Composed read-halo for the streamed filter chain, in lattice rows.
#
# `PHASE1-HALO-CALCULUS.md` §5 derives 2 rows for despeckle -> carve -> restore
# on its own (1 + 0 + 1; the halos add, they do not max), and §6 raises the
# symmetric figure to 3 so the same band also owns correct final keep state for
# the row triangulation reaches down into. 3 is therefore the number a streamed
# producer passes to `e57_reader.iter_row_bands`, and the default here, so the
# filtering stage is never the reason a band diverges.
#
# Filtering alone is exact from 2 upwards; the third row is triangulation's, and
# `tests/test_band_local_clean.py` measures both facts rather than assuming them.
COMPOSED_FILTER_HALO = 3


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

    for plan in _judged_bands(grid.rows, band_rows, halo=1):
        _despeckle_judge(
            grid,
            keep,
            plan,
            wrap=wrap,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            min_support=min_support,
        )

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

    out = dropped.copy()
    if not np.any(dropped):
        return out
    wrap = columns_wrap(grid.scan)

    for plan in _judged_bands(grid.rows, band_rows, halo=1):
        _restore_judge(
            grid,
            dropped,
            out,
            plan,
            wrap=wrap,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            min_support=min_support,
        )

    return out


@dataclass(frozen=True)
class _Settings:
    """The tuned constants `clean` and `clean_bands` share.

    One definition rather than two sets of `kwargs.get` calls: the whole point
    of the band-local path is that it agrees with the in-memory one, and a
    default that drifts between them would look like a halo defect.
    """

    tol_abs: float = 0.02
    tol_rel: float = 0.006
    min_support: int = 2
    clear_margin: float = 0.4
    relative_margin: float = 1.02
    min_votes: int = 1
    restore_support: int = 5


def _settings(kwargs: dict[str, float | int]) -> _Settings:
    base = _Settings()
    return _Settings(
        tol_abs=float(kwargs.get("tol_abs", base.tol_abs)),
        tol_rel=float(kwargs.get("tol_rel", base.tol_rel)),
        min_support=int(kwargs.get("min_support", base.min_support)),
        clear_margin=float(kwargs.get("clear_margin", base.clear_margin)),
        relative_margin=float(kwargs.get("relative_margin", base.relative_margin)),
        min_votes=int(kwargs.get("min_votes", base.min_votes)),
        restore_support=int(kwargs.get("restore_support", base.restore_support)),
    )


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

    cfg = _settings(kwargs)
    stats_kw: dict[str, int] = {}
    n0 = len(scan)
    current = scan

    if despeckle:
        grid = ScanGrid.build(current)
        keep = isolation_despeckle(
            grid,
            tol_abs=cfg.tol_abs,
            tol_rel=cfg.tol_rel,
            min_support=cfg.min_support,
        )
        stats_kw["dropped_despeckle"] = int(np.count_nonzero(~keep))
        current = select(grid.scan, keep)

    if others:
        initial_drop = ~carve_movers(
            current,
            others,
            clear_margin=cfg.clear_margin,
            relative_margin=cfg.relative_margin,
            min_votes=cfg.min_votes,
        )
        drop = initial_drop
        if restore and np.any(initial_drop):
            drop = restore_parallax_carve(
                ScanGrid.build(current),
                drop,
                min_support=cfg.restore_support,
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
# band-local filtering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BandFilterResult:
    """One band's core-owned output. Halo rows contributed evidence only.

    `retained` carries **only** samples whose lattice row is in the core, so
    concatenating every band reproduces the scan exactly once, and the three
    counts can be summed into a ledger that stays exclusive
    (`PHASE1-HALO-CALCULUS.md` §7).
    """

    plan: BandPlan
    retained: StructuredScan
    dropped_despeckle: int
    dropped_mover_carve: int
    restored_from_carve: int


def iter_clean_bands(
    scan: StructuredScan,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    restore: bool = True,
    *,
    band_rows: int = 256,
    halo: int = COMPOSED_FILTER_HALO,
    bands: Sequence[BandPlan] | None = None,
    **kwargs: float | int,
) -> Iterator[BandFilterResult]:
    """Run `clean`'s filter chain one band at a time, yielding core-owned output.

    Same three stages in the same order as `clean` — despeckle, carve, restore —
    but each band judges only its own core and reads the rest as evidence. On a
    fixture the two paths produce the same retained scan and the same counts;
    that is measured in `tests/test_band_local_clean.py`, not asserted here.

    Why the intermediate window is wider than the core
    --------------------------------------------------
    Restore reads the *post-carve* state of the row above and below each core
    row, so despeckle and carve must both be correct over
    ``[core_start - 1, core_stop + 1)``; despeckle in turn reads plus or minus
    one row around each of those. That is the 1 + 0 + 1 composition of
    `PHASE1-HALO-CALCULUS.md` §5, and it is why a `halo` below 2 changes
    band-edge decisions while 2 and above do not. The default is
    `COMPOSED_FILTER_HALO` (3), because a streamed producer's band also has to
    own triangulation's down-reach row.

    `bands` overrides the computed plan, so a caller can drive filtering from
    the bands `e57_reader.iter_row_bands` actually emitted and keep the chunk
    axis separate from the band axis.

    Scope: this is PLAN.md §5 item 6 only. `others` must be complete neighbour
    grids, exactly as `pipeline.carve_grids` builds them today — streaming
    carve-grid construction is item 7 and is not attempted here.
    """
    import numpy as np

    cfg = _settings(kwargs)
    grid = ScanGrid.build(scan)
    # Scan-level predicate. Recomputing it per band would judge a band of a full
    # sweep as a partial-FOV scan and drop the wrap seam
    # (`PHASE1-HALO-CALCULUS.md` §3).
    wrap = columns_wrap(grid.scan)
    plans = (
        list(bands) if bands is not None else _judged_bands(grid.rows, band_rows, halo)
    )

    for plan in plans:
        band_grid = ScanGrid.build(
            select_rows(grid, plan.data_row_start, plan.data_row_stop)
        )
        band = band_grid.scan
        if len(band) == 0:
            continue
        b0, b1 = plan.core_row_start, plan.core_row_stop

        # Rows whose post-carve state restore is allowed to read for this core.
        w0 = max(b0 - 1, plan.data_row_start)
        w1 = min(b1 + 1, plan.data_row_stop)
        window = BandPlan(
            core_row_start=w0,
            core_row_stop=w1,
            data_row_start=max(w0 - 1, plan.data_row_start),
            data_row_stop=min(w1 + 1, plan.data_row_stop),
        )

        keep = np.ones(len(band), dtype=bool)
        if despeckle:
            _despeckle_judge(
                band_grid,
                keep,
                window,
                wrap=wrap,
                tol_abs=cfg.tol_abs,
                tol_rel=cfg.tol_rel,
                min_support=cfg.min_support,
            )
        core_band = (band.row >= b0) & (band.row < b1)
        dropped_despeckle = int(np.count_nonzero(core_band & ~keep))

        in_window = (band.row >= w0) & (band.row < w1)
        current_grid = ScanGrid.build(select(band, keep & in_window))
        current = current_grid.scan

        drop = np.zeros(len(current), dtype=bool)
        initial_drop = drop
        if others:
            initial_drop = ~carve_movers(
                current,
                others,
                clear_margin=cfg.clear_margin,
                relative_margin=cfg.relative_margin,
                min_votes=cfg.min_votes,
            )
            drop = initial_drop
            if restore and np.any(initial_drop):
                drop = initial_drop.copy()
                _restore_judge(
                    current_grid,
                    initial_drop,
                    drop,
                    BandPlan(
                        core_row_start=b0,
                        core_row_stop=b1,
                        data_row_start=w0,
                        data_row_stop=w1,
                    ),
                    wrap=wrap,
                    tol_abs=cfg.tol_abs,
                    tol_rel=cfg.tol_rel,
                    min_support=cfg.restore_support,
                )

        core_current = (current.row >= b0) & (current.row < b1)
        yield BandFilterResult(
            plan=plan,
            retained=select(current, core_current & ~drop),
            dropped_despeckle=dropped_despeckle,
            dropped_mover_carve=int(np.count_nonzero(core_current & drop)),
            restored_from_carve=int(
                np.count_nonzero(core_current & initial_drop & ~drop)
            ),
        )


def clean_bands(
    scan: StructuredScan,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    restore: bool = True,
    *,
    band_rows: int = 256,
    halo: int = COMPOSED_FILTER_HALO,
    bands: Sequence[BandPlan] | None = None,
    **kwargs: float | int,
) -> tuple[StructuredScan, FilterStats]:
    """Band-local `clean`: same arguments, same two return values.

    The ledger it returns is the same *partial* one `clean` returns — `retained`
    is left at 0 and filled in by `pipeline.mesh_station`, which is the only
    place that knows how many samples survived meshing. Matching that shape is
    deliberate: the equivalence harness compares this against the
    version-matched in-memory reference, and a ledger differing only in
    convention would read as a real divergence.

    Reassembling the retained scan here is a convenience for the equivalence
    gate, not the streamed design — it holds the whole retained set in memory.
    Incremental output is PLAN.md §5 item 8; consume `iter_clean_bands`
    directly to avoid it.
    """
    import numpy as np

    parts: list[StructuredScan] = []
    dropped_despeckle = 0
    dropped_mover_carve = 0
    restored_from_carve = 0
    for result in iter_clean_bands(
        scan,
        others,
        despeckle,
        restore,
        band_rows=band_rows,
        halo=halo,
        bands=bands,
        **kwargs,
    ):
        parts.append(result.retained)
        dropped_despeckle += result.dropped_despeckle
        dropped_mover_carve += result.dropped_mover_carve
        restored_from_carve += result.restored_from_carve

    retained = concat(parts) if parts else select(scan, np.zeros(len(scan), dtype=bool))

    source_count = scan.source_sample_count
    input_points = (
        source_count
        if source_count is not None
        else len(scan) + scan.dropped_no_return + scan.dropped_other
    )
    return retained, FilterStats(
        input_points=input_points,
        dropped_no_return=scan.dropped_no_return,
        dropped_other=scan.dropped_other,
        dropped_despeckle=dropped_despeckle,
        dropped_mover_carve=dropped_mover_carve,
        restored_from_carve=restored_from_carve,
    )


# --------------------------------------------------------------------------
# banding helpers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BandPlan:
    """One band's judged core and the padded rows it is allowed to read.

    Same four numbers, same meaning and same arithmetic as
    `e57_reader.RawRowBand`, so a plan computed here and a band produced by
    `iter_row_bands` describe the same window. The duplication is deliberate:
    `iter_row_bands` also carries the point data, and the filter stage needs the
    geometry of a band without depending on the reader.
    """

    core_row_start: int
    core_row_stop: int
    data_row_start: int
    data_row_stop: int


def _judged_bands(rows: int, band_rows: int, halo: int) -> list[BandPlan]:
    """Bands to *judge*, tiling the lattice cores with no overlap.

    Distinct from `ScanGrid.bands`, which produces overlapping ranges for
    triangulation. Here the halo is loaded but never judged, so **cores** must
    not overlap or a sample gets evaluated twice — harmless for a keep-mask,
    but it would double-count in the statistics, and the statistics are how
    thresholds get tuned.

    `halo` sizes the padded `data_row_*` window each core is loaded with; it
    deliberately does **not** move the core boundaries. Judge-once
    (`PHASE1-HALO-CALCULUS.md` §7) requires the core tiling to be independent of
    how much evidence a band reads, or the ledger stops being exclusive the
    moment the halo changes.

    Until this package the parameter was accepted and never read, and the ±1
    load was hardcoded at both call sites — so raising the halo was silently
    ineffective (`PHASE1-HALO-CALCULUS.md` §4 Trap 1). That is now wired, and
    §9's secondary falsifier ("a test asserting the halo parameter has no
    effect... if that test fails, the parameter has been wired up") is expected
    to fail from here on.
    """
    step = max(band_rows, 1)
    return [
        BandPlan(
            core_row_start=b,
            core_row_stop=min(b + step, rows),
            data_row_start=max(b - halo, 0),
            data_row_stop=min(min(b + step, rows) + halo, rows),
        )
        for b in range(0, rows, step)
    ]


def _despeckle_judge(
    grid: ScanGrid,
    keep: BOOL,
    plan: BandPlan,
    *,
    wrap: bool,
    tol_abs: float,
    tol_rel: float,
    min_support: int,
) -> None:
    """Write despeckle keep decisions for `plan`'s core, reading its data rows.

    One kernel, called by both the full-scan path and the band-local one, so
    "the streamed path agrees with the in-memory path" is a property of shared
    code rather than of two implementations kept in step by hand.

    Reads only rows inside `plan`'s data window. When that window is short of
    the ±1 the 8-neighbourhood needs, the missing row reads as absent rather
    than as evidence — which is exactly the divergence the composed halo exists
    to prevent, so it is visible in the output instead of being papered over.
    """
    import numpy as np

    d0, d1 = plan.data_row_start, plan.data_row_stop
    idx = grid.dense_rows(d0, d1)
    if idx.size == 0:
        return
    R = np.where(idx >= 0, grid.scan.rng[np.maximum(idx, 0)], np.float32(np.inf))
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

    s0, s1 = plan.core_row_start - d0, plan.core_row_stop - d0
    drop = (
        (idx[s0:s1] >= 0)
        & (available[s0:s1] >= min_support)
        & (support[s0:s1] < min_support)
    )
    keep[idx[s0:s1][drop]] = False


def _restore_judge(
    grid: ScanGrid,
    dropped: BOOL,
    out: BOOL,
    plan: BandPlan,
    *,
    wrap: bool,
    tol_abs: float,
    tol_rel: float,
    min_support: int,
) -> None:
    """Un-drop `plan`'s core samples whose neighbourhood survived the carve.

    `dropped` is the carve verdict being judged and is never written; `out` is
    the revised mask. Keeping them separate is what makes restoration a single
    pass rather than one that creeps inwards from a mover's boundary.
    """
    import numpy as np

    d0, d1 = plan.data_row_start, plan.data_row_stop
    idx = grid.dense_rows(d0, d1)
    if idx.size == 0:
        return
    present = idx >= 0
    safe = np.maximum(idx, 0)
    R = np.where(present, grid.scan.rng[safe], np.float32(np.inf))
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

    s0, s1 = plan.core_row_start - d0, plan.core_row_stop - d0
    sub = idx[s0:s1]
    restore = (sub >= 0) & dropped[np.maximum(sub, 0)] & (support[s0:s1] >= min_support)
    out[sub[restore]] = False


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
