"""
The azimuth wrap seam — WP-1.G1b Part B1, `PHASE1-DETERMINISM-SPEC.md` §6.

What was already covered: `columns_wrap` (`filters.py:83`) decides whether column
`cols-1` neighbours column `0`, and `tests/test_pipeline.py` asserts the
**predicate** — true for a full sweep, false for a quarter-FOV lattice.

What was not covered, and is here: **the geometry at the seam.** The wrap is
consumed in two places that nothing asserted. `_shift` rolls columns and only
masks the wrapped edge when `wrap` is false (`filters.py`), so despeckle and
restore judge *across* the seam; `_band_triangles` forms the B and D corners
with `np.roll(idx, -1, axis=1)` and invalidates the final column only when
`not wrap` (`triangulate.py:150-161`), so triangulation bridges it. Nothing
asserted that the quads joining `cols-1` to `0` are produced, produced once,
wound like their neighbours, or judged by the same filter decision.

Why a cylinder
--------------
The fixture is a right cylinder with the scanner on its axis. Range depends only
on elevation — `R / cos(e)` — so **every column is identical**. That is the whole
point: on a rotationally symmetric scene the seam column pair is geometrically
indistinguishable from every interior pair, so any difference between them is a
bug and cannot be a property of the scene. There is no range noise, deliberately;
noise would break the exact symmetry that makes "equals, exactly" assertable
instead of "equals, roughly".

Honesty condition (spec §6, standing rule 8)
--------------------------------------------
Reading `triangulate.py:150-161` and `filters.py`, the wrap handling looks
correct, so this suite is expected to pass on first run. If it does it is a
**regression test, not a bug-fix test**, and the report says so. It is not
presented as a proven-failing test.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from rapidmesh.filters import columns_wrap, isolation_despeckle
from rapidmesh.grid import ScanGrid
from rapidmesh.triangulate import triangulate
from rapidmesh.types import LatticeInfo, LatticeSource, ScanPose, StructuredScan

RADIUS = 5.0
ROWS = 41
COLS = 120
EL_HALF_DEG = 20.0

# A quarter-FOV lattice at the same angular pitch: 30 of the 120 columns, so the
# span is pi/2 and `columns_wrap` must read false.
PARTIAL_COLS = 30

# Band sizes for the invariance assertion. 512 exceeds the row count, so it is
# the single-band control; 4 and 7 cut the lattice into many bands and 7 does not
# divide 41 evenly, which is the case a fencepost error survives.
BAND_ROWS_MATRIX = (4, 7, 512)


def cylinder_scan(
    *,
    rows: int = ROWS,
    cols: int = COLS,
    emit_cols: int | None = None,
    radius: float = RADIUS,
    el_half_deg: float = EL_HALF_DEG,
    speckle: tuple[int, int] | None = None,
    speckle_push: float = 1.0,
) -> StructuredScan:
    """A right cylinder viewed from its axis — every column identical.

    `emit_cols` narrows the lattice without changing the angular pitch, which is
    how the partial-FOV case is built: the same scanner, a 90 degree sweep, so
    `columns_wrap` reads false for a reason the fixture actually has.

    `speckle` pushes one cell's range out by `speckle_push` metres, making it an
    isolated return with no agreeing neighbour. It is the only thing that breaks
    the rotational symmetry, and it exists so the despeckle assertion has a
    decision to compare rather than an all-true mask.
    """
    az_step = 2.0 * math.pi / cols
    used = cols if emit_cols is None else emit_cols
    el_half = math.radians(el_half_deg)
    el_step = (2.0 * el_half) / max(rows - 1, 1)
    az0, el0 = -math.pi, -el_half

    row_i, col_i = np.meshgrid(
        np.arange(rows, dtype=np.int64), np.arange(used, dtype=np.int64),
        indexing="ij",
    )
    row_i, col_i = row_i.ravel(), col_i.ravel()          # row-major, as required
    az = az0 + col_i * az_step
    el = el0 + row_i * el_step

    rng = radius / np.cos(el)
    if speckle is not None:
        hit = (row_i == speckle[0]) & (col_i == speckle[1] % used)
        rng = rng + hit * speckle_push

    xyz = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1
    ) * rng[:, None]

    return StructuredScan(
        row=row_i.astype(np.int32),
        col=col_i.astype(np.int32),
        xyz=xyz.astype(np.float32),
        rng=rng.astype(np.float32),
        pose=ScanPose(translation=np.zeros(3), rotation=np.eye(3)),
        lattice=LatticeInfo(
            rows=rows, cols=used, az_step=az_step, el_step=el_step,
            az0=az0, el0=el0, source=LatticeSource.SYNTHETIC,
        ),
        station_id="wrapseam",
        sample_id=(row_i * used + col_i).astype(np.int64),
        source_sample_count=int(row_i.size),
    )


# ---------------------------------------------------------------------------
# helpers over a triangle set
# ---------------------------------------------------------------------------


def _tri_columns(scan: StructuredScan, tris: Any) -> Any:
    return scan.col[tris].astype(np.int64)


def _pair_of(tri_cols: Any, cols: int) -> Any:
    """The (low, high) column pair each triangle spans, seam-aware.

    A quad's triangle touches exactly two adjacent columns. The seam pair is
    reported as `(cols - 1, 0)` rather than sorted, so it cannot be confused
    with the interior pair `(0, 1)`.
    """
    lo = tri_cols.min(axis=1)
    hi = tri_cols.max(axis=1)
    seam = (lo == 0) & (hi == cols - 1)
    out = np.stack([lo, hi], axis=1)
    out[seam] = [cols - 1, 0]
    return out


def _counts_by_pair(scan: StructuredScan, tris: Any) -> dict[tuple[int, int], int]:
    cols = scan.lattice.cols
    tri_cols = _tri_columns(scan, tris)
    distinct = np.array([len(set(r.tolist())) for r in tri_cols])
    assert set(distinct.tolist()) <= {2}, (
        "a lattice quad's triangle must touch exactly two columns; "
        f"saw {sorted(set(distinct.tolist()))}"
    )
    pairs = _pair_of(tri_cols, cols)
    out: dict[tuple[int, int], int] = {}
    for lo, hi in pairs:
        out[(int(lo), int(hi))] = out.get((int(lo), int(hi)), 0) + 1
    return out


def _seam_mask(scan: StructuredScan, tris: Any) -> Any:
    cols = scan.lattice.cols
    tri_cols = _tri_columns(scan, tris)
    return (tri_cols.min(axis=1) == 0) & (tri_cols.max(axis=1) == cols - 1)


@pytest.fixture(scope="module")
def full() -> tuple[StructuredScan, Any]:
    scan = cylinder_scan()
    grid = ScanGrid.build(scan)
    return scan, triangulate(grid)


# ---------------------------------------------------------------------------
# spec §6, assertion by assertion
# ---------------------------------------------------------------------------


def test_the_fixture_is_a_full_sweep_and_wraps() -> None:
    """Precondition. Every assertion below is vacuous if the fixture is not a
    full sweep, so the predicate is pinned before the geometry is."""
    scan = cylinder_scan()
    assert columns_wrap(scan) is True
    assert scan.lattice.cols == COLS
    assert len(scan) == ROWS * COLS


def test_1_seam_triangles_exist(full: tuple[StructuredScan, Any]) -> None:
    """§6 assertion 1 — triangles span column cols-1 and column 0."""
    scan, tris = full
    assert tris.shape[0] > 0
    assert int(_seam_mask(scan, tris).sum()) > 0


def test_2_seam_count_equals_an_interior_column_pair(
    full: tuple[StructuredScan, Any]
) -> None:
    """§6 assertion 2 — equal *exactly*, and equal for every interior pair.

    The stronger form is asserted on purpose: comparing the seam against one
    arbitrary interior pair would pass on a fixture that was not actually
    symmetric. Requiring every interior pair to agree first establishes the
    symmetry, and only then is the seam held to it.
    """
    scan, tris = full
    counts = _counts_by_pair(scan, tris)
    interior = {p: n for p, n in counts.items() if p != (COLS - 1, 0)}
    assert len(interior) == COLS - 1, sorted(interior)
    unique = set(interior.values())
    assert len(unique) == 1, f"interior pairs disagree: {sorted(unique)}"
    assert counts[(COLS - 1, 0)] == unique.pop()


def test_3_seam_winding_matches_the_interior(
    full: tuple[StructuredScan, Any]
) -> None:
    """§6 assertion 3 — the same facing test `build_mesh` applies.

    `build_mesh` orients by `dot(normal, -local_vertex)`, the scanner sitting at
    the local origin. Applied per face here: every triangle must face the
    scanner with the same sign, and the seam must not be the exception.
    """
    scan, tris = full
    verts = scan.xyz.astype(np.float64)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    facing = np.einsum("ij,ij->i", normal, -centroid)
    assert np.all(np.abs(facing) > 0.0), "a degenerate face has no orientation"

    seam = _seam_mask(scan, tris)
    assert np.any(seam) and np.any(~seam)
    signs_interior = set(np.sign(facing[~seam]).tolist())
    signs_seam = set(np.sign(facing[seam]).tolist())
    assert len(signs_interior) == 1, signs_interior
    assert signs_seam == signs_interior


def test_4_each_seam_triangle_appears_exactly_once(
    full: tuple[StructuredScan, Any]
) -> None:
    """§6 assertion 4 — no duplicate, which is what a double-counted wrap
    column would produce and what nothing previously checked."""
    scan, tris = full
    seam = tris[_seam_mask(scan, tris)]
    keys = np.sort(seam, axis=1)
    unique = np.unique(keys, axis=0)
    assert unique.shape[0] == seam.shape[0], (
        f"{seam.shape[0] - unique.shape[0]} seam triangle(s) duplicated"
    )
    whole = np.unique(np.sort(tris, axis=1), axis=0)
    assert whole.shape[0] == tris.shape[0], "duplicate triangles in the mesh"


def test_5_a_partial_fov_lattice_has_no_seam_triangle() -> None:
    """§6 assertion 5 — the failure mode is a sheet of triangles across the
    middle of the room, obvious once seen and invisible until then."""
    scan = cylinder_scan(emit_cols=PARTIAL_COLS)
    assert columns_wrap(scan) is False
    tris = triangulate(ScanGrid.build(scan))
    assert tris.shape[0] > 0, "the partial fixture produced no triangles at all"
    assert int(_seam_mask(scan, tris).sum()) == 0


@pytest.mark.parametrize("band_rows", BAND_ROWS_MATRIX)
def test_6_seam_count_is_invariant_across_band_sizes(band_rows: int) -> None:
    """§6 assertion 6 — bands are row ranges, so the seam lies *inside* every
    band; this asserts the two seams never interact."""
    scan = cylinder_scan()
    grid = ScanGrid.build(scan)
    tris = triangulate(grid, band_rows=band_rows)
    reference = triangulate(grid, band_rows=max(BAND_ROWS_MATRIX))
    assert int(_seam_mask(scan, tris).sum()) == int(
        _seam_mask(scan, reference).sum()
    )
    assert tris.shape[0] == reference.shape[0]


def test_6b_the_band_matrix_actually_splits_the_lattice() -> None:
    """Guard on the guard: invariance across three band sizes that all produced
    one band would assert nothing."""
    scan = cylinder_scan()
    grid = ScanGrid.build(scan)
    counts = {b: len(list(grid.bands(band_rows=b, overlap=1))) for b in BAND_ROWS_MATRIX}
    assert counts[512] == 1, counts
    assert counts[4] > 3 and counts[7] > 3, counts
    assert ROWS % 7 != 0, "pick a band size that does not divide the row count"


def test_7_seam_column_filter_decisions_match_an_interior_column() -> None:
    """§6 assertion 7 — despeckle judges across the seam exactly as inside.

    An isolated return is planted at column 0, whose 8-neighbourhood reaches
    across the seam into column `cols-1`, and again at an interior column. On a
    rotationally symmetric cylinder the two cases are the same case rotated, so
    rolling one keep-mask onto the other must reproduce it exactly. If `_shift`
    masked the wrapped edge when it should not, the column-0 neighbourhood would
    be truncated and the two would differ.
    """
    row = ROWS // 2
    interior_col = COLS // 2

    at_seam = cylinder_scan(speckle=(row, 0))
    at_interior = cylinder_scan(speckle=(row, interior_col))

    keep_seam = isolation_despeckle(ScanGrid.build(at_seam)).reshape(ROWS, COLS)
    keep_int = isolation_despeckle(ScanGrid.build(at_interior)).reshape(ROWS, COLS)

    # The planted point must actually be dropped, or the comparison is between
    # two all-true masks and proves nothing.
    assert not keep_seam[row, 0]
    assert not keep_int[row, interior_col]

    rolled = np.roll(keep_int, -interior_col, axis=1)
    assert np.array_equal(rolled, keep_seam), (
        "the seam column's despeckle decision differs from the same case "
        "rotated to an interior column"
    )


def test_7b_an_unspeckled_cylinder_keeps_every_column_identically() -> None:
    """The weaker companion, which pins the symmetry the rest relies on."""
    scan = cylinder_scan()
    keep = isolation_despeckle(ScanGrid.build(scan)).reshape(ROWS, COLS)
    first = keep[:, :1]
    assert np.array_equal(keep, np.repeat(first, COLS, axis=1)), (
        "columns of an unspeckled cylinder disagree; the fixture is not "
        "rotationally symmetric and every seam assertion is unsound"
    )
