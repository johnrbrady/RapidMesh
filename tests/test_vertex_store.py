"""
The station-wide vertex store leaves the production path — WP-B, DEC-022.

WP-10m's four-term decomposition names T1 as the station-wide floor: the mesh
vertex store `retained` (xyz, row, col, sample_id, rng, rgb) plus `cells`,
`keep_idx`, `vertex_tile` and the two membership masks — **783,268,950 B on
ordinal 20**, live from the `retained` stage to the end of QA, 1.53× the whole
working-memory budget on its own. No transient cut reaches it, because it is not
a transient: it is the pass holding the station so that three later steps can
index it.

Three things read it, and each is given a bounded source instead:

* **area reduction** indexed `retained.xyz` to compute a cross product per
  triangle. DEC-022 carries the triangle's own f32 area in the `tri` segment
  record instead, computed in Pass A on the band's positions;
* **tile assembly** indexed `retained.xyz[keep_idx]` for positions, colour and
  sample ids. A per-tile vertex spool, filled by one pass over `pos`, carries
  them;
* **the observation store** masked every field of `retained` by `final`. It is
  written band by band from `pos` against the surviving-cell bitset.

What must not move is everything else. The area field is the *same f32 value* —
same expression, same positions, same winding — so component totals, the cull
decision and the exactness ratio are bit-identical, not merely close. The global
vertex index stays rank among surviving cells, which is the ascending-cell
compaction order `build_mesh` already walked.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed

BAND_ROWS = 16
CHUNK_POINTS = 5_000
HALO = 3


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="vstore",
    )


# ---------------------------------------------------------------------------
# DEC-022 — the area field is the reference's own f32 summand
# ---------------------------------------------------------------------------


def test_the_tri_record_carries_an_f32_area(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Red before WP-B: `_TRI_FIELDS` has four int64 columns and no area.

    The width is asserted because the whole point is that this is the
    *reference's own float32 summand*. Storing it as f64 would make it a
    different number that happens to be close, and the equality below would then
    be testing this test's arithmetic rather than Pass A's.
    """
    from rapidmesh.segments_io import _TRI_FIELDS

    fields = dict((name, dtype) for name, dtype in _TRI_FIELDS)
    assert "area" in fields, (
        f"the tri record has no area column: {sorted(fields)}; DEC-022 requires "
        "Pass A to carry the triangle's own f32 area"
    )
    assert fields["area"] == "<f4", (
        f"the tri area column is {fields['area']}, not '<f4' — it must be the "
        "reference's own float32 summand, not a widened recomputation"
    )


def test_the_area_field_is_bitwise_the_old_cross_product(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """DEC-022's falsifier, at the only tolerance that means anything: none.

    Red before WP-B: `read_tri_segment` returns two arrays and there is no third
    to compare.

    `_block_areas` is the expression the culled component totals were built from
    — `0.5 * norm(cross(b - a, c - a))` on float32 positions. Pass A now computes
    that on the band's own positions, in the winding `_band_triangles` produced,
    before the merge canonicalises the triple. Compared as **raw uint32 bits**,
    because `cross(B-A, C-A)` and `cross(C-B, A-B)` differ by an ULP and an
    almost-equal test would let exactly that through — the trap
    `_block_indices`' docstring records having been measured once already.

    The segments are read off disk from a real streamed run rather than from a
    hand-driven Pass A, so what is compared is what the production path wrote.
    """
    from rapidmesh.pass_b_area import _block_areas
    from rapidmesh.segments_io import read_pos_segment, read_tri_segment

    work = tmp_path / "work"
    work.mkdir()
    result = mesh_station_streamed(
        station.scan, band_rows=BAND_ROWS, chunk_points=CHUNK_POINTS, halo=HALO,
        measure=False, out_dir=str(tmp_path / "out"), work_dir=str(work),
    )
    assert result.tiles is not None

    seg = work / "seg"
    pos_files = sorted(seg.glob("*.rmseg.pos"))
    tri_files = sorted(seg.glob("*.rmseg.tri"))
    assert pos_files and tri_files, "the run left no segments to read"

    # The whole retained set, keyed by cell — this is the array `_block_areas`
    # used to be handed as `retained.xyz`, rebuilt here only so the reference
    # expression has something to run against.
    cell_parts = []
    xyz_parts = []
    for path in pos_files:
        record = read_pos_segment(path)
        cell_parts.append(record["cell"])
        xyz_parts.append(record["xyz"])
    cells = np.concatenate(cell_parts)
    xyz = np.concatenate(xyz_parts).astype(np.float32)
    order = np.argsort(cells, kind="stable")
    cells, xyz = cells[order], xyz[order]

    compared = 0
    for path in tri_files:
        got = read_tri_segment(path)
        assert len(got) == 3, (
            f"read_tri_segment returns {len(got)} arrays; WP-B adds the area"
        )
        triple, _component, area = got
        if triple.shape[0] == 0:
            continue
        # The winding as written, not canonicalised: this is the same triple
        # order `band_triangle_indices` produced, which is what DEC-022 pins.
        slots = np.searchsorted(cells, triple).astype(np.int64)
        assert np.array_equal(cells[slots], triple), "a triangle names an unknown cell"
        expected = _block_areas(slots, xyz)
        assert np.array_equal(
            np.asarray(area, np.float32).view(np.uint32),
            expected.astype(np.float32).view(np.uint32),
        ), f"{path.name}: area bits differ from _block_areas"
        compared += int(triple.shape[0])

    assert compared > 0, "the fixture produced no triangles; the test proves nothing"


def test_the_area_pass_does_not_touch_positions(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """The cut itself, made impossible to hide.

    Red before WP-B: `stream_component_areas(runs, cells, xyz)` takes the
    station's positions and indexes them. After it, the signature has no
    positions to take — a reduction that cannot reach the vertex store cannot
    hold it alive.
    """
    import inspect

    from rapidmesh.pass_b_area import exact_component_areas, stream_component_areas

    for fn in (stream_component_areas, exact_component_areas):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"xyz", "cells"}, (
            f"{fn.__name__} still takes {sorted(names & {'xyz', 'cells'})}; the "
            "area reduction must read the record's own area field"
        )


# ---------------------------------------------------------------------------
# the tiled path never rebuilds the station
# ---------------------------------------------------------------------------


def test_the_tiled_path_never_builds_a_station_wide_vertex_store(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural assertion, as a detonator rather than a peak.

    `retained_scan_for_cells` is the one function that materialises the mesh
    vertex store. Replacing it with something that raises is stronger than
    watching a memory figure: a peak can be small because the fixture is small,
    but a call is a call. This is `test_tiles.py`'s `build_mesh` detonator
    applied one layer down.

    Red before WP-B: `pass_b_finalise` calls it on line 240 of `pass_b.py`.
    """
    import rapidmesh.pass_b as pass_b

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError(
            "the tiled path rebuilt the station-wide vertex store"
        )

    monkeypatch.setattr(pass_b, "retained_scan_for_cells", forbidden, raising=False)
    import rapidmesh.retained_io as retained_io

    monkeypatch.setattr(retained_io, "retained_scan_for_cells", forbidden)

    result = mesh_station_streamed(
        station.scan, band_rows=BAND_ROWS, chunk_points=CHUNK_POINTS, halo=HALO,
        measure=True, measure_samples=2_000,
        out_dir=str(tmp_path / "out"),
    )
    assert result.tiles is not None
    assert result.mesh is None
    assert result.tiles.triangle_count > 0


def test_assemble_tile_is_not_handed_the_station(
    station: synthetic.SyntheticScan
) -> None:
    """`assemble_tile` reads a vertex spool, not `retained` and `keep_idx`.

    Red before WP-B: its signature is
    `(tile_id, records, retained, keep_idx, vertex_tile, grid)`.
    """
    import inspect

    from rapidmesh.tile_assemble import assemble_tile

    names = set(inspect.signature(assemble_tile).parameters)
    assert not names & {"retained", "keep_idx", "vertex_tile"}, (
        f"assemble_tile still takes {sorted(names & {'retained', 'keep_idx', 'vertex_tile'})}"
        "; the tile must be assembled from its own spooled vertices"
    )


def test_the_generation_is_unchanged_by_the_cut(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """The identity guard: the mesh is the same mesh.

    Reassembly against the resident reference run is stronger than a stored
    digest — a digest proves "same as last time", this proves "same as the
    implementation the tiled path is a memory optimisation of". Positions,
    normals, colour and source identity, bitwise; the ledger; both QA reports.
    """
    from rapidmesh.equivalence import compare_mesh_results
    from rapidmesh.pipeline import mesh_station
    from rapidmesh.tile_equivalence import reconstitute_mesh

    resident = mesh_station(station.scan, measure=True, measure_samples=2_000)
    tiled = mesh_station_streamed(
        station.scan, band_rows=BAND_ROWS, chunk_points=CHUNK_POINTS, halo=HALO,
        measure=True, measure_samples=2_000,
        out_dir=str(tmp_path / "out"), window=(16, 64),
    )
    assert tiled.tiles is not None
    rebuilt = reconstitute_mesh(tiled.tiles)
    for name in ("origin", "vertices", "normals", "rgb", "source_sample_id"):
        assert np.array_equal(
            getattr(rebuilt, name), getattr(resident.mesh, name)
        ), name
    assert compare_mesh_results(resident, tiled, source_sha256="c" * 64).ok
    tiled.stats.require_balanced()
    assert tiled.stats == resident.stats
    for name in ("rms", "mean", "p95", "p99_9", "maximum"):
        assert getattr(tiled.deviation, name) == getattr(resident.deviation, name)
        assert getattr(tiled.mesh_to_source, name) == getattr(
            resident.mesh_to_source, name
        )
