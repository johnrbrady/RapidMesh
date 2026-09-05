"""
Round 4e — Pass B's per-component working set (ITEM-014).

Round 4c bounded Pass B's per-*sample* term. The Round 6 campaign then measured
what was left: working memory scaling with the station's **component** count,
putting 2 of 30 stations over the 512 MB budget. Phase 1 attributed it to three
per-component Python structures that Pass B built and did not need —

    read_component_table    409-422 B a row, measured across a tenfold range of
                            fragmentation, resident for the whole pass with no
                            reader after the alias was built
    write_finalised_...     a second row list, a lookup dict and a third list of
                            finalised rows, all live at once
    PassAResult.table       Pass A's union-find, id ints, triangle-count dict
                            and retired set, dead once Pass A wrote the table to
                            disk but reachable for the whole of Pass B

— plus one transient, `np.unique(np.concatenate(chunks))` in
`build_triangle_runs`, which was the only call in the pass that raised the
process peak at all on the worst station.

So these tests come in the same pairs the Round 4c ones do: the column-wise
finalise must produce **byte-identical** output to the row-wise form it
replaces, and Pass B must not build the row objects at all. The row-wise form
is kept here as the reference, exactly as `component_area_v1` is kept for the
area reduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.islands import pass_a_sweep
from rapidmesh.pass_b import build_triangle_runs, stream_component_areas
from rapidmesh.segments_io import (
    ComponentRecord,
    SegmentError,
    read_component_alias,
    read_component_table,
    read_tri_segment,
    write_component_table,
    write_finalised_component_table,
)
from rapidmesh.triangulate import MIN_COMPONENT_TRIANGLES

MIN_AREA = 0.005


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=80, cols=320,
        dropout=0.01, range_noise=0.002, seed=7, station_id="r4e",
    )


def _pass_a(station: synthetic.SyntheticScan, tmp_path: Path, band_rows: int = 16) -> Any:
    return pass_a_sweep(station.scan, work_dir=tmp_path, band_rows=band_rows, halo=3)


def _alias_row_wise(pass_a: Any) -> np.ndarray:
    """The pre-Round-4e alias build, kept as the reference."""
    records = read_component_table(pass_a.component_table_path)
    alias = np.full(len(records) + 1, -1, np.int64)
    for record in records:
        alias[record.component_id] = record.root
    return alias


def _finalised_row_wise(
    work_dir: Path,
    provisional_table: Path,
    area: Any,
    *,
    min_component_area: float,
    min_triangles: int,
) -> Path:
    """The pre-Round-4e row-wise finalise, kept as the reference."""
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


def _area(station: synthetic.SyntheticScan, pass_a: Any, tmp_path: Path) -> Any:
    # DEC-022: the reduction reads the record's own area, so this no longer
    # needs the retained scan or a cell index to hand it.
    runs, _ = build_triangle_runs(pass_a.segments, _alias_row_wise(pass_a), tmp_path)
    return stream_component_areas(runs).as_area_result(set())


# ---------------------------------------------------------------------------
# the alias, without the rows
# ---------------------------------------------------------------------------


def test_read_component_alias_matches_the_row_wise_build(
    station: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """Same array, including the -1 fill and the `rows + 1` length."""
    pass_a = _pass_a(station, tmp_path)
    expected = _alias_row_wise(pass_a)
    actual = read_component_alias(pass_a.component_table_path)
    assert actual.dtype == np.int64
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_read_component_alias_on_an_empty_table(tmp_path: Path) -> None:
    path = write_component_table(tmp_path, [])
    alias = read_component_alias(path)
    assert alias.shape == (1,)
    assert int(alias[0]) == -1


def test_pass_b_must_not_materialise_the_component_record_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required-red: if Pass B builds `ComponentRecord` rows again, fail.

    The per-component bound Round 4e claims is structural — read the two
    columns Pass B needs and rewrite the table column-wise — and this is the
    tripwire for that claim. A synthetic station has few enough components that
    a memory comparison would not catch the regression, so the tripwire is on
    the call rather than on a byte count.

    Patched on `segments_io`, which is where the row-wise reader lives.
    `pass_b_finalise` imports from it inside the function body, so a regression
    resolves the name at call time and trips this.
    """
    import rapidmesh.segments_io as segments_io

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "pass_b must not build ComponentRecord rows; "
            "use read_component_alias and the column-wise finalise"
        )

    monkeypatch.setattr(segments_io, "read_component_table", _boom)

    from rapidmesh.pipeline import mesh_station_streamed

    built = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=40, cols=160,
        dropout=0.01, range_noise=0.002, seed=3, station_id="r4e-red",
    )
    result = mesh_station_streamed(
        built.scan, band_rows=16, chunk_points=50_000, halo=3, measure=False,
    )
    assert result.stats.balanced
    assert result.mesh is not None
    assert result.mesh.vertex_count > 0


# ---------------------------------------------------------------------------
# the finalised table, column-wise
# ---------------------------------------------------------------------------


def test_the_column_wise_finalise_is_byte_identical_to_the_row_wise_form(
    station: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """Not "equivalent" — the same bytes, header digest included."""
    pass_a = _pass_a(station, tmp_path)
    area = _area(station, pass_a, tmp_path)

    column_dir = tmp_path / "column"
    row_dir = tmp_path / "row"
    column_dir.mkdir()
    row_dir.mkdir()

    written = write_finalised_component_table(
        column_dir, pass_a.component_table_path, area,
        min_component_area=MIN_AREA, min_triangles=MIN_COMPONENT_TRIANGLES,
    )
    reference = _finalised_row_wise(
        row_dir, pass_a.component_table_path, area,
        min_component_area=MIN_AREA, min_triangles=MIN_COMPONENT_TRIANGLES,
    )
    assert written.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("min_area", (0.0, -1.0, MIN_AREA, 1e9))
def test_the_column_wise_finalise_matches_at_every_cull_threshold(
    station: synthetic.SyntheticScan, tmp_path: Path, min_area: float
) -> None:
    """`min_component_area <= 0` keeps everything; a huge one culls everything."""
    pass_a = _pass_a(station, tmp_path)
    area = _area(station, pass_a, tmp_path)
    column_dir = tmp_path / "column"
    row_dir = tmp_path / "row"
    column_dir.mkdir()
    row_dir.mkdir()

    written = write_finalised_component_table(
        column_dir, pass_a.component_table_path, area,
        min_component_area=min_area, min_triangles=MIN_COMPONENT_TRIANGLES,
    )
    reference = _finalised_row_wise(
        row_dir, pass_a.component_table_path, area,
        min_component_area=min_area, min_triangles=MIN_COMPONENT_TRIANGLES,
    )
    assert written.read_bytes() == reference.read_bytes()


def test_the_finalise_refuses_unsorted_area_roots(
    station: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """`searchsorted` replaced a dict, so the ordering is now load-bearing.

    An unsorted `root_ids` would mis-assign areas to components rather than
    fail, which is the one failure mode the dict could not have.
    """
    pass_a = _pass_a(station, tmp_path)
    area = _area(station, pass_a, tmp_path)
    if area.root_ids.size < 2:
        pytest.skip("needs at least two components to unsort")
    area.root_ids[[0, 1]] = area.root_ids[[1, 0]]
    with pytest.raises(SegmentError, match="strictly increasing"):
        write_finalised_component_table(
            tmp_path / "x", pass_a.component_table_path, area,
            min_component_area=MIN_AREA, min_triangles=MIN_COMPONENT_TRIANGLES,
        )


# ---------------------------------------------------------------------------
# the mesh vertex store cells, without the three-copy transient
# ---------------------------------------------------------------------------


def test_triangle_run_cells_are_the_sorted_unique_named_set(
    station: synthetic.SyntheticScan, tmp_path: Path
) -> None:
    """The in-place dedup must return exactly what `np.unique` returned."""
    pass_a = _pass_a(station, tmp_path)
    _runs, cells = build_triangle_runs(
        pass_a.segments, _alias_row_wise(pass_a), tmp_path
    )
    assert cells.dtype == np.int64
    assert cells.size > 0
    assert bool(np.all(np.diff(cells) > 0))

    named: set[int] = set()
    for segment in pass_a.segments:
        tri_cells, _, _ = read_tri_segment(segment.tri_path)
        named.update(int(c) for c in tri_cells.ravel())
    assert {int(c) for c in cells} == named
