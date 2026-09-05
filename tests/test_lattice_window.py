"""
The intermediate partition is the lattice, not the floor — WP-A, DEC-021.

A metric cell does not bound how many samples fall in it. Sample density on a
terrestrial range image goes as 1/r², so a 4 m cell holds a handful of samples at
40 m and a million at 2 m. WP-10m through 11m.t measured the consequence and hit
its floor: ordinal 20 packs its surface into **5** tiles whose largest holds
9,341,715 vertices, while ordinal 3 — a comparable station at longer range —
spreads the same work over 364.

The fixture here is that pathology in miniature and is the falsifier for the
whole change: a close-range room on a 500 × 2000 lattice, which the 4 m grid
piles into one cell. Under lattice windows the same station cannot exceed
`W × H` owned vertices **by construction** — the cap is arithmetic on the cell
id, not a property of where the walls happen to be.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from rapidmesh import synthetic
from rapidmesh.pipeline import mesh_station_streamed
from rapidmesh.tiles import DEFAULT_WINDOW_COLS, DEFAULT_WINDOW_ROWS

# A room small enough that every wall is inside one 4 m cell, sampled on a
# lattice big enough that the cell holds most of a million vertices.
CLOSE_ROOM = dict(
    half_x=1.0, half_y=0.75, floor_z=-0.4, ceil_z=0.35,
    scanner=(0.09, -0.05, 0.0), mover=True,
)
LATTICE_ROWS, LATTICE_COLS = 500, 2000


@pytest.fixture(scope="module")
def close_station() -> Any:
    return synthetic.generate(
        synthetic.RoomScene(**CLOSE_ROOM), rows=LATTICE_ROWS, cols=LATTICE_COLS,
        dropout=0.01, range_noise=0.002, seed=11, station_id="close",
    )


def _build(station: Any, out: pathlib.Path, **kw: Any) -> Any:
    return mesh_station_streamed(
        station.scan, band_rows=56, chunk_points=250_000, halo=3,
        measure=False, out_dir=str(out), **kw,
    )


def _owned_and_resident(store: Any) -> tuple[int, int, int]:
    owned = [int(e["owned_count"]) for e in store.manifest["tiles"]]
    resident = [int(e["vertex_count"]) for e in store.manifest["tiles"]]
    return max(owned), max(resident), len(owned)


def test_the_close_range_fixture_is_pathological_under_metric_tiles(
    close_station: Any, tmp_path: pathlib.Path
) -> None:
    """The control. If this stops failing, the fixture has stopped being a test.

    A green run of the lattice test below means nothing unless the same station
    is genuinely unbounded under the partition it replaces. This holds that line:
    the metric grid must still pile this room into a handful of cells with far
    more than `W × H` vertices in the largest.
    """
    result = _build(close_station, tmp_path / "metric", tile_size=4.0,
                    partition="metric")
    assert result.tiles is not None
    owned, _, tiles = _owned_and_resident(result.tiles)
    cap = DEFAULT_WINDOW_ROWS * DEFAULT_WINDOW_COLS
    assert tiles <= 8, f"the fixture spread over {tiles} metric tiles; it should pile"
    assert owned > cap, (
        f"the metric grid held {owned:,} owned vertices in its largest tile, "
        f"which is inside the {cap:,} cap — this fixture no longer reproduces "
        "the pathology and the test below proves nothing"
    )


def test_lattice_windows_bound_owned_vertices(
    close_station: Any, tmp_path: pathlib.Path
) -> None:
    """Red before WP-A: production tiling is metric, so this station's largest
    tile holds 938,047 owned vertices against a 262,144 cap.

    The bound is structural. A window is `W × H` lattice cells and a vertex
    belongs to the window containing its own cell, so the owned count cannot
    exceed `W × H` whatever the geometry does. The resident count adds the
    1-cell ring the halo needs, so it cannot exceed `(W+2) × (H+2)`.
    """
    result = _build(close_station, tmp_path / "lattice")
    assert result.tiles is not None
    owned, resident, tiles = _owned_and_resident(result.tiles)

    owned_cap = DEFAULT_WINDOW_ROWS * DEFAULT_WINDOW_COLS
    resident_cap = (DEFAULT_WINDOW_ROWS + 2) * (DEFAULT_WINDOW_COLS + 2)
    assert owned <= owned_cap, (
        f"largest window owns {owned:,} vertices against the {owned_cap:,} "
        f"cap ({tiles} windows)"
    )
    assert resident <= resident_cap, (
        f"largest window is resident for {resident:,} vertices against the "
        f"{resident_cap:,} ring cap"
    )
    # Not vacuous: the station really is big enough to have breached the cap.
    assert sum(int(e["owned_count"]) for e in result.tiles.manifest["tiles"]) > owned_cap


def test_the_manifest_records_the_partition(
    close_station: Any, tmp_path: pathlib.Path
) -> None:
    """A generation must say how it was cut, or a reader cannot reproduce it.

    The window is a setting, like `band_rows`. `PHASE1-DETERMINISM-SPEC.md` §3
    requires every such setting to be explicit and recorded, and a tile store
    whose partition is implicit would make two generations with different cuts
    indistinguishable on disk.
    """
    result = _build(close_station, tmp_path / "manifest")
    assert result.tiles is not None
    grid = result.tiles.manifest["grid"]
    assert grid["kind"] == "lattice"
    assert grid["window_rows"] == DEFAULT_WINDOW_ROWS
    assert grid["window_cols"] == DEFAULT_WINDOW_COLS
    assert grid["lattice_rows"] == LATTICE_ROWS
    assert grid["lattice_cols"] == LATTICE_COLS
    assert result.tiles.manifest["contract"] == "rapidmesh-tile-v1"
