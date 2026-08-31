"""
Incremental full-resolution tile output — WP-3.2, ADR-006 Decision 2.

ADR-006's arithmetic is the reason this package exists: the reference station's
finished mesh is 628.9 MB of *output alone* against a 512 MB working-memory
budget, so the output has to be written as it is produced rather than assembled
and then saved. The claim under test is therefore not "tiles are produced" but
"the station's mesh is never held".

Three things have to be true at once, and they pull against each other in the
same way `test_pass_b_bound.py`'s pair does:

* the tiled generation must reassemble to **exactly** what the resident path
  produced — bit for bit on positions, normals, colour, source identity and the
  canonical oriented triangle multiset;
* nothing in the tiled path may materialise a whole-station vertex array and a
  whole-station triangle array at once, and no `MeshData` for the station at all;
* the tile-independent outputs must not depend on the tile size, because ADR-006
  Decision 2a makes tile size a **measured parameter** and anything that moved
  with it would be a number with no fixed meaning.

Equality is `np.array_equal`, never `np.allclose`. Ownership rules are asserted
as counts rather than described: ADR-006 2a calls deterministic ownership a
reproducibility requirement, and a rule that is only documented is a rule that
drifts.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.equivalence import compare_mesh_results
from rapidmesh.memory import (
    PEAK_RSS_BUDGET_BYTES,
    WORKING_MEMORY_BUDGET_BYTES,
    measure_in_child,
)
from rapidmesh.pipeline import mesh_station, mesh_station_streamed
from rapidmesh.tile_equivalence import TileReassemblyError, reconstitute_mesh
from rapidmesh.tile_io import TileError, read_tile, read_tile_join
from rapidmesh.tile_spool import TileSpoolSet
from rapidmesh.tiles import CURRENT_NAME, MANIFEST_NAME, TileGrid, TileStore

TOOLS = str(pathlib.Path(__file__).resolve().parent.parent / "tools")

# The WP-3.2 gate ladder. Smaller than the QA ladder because every row here runs
# the whole streamed pipeline rather than QA against a mesh loaded from disk.
LADDER = ((120, 480), (240, 960), (400, 1600))
GATE_TILE_SIZE = 4.0

# WP-3.3: QA's peak above the tile working set.
QA_MARGINAL_GATE_BYTES = 128_000_000


@pytest.fixture(scope="module")
def station() -> synthetic.SyntheticScan:
    return synthetic.generate(
        synthetic.RoomScene(mover=True), rows=48, cols=192,
        dropout=0.01, range_noise=0.002, seed=11, station_id="tiles",
    )


@pytest.fixture(scope="module")
def resident(station: synthetic.SyntheticScan) -> Any:
    return mesh_station(station.scan, measure=True, measure_samples=20_000)


def _tiled(
    station: synthetic.SyntheticScan,
    out: pathlib.Path,
    *,
    tile_size: float = 2.0,
    band_rows: int = 16,
    measure: bool = True,
) -> Any:
    return mesh_station_streamed(
        station.scan, band_rows=band_rows, chunk_points=5_000, halo=3,
        measure=measure, measure_samples=20_000,
        out_dir=str(out), tile_size=tile_size,
    )


# ---------------------------------------------------------------------------
# the station's mesh is never held
# ---------------------------------------------------------------------------


def test_the_tiled_run_keeps_no_resident_mesh(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    result = _tiled(station, tmp_path / "out")
    assert result.mesh is None
    assert result.tiles is not None
    assert result.tiles.triangle_count > 0
    assert len(result.tiles.tile_ids) > 1


def test_the_tiled_run_never_calls_build_mesh(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural assertion, made impossible to hide.

    `build_mesh` is the one function that assembles a whole-station `MeshData`.
    Replacing it with a detonator is stronger than inspecting a peak: a peak can
    be small because the fixture is small, but a call is a call.
    """
    import rapidmesh.triangulate as triangulate

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("the tiled path assembled a whole-station mesh")

    monkeypatch.setattr(triangulate, "build_mesh", forbidden)
    result = _tiled(station, tmp_path / "out", measure=True)
    assert result.tiles is not None


def test_the_resident_branch_is_still_reachable_and_still_differs(
    station: synthetic.SyntheticScan
) -> None:
    """Without an output directory the old shape runs, and says so.

    Kept deliberately — it is the reference the tiled path is compared against,
    the way `qa_reference` is for forward QA — so a change that silently removed
    it would take the comparison with it.
    """
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3, measure=False
    )
    assert result.mesh is not None
    assert result.tiles is None


# ---------------------------------------------------------------------------
# it reassembles to exactly the resident mesh
# ---------------------------------------------------------------------------


def test_the_generation_reassembles_bitwise(
    station: synthetic.SyntheticScan, resident: Any, tmp_path: pathlib.Path
) -> None:
    tiled = _tiled(station, tmp_path / "out")
    rebuilt = reconstitute_mesh(tiled.tiles)
    assert rebuilt.vertex_count == resident.mesh.vertex_count
    assert rebuilt.triangle_count == resident.mesh.triangle_count
    for name in ("origin", "vertices", "normals", "rgb", "source_sample_id"):
        assert np.array_equal(
            getattr(rebuilt, name), getattr(resident.mesh, name)
        ), name


def test_both_qa_reports_match_the_resident_run(
    station: synthetic.SyntheticScan, resident: Any, tmp_path: pathlib.Path
) -> None:
    """WP-3.3's gate: QA reads the tile store and does not move a figure."""
    tiled = _tiled(station, tmp_path / "out")
    assert compare_mesh_results(resident, tiled, source_sha256="a" * 64).ok
    for name in ("rms", "mean", "p95", "p99_9", "maximum"):
        assert getattr(tiled.deviation, name) == getattr(resident.deviation, name)
        assert getattr(tiled.mesh_to_source, name) == getattr(
            resident.mesh_to_source, name
        )


def test_the_reassembly_would_notice_a_missing_vertex(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Sensitivity break for the adapter.

    A generation with a tile removed no longer covers the global index range,
    and must fail rather than reassemble a shorter mesh — a shorter mesh is
    exactly the silent geometry loss `segments_io` refuses elsewhere.
    """
    tiled = _tiled(station, tmp_path / "out", measure=False)
    store = tiled.tiles
    trimmed = TileStore(
        root=store.root,
        manifest={**store.manifest, "tiles": store.manifest["tiles"][1:]},
    )
    with pytest.raises(TileReassemblyError):
        reconstitute_mesh(trimmed)


def test_a_station_that_meshes_to_nothing_still_publishes_a_generation(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """CLAUDE.md §4 rule 7: a scan that fails to mesh is not a failed scan.

    Culling everything is a real production outcome, and it is where a tiled
    generation has the least to work with: no tiles, so nothing to read an origin
    or a colour policy off. Both are recorded in the manifest for exactly that
    reason, and this asserts the empty generation still compares equal to the
    resident run rather than differing on fields neither has data for.
    """
    resident = mesh_station(station.scan, measure=False, min_component_area=1e9)
    assert resident.mesh is not None
    assert resident.mesh.triangle_count == 0

    tiled = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=5_000, halo=3,
        measure=False, min_component_area=1e9,
        out_dir=str(tmp_path / "out"), tile_size=2.0,
    )
    assert tiled.tiles is not None
    assert len(tiled.tiles.tile_ids) == 0
    assert np.array_equal(tiled.tiles.origin, resident.mesh.origin)
    assert compare_mesh_results(resident, tiled, source_sha256="e" * 64).ok
    tiled.stats.require_balanced()
    assert tiled.stats == resident.stats


def test_the_published_generation_has_no_scratch_left_in_it(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """The spool directory is scratch and must not outlive the spools."""
    out = tmp_path / "out"
    _tiled(station, out, measure=False)
    assert not (out / "tilespool").exists()
    assert not list(out.rglob("*.rmspool"))


# ---------------------------------------------------------------------------
# ownership, determinism and tile-size invariance
# ---------------------------------------------------------------------------


def test_every_vertex_and_triangle_is_owned_exactly_once(
    station: synthetic.SyntheticScan, resident: Any, tmp_path: pathlib.Path
) -> None:
    """ADR-006 2a as counts, not as prose."""
    store = _tiled(station, tmp_path / "out", measure=False).tiles
    owned: list[Any] = []
    triangles = 0
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        owned.append(join.global_vertex_index[np.asarray(join.owned, bool)])
        triangles += payload.triangle_count
    index = np.sort(np.concatenate(owned))
    assert np.array_equal(index, np.arange(resident.mesh.vertex_count))
    assert triangles == resident.mesh.triangle_count


def test_boundary_vertices_are_duplicated_with_identical_bits(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """The locked boundary ring: neighbours must agree on the seam exactly.

    ADR-006 2a's locked boundary exists so adjacent tiles cannot crack apart. If
    two tiles held a shared vertex at even one differing float32 bit, the seam
    would be a real gap, so the agreement is asserted bitwise.
    """
    store = _tiled(station, tmp_path / "out", measure=False).tiles
    seen: dict[int, Any] = {}
    duplicates = 0
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        join = store.join(tile_id)
        for slot, global_id in enumerate(join.global_vertex_index):
            key = int(global_id)
            position = payload.positions[slot]
            if key in seen:
                duplicates += 1
                assert np.array_equal(seen[key], position), key
            else:
                seen[key] = position
    assert duplicates > 0, "the fixture produced no cross-tile boundary at all"


def test_two_runs_produce_identical_tiles(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Deterministic ownership is a reproducibility requirement, so the digests
    are compared, not just the geometry."""
    first = _tiled(station, tmp_path / "a", measure=False).tiles
    second = _tiled(station, tmp_path / "b", measure=False).tiles
    assert first.tile_ids == second.tile_ids
    left = [(e["tile_id"], e["sha256"]) for e in first.manifest["tiles"]]
    right = [(e["tile_id"], e["sha256"]) for e in second.manifest["tiles"]]
    assert left == right


@pytest.mark.parametrize("tile_size", (1.5, 3.0, 6.0))
def test_tile_independent_outputs_do_not_move_with_tile_size(
    station: synthetic.SyntheticScan, resident: Any, tile_size: float,
    tmp_path: pathlib.Path,
) -> None:
    """The tile parameter may change the tiling and nothing else.

    ADR-006 2a makes tile size a measured parameter, which is only meaningful if
    varying it cannot move an output. Three sizes, spanning a factor of four, all
    reassembling to the same mesh and reporting the same ledger and the same two
    QA figures.
    """
    tiled = _tiled(station, tmp_path / "out", tile_size=tile_size)
    rebuilt = reconstitute_mesh(tiled.tiles)
    assert np.array_equal(rebuilt.vertices, resident.mesh.vertices)
    assert np.array_equal(rebuilt.normals, resident.mesh.normals)
    assert compare_mesh_results(resident, tiled, source_sha256="b" * 64).ok
    tiled.stats.require_balanced()
    assert tiled.stats == resident.stats


def test_the_tile_size_actually_changes_the_tiling(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Guard on the guard above: invariance across sizes that produced the same
    tiling would be invariance across nothing."""
    small = _tiled(station, tmp_path / "s", tile_size=1.5, measure=False).tiles
    large = _tiled(station, tmp_path / "l", tile_size=6.0, measure=False).tiles
    assert len(small.tile_ids) > len(large.tile_ids)


# ---------------------------------------------------------------------------
# the on-disk contract
# ---------------------------------------------------------------------------


def test_the_generation_publishes_its_pointer_last(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """Contract §3 and islands §4.5: a reader sees a complete generation."""
    out = tmp_path / "out"
    store = _tiled(station, out, measure=False).tiles
    pointer = json.loads((out / CURRENT_NAME).read_text(encoding="utf-8"))
    generation = out / "generations" / pointer["generation"]
    assert (generation / MANIFEST_NAME).exists()
    for entry in store.manifest["tiles"]:
        assert store.tile_path(int(entry["tile_id"])).exists()
        assert store.join_path(int(entry["tile_id"])).exists()
    assert not list(out.rglob("*.tmp"))


def test_a_corrupt_tile_fails_closed(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """A truncated or edited payload must raise, not read short.

    `segments_io`'s rule, applied to the final tier: a short read that silently
    yields fewer triangles would delete survey geometry and report success.
    """
    store = _tiled(station, tmp_path / "out", measure=False).tiles
    victim = store.tile_path(store.tile_ids[0])
    raw = bytearray(victim.read_bytes())
    raw[-1] ^= 0xFF
    victim.write_bytes(bytes(raw))
    with pytest.raises(TileError):
        read_tile(victim)


def test_the_client_payload_carries_no_source_identity(
    station: synthetic.SyntheticScan, tmp_path: pathlib.Path
) -> None:
    """DEC-004's boundary is the file split, not a serialiser's good manners.

    `read_tile` returns positions, normals, colour and local triangle indices and
    has no field for a stable sample id, a lattice row or a global vertex index;
    those live only in the separate join file. A reader handed the payload alone
    cannot rebuild the evidence join.
    """
    store = _tiled(station, tmp_path / "out", measure=False).tiles
    tile_id = store.tile_ids[0]
    payload = read_tile(store.tile_path(tile_id))
    fields = set(vars(payload))
    assert fields == {
        "tile_id", "origin", "bounds", "positions", "normals", "triangles", "rgb"
    }
    assert int(payload.triangles.max()) < payload.vertex_count
    assert payload.normals.shape[0] == payload.owned_count <= payload.vertex_count
    join = read_tile_join(store.join_path(tile_id))
    assert join.source_sample_id.shape[0] == payload.vertex_count


def test_the_grid_is_x_major_so_lower_id_is_lower_xyz() -> None:
    """ADR-006 2a's ownership rule depends on this ordering being the stated
    one, so it is checked rather than assumed."""
    grid = TileGrid.covering(
        np.zeros(3), np.array([4.0, 4.0, 4.0]), size=2.0
    )
    assert grid.shape == (2, 2, 2)
    corners = np.array([
        [0.5, 0.5, 0.5], [0.5, 0.5, 3.5], [0.5, 3.5, 0.5], [3.5, 0.5, 0.5],
    ])
    ids = grid.index_of(corners)
    assert list(ids) == [0, 1, 2, 4]
    assert grid.index_of(np.array([[-99.0, -99.0, -99.0]]))[0] == 0
    assert grid.index_of(np.array([[99.0, 99.0, 99.0]]))[0] == grid.count - 1


def test_the_spool_preserves_append_order_within_a_tile(
    tmp_path: pathlib.Path
) -> None:
    """Per-tile normal accumulation depends on it, and a sort would hide the
    loss. The open-file cap is set to one so eviction and reopening happen on
    every append."""
    dtype = np.dtype([("v", "<i4")])
    spool = TileSpoolSet(tmp_path, "t", dtype, open_limit=1, buffer_bytes=8)
    rng = np.random.default_rng(5)
    tiles = rng.integers(0, 4, 500).astype(np.int64)
    values = np.arange(500, dtype=np.int32)
    for start in range(0, 500, 37):
        block = np.empty(min(37, 500 - start), dtype)
        block["v"] = values[start : start + 37]
        spool.append(tiles[start : start + 37], block)
    spool.finish()
    for tile_id in spool.tile_ids():
        got = np.concatenate([b["v"] for b in spool.read(tile_id)])
        assert np.array_equal(got, values[tiles == tile_id])
        assert got.size == spool.count(tile_id)


# ---------------------------------------------------------------------------
# the memory gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ladder(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Three stations through the whole tiled pipeline, one child process each."""
    root = tmp_path_factory.mktemp("tile-ladder")
    rows: list[dict[str, Any]] = []
    for lattice_rows, cols in LADDER:
        fixture = str(root / f"scan-{lattice_rows}x{cols}.npz")
        made = measure_in_child(
            "measure_peak_memory", "prepare_fixture",
            {"path": fixture, "rows": lattice_rows, "cols": cols},
            label="fixture", sys_path=[TOOLS],
        )
        quiet = measure_in_child(
            "measure_peak_memory", "tiled_station",
            {"fixture": fixture, "band_rows": 64, "tile_size": GATE_TILE_SIZE,
             "measure": False},
            label=f"tiles {lattice_rows}x{cols}", sys_path=[TOOLS], timeout=1800.0,
        )
        run = measure_in_child(
            "measure_peak_memory", "tiled_station",
            {"fixture": fixture, "band_rows": 64, "tile_size": GATE_TILE_SIZE,
             "measure": True},
            label=f"tiled {lattice_rows}x{cols}", sys_path=[TOOLS], timeout=1800.0,
        )
        rows.append({
            "samples": int(made.detail["samples"]),
            "working": run.working_set_delta_bytes,
            "peak": run.peak_rss_bytes,
            "tiles_only_peak": quiet.peak_rss_bytes,
            "qa_marginal": run.peak_rss_bytes - quiet.peak_rss_bytes,
            "tile_bytes": int(run.detail["tile_bytes"]),
            "tiles": int(run.detail["tiles"]),
            "triangles": int(run.detail["triangles"]),
            "resident_mesh": bool(run.detail["resident_mesh"]),
            "seconds": run.seconds,
        })
    return rows


def test_the_tiled_run_meets_both_adr_006_budgets(
    ladder: list[dict[str, Any]]
) -> None:
    """ADR-006 Decision 1's two separate budgets, measured separately.

    Bytes already written to tile files are excluded from the working figure by
    construction — they are on disk and not in the process — and the amount
    excluded is asserted to be non-zero, because an exclusion nobody can size is
    not an exclusion.
    """
    for row in ladder:
        assert not row["resident_mesh"], row
        assert row["tile_bytes"] > 0, row
        assert row["working"] <= WORKING_MEMORY_BUDGET_BYTES, row
        assert row["peak"] <= PEAK_RSS_BUDGET_BYTES, row


def test_the_tile_ladder_spans_a_range(ladder: list[dict[str, Any]]) -> None:
    assert len(ladder) >= 3
    assert ladder[-1]["samples"] >= 8 * ladder[0]["samples"]
    assert all(row["tiles"] > 1 for row in ladder)


def test_qa_against_tiles_stays_under_its_own_gate(
    ladder: list[dict[str, Any]]
) -> None:
    """WP-3.3's gate: `peak(tiles + QA) - peak(tiles)`, both directions.

    Measured as the difference between two whole tiled runs in fresh children,
    one with `measure=False`, so what is attributed to QA is what turning
    measurement on actually costs — the same construction WP-1.10 used, and the
    reason it is a difference of two runs rather than an instrumented one is
    that a peak is a high-water mark and two workloads in one process share one.
    """
    over = [row for row in ladder if row["qa_marginal"] > QA_MARGINAL_GATE_BYTES]
    assert not over, "; ".join(
        f"{row['samples']:,} samples: QA marginal {row['qa_marginal']:,} B "
        f"> {QA_MARGINAL_GATE_BYTES:,} B"
        for row in over
    )
