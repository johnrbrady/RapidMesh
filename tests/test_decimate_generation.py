"""
Decimating a written generation end to end — WP-3.4, Round 10.

`test_decimate.py` covers one patch and `test_decimate_seams.py` covers a
hand-built pair. This runs the real path: a synthetic station through
`mesh_station_streamed` into a real lattice-partitioned `TileStore`, then
`decimate_generation` over it, then both DEC-013 baselines over the result.

What it pins that the smaller suites cannot
--------------------------------------------
* the locked set is derived correctly from a **real** tile's owned count and
  triangle list, rather than from a fixture that was built to suit it;
* the seam ledger comes out clean over a station with more than two tiles and a
  boundary that includes genuine holes, not only a rectangle's rim;
* two runs of the whole generation produce byte-identical tiles — the
  determinism claim at the level the report makes it;
* both baselines are producible from what Pass B writes beside the tiles, with
  no second read of the source.

Honesty condition (standing rule 8)
------------------------------------
Regression tests on new code, expected to pass on first run and not presented as
proven-failing. The sensitivity evidence for the seam check is in
`tests/test_decimate_seams.py`, where the same functions are shown red.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
The figures here are fixture figures and are never quoted as real-data results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rapidmesh import decimate_kernel, decimate_sweep, synthetic
from rapidmesh.decimate import DecimationSettings
from rapidmesh.decimate_qa import (
    against_observations,
    against_pre_decimation,
    open_observations,
)
from rapidmesh.decimate_tiles import decimate_generation, read_decimated
from rapidmesh.pipeline import mesh_station_streamed

FIXTURE_ROWS, FIXTURE_COLS = 260, 520
REDUCTION = 4


@pytest.fixture(scope="module")
def generation(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("decimate-generation")
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=FIXTURE_ROWS, cols=FIXTURE_COLS,
        dropout=0.01, range_noise=0.002, seed=17, station_id="decimate",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=25_000, halo=3,
        measure=False, out_dir=str(root / "out"),
    )
    assert result.tiles is not None
    assert len(result.tiles.tile_ids) > 1, "the fixture must span more than one tile"
    return result.tiles, root


def run(generation: Any, out: str, **kwargs: Any) -> Any:
    store, root = generation
    return decimate_generation(
        store, root / out, DecimationSettings(), reduction=REDUCTION, **kwargs
    )


def test_a_real_generation_decimates_with_the_seams_intact(generation: Any) -> None:
    report = run(generation, "first")
    assert report.triangles_out < report.triangles_in
    assert report.seam_boundary_edges_before > 0, "a station with no boundary at all"
    assert report.seam_edges_lost == 0
    assert report.seam_edges_gained == 0
    assert report.shared_vertex_position_conflicts == 0
    assert report.seams_clean


def edge_use(triangles: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0, return_counts=True)[1]


def union_faces(store: Any) -> np.ndarray:
    """The pre-decimation station in global ids, for the comparison only."""
    out = []
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        gids = np.asarray(store.join(tile_id).global_vertex_index, np.int64)
        out.append(gids[payload.triangles.astype(np.int64)])
    return np.concatenate(out)


def test_the_joined_mesh_is_as_manifold_as_the_one_it_came_from(
    generation: Any,
) -> None:
    """Joining the tiles by global id must not reveal a fault decimation made."""
    store, root = generation
    run(generation, "joined")
    positions, triangles, ids = read_decimated(root / "joined")
    assert positions.shape[0] == ids.shape[0]
    assert triangles.shape[0] > 0
    assert int(edge_use(union_faces(store)).max()) <= 2, "the fixture must start clean"
    assert int(edge_use(triangles).max()) <= 2, "the join produced a non-manifold edge"
    assert int(store.triangle_count) > int(triangles.shape[0])


def test_without_the_locked_join_rule_two_tiles_emit_the_same_face(
    generation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The red case for `_would_join_locked`, and the defect it was found by.

    Removing the rule leaves each tile's own patch manifold and the boundary set
    unchanged — the seam check still reads clean — while the union gains edges
    used by four triangles, because two tiles closed the same fan onto the same
    pair of ring vertices. That is why the rule lives in `decimate_sweep.py`
    and not in the seam check.

    The patch is on the class, not on a module attribute, so it reaches the
    sweep however `decimate_patch` imported it — and it reaches only the Python
    sweep, which is why `decimate_kernel.DEFAULT_CHOICE` stays `python`: a
    monkeypatch cannot enter the Rust one, and a default of `rust` would make
    this test silently stop testing anything.

    For the same reason the kernel is pinned here rather than inherited: with
    `RAPIDMESH_DECIMATE_KERNEL=rust` in the environment the patch never lands
    and this red case fails for a reason unrelated to the rule (ITEM-031).
    """
    _, root = generation
    monkeypatch.setenv(decimate_kernel.ENVIRONMENT_VARIABLE, "python")
    monkeypatch.setattr(
        decimate_sweep._PatchState, "_would_join_locked", lambda *_args: False
    )
    report = run(generation, "unruled")
    _, triangles, _ = read_decimated(root / "unruled")
    assert report.seams_clean, "the boundary check cannot see this fault"
    assert int(edge_use(triangles).max()) > 2


def test_two_runs_of_a_whole_generation_agree_byte_for_byte(generation: Any) -> None:
    _, root = generation
    run(generation, "run_a")
    run(generation, "run_b")

    def digest(folder: Path) -> str:
        h = hashlib.sha256()
        for path in sorted(folder.glob("*.npz")):
            with np.load(path) as data:
                for key in sorted(data.files):
                    h.update(np.ascontiguousarray(data[key]).tobytes())
        return h.hexdigest()

    assert digest(root / "run_a") == digest(root / "run_b")


def test_both_dec_013_baselines_are_producible_side_by_side(generation: Any) -> None:
    store, root = generation
    run(generation, "measured")
    positions, triangles, _ = read_decimated(root / "measured")

    own = against_pre_decimation(
        store, positions, triangles, samples=20_000, seed=0
    )
    end_to_end = against_observations(
        open_observations(store.root), positions, triangles,
        samples=20_000, seed=0,
    )

    assert own.samples > 0 and end_to_end.samples > 0
    assert own.baseline != end_to_end.baseline
    # No ordering is asserted between the two. They are weighted differently —
    # baseline 1 by surface area, baseline 2 by observation density — and on a
    # range image those disagree by construction, so which is larger is a
    # property of the station rather than of the metric. The first draft of this
    # test asserted an ordering and was wrong; the correction is recorded here
    # rather than quietly deleted.
    assert own.rms > 0.0 and end_to_end.rms > 0.0
    # The ITEM-015 spike: observations that are still mesh vertices read exactly
    # zero, and the metric has to be able to say how many.
    assert 0.0 < end_to_end.zero_fraction < 1.0
    assert own.zero_fraction < end_to_end.zero_fraction


def test_the_decimated_tier_is_smaller_than_the_generation_it_came_from(
    generation: Any,
) -> None:
    store, root = generation
    report = run(generation, "sized")
    assert report.vertices_out < report.vertices_in
    assert report.vertices_in == store.vertex_count
    assert report.triangles_in == store.triangle_count
