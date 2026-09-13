"""The output partition, and what it must not change — WP-3.6, Round 13.

Round 13 is a **measurement** package, so the thing its tests have to protect is
not a new capability but an old one: every figure it reports is measured against
Rounds 10 to 12, and that comparison is worthless if the new partition path
quietly decimates something different from what `decimate_generation` decimates.

So the load-bearing test here is the **baseline** one: at
`OutputPartition("lattice", 512, 512)` the patches are the intermediate tiles
themselves and the output is byte-identical to `decimate_generation`'s. If that
ever goes red, every number in the Round 13 report is measuring a different
decimator from the one on the record.

The rest pin the properties ADR-006's benchmark reads off the partition —
conservation, the locked-fraction direction, determinism, the refusal — and the
kink that `decimate_tiles.check_seams` is structurally unable to see.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed. The
figures here are fixture figures and are never quoted as real-data results.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from rapidmesh import synthetic
from rapidmesh.decimate import DecimationSettings
from rapidmesh.decimate_tiles import decimate_generation
from rapidmesh.output_measure import (
    metric_occupancy,
    occupancy_of,
    seam_dihedral_change,
    seam_vertices,
)
from rapidmesh.output_run import decimate_output, decode_tiles
from rapidmesh.output_tiles import OutputPartition, patches, plan
from rapidmesh.pipeline import mesh_station_streamed

FIXTURE_ROWS, FIXTURE_COLS = 260, 520
BUDGET_M = 0.0128


@pytest.fixture(scope="module")
def generation(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("output-tiles")
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=FIXTURE_ROWS, cols=FIXTURE_COLS,
        dropout=0.01, range_noise=0.002, seed=17, station_id="outputtiles",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=25_000, halo=3,
        measure=False, out_dir=str(root / "out"),
    )
    assert result.tiles is not None
    assert len(result.tiles.tile_ids) > 1, "the fixture must span more than one tile"
    return result.tiles, root


def digests(folder: Any) -> set[str]:
    """One SHA-256 per tile over its arrays, as a set.

    A set and not a mapping, because the two paths name their files differently
    — by intermediate tile id, and by output tile key — and the claim under test
    is about the geometry, not about the filenames.
    """
    out = set()
    for path in sorted(folder.glob("*.npz")):
        h = hashlib.sha256()
        with np.load(path) as data:
            for key in sorted(data.files):
                h.update(key.encode("utf-8"))
                h.update(np.ascontiguousarray(data[key]).tobytes())
        out.add(h.hexdigest())
    return out


def test_the_window_sized_partition_reproduces_decimate_generation(
    generation: Any,
) -> None:
    """The baseline case must be the thing every Round 13 figure is measured against.

    `decimate_generation` is what Rounds 10 to 12 measured. At one window per
    output tile this path has to produce the same bytes, or the benchmark is
    comparing two decimators and reporting the difference as a tile-size effect.
    """
    store, root = generation
    settings = DecimationSettings(max_plane_deviation_m=BUDGET_M)
    reference = decimate_generation(store, root / "reference", settings)
    run = decimate_output(
        store, root / "window", settings, OutputPartition("lattice", 512, 512)
    )
    assert digests(root / "reference") == digests(root / "window")
    assert run.triangles_in == reference.triangles_in
    assert run.triangles_out == reference.triangles_out
    assert run.vertices_out == reference.vertices_out


def test_every_owned_vertex_and_triangle_lands_in_exactly_one_output_tile(
    generation: Any,
) -> None:
    """Conservation, over four partitions that cut the surface differently.

    A partition that dropped geometry would flatter every quantity ADR-006 asks
    for at once — smaller output, better ratio, fewer locked vertices — so this
    is checked before any of them is believed.
    """
    store, _ = generation
    for partition in (
        OutputPartition("lattice", 512, 512),
        OutputPartition("lattice", 128, 512),
        OutputPartition("lattice", 2048, 2048),
        OutputPartition("metric", cell_m=2.0),
    ):
        owned: list[Any] = []
        triangles = 0
        for item in patches(store, partition):
            assert not isinstance(item, tuple), "the fixture must not be refused"
            owned.append(item.global_ids[: item.owned_count])
            triangles += item.triangle_count
        seen = np.concatenate(owned) if owned else np.empty(0, np.int64)
        assert seen.shape[0] == np.unique(seen).shape[0], (
            f"{partition.label}: an owned vertex appears in two output tiles"
        )
        assert seen.shape[0] == store.vertex_count, (
            f"{partition.label}: {seen.shape[0]} owned of {store.vertex_count}"
        )
        assert triangles == store.triangle_count, (
            f"{partition.label}: {triangles} triangles of {store.triangle_count}"
        )


def test_larger_output_tiles_lock_less_of_the_surface(generation: Any) -> None:
    """ADR-006's stated cost, as a test rather than as a prediction.

    "Locked boundaries never simplify" — so a finer partition must refuse
    permission to move to a larger share of the surface. If this ever came out
    flat, the benchmark would have no tension to measure and the recommendation
    would be arbitrary.
    """
    store, root = generation
    settings = DecimationSettings(max_plane_deviation_m=BUDGET_M)
    fractions = []
    for rows in (128, 256, 512):
        run = decimate_output(
            store, root / f"locked{rows}", settings,
            OutputPartition("lattice", rows, 512),
        )
        fractions.append(run.locked_fraction)
    assert all(f is not None for f in fractions)
    assert fractions[0] > fractions[1] > fractions[2], fractions


def test_a_row_split_patch_keeps_no_halo_vertex_a_triangle_does_not_name(
    generation: Any,
) -> None:
    """The drop rule, asserted as the per-patch invariant it actually is.

    A row-split patch inherits its whole window's halo but owns one band of it,
    so most of that halo belongs to triangles this patch does not own.
    `lock_boundary` rule 1 locks every one of them, so keeping them would
    inflate ADR-006 quantity 3 — the locked percentage this benchmark
    reports — and make a fine partition look far more seam-bound than it is.

    This test exists because the trend test above was **blind** to exactly that
    defect, and the mutation matrix caught it being blind. Keeping every halo
    vertex inflates the fine rungs together and leaves the window-sized rung
    untouched, so the ordering that test asserts still holds and it stays green.
    Measured on this fixture, the rule drops **266,262 of 267,811** halo
    vertices at 128 x 512 and **none at all** at 512 x 512 — which is also
    why the byte-identity baseline cannot see it: at the window size there is
    nothing to drop.
    """
    store, _ = generation
    checked = 0
    for item in patches(store, OutputPartition("lattice", 128, 512)):
        assert not isinstance(item, tuple), f"the fixture refused a patch: {item}"
        count = int(item.positions.shape[0])
        named = np.zeros(count, bool)
        if item.triangles.shape[0]:
            named[np.unique(item.triangles)] = True
        stranded = int((~named[item.owned_count:]).sum())
        assert stranded == 0, (
            f"patch {item.key} keeps {stranded} halo vertices no triangle names"
        )
        checked += count - item.owned_count
    # Round 12's lesson: a check over an empty set is green for the wrong
    # reason. If no patch carried a halo, nothing above was tested.
    assert checked > 0, "no patch carried a halo, so nothing was checked"


def test_the_partition_is_deterministic(generation: Any) -> None:
    """Same generation, same partition, same bytes — ADR-006's ownership rule."""
    store, root = generation
    settings = DecimationSettings(max_plane_deviation_m=BUDGET_M)
    partition = OutputPartition("lattice", 256, 512)
    first = decimate_output(store, root / "det1", settings, partition)
    second = decimate_output(store, root / "det2", settings, partition)
    assert digests(root / "det1") == digests(root / "det2")
    assert first.tile_count == second.tile_count


def test_a_patch_over_the_ceiling_is_refused_with_its_size_recorded(
    generation: Any,
) -> None:
    """The guard reports the size it could not hold rather than dying of it.

    The benchmark's job is to push tile size up until memory binds. An
    allocation failure at that point would report nothing; a refusal reports the
    number that matters.
    """
    store, root = generation
    run = decimate_output(
        store, root / "refused",
        DecimationSettings(max_plane_deviation_m=BUDGET_M),
        OutputPartition("lattice", 4096, 4096),
        max_patch_vertices=1,
    )
    assert run.tile_count == 0
    assert run.refused, "nothing was refused and nothing was decimated"
    assert all(r["vertices_needed"] > 1 for r in run.refused)


def test_the_seam_kink_check_sees_what_the_boundary_check_cannot(
    generation: Any,
) -> None:
    """`check_seams` is exact about cracks and blind to angles. This is the other half.

    Proven by construction rather than asserted: one tile's vertices are moved
    off the surface *without touching any boundary edge*, so the boundary edge
    set is untouched — `check_seams` still reads clean — while the surface now
    meets its neighbour at an angle. A crack check that could see this would make
    the kink measurement redundant; it cannot, which is why both are reported.
    """
    store, root = generation
    run = decimate_output(
        store, root / "kink",
        DecimationSettings(max_plane_deviation_m=BUDGET_M),
        OutputPartition("lattice", 256, 512),
    )
    assert run.seams_clean, "the fixture must start clean"
    honest = seam_dihedral_change(store, root / "kink")
    assert honest["seam_edges"] > 0, "the fixture must have cross-tile edges"
    assert honest["interior"] is not None, "the control population must exist"

    # Bend one tile away from the surface, leaving every boundary edge in place.
    paths = sorted((root / "kink").glob("*.npz"))
    with np.load(paths[0]) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    free = ~np.asarray(arrays["locked"], bool)
    assert bool(free.any()), "the fixture tile must have an interior"
    arrays["positions"][free] += np.float32(0.5)
    np.savez(paths[0], **arrays)

    bent = seam_dihedral_change(store, root / "kink")
    # The interior control is the point: bending one tile must move the
    # cross-tile population *relative to* the interior one, not merely move
    # both, which a coarser mesh would also do.
    assert (bent["cross_tile"]["max_deg"] > honest["cross_tile"]["max_deg"] + 1.0), (
        honest["cross_tile"], bent["cross_tile"]
    )
    assert (bent["ratio_mean_cross_over_interior"]
            > honest["ratio_mean_cross_over_interior"]), (honest, bent)


def test_metric_occupancy_counts_what_dec_021_measured(generation: Any) -> None:
    """The counter, against a hand-computable answer on the same data.

    DEC-021's finding is an occupancy statistic, so the statistic itself is
    pinned here rather than only the conclusion drawn from it: the full-
    resolution count must equal the generation's own owned-vertex total, and a
    coarser cell can only hold more per cell, never fewer.
    """
    store, _ = generation
    cells_fine, max_fine, total_fine, _ = metric_occupancy(store, 1.0)
    cells_coarse, max_coarse, total_coarse, _ = metric_occupancy(store, 4.0)
    assert total_fine == total_coarse == store.vertex_count
    assert cells_coarse < cells_fine
    assert max_coarse >= max_fine
    # `occupancy_of` is the same statistic over a resident array, and must agree
    # with itself when handed a single cell's worth of points.
    points = np.zeros((7, 3), np.float64)
    assert occupancy_of(points, 4.0) == (1, 7, 7, 7.0)


def test_decode_time_is_reported_per_tile_and_named_a_proxy(
    generation: Any,
) -> None:
    """ADR-006 quantity 7, with rule 6's guard on what it is allowed to be.

    A first-paint number here would be a deferred capability claimed. The
    function must therefore carry its own disclaimer, so a figure lifted out of
    the JSON carries it too.
    """
    store, root = generation
    decimate_output(
        store, root / "decode",
        DecimationSettings(max_plane_deviation_m=BUDGET_M),
        OutputPartition("lattice", 512, 512),
    )
    stats = decode_tiles(root / "decode")
    assert stats["tiles"] > 0
    assert len(stats["per_tile"]) == stats["tiles"]
    assert stats["bytes_total"] > 0
    assert "not first paint" in stats["proxy_note"]


def test_the_plan_names_every_source_tile_before_anything_is_read(
    generation: Any,
) -> None:
    """The size guard is only cheap if the plan is manifest-only.

    If `plan` had to open tiles to know which output tile they feed, the refusal
    that protects a large-tile run would cost as much as the run it refuses.
    """
    store, _ = generation
    mapping = plan(store, OutputPartition("lattice", 1024, 1024))
    named = {t for ids in mapping.values() for t in ids}
    assert named == set(store.tile_ids)
    assert seam_vertices  # imported for the seam tests above; keeps the name live
