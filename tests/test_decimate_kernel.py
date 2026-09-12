"""
The Rust collapse kernel against the Python reference — WP-3.4b, Round 11.

The whole of this package's correctness claim is one sentence: **the native
sweep and `decimate.py`'s sweep produce the same bytes.** `decimate.py` is the
equivalence reference, exactly as `pipeline.mesh_station` is for the streamed
path (DEC-012), and these tests are what hold the two together.

So the comparisons here are on **bytes, not values**. `np.allclose` would pass a
kernel that had drifted in the last bits, and a kernel that has drifted in the
last bits is one whose output is no longer predictable from the reference — at
which point the reference stops being worth keeping and every Round 10 figure
stops transferring. `positions` are compared with `tobytes()` so that `-0.0`
against `0.0`, and any NaN at all, is a failure rather than a pass.

What is covered
---------------
* byte-identity on a curved patch, on a noisy patch under an error cap, and
  under `placement="endpoint"` — the three branches of `_place`;
* byte-identity per tile on a **real written generation**, which is the only
  place the locked set comes from a real tile rather than from a fixture built
  to suit it;
* determinism of the native path, two runs compared rather than asserted;
* the locked-join rule of Round 10 §2.3 firing on the native path, and the
  joined mesh staying manifold — the defect that was invisible to the seam
  check.

Honesty condition (standing rule 8)
------------------------------------
A test that a new module agrees with an old one passes trivially when the new
module is absent, and these skip when the crate is not built. The evidence that
outranks a green run is the **sensitivity demonstration** recorded in the Round
11 report: the kernel was perturbed by single characters — the tie-break in
`cheapest` from `<` to `<=`, and the queue's secondary key — and
`assert_identical_patch` went red on the first fixture each time. These tests
detect a kernel that is *nearly* right, which is the only kind of wrong a
transcription is likely to be.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rapidmesh import decimate_kernel, synthetic
from rapidmesh.decimate import DecimatedPatch, DecimationSettings, decimate_patch
from rapidmesh.decimate_tiles import decimate_generation, lock_boundary, read_decimated
from rapidmesh.pipeline import mesh_station_streamed

FIXTURE_ROWS, FIXTURE_COLS = 260, 520
REDUCTION = 4

needs_kernel = pytest.mark.skipif(
    not decimate_kernel.available(),
    reason=f"rapidmesh_kernel not built: {decimate_kernel.unavailable_reason()}",
)


def lattice_patch(
    rows: int, cols: int, bump: float = 0.05, noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A gently curved grid with its outer rim locked.

    The same shape as `test_decimate.py`'s fixture and kept separately on
    purpose: this module is the oracle that decides whether the two sweeps
    agree, and an oracle that imports its inputs from another test file fails
    for reasons that have nothing to do with the kernel. `tests/` is not a
    package in this project and no other test imports a sibling.
    """
    xs, ys = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    z = bump * np.sin(xs * 0.3) * np.cos(ys * 0.3)
    if noise:
        z = z + noise * np.random.default_rng(seed).standard_normal(z.shape)
    positions = np.stack(
        [xs.ravel() * 0.01, ys.ravel() * 0.01, z.ravel() + 2.0], axis=1
    ).astype(np.float32)
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    triangles = np.concatenate(
        [np.stack([a, a + 1, a + cols + 1], 1), np.stack([a, a + cols + 1, a + cols], 1)]
    ).astype(np.uint32)
    locked = np.zeros(positions.shape[0], bool)
    index = np.arange(positions.shape[0]).reshape(rows, cols)
    for edge in (index[0, :], index[-1, :], index[:, 0], index[:, -1]):
        locked[edge] = True
    return positions, triangles, locked


def assert_identical_patch(rust: DecimatedPatch, python: DecimatedPatch) -> None:
    """Every field, compared as bytes where it is an array."""
    assert rust.positions.dtype == python.positions.dtype
    assert rust.positions.shape == python.positions.shape
    assert rust.positions.tobytes() == python.positions.tobytes(), "positions differ"
    assert rust.triangles.tobytes() == python.triangles.tobytes(), "triangles differ"
    assert rust.source_index.tobytes() == python.source_index.tobytes()
    assert rust.moved.tobytes() == python.moved.tobytes()
    assert rust.collapses == python.collapses
    assert rust.rejected_link == python.rejected_link
    assert rust.rejected_seam == python.rejected_seam
    assert rust.rejected_turn == python.rejected_turn
    assert rust.rejected_error == python.rejected_error
    assert rust.locked_vertices == python.locked_vertices
    # The accepted-bound figure is a reported number in its own right
    # (Round 10 §3.4b), so it is compared exactly and not to a tolerance.
    assert rust.max_accepted_error_m.hex() == python.max_accepted_error_m.hex()


def both_kernels(
    positions: Any, triangles: Any, locked: Any, settings: DecimationSettings
) -> tuple[DecimatedPatch, DecimatedPatch]:
    rust = decimate_patch(positions, triangles, locked, settings, kernel="rust")
    python = decimate_patch(positions, triangles, locked, settings, kernel="python")
    return rust, python


@needs_kernel
def test_the_two_kernels_agree_byte_for_byte_on_a_curved_patch() -> None:
    positions, triangles, locked = lattice_patch(90, 120)
    settings = DecimationSettings(target_triangles=triangles.shape[0] // 4)
    rust, python = both_kernels(positions, triangles, locked, settings)
    assert rust.collapses > 0, "the fixture must actually decimate"
    assert_identical_patch(rust, python)


@needs_kernel
def test_the_two_kernels_agree_under_an_error_cap_on_a_noisy_patch() -> None:
    """The `max_error_m` branch, which is also the only path that counts
    `rejected_error` — a counter the ratio-driven sweeps never exercise."""
    positions, triangles, locked = lattice_patch(80, 110, noise=0.004, seed=3)
    settings = DecimationSettings(max_error_m=0.002)
    rust, python = both_kernels(positions, triangles, locked, settings)
    assert rust.rejected_error > 0, "the cap must actually bite on this fixture"
    assert_identical_patch(rust, python)


@needs_kernel
def test_the_two_kernels_agree_with_endpoint_placement() -> None:
    """`placement="endpoint"` takes a different branch of `_place`, including
    Python's `min(..., key=...)` first-wins tie-break."""
    positions, triangles, locked = lattice_patch(70, 90)
    settings = DecimationSettings(
        placement="endpoint", target_triangles=triangles.shape[0] // 3
    )
    rust, python = both_kernels(positions, triangles, locked, settings)
    assert rust.collapses > 0
    assert_identical_patch(rust, python)


@needs_kernel
def test_a_flat_patch_agrees_where_every_quadric_is_singular() -> None:
    """A plane sends every candidate down the endpoint/midpoint ladder, so this
    is the fixture that exercises `solve_optimal` returning `None`."""
    positions, triangles, locked = lattice_patch(60, 80, bump=0.0)
    settings = DecimationSettings(target_triangles=triangles.shape[0] // 2)
    rust, python = both_kernels(positions, triangles, locked, settings)
    assert_identical_patch(rust, python)


@pytest.fixture(scope="module")
def generation(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("kernel-generation")
    station = synthetic.generate(
        synthetic.RoomScene(mover=True), rows=FIXTURE_ROWS, cols=FIXTURE_COLS,
        dropout=0.01, range_noise=0.002, seed=17, station_id="kernel",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=25_000, halo=3,
        measure=False, out_dir=str(root / "out"),
    )
    assert result.tiles is not None
    assert len(result.tiles.tile_ids) > 1, "the fixture must span more than one tile"
    return result.tiles, root


@needs_kernel
def test_the_two_kernels_agree_on_every_tile_of_a_real_generation(
    generation: Any,
) -> None:
    """The locked set here comes from a real tile's owned count and triangle
    list, which is the one thing the hand-built fixtures cannot supply."""
    store, _ = generation
    seen_locked = 0
    for tile_id in store.tile_ids:
        payload = store.payload(tile_id)
        locked = lock_boundary(
            payload.triangles, payload.vertex_count, payload.owned_count,
            lock_edges=True,
        )
        seen_locked += int(np.count_nonzero(locked))
        settings = DecimationSettings(
            target_triangles=-(-payload.triangle_count // REDUCTION)
        )
        rust, python = both_kernels(
            payload.positions, payload.triangles, locked, settings
        )
        assert_identical_patch(rust, python)
    assert seen_locked > 0, "no tile had a locked boundary; the fixture proves nothing"


@needs_kernel
def test_two_native_runs_agree_byte_for_byte(generation: Any) -> None:
    """Determinism demonstrated, not asserted."""
    store, _ = generation
    tile_id = store.tile_ids[0]
    payload = store.payload(tile_id)
    locked = lock_boundary(
        payload.triangles, payload.vertex_count, payload.owned_count, lock_edges=True
    )
    settings = DecimationSettings(
        target_triangles=-(-payload.triangle_count // REDUCTION)
    )
    first = decimate_patch(
        payload.positions, payload.triangles, locked, settings, kernel="rust"
    )
    second = decimate_patch(
        payload.positions, payload.triangles, locked, settings, kernel="rust"
    )
    assert_identical_patch(first, second)


def edge_use(triangles: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0, return_counts=True)[1]


@needs_kernel
def test_the_locked_join_rule_fires_on_the_native_path(
    generation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 10 §2.3's defect, checked on the kernel that replaced the loop.

    Two assertions, and both are needed. The union staying manifold is the
    property; `rejected_seam > 0` is the evidence that the rule was *reached* on
    this path rather than the fixture never posing the question.

    The environment variable is set explicitly because `decimate_generation`
    takes no kernel argument and the default is `python`
    (`decimate_kernel.DEFAULT_CHOICE`). Without it this test would quietly
    exercise the Python sweep and assert nothing about the native one — which is
    the same trap, in the other direction, that the default exists to close.
    """
    monkeypatch.setenv(decimate_kernel.ENVIRONMENT_VARIABLE, "rust")
    store, root = generation
    report = decimate_generation(
        store, root / "native", DecimationSettings(), reduction=REDUCTION
    )
    _, triangles, _ = read_decimated(root / "native")
    rejected = sum(tile.rejected_seam for tile in report.tiles)
    assert rejected > 0, "the locked-join rule never fired; the fixture proves nothing"
    assert int(edge_use(triangles).max()) <= 2, "the join produced a doubled surface"
    assert report.seams_clean


def test_an_unknown_kernel_name_is_refused() -> None:
    positions, triangles, locked = lattice_patch(20, 20)
    with pytest.raises(ValueError, match="unknown kernel"):
        decimate_patch(positions, triangles, locked, kernel="fortran")


def test_demanding_an_unbuilt_kernel_fails_loudly_rather_than_falling_back() -> None:
    """`"rust"` is a demand. Silence here would turn a missing build into a
    throughput measurement of the Python path reported as the kernel's."""
    if decimate_kernel.available():
        pytest.skip("the kernel is built in this tree, so it cannot be missing")
    positions, triangles, locked = lattice_patch(20, 20)
    with pytest.raises(RuntimeError, match="unavailable"):
        decimate_patch(positions, triangles, locked, kernel="rust")


def test_a_tree_without_the_crate_still_decimates() -> None:
    """The fallback, which is what keeps a clean clone working without Rust."""
    positions, triangles, locked = lattice_patch(40, 50)
    settings = DecimationSettings(target_triangles=triangles.shape[0] // 2)
    patch = decimate_patch(positions, triangles, locked, settings, kernel="python")
    assert patch.collapses > 0
    assert patch.triangle_count < triangles.shape[0]


def test_the_kernel_describes_itself_for_the_record() -> None:
    """A measurement record has to be able to say which path produced it."""
    description = decimate_kernel.describe()
    assert set(description) == {"available", "reason", "contract_version"}
    assert isinstance(description["available"], bool)
    assert description["available"] is (description["reason"] is None)
