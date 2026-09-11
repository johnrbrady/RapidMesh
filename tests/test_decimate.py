"""
The quadric decimator's own invariants — WP-3.4, Round 10.

Five properties, each of which the collapse loop can break silently and none of
which a triangle count would reveal:

1. **Determinism.** Two runs at the same settings return the same bytes
   (`PHASE1-DETERMINISM-SPEC.md` §3).
2. **Locked vertices are locked.** Every one survives, with its float32 bits
   unchanged, and none is ever flagged as moved. That, and nothing else, is what
   makes two tiles agree on a seam.
3. **The error bound is a bound.** No accepted collapse exceeds
   `max_error_m`, and tightening it monotonically reduces how far the sweep
   gets.
4. **The result is still a manifold.** No boundary edge is created and no edge
   ends up used by more than two triangles — the link condition's job.
5. **`placement="endpoint"` invents no position.** Every output vertex sits
   exactly on an input vertex.

Honesty condition (standing rule 8)
------------------------------------
These are **regression tests, not bug-fix tests**: the code they cover is new in
this package, so they are expected to pass on first run and are not presented as
proven-failing. The sensitivity evidence that outranks a green run lives in
`tests/test_decimate_seams.py`, where the seam check is shown red against a
deliberately broken seam. Property 3 here also carries its own red case: the
tight bound is asserted to stop the sweep short of the loose one, so a bound that
was not being applied would fail rather than merely look plausible.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidmesh.decimate import DecimationSettings, decimate_patch


def lattice_patch(
    rows: int, cols: int, bump: float = 0.05, noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A gently curved grid with its outer rim locked.

    Curved rather than flat on purpose: on a plane every quadric is singular,
    the optimal solve never runs, and the fallback ladder would be the only path
    exercised. The bump is small enough that the surface stays a height field.
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


def edge_use_counts(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tris = np.asarray(triangles, np.int64)
    pairs = np.concatenate((tris[:, (0, 1)], tris[:, (1, 2)], tris[:, (2, 0)]), axis=0)
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0, return_counts=True)


def test_two_runs_at_the_same_settings_return_the_same_bytes() -> None:
    positions, triangles, locked = lattice_patch(48, 60, noise=0.001)
    settings = DecimationSettings(target_triangles=triangles.shape[0] // 4)
    first = decimate_patch(positions, triangles, locked, settings)
    second = decimate_patch(positions, triangles, locked, settings)

    assert first.collapses == second.collapses
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.source_index, second.source_index)
    # Bit equality, not `allclose`: a decimator whose output drifts in the last
    # place is a decimator whose two LODs disagree about where a seam is.
    assert first.positions.tobytes() == second.positions.tobytes()


def test_locked_vertices_all_survive_with_their_bits_unchanged() -> None:
    positions, triangles, locked = lattice_patch(40, 50, noise=0.001)
    out = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 8),
    )
    lock_ids = np.flatnonzero(locked)
    assert np.isin(lock_ids, out.source_index).all()

    slot = {int(s): i for i, s in enumerate(out.source_index)}
    for source in lock_ids:
        row = slot[int(source)]
        assert out.positions[row].tobytes() == positions[source].tobytes()
        assert not bool(out.moved[row])


def test_no_accepted_collapse_exceeds_the_error_bound() -> None:
    positions, triangles, locked = lattice_patch(40, 50, noise=0.001)
    bound = 0.0005
    out = decimate_patch(
        positions, triangles, locked, DecimationSettings(max_error_m=bound)
    )
    assert out.max_accepted_error_m <= bound
    assert out.rejected_error > 0, "the bound should have refused some candidate"


def test_a_tighter_bound_stops_the_sweep_earlier() -> None:
    """The bound's red case: if it were ignored, both runs would land together."""
    positions, triangles, locked = lattice_patch(40, 50, noise=0.001)
    tight = decimate_patch(
        positions, triangles, locked, DecimationSettings(max_error_m=0.0001)
    )
    loose = decimate_patch(
        positions, triangles, locked, DecimationSettings(max_error_m=0.01)
    )
    assert tight.collapses < loose.collapses
    assert tight.triangle_count > loose.triangle_count


def test_the_decimated_patch_is_still_a_manifold() -> None:
    positions, triangles, locked = lattice_patch(40, 50, noise=0.001)
    before_edges, before_counts = edge_use_counts(triangles)
    out = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(target_triangles=triangles.shape[0] // 6),
    )
    after_edges, after_counts = edge_use_counts(out.triangles)
    assert int(after_counts.max()) <= 2, "a collapse created a non-manifold edge"

    # The boundary is the rim, and the rim is locked, so it must come through
    # unchanged once both sides are expressed in input indices.
    original = {
        (int(u), int(v)) for u, v in before_edges[before_counts == 1].tolist()
    }
    survived = {
        (int(u), int(v))
        for u, v in out.source_index[after_edges[after_counts == 1]].tolist()
    }
    assert survived == original


def test_endpoint_placement_never_invents_a_position() -> None:
    positions, triangles, locked = lattice_patch(30, 36, noise=0.001)
    out = decimate_patch(
        positions, triangles, locked,
        DecimationSettings(
            target_triangles=triangles.shape[0] // 4, placement="endpoint"
        ),
    )
    seen = {positions[i].tobytes() for i in range(positions.shape[0])}
    assert all(out.positions[i].tobytes() in seen for i in range(out.vertex_count))


def test_a_fully_locked_patch_is_returned_untouched() -> None:
    positions, triangles, _ = lattice_patch(12, 14)
    locked = np.ones(positions.shape[0], bool)
    out = decimate_patch(positions, triangles, locked, DecimationSettings())
    assert out.collapses == 0
    assert out.triangle_count == triangles.shape[0]
    assert out.positions.tobytes() == positions.tobytes()


def test_an_empty_patch_is_not_an_error() -> None:
    out = decimate_patch(
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.uint32),
        np.zeros(0, bool),
        DecimationSettings(),
    )
    assert out.vertex_count == 0
    assert out.triangle_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"placement": "nearest"},
        {"max_error_m": 0.0},
        {"target_triangles": -1},
        {"max_normal_turn_deg": 0.0},
    ],
)
def test_settings_reject_a_value_that_could_not_be_honoured(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DecimationSettings(**kwargs)


def test_a_triangle_naming_a_vertex_outside_the_patch_is_refused() -> None:
    positions, triangles, locked = lattice_patch(8, 8)
    broken = triangles.copy()
    broken[0, 0] = np.uint32(positions.shape[0])
    with pytest.raises(ValueError, match="outside the patch"):
        decimate_patch(positions, broken, locked, DecimationSettings())
