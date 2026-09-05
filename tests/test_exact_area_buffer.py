"""
What the exact-area second pass buffers — WP-12b, ITEM-022.

WP-12a sized it: on ordinal 20 one flagged component holds 25,598,786 triangles,
and `exact_component_areas` kept a CPython float plus a list slot for each of
them — 819,161,152 B, 55.5% of that prefix's tracked peak. The same values as a
float64 array are 204,790,288 B.

This is a storage change and nothing else. `math.fsum` still runs, over the same
summands, in the same merge order, so the value it returns must not move by a
single bit — which is what `GOLDEN_BITS` pins, recorded from the list
implementation before it was replaced.

Synthetic fixtures only. No `H:\\Sample` access is made and none is claimed.
"""

from __future__ import annotations

import math
import pathlib
import struct
from typing import Any

import numpy as np
import pytest

from rapidmesh.pass_b import TRIANGLE_RUN_FIELDS, TRIANGLE_RUN_KEY
from rapidmesh.pass_b_area import (
    BOUNDARY_SAFETY,
    EXACTNESS_RATIO_LIMIT,
    exact_component_areas,
    stream_component_areas,
)
from rapidmesh.pass_b_merge import write_runs

# Five lattice cells, five vertices. Two of them sit 1e-5 m from the origin, so
# the triangle they form with it has an area ~1e10 times smaller than the
# component's total — which is how a component earns the exact fallback. Real
# stations earn it the same way: ordinal 20's flagged component spans 94% of the
# station, so its total dwarfs its smallest face.
CELLS = np.array([10, 20, 30, 40, 50], np.int64)
XYZ = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
     [1.0e-5, 0.0, 0.0], [0.0, 1.0e-5, 0.0]],
    np.float32,
)
TRIPLES = ((10, 20, 30), (10, 40, 50), (20, 30, 40), (10, 20, 40))

# `math.fsum` over the four areas in merge order, from the list implementation
# this package replaced. Stored as bits because the claim is bitwise.
GOLDEN_BITS = "38df0680f5ffef3f"
GOLDEN_VALUE = struct.unpack("<d", bytes.fromhex(GOLDEN_BITS))[0]
FLAGGED_ROOT = 0
FLAGGED_TRIANGLES = 4


def _runs(work: pathlib.Path) -> Any:
    dtype = np.dtype(TRIANGLE_RUN_FIELDS)
    record = np.empty(len(TRIPLES), dtype)
    record["root"] = FLAGGED_ROOT
    record["rot"] = 0
    for index, (c0, c1, c2) in enumerate(TRIPLES):
        record["c0"][index] = c0
        record["c1"][index] = c1
        record["c2"][index] = c2
    return write_runs([record], work, "tri", TRIANGLE_RUN_KEY)


def test_the_fixture_actually_trips_the_exactness_condition(
    tmp_path: pathlib.Path,
) -> None:
    """Otherwise every other test here would pass on the empty early return.

    `exact_component_areas` returns `{}` when nothing is flagged, so a fixture
    that failed to trip the condition would make the buffer tests vacuous
    without failing.
    """
    accumulator = stream_component_areas(_runs(tmp_path), CELLS, XYZ)
    ratios = accumulator.ratios()
    assert ratios[FLAGGED_ROOT] > EXACTNESS_RATIO_LIMIT * BOUNDARY_SAFETY
    assert accumulator.needs_exact() == {FLAGGED_ROOT}
    assert accumulator.counts[FLAGGED_ROOT] == FLAGGED_TRIANGLES


def _fsum_arguments(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> list[Any]:
    """Whatever `exact_component_areas` hands to `math.fsum`, as it hands it.

    The function does `import math` in its own body, so the name resolves to the
    stdlib module attribute at call time and replacing that attribute is enough
    to see the argument without touching the function.
    """
    seen: list[Any] = []
    real = math.fsum

    def spy(values: Any) -> float:
        seen.append(values)
        return real(values)

    monkeypatch.setattr(math, "fsum", spy)
    runs = _runs(tmp_path)
    accumulator = stream_component_areas(runs, CELLS, XYZ)
    exact_component_areas(runs, CELLS, XYZ, accumulator.needs_exact())
    return seen


def test_the_flagged_buffer_is_a_float64_array_not_a_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The storage claim, observed at the moment of summation.

    Red before WP-12b: the argument is a `list` of CPython floats — 32 B a
    triangle against the 8 B the same value needs as float64.
    """
    seen = _fsum_arguments(tmp_path, monkeypatch)

    assert len(seen) == 1, f"expected one flagged component, summed once; got {len(seen)}"
    buffer = seen[0]
    assert isinstance(buffer, np.ndarray), (
        f"the exact-area buffer is still {type(buffer).__name__}; "
        "WP-12b replaces it with a float64 array"
    )
    assert buffer.dtype == np.float64
    assert not isinstance(buffer, list)


def test_the_buffer_is_sized_exactly_rather_than_grown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One allocation of the right length, not a doubling sequence.

    The brief forbids guessing lengths by growing, so the array handed to
    `fsum` must be exactly the component's triangle count — no spare capacity,
    no trailing zeros that would silently join the sum.
    """
    buffer = _fsum_arguments(tmp_path, monkeypatch)[0]
    assert buffer.shape == (FLAGGED_TRIANGLES,), (
        f"buffer is {buffer.shape}, expected exactly {FLAGGED_TRIANGLES} entries"
    )
    # Not "every entry is positive": one of the four triples here is collinear
    # and has area exactly 0.0, which is real geometry rather than a spare slot.
    # Trailing zeros are invisible to `fsum` — adding 0.0 changes nothing — so
    # the size guard has to be tested directly, below.


def test_a_wrong_size_is_refused_rather_than_summed(
    tmp_path: pathlib.Path,
) -> None:
    """The over-allocation guard, shown failing.

    A buffer one slot too long would sum to the same value, because the spare
    slot holds 0.0 and `fsum` is indifferent to it — so a green run of the test
    above only means something if the guard it relies on can still fire. Here
    it is handed a count that is deliberately wrong.
    """
    runs = _runs(tmp_path)
    with pytest.raises(ValueError, match="were counted"):
        exact_component_areas(
            runs, CELLS, XYZ, {FLAGGED_ROOT},
            counts={FLAGGED_ROOT: FLAGGED_TRIANGLES + 1},
        )


def test_the_fsum_value_is_bitwise_what_the_list_path_produced(
    tmp_path: pathlib.Path,
) -> None:
    """The value, to the last bit, against the implementation this replaced.

    A guard rather than a red-first test: it passed before the change and must
    keep passing. `PHASE1-ISLANDS-FINALISATION.md` §4.1 makes this sum
    authoritative for a component the streamed total cannot vouch for, and it
    lands on a keep-or-delete decision about survey geometry, so `np.isclose`
    would be the wrong instrument.
    """
    runs = _runs(tmp_path)
    accumulator = stream_component_areas(runs, CELLS, XYZ)
    exact = exact_component_areas(runs, CELLS, XYZ, accumulator.needs_exact())

    assert set(exact) == {FLAGGED_ROOT}
    value = exact[FLAGGED_ROOT]
    assert struct.pack("<d", value).hex() == GOLDEN_BITS, (
        f"exact area moved: {value!r} against the recorded {GOLDEN_VALUE!r}"
    )


def test_nothing_flagged_still_returns_the_empty_dict(
    tmp_path: pathlib.Path,
) -> None:
    """The early return is the path every synthetic fixture in the repo takes.

    It must not start allocating, and it must not start a counting pass over
    the merge for an empty root set.
    """
    assert exact_component_areas(_runs(tmp_path), CELLS, XYZ, set()) == {}
