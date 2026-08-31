"""
`reverse-qa-v2`'s systematic selection, without the station-sized cumulative.

WP-3.1. `reverse_qa.run_reverse_qa` used to materialise two float64 arrays over
the whole station to draw a bounded sample from it: the canonical-order areas,
and their running total. Sixteen bytes per triangle, for a selection of at most
`max_samples` records.

**Why it can be streamed at all, and exactly.** `np.cumsum` is a strict
left-to-right recurrence — pairwise summation is `np.sum`'s trick, not
`cumsum`'s — so every partial sum is `c[i] = c[i-1] + a[i]` in float64. A block
boundary is therefore invisible if the carry is *inside* the recurrence rather
than added to it afterwards:

    np.cumsum(np.concatenate(([carry], block)))[1:]

is the continuation, bit for bit, while `carry + np.cumsum(block)` is not —
float addition does not associate, and using it would move the selection by a
rounding step on some triangle somewhere and be nearly impossible to find.
`tests/test_qa_bound.py` asserts the streamed cumulative against the resident
one at several block sizes rather than taking the argument on trust.

Two merge passes replace the two arrays: one to reach the total the sample
targets are scaled by, one to resolve every target and gather only the records
it selected. That is the same number of passes the resident form needed — it
already re-merged to gather — so the reduction costs no extra I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]


def cumulative_total(runs: Any) -> float:
    """`float(np.cumsum(area)[-1])` over canonical order, without the array.

    Zero for an empty run set, which is the value the caller needs to refuse
    the selection rather than divide by it.
    """
    import numpy as np

    from .pass_b_merge import merge_runs

    carry = 0.0
    for block in merge_runs(runs):
        if block.shape[0] == 0:
            continue
        carry = float(
            np.cumsum(np.concatenate((np.array([carry], np.float64), block["area"])))[-1]
        )
    return carry


def selection_targets(total: float, n: int) -> F64:
    """The `n` systematic sample positions along the area axis.

    Kept here beside the recurrence it is compared against: the targets and the
    cumulative have to be produced from the same `total`, and separating them
    is how a selection quietly stops being reproducible.
    """
    import numpy as np

    out: F64 = (np.arange(n, dtype=np.float64) + 0.5) * total / n
    return out


def select_and_gather(runs: Any, targets: F64) -> tuple[F64, I64, I64]:
    """Resolve every target against the streamed cumulative and gather it.

    Returns `(corners, canonical triples, vertex indices)` for the selected
    samples, in target order. `targets` is ascending and the cumulative is
    non-decreasing, so each block resolves a contiguous prefix of what is left
    and one pass suffices.

    The selection rule is the resident one unchanged:
    `clip(searchsorted(cumulative, t, side="right"), 0, count - 1)`. The clip's
    upper arm is the tail case — a target that rounds past the final cumulative
    value takes the last record — and it is applied here by remembering that
    record rather than by holding the array it used to be read from.

    Corner positions come out of the record itself (WP-3.3), not from a vertex
    array indexed by the selection. That is what lets a tile store answer this at
    all — it has no station-wide vertex array to index — and it is bit for bit
    the same value, because the record stores the float32 the surface holds and
    float32 widens to float64 exactly.
    """
    import numpy as np

    from .pass_b_merge import merge_runs

    n = int(targets.shape[0])
    corners = np.empty((n, 3, 3), np.float64)
    triples = np.empty((n, 3), np.int64)
    vertex_index = np.empty((n, 3), np.int64)
    if n == 0:
        return corners, triples, vertex_index

    cursor = 0
    carry = 0.0
    last: Any = None
    for block in merge_runs(runs):
        size = int(block.shape[0])
        if size == 0:
            continue
        running = np.cumsum(
            np.concatenate((np.array([carry], np.float64), block["area"]))
        )[1:]
        carry = float(running[-1])
        last = block[-1:]
        if cursor >= n:
            continue
        local = np.searchsorted(running, targets[cursor:], side="right")
        take = int(np.searchsorted(local, size, side="left"))
        if take:
            _fill(corners, triples, vertex_index, cursor, block[local[:take]])
            cursor += take

    if cursor < n:
        if last is None:
            raise ValueError("reverse-QA selection ran with no records to select from")
        # Targets past the final cumulative value. `np.repeat` rather than a
        # loop so the tail costs the same whether it is one sample or many.
        _fill(
            corners, triples, vertex_index, cursor, np.repeat(last, n - cursor)
        )
    return corners, triples, vertex_index


def _fill(
    corners: F64, triples: I64, vertex_index: I64, at: int, chosen: Any
) -> None:
    """Write one contiguous run of selected records into the output arrays."""
    import numpy as np

    size = int(chosen.shape[0])
    vertex_index[at : at + size] = np.stack(
        [chosen["i0"], chosen["i1"], chosen["i2"]], axis=1
    ).astype(np.int64)
    corners[at : at + size] = chosen["p"].astype(np.float64)
    triples[at : at + size] = np.stack(
        [chosen["s0"], chosen["s1"], chosen["s2"]], axis=1
    ).astype(np.int64)
