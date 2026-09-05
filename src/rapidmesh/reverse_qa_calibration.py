"""
Calibrating reverse QA's row window — split from `reverse_qa.py` for DEC-010.

A coherent boundary rather than a convenient one: `reverse_qa.py` computes the
metric, and this answers a different question — *is the bounded candidate set
the same answer as the unbounded one?* It is a measurement tool used by
`tests/test_reverse_qa.py` and by window calibration, never by the pipeline, and
it is the only caller that deliberately runs the unbounded path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .reverse_qa import DEFAULT_QA_WINDOW_ROWS, run_reverse_qa

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .types import MeshData

    F32 = npt.NDArray[np.float32]
    I32 = npt.NDArray[np.int32]


def calibrate_window(
    mesh: MeshData | Any,
    source_points: F32,
    *,
    source_rows: I32,
    qa_window_rows: int = DEFAULT_QA_WINDOW_ROWS,
    max_samples: int = 500_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bounded against unbounded, on the *same* samples.

    Selection and interior points are deterministic functions of the mesh, so
    the two runs measure identical points and only the candidate set differs.
    Any disagreement is therefore attributable to the window and to nothing
    else. **"Match" means bitwise-equal float64 distance**, not "close": a
    window that moved an answer by a rounding step would still have moved it.
    """
    import numpy as np

    near, evidence = run_reverse_qa(
        mesh, source_points, source_rows=source_rows,
        qa_window_rows=qa_window_rows, max_samples=max_samples, seed=seed,
    )
    far, unbounded = run_reverse_qa(
        mesh, source_points, source_rows=source_rows,
        qa_window_rows=None, max_samples=max_samples, seed=seed,
    )
    if near.size != far.size:
        raise ValueError(
            "bounded and unbounded runs measured different sample counts: "
            f"{near.size} != {far.size}"
        )
    identical = int(np.count_nonzero(near == far))
    difference = near - far
    return {
        "samples": float(near.size),
        "identical": float(identical),
        "identical_fraction": identical / near.size if near.size else 1.0,
        "max_overestimate_m": float(difference.max()) if near.size else 0.0,
        "min_difference_m": float(difference.min()) if near.size else 0.0,
        "unmatched": float(evidence.samples_unmatched),
        "largest_candidate_set": float(evidence.largest_window_candidates),
        "unbounded_candidate_set": float(unbounded.largest_window_candidates),
    }
