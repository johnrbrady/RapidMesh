"""
Bounded reverse QA — `reverse-qa-v2`, PLAN.md §5 item 9, specified in
`PHASE1-DETERMINISM-SPEC.md` §5(h).

**What was wrong with v1.** `qa.mesh_to_source_report` selects triangles with
`rs.choice(tri_ids, p=weights/weights.sum())` over a probability vector in
*triangle array order*. Inverse-CDF sampling over a reordered vector picks
different triangles, so the metric moved when nothing about the data moved.
SPEC §5(h) measured 1.371 mm of p99.9 spread from `triangulate`'s internal band
size alone, and on this repository's own 12 x 48 harness fixture the in-memory
and streamed paths disagreed by about 5% of RMS at every band size while
forward QA was bitwise identical. No tolerance was offered for that, and none
is offered here: one wide enough to absorb it would be comparable to the
quantity being tested.

**What v2 does instead.** Every step is a function of the mesh's own content,
never of array order:

1. **Identity** — each positive-area triangle becomes its canonical oriented
   `source_sample_id` triple (SPEC §4). Reversed winding is a different
   identity, never the same one.
2. **Canonical order** — triangles are sorted by that triple.
3. **Systematic area-weighted selection** — one strict left-to-right float64
   cumulative recurrence over canonical order; sample `j` takes the triangle
   whose half-open cumulative interval contains `(j + 0.5) x total / n`. No
   RNG, so no process-global state and nothing to reseed.
4. **Interior point** — SHA-256 over a domain separator, the seed, the
   canonical triple and the global sample ordinal, then the same square-root
   barycentric transform v1 used.
5. **Fixed row window** — correspondence is built only against retained
   observations in `[min_row - qa_window_rows, max_row + qa_window_rows]`,
   clipped to the scan. `qa_window_rows` defaults to 8 and is recorded.

**What "exact within the window" means, precisely.** The reported distance is
the exact distance to the nearest retained observation *inside the window*. If
the true nearest observation lies outside it, the reported figure is an
**over-estimate** — never an under-estimate, and never an invented match. A
sample whose window holds no retained observation at all is **excluded and
counted**, not assigned a distance. `calibrate_window` measures how often the
window changes the answer on a fixture small enough to compute both.

**Not in scope.** The reservoir strategy is the Pipeline B fallback
(PLAN.md §5 item 9) and is **not implemented**; a caller needing it must treat
it as unsupported.

SPEC §5(h) step 2's bounded k-way merge over persisted runs **is** implemented
now (ITEM-009): the canonical order comes from `pass_b_merge`, so neither the
sort nor the `(T,3,3)` corner array is resident for the station.
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import DeviationReport, MeshData

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I32 = npt.NDArray[np.int32]
    I64 = npt.NDArray[np.int64]

REVERSE_QA_VERSION = "reverse-qa-v2"
REVERSE_QA_METRIC = "mesh-to-retained-source-v2"
CANONICAL_ORDER_VERSION = "canonical-oriented-triple-v1"

# `PHASE1-DETERMINISM-SPEC.md` §5(h) step 6: a recorded semantic setting, not
# the processing band size. 8 is the spec's initial calibration candidate.
DEFAULT_QA_WINDOW_ROWS = 8

_DOMAIN = b"rapidmesh-reverse-qa-v2"
_TWO_64 = float(1 << 64)


@dataclass(frozen=True)
class ReverseQAEvidence:
    """Everything needed to reproduce or audit one reverse-QA figure.

    Carried beside the `DeviationReport` rather than inside it: adding fields
    to the exported report is a schema change, and the window is a property of
    *how* the figure was produced, not of the mesh.
    """

    metric_version: str
    canonical_order_version: str
    seed: int
    max_samples: int
    samples_selected: int
    samples_measured: int
    samples_unmatched: int
    qa_window_rows: int | None
    positive_area_triangles: int
    windows_used: int
    largest_window_candidates: int

    @property
    def bounded(self) -> bool:
        """False for the unbounded calibration reference."""
        return self.qa_window_rows is not None

    def describe(self) -> str:
        window = "unbounded" if self.qa_window_rows is None else f"+/-{self.qa_window_rows} rows"
        return (
            f"{self.metric_version} window={window} "
            f"selected={self.samples_selected} measured={self.samples_measured} "
            f"unmatched={self.samples_unmatched} windows={self.windows_used} "
            f"largest_candidate_set={self.largest_window_candidates}"
        )

    def as_settings(self) -> tuple[tuple[str, str], ...]:
        """The subset that belongs in `QAReportMetadata.settings`."""
        return (
            ("reverse_qa_version", self.metric_version),
            ("qa_window_rows", str(self.qa_window_rows)),
        )


def mesh_to_source_report_v2(
    mesh: MeshData,
    source_points: F32,
    *,
    source_rows: I32,
    qa_window_rows: int | None = DEFAULT_QA_WINDOW_ROWS,
    max_samples: int = 500_000,
    seed: int = 0,
    work_dir: Path | None = None,
) -> tuple[DeviationReport, ReverseQAEvidence]:
    """Sample triangle interiors and measure to retained observations, bounded.

    `source_rows` is the lattice row of each entry of `source_points`, in the
    same order. It is what makes the window addressable: the mesh knows which
    rows a triangle came from, so it knows which observations could plausibly
    be its nearest neighbour without consulting the whole station.
    """
    from .qa import _summarise

    distances, evidence = run_reverse_qa(
        mesh, source_points, source_rows=source_rows,
        qa_window_rows=qa_window_rows, max_samples=max_samples, seed=seed,
        work_dir=work_dir,
    )
    report = _summarise(
        distances, metric=REVERSE_QA_METRIC, population=mesh.triangle_count,
        exact=False, source_of_truth="retained source observations",
    )
    return report, evidence


def run_reverse_qa(
    mesh: MeshData,
    source_points: F32,
    *,
    source_rows: I32,
    qa_window_rows: int | None = DEFAULT_QA_WINDOW_ROWS,
    max_samples: int = 500_000,
    seed: int = 0,
    work_dir: Path | None = None,
) -> tuple[F64, ReverseQAEvidence]:
    """The per-sample distances and the evidence behind one reverse-QA figure.

    One implementation, used by the report and by the calibration, so the two
    cannot drift into measuring different things.

    `work_dir` holds the bounded merge runs. When absent a temporary directory
    is used and removed; Pass B passes its own so the runs sit beside the
    segments they were derived from.
    """
    import numpy as np

    from .pass_b_merge import merge_runs

    tris = np.asarray(mesh.triangles, dtype=np.int64)
    ids = mesh.source_sample_id
    if (
        mesh.triangle_count == 0
        or source_points.shape[0] == 0
        or max_samples <= 0
        or ids is None
    ):
        return np.empty(0, np.float64), _no_evidence(qa_window_rows, max_samples, seed)

    # ITEM-009. The canonical order comes from the bounded external merge, not
    # a resident sort: `verts[tris]` alone is a (T,3,3) float64 array — 72 bytes
    # per triangle — and it used to be built for the whole station just to draw
    # a bounded sample from it.
    with _run_directory(work_dir) as merge_dir:
        runs = _build_reverse_runs(mesh, tris, np.asarray(ids, np.int64), merge_dir)
        if runs.record_count == 0:
            return np.empty(0, np.float64), _no_evidence(
                qa_window_rows, max_samples, seed
            )

        # Areas in canonical order, then one strict left-to-right recurrence —
        # the same values and the same summation the resident path performed,
        # so the selection cannot shift by a rounding step.
        count = runs.record_count
        area = np.empty(count, np.float64)
        at = 0
        for block in merge_runs(runs):
            area[at : at + block.shape[0]] = block["area"]
            at += block.shape[0]
        cumulative = np.cumsum(area)
        del area

        n = min(max_samples, max(count, min(10_000, max_samples)))
        targets = (np.arange(n, dtype=np.float64) + 0.5) * float(cumulative[-1]) / n
        picked = np.clip(
            np.searchsorted(cumulative, targets, side="right"), 0, count - 1
        )
        del cumulative

        corners, triples, vertex_index = _gather_selected(runs, picked, mesh)

    samples = _interior_points(corners, triples, seed)
    rows = np.asarray(source_rows, dtype=np.int64)
    tri_rows = rows[vertex_index]
    distances, unmatched, windows, largest = _windowed_distances(
        samples, np.asarray(source_points, np.float64), rows,
        tri_rows.min(axis=1), tri_rows.max(axis=1), qa_window_rows,
    )
    return distances, ReverseQAEvidence(
        metric_version=REVERSE_QA_VERSION,
        canonical_order_version=CANONICAL_ORDER_VERSION,
        seed=seed, max_samples=max_samples,
        samples_selected=int(n), samples_measured=int(distances.size),
        samples_unmatched=int(unmatched), qa_window_rows=qa_window_rows,
        positive_area_triangles=count, windows_used=int(windows),
        largest_window_candidates=int(largest),
    )


def _no_evidence(
    qa_window_rows: int | None, max_samples: int, seed: int
) -> ReverseQAEvidence:
    return ReverseQAEvidence(
        metric_version=REVERSE_QA_VERSION,
        canonical_order_version=CANONICAL_ORDER_VERSION,
        seed=seed, max_samples=max_samples, samples_selected=0, samples_measured=0,
        samples_unmatched=0, qa_window_rows=qa_window_rows,
        positive_area_triangles=0, windows_used=0, largest_window_candidates=0,
    )


def _canonical_triples(triples: I64) -> I64:
    """Rotate each triple to start at its smallest id. Rotation only — never
    reversal, because winding carries orientation (SPEC §4 clause 3)."""
    import numpy as np

    pick = np.argmin(triples, axis=1)
    rows = np.arange(triples.shape[0])
    out: I64 = np.stack(
        [triples[rows, (pick + k) % 3] for k in (0, 1, 2)], axis=1
    )
    return out


def _interior_points(corners: F64, triples: I64, seed: int) -> F64:
    """One deterministic interior point per selected sample.

    The two uniforms come from SHA-256 over the domain separator, the seed, the
    canonical triple and the sample's global ordinal, so a sample's position
    depends on its identity and not on when it was drawn. No process-global RNG
    is touched, which is what makes two runs — and two pipelines — agree.
    """
    import numpy as np

    n = triples.shape[0]
    u1 = np.empty(n, np.float64)
    u2 = np.empty(n, np.float64)
    for j in range(n):
        a, b, c = triples[j]
        digest = hashlib.sha256(
            _DOMAIN + struct.pack("<QqqqQ", seed, int(a), int(b), int(c), j)
        ).digest()
        u1[j] = (int.from_bytes(digest[0:8], "little") + 0.5) / _TWO_64
        u2[j] = (int.from_bytes(digest[8:16], "little") + 0.5) / _TWO_64

    root = np.sqrt(u1)
    out: F64 = (
        (1.0 - root)[:, None] * corners[:, 0]
        + (root * (1.0 - u2))[:, None] * corners[:, 1]
        + (root * u2)[:, None] * corners[:, 2]
    )
    return out


def _windowed_distances(
    samples: F64,
    source_points: F64,
    source_rows: I64,
    lo_rows: I64,
    hi_rows: I64,
    qa_window_rows: int | None,
) -> tuple[F64, int, int, int]:
    """Nearest retained observation within each sample's own row window.

    Samples are grouped by their exact `(min_row, max_row)` pair, so every
    member of a group has the *identical* candidate set — the only coalescing
    SPEC §5(h) step 6 permits. Widening a window to serve several groups would
    add candidates and could only shorten distances, which is precisely the
    silent improvement this metric must not make.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    if qa_window_rows is None:
        tree = cKDTree(source_points)
        distances, _ = tree.query(samples, k=1, workers=1)
        return np.asarray(distances, np.float64), 0, 1, int(source_points.shape[0])

    out = np.full(samples.shape[0], np.nan, np.float64)
    keys = np.stack([lo_rows, hi_rows], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(source_rows, kind="stable")
    sorted_rows = source_rows[order]
    largest = 0
    for group in range(unique.shape[0]):
        members = np.nonzero(inverse == group)[0]
        lo = int(unique[group, 0]) - qa_window_rows
        hi = int(unique[group, 1]) + qa_window_rows
        start, stop = np.searchsorted(sorted_rows, [lo, hi + 1])
        if stop <= start:
            continue                      # no retained observation in the window
        candidates = order[start:stop]
        largest = max(largest, int(candidates.size))
        tree = cKDTree(source_points[candidates])
        distances, _ = tree.query(samples[members], k=1, workers=1)
        out[members] = distances

    matched = np.isfinite(out)
    return out[matched], int((~matched).sum()), int(unique.shape[0]), largest


def calibrate_window(
    mesh: MeshData,
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


# ---------------------------------------------------------------------------
# bounded canonical order — the shared merge layer, second view
# ---------------------------------------------------------------------------

REVERSE_RUN_FIELDS = [
    ("s0", "<i8"), ("s1", "<i8"), ("s2", "<i8"),
    ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4"),
    ("area", "<f8"),
]
REVERSE_RUN_KEY = ("s0", "s1", "s2")

# Triangles per run-creation block. Bounded so the (block, 3, 3) float64 corner
# array is a fixed cost rather than a multiple of the station.
REVERSE_BLOCK_TRIANGLES = 65_536


@contextmanager
def _run_directory(work_dir: Path | None) -> Iterator[Path]:
    """Where the merge runs live. A caller's directory, or a temporary one."""
    import shutil
    import tempfile
    from pathlib import Path as _Path

    if work_dir is not None:
        yield work_dir
        return
    made = tempfile.mkdtemp(prefix="rapidmesh-reverseqa-")
    try:
        yield _Path(made)
    finally:
        shutil.rmtree(made, ignore_errors=True)


def _build_reverse_runs(
    mesh: MeshData, tris: I64, ids: I64, work_dir: Path
) -> Any:
    """Sorted runs keyed by canonical `source_sample_id` triple.

    This is `PHASE1-TILE-CONTRACT-V0.md` §2's *second* merge view — triple
    alone, no component root — and it is a different key on different files, so
    it cannot be mistaken for the area view.

    The stored vertex indices keep the mesh's **original winding**. Rotating
    them to canonical form would be invisible to the T2 triangle comparison and
    would still move every barycentric interior point, so the canonical form is
    the sort key only and never the payload.
    """
    import numpy as np

    from .pass_b_merge import write_runs

    def blocks() -> Iterator[Any]:
        for start in range(0, tris.shape[0], REVERSE_BLOCK_TRIANGLES):
            piece = tris[start : start + REVERSE_BLOCK_TRIANGLES]
            corners = mesh.vertices.astype(np.float64)[piece]
            area = 0.5 * np.linalg.norm(
                np.cross(
                    corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
                ), axis=1,
            )
            positive = np.nonzero(area > 0.0)[0]
            if positive.size == 0:
                continue
            kept = piece[positive]
            triples = _canonical_triples(ids[kept])
            record = np.empty(positive.size, dtype=np.dtype(REVERSE_RUN_FIELDS))
            for column, name in enumerate(("s0", "s1", "s2")):
                record[name] = triples[:, column]
            for column, name in enumerate(("i0", "i1", "i2")):
                record[name] = kept[:, column]
            record["area"] = area[positive]
            yield record

    return write_runs(blocks(), work_dir, "reverseqa", REVERSE_RUN_KEY)


def _gather_selected(runs: Any, picked: I64, mesh: MeshData) -> tuple[F64, I64, I64]:
    """Corners, canonical triples and vertex indices for the selected samples.

    `picked` is ascending, so one more merged pass suffices: each block covers a
    contiguous span of canonical positions and the selections inside it are
    taken by offset. Only the sampled triangles are ever materialised, which is
    the whole reduction — `n` is capped by the caller, the station is not.
    """
    import numpy as np

    from .pass_b_merge import merge_runs

    verts = mesh.vertices.astype(np.float64)
    corners = np.empty((picked.size, 3, 3), np.float64)
    triples = np.empty((picked.size, 3), np.int64)
    vertex_index = np.empty((picked.size, 3), np.int64)

    at = cursor = 0
    for block in merge_runs(runs):
        stop = at + block.shape[0]
        while cursor < picked.size and picked[cursor] < stop:
            record = block[int(picked[cursor]) - at]
            index = np.array([record["i0"], record["i1"], record["i2"]], np.int64)
            vertex_index[cursor] = index
            corners[cursor] = verts[index]
            triples[cursor] = (record["s0"], record["s1"], record["s2"])
            cursor += 1
        at = stop
        if cursor >= picked.size:
            break
    if cursor != picked.size:
        raise ValueError(f"reverse-QA merge left {picked.size - cursor} selections unresolved")
    return corners, triples, vertex_index
