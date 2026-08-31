"""
Equivalence predicates for the streamed ≡ in-memory harness.

Contract: PHASE1-DETERMINISM-SPEC.md §7 (tiers T1 / T2 / T3).
T1 uses ``np.array_equal``, never ``np.allclose``. T2 is the canonical
oriented triangle multiset on ``source_sample_id`` triples. T3 is only the
directed-normal rule of §5(d): provisional ≤ 1e-6 rad.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .pipeline import MeshResult
    from .types import (
        DeviationReport,
        FilterStats,
        MeshData,
        QAReportMetadata,
    )

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]
    U32 = npt.NDArray[np.uint32]
    ArrayLike = npt.NDArray[np.generic]

NORMAL_ANGLE_TOL_RAD = 1e-6
NORMAL_UNIT_TOL = 1e-6
ORIENTATION_AMBIGUITY_REL = 1e-12

# Reverse QA was excluded from the comparison until PLAN.md §5 item 9 existed,
# because v1's triangle selection followed array order and the figure moved
# when nothing about the data moved (SPEC §5(h)). `reverse-qa-v2` makes the
# selection a function of the mesh's own content, so the reverse report is now
# compared at T1 like the forward one — SPEC §7 says "T1 after reverse-qa-v2
# passes §5(h)", and this is that lift.
REVERSE_QA_STATUS = "compared_reverse-qa-v2"

# QAReportMetadata fields compared at T1. Everything the evidence envelope
# records is here except the three in `EXCLUDED_FROM_COMPARISON`: timing and
# peak RSS measure the machine rather than the mesh (SPEC §5(i)), and the
# streaming axes are the configuration the two compared runs deliberately
# differ in — requiring those to match would be requiring the harness to
# compare a run with itself.
_QA_METADATA_COMPARED = (
    "source_sha256",
    "rapidmesh_version",
    "settings",
    "exclusions",
    "envelope_version",
    "commit_sha",
    "working_tree",
    "platform",
    "libraries",
    "seeds",
    "threads",
    "frame_path",
    "versions",
    "reverse_qa",
)
_QA_METADATA_EXCLUDED = ("processing_seconds", "peak_rss_bytes")


class EquivalenceMismatch(Exception):
    """First differing field (and index, when applicable)."""

    def __init__(
        self,
        field: str,
        index: int | None = None,
        detail: str = "",
    ) -> None:
        self.field = field
        self.index = index
        self.detail = detail
        loc = f"[{index}]" if index is not None else ""
        msg = f"{field}{loc}: {detail}" if detail else f"{field}{loc}"
        super().__init__(msg)


@dataclass(frozen=True)
class EquivalenceReport:
    """Outcome of a full MeshResult comparison under SPEC §7."""

    ok: bool
    reverse_qa_status: str
    triangle_rotations_left: int = 0
    triangle_rotations_right: int = 0
    orientation_ambiguous_left: int = 0
    orientation_ambiguous_right: int = 0


def compare_t1_array(
    field_name: str,
    left: ArrayLike | None,
    right: ArrayLike | None,
) -> None:
    """T1: bitwise / exact array equality via ``np.array_equal``.

    ``np.allclose`` is forbidden here. A single-ULP float difference must
    raise ``EquivalenceMismatch``.
    """
    import numpy as np

    if left is None and right is None:
        return
    if left is None or right is None:
        raise EquivalenceMismatch(
            field_name, detail="one side is None and the other is not"
        )
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise EquivalenceMismatch(
            field_name,
            detail=f"shape {a.shape} != {b.shape}",
        )
    if a.dtype != b.dtype:
        raise EquivalenceMismatch(
            field_name,
            detail=f"dtype {a.dtype} != {b.dtype}",
        )
    if np.array_equal(a, b):
        return
    # Locate the first differing flat index for the report.
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    # For structured row comparisons, report the first differing row.
    if a.ndim >= 1 and a.shape[0] > 0 and a.size == flat_a.size:
        # Prefer row index when trailing dimensions match.
        if a.ndim >= 2:
            row_eq = np.all(a.reshape(a.shape[0], -1) == b.reshape(b.shape[0], -1), axis=1)
            bad = np.flatnonzero(~row_eq)
            idx = int(bad[0]) if bad.size else 0
        else:
            bad = np.flatnonzero(flat_a != flat_b)
            idx = int(bad[0]) if bad.size else 0
    else:
        bad = np.flatnonzero(flat_a != flat_b)
        idx = int(bad[0]) if bad.size else 0
    raise EquivalenceMismatch(field_name, index=idx, detail="T1 array_equal failed")


def canonical_oriented_triple(
    a: int, b: int, c: int
) -> tuple[tuple[int, int, int], bool]:
    """Lexicographically smallest cyclic rotation; not a reversal.

    Returns the canonical triple and whether a non-identity rotation was used.
    """
    candidates = ((a, b, c), (b, c, a), (c, a, b))
    best = min(candidates)
    rotated = best != (a, b, c)
    return best, rotated


def canonical_oriented_triples(
    source_sample_id: I64,
    triangles: U32,
) -> tuple[npt.NDArray[np.int64], int]:
    """Map vertex-index triangles to sorted canonical source-id triples.

    Comparison must never use vertex indices (SPEC §4). Returns the sorted
    multiset array and the count of triples that required a cyclic rotation.
    """
    import numpy as np

    ids = np.asarray(source_sample_id, dtype=np.int64)
    tris = np.asarray(triangles)
    if tris.size == 0:
        return np.empty((0, 3), dtype=np.int64), 0
    out = np.empty((tris.shape[0], 3), dtype=np.int64)
    rotations = 0
    for i, (ia, ib, ic) in enumerate(tris):
        triple, rotated = canonical_oriented_triple(
            int(ids[int(ia)]), int(ids[int(ib)]), int(ids[int(ic)])
        )
        out[i] = triple
        if rotated:
            rotations += 1
    order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
    return out[order], rotations


def compare_source_id_triples_t2(
    left_ids: I64,
    left_tris: U32,
    right_ids: I64,
    right_tris: U32,
) -> tuple[int, int]:
    """T2: exact canonical oriented multiset including multiplicity."""
    import numpy as np

    left, rot_l = canonical_oriented_triples(left_ids, left_tris)
    right, rot_r = canonical_oriented_triples(right_ids, right_tris)
    if left.shape != right.shape or not np.array_equal(left, right):
        # First row that differs, or length mismatch at index 0.
        n = min(left.shape[0], right.shape[0])
        idx: int | None = None
        if left.shape[0] != right.shape[0]:
            idx = n  # first missing / extra slot
            detail = f"multiset length {left.shape[0]} != {right.shape[0]}"
        else:
            row_eq = np.all(left == right, axis=1)
            bad = np.flatnonzero(~row_eq)
            idx = int(bad[0]) if bad.size else 0
            detail = "canonical oriented multiset differs"
        raise EquivalenceMismatch("triangles", index=idx, detail=detail)
    return rot_l, rot_r


def directed_angle_rad(a: F32 | F64, b: F32 | F64) -> npt.NDArray[np.float64]:
    """Directed angle ``atan2(‖a × b‖, a · b)`` per corresponding row.

    No absolute value around the dot product: a sign reversal is ~π rad.
    """
    import numpy as np

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    cross = np.cross(aa, bb)
    cross_n = np.linalg.norm(cross, axis=-1)
    dot = np.einsum("ij,ij->i", aa, bb)
    out = np.arctan2(cross_n, dot)
    return np.asarray(out, dtype=np.float64)


def orientation_ambiguity_count(
    normals_local: F32 | F64,
    local_verts: F32 | F64,
) -> int:
    """Count vertices with ``|facing| ≤ 1e-12 · ‖n‖ · ‖v‖`` (SPEC §5(d))."""
    import numpy as np

    n = np.asarray(normals_local, dtype=np.float64)
    v = np.asarray(local_verts, dtype=np.float64)
    facing = np.einsum("ij,ij->i", n, -v)
    thresh = ORIENTATION_AMBIGUITY_REL * (
        np.linalg.norm(n, axis=1) * np.linalg.norm(v, axis=1)
    )
    return int(np.count_nonzero(np.abs(facing) <= thresh))


def compare_normals_t3(
    left: F32 | None,
    right: F32 | None,
    *,
    angle_tol: float = NORMAL_ANGLE_TOL_RAD,
) -> None:
    """T3: finite, unit, and directed angle ≤ ``angle_tol`` (default 1e-6)."""
    import numpy as np

    if left is None and right is None:
        return
    if left is None or right is None:
        raise EquivalenceMismatch(
            "normals", detail="one side is None and the other is not"
        )
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise EquivalenceMismatch(
            "normals", detail=f"shape {a.shape} != {b.shape}"
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        bad = np.flatnonzero(~(np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)))
        raise EquivalenceMismatch(
            "normals",
            index=int(bad[0]) if bad.size else 0,
            detail="non-finite normal",
        )
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    unit_ok = (np.abs(na - 1.0) <= NORMAL_UNIT_TOL) & (
        np.abs(nb - 1.0) <= NORMAL_UNIT_TOL
    )
    if not np.all(unit_ok):
        bad = np.flatnonzero(~unit_ok)
        raise EquivalenceMismatch(
            "normals",
            index=int(bad[0]),
            detail="normal not unit length within 1e-6",
        )
    angles = directed_angle_rad(a, b)
    within = angles <= angle_tol
    if not np.all(within):
        bad = np.flatnonzero(~within)
        idx = int(bad[0])
        raise EquivalenceMismatch(
            "normals",
            index=idx,
            detail=f"directed angle {angles[idx]} rad exceeds {angle_tol}",
        )


def compare_filter_stats_t1(left: FilterStats, right: FilterStats) -> None:
    """Every FilterStats field at T1; both ledgers must balance."""
    left.require_balanced()
    right.require_balanced()
    for f in fields(left):
        lv = getattr(left, f.name)
        rv = getattr(right, f.name)
        if lv != rv:
            raise EquivalenceMismatch(
                f"stats.{f.name}",
                detail=f"{lv} != {rv}",
            )


def compare_deviation_report_t1(
    field_name: str,
    left: DeviationReport | None,
    right: DeviationReport | None,
) -> None:
    """Forward DeviationReport at T1 (exact field equality)."""
    if left is None and right is None:
        return
    if left is None or right is None:
        raise EquivalenceMismatch(
            field_name, detail="one side is None and the other is not"
        )
    for f in fields(left):
        lv = getattr(left, f.name)
        rv = getattr(right, f.name)
        if lv != rv:
            raise EquivalenceMismatch(
                f"{field_name}.{f.name}",
                detail=f"{lv!r} != {rv!r}",
            )


def compare_qa_metadata_t1(
    left: QAReportMetadata, right: QAReportMetadata
) -> None:
    """T1 on envelope fields except processing_seconds and peak_rss_bytes."""
    for name in _QA_METADATA_COMPARED:
        lv = getattr(left, name)
        rv = getattr(right, name)
        if lv != rv:
            raise EquivalenceMismatch(
                f"metadata.{name}",
                detail=f"{lv!r} != {rv!r}",
            )
    # Every metadata field is either compared or named as excluded. A field
    # added to the envelope and to neither list is a silent gap in the
    # comparison, so this fails rather than ignoring it.
    from .evidence import EXCLUDED_FROM_COMPARISON

    assert _QA_METADATA_EXCLUDED == ("processing_seconds", "peak_rss_bytes")
    covered = set(_QA_METADATA_COMPARED) | set(EXCLUDED_FROM_COMPARISON)
    missing = {f.name for f in fields(left)} - covered
    if missing:
        raise EquivalenceMismatch(
            "metadata",
            detail=f"envelope fields neither compared nor excluded: {sorted(missing)}",
        )


def compare_mesh_data(
    left: MeshData,
    right: MeshData,
) -> tuple[int, int]:
    """Compare MeshData fields under SPEC §7 tiers. Returns T2 rotation counts."""
    compare_t1_array("origin", left.origin, right.origin)
    compare_t1_array("vertices", left.vertices, right.vertices)
    if left.source_sample_id is None or right.source_sample_id is None:
        raise EquivalenceMismatch(
            "source_sample_id",
            detail="T2 requires source_sample_id on both meshes",
        )
    compare_t1_array(
        "source_sample_id", left.source_sample_id, right.source_sample_id
    )
    rot_l, rot_r = compare_source_id_triples_t2(
        left.source_sample_id,
        left.triangles,
        right.source_sample_id,
        right.triangles,
    )
    compare_normals_t3(left.normals, right.normals)
    compare_t1_array("rgb", left.rgb, right.rgb)
    return rot_l, rot_r


def _mesh_of(result: MeshResult) -> MeshData:
    """The run's mesh, reassembled from its tiles when it did not keep one."""
    if result.mesh is not None:
        return result.mesh
    if result.tiles is None:
        raise EquivalenceMismatch(
            "mesh", detail="the run produced neither a resident mesh nor tiles"
        )
    from .tile_equivalence import reconstitute_mesh

    return reconstitute_mesh(result.tiles)


def compare_mesh_results(
    left: MeshResult,
    right: MeshResult,
    *,
    source_sha256: str,
    peak_rss_left: int | None = None,
    peak_rss_right: int | None = None,
) -> EquivalenceReport:
    """Full SPEC §7 comparison; reverse DeviationReport is recorded as blocked."""
    rot_l, rot_r = compare_mesh_data(_mesh_of(left), _mesh_of(right))
    compare_filter_stats_t1(left.stats, right.stats)
    compare_deviation_report_t1("deviation", left.deviation, right.deviation)
    # Reverse QA, at T1 now that `reverse-qa-v2` is the metric on both paths.
    # A run that produced no reverse figure on either side is not a mismatch;
    # one that produced it on exactly one side is.
    compare_deviation_report_t1(
        "mesh_to_source", left.mesh_to_source, right.mesh_to_source
    )
    left_rep = left.evidence_report(source_sha256, peak_rss_bytes=peak_rss_left)
    right_rep = right.evidence_report(source_sha256, peak_rss_bytes=peak_rss_right)
    compare_qa_metadata_t1(left_rep.metadata, right_rep.metadata)
    if left_rep.lattice_source != right_rep.lattice_source:
        raise EquivalenceMismatch(
            "lattice_source",
            detail=f"{left_rep.lattice_source!r} != {right_rep.lattice_source!r}",
        )
    return EquivalenceReport(
        ok=True,
        reverse_qa_status=REVERSE_QA_STATUS,
        triangle_rotations_left=rot_l,
        triangle_rotations_right=rot_r,
    )
