"""
The exported evidence envelope — PLAN.md §5 item 11, `CLAUDE.md` §9 item 1.

`PHASE1-DETERMINISM-SPEC.md` §2 lists the conditions that must all be equal for
two runs to be compared bit for bit, and against each one a column saying where
it is recorded today. Most said **not recorded**. Until they are, a figure this
pipeline produces cannot be re-created by anyone else, which is why §9 item 1
blocks every real-data fidelity claim on this work rather than on more geometry.

What the envelope is for
------------------------
Not decoration, and not a log. It is the answer to one question: *given this
number, what would I have to set up to get it again?* So every field here is
either an input to that setup or a version of something that could change the
answer. A field that is merely interesting does not belong.

Three fields are deliberately **not** comparable between two runs of the same
station, and each is excluded for a stated reason rather than by omission:

* `processing_seconds` and `peak_rss_bytes` — measurements of the machine, not
  of the mesh (SPEC §5(i));
* `streaming` — the band, chunk and halo axes. The equivalence harness compares
  a resident run against a streamed one *on purpose*: they are different
  configurations of the same station, so requiring these to match would be
  requiring the harness to compare a run with itself.

`EXCLUDED_FROM_COMPARISON` names all three in one place, and
`equivalence.compare_qa_metadata_t1` asserts against that name, so dropping a
field from the comparison is a visible edit rather than a silent one.

What is recorded honestly rather than fabricated
------------------------------------------------
`commit_sha` is `"unknown"` outside a git checkout. `frame_path` is
`"unrecorded"` when the caller did not supply one — a synthetic fixture never
goes through `_resolve_frame` at all, so claiming a branch for it would be a
fiction. `peak_rss_bytes` is `None` unless something actually measured it. None
of these is filled with a plausible-looking default.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .pipeline import MeshResult
    from .types import StationQAReport

EVIDENCE_ENVELOPE_VERSION = "evidence-envelope-v1"

# Recorded but never compared across runs. Named here so the exclusion is one
# decision in one place — see the module docstring.
EXCLUDED_FROM_COMPARISON = ("processing_seconds", "peak_rss_bytes", "streaming")

# `PHASE1-DETERMINISM-SPEC.md` §3: evidence and gate runs pin every thread pool
# to an explicit positive count. `-1` means "all processors this machine
# happens to have", which is the opposite of a recorded condition.
DEFAULT_QA_WORKERS = 1
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

FRAME_PATH_UNRECORDED = "unrecorded"


class EvidencePolicyError(ValueError):
    """A run configured in a way that cannot be used as evidence."""


def resolve_qa_workers(workers: int) -> int:
    """Validate a KD-tree worker count for an evidence run.

    `scipy` accepts `-1` for "every processor", and SPEC §3 forbids it here:
    the whole point of the envelope is that the run can be set up again, and a
    count that depends on the machine cannot be. A positive integer is
    required, and it is recorded.
    """
    if workers == -1:
        raise EvidencePolicyError(
            "qa_workers=-1 means 'all processors on this machine' and cannot be "
            "recorded as a reproduction condition (PHASE1-DETERMINISM-SPEC.md §3). "
            "Pass an explicit positive count; evidence and gate runs use 1."
        )
    if workers < 1:
        raise EvidencePolicyError(f"qa_workers must be a positive integer, got {workers}")
    return int(workers)


def pinned_thread_environment(threads: int = 1) -> dict[str, str]:
    """The environment a measured child must be started with.

    Returned rather than applied. SPEC §3 is explicit that these have to be set
    **before NumPy and SciPy are imported** — a library that has already sized
    its pool does not resize when the variable changes, so calling
    `os.environ[...] = "1"` from inside a running process records an intention
    and changes nothing.
    """
    value = str(int(threads))
    return {name: value for name in BLAS_THREAD_VARIABLES}


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def package_version() -> str:
    from . import __version__

    return str(__version__)


def _repository_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*arguments: str) -> str | None:
    """One git command's stdout, or `None` if it could not be run.

    `None` rather than `""`: a command that failed and a command that printed
    nothing are different facts, and `git status --porcelain` uses the empty
    string to mean "clean".
    """
    try:
        result = subprocess.run(
            ["git", "-C", _repository_root(), *arguments],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):       # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def commit_sha() -> str:
    """The commit this code is running from, or `"unknown"`.

    Never a guess: a working tree that is not a git checkout, or a git that is
    not on PATH, produces `"unknown"` rather than something that looks like a
    hash.
    """
    out = _git("rev-parse", "HEAD")
    if out is None:
        return "unknown"
    sha = out.strip()
    return sha if len(sha) == 40 else "unknown"


def working_tree_state() -> str:
    """`clean`, `modified`, or `unknown` — the other half of `commit_sha`.

    A hash on its own is a *false* reproduction condition when the tree it was
    read from has uncommitted edits: it names code that is not the code that
    ran. Recording the state costs one command and turns a confident wrong
    answer into an honest one. Untracked files count as modified — an
    untracked module is still code the run could have imported.
    """
    out = _git("status", "--porcelain")
    if out is None:
        return "unknown"
    return "modified" if out.strip() else "clean"


def platform_fields() -> tuple[tuple[str, str], ...]:
    return (
        ("system", platform.system()),
        ("release", platform.release()),
        ("machine", platform.machine()),
        ("python", platform.python_version()),
        ("python_implementation", platform.python_implementation()),
    )


def library_versions() -> tuple[tuple[str, str], ...]:
    """Installed versions of every library that can move a last bit.

    `importlib.metadata` rather than a module's `__version__` attribute: pye57
    exposes `__version__` as a *module*, not a string, and reading it naively
    records the repr of a module object as the version.
    """
    from importlib.metadata import PackageNotFoundError, version

    out: list[tuple[str, str]] = []
    for name in ("numpy", "scipy", "pye57"):
        try:
            out.append((name, version(name)))
        except PackageNotFoundError:
            out.append((name, "absent"))
    return tuple(out)


def thread_fields(requested_qa_workers: int) -> tuple[tuple[str, str], ...]:
    """Requested and effective thread counts.

    "Effective" is what the environment actually said when the numerical
    libraries were imported, read back rather than assumed. A variable that is
    unset is recorded as `unset`, which is a different fact from `1`.
    """
    fields = [
        ("qa_workers_requested", str(int(requested_qa_workers))),
        ("qa_workers_effective", str(int(requested_qa_workers))),
    ]
    fields.extend(
        (name.lower(), os.environ.get(name, "unset")) for name in BLAS_THREAD_VARIABLES
    )
    return tuple(fields)


def metric_versions() -> tuple[tuple[str, str], ...]:
    """Every versioned metric and contract whose change would move a figure."""
    from .reverse_qa import CANONICAL_ORDER_VERSION, REVERSE_QA_VERSION
    from .segments_io import SEGMENT_CONTRACT_VERSION
    from .triangulate import COMPONENT_AREA_VERSION

    return (
        ("envelope", EVIDENCE_ENVELOPE_VERSION),
        ("component_area", COMPONENT_AREA_VERSION),
        ("canonical_order", CANONICAL_ORDER_VERSION),
        ("reverse_qa", REVERSE_QA_VERSION),
        ("segment_contract", str(SEGMENT_CONTRACT_VERSION)),
    )


def as_pairs(values: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Any mapping as sorted `(name, str(value))` pairs.

    Sorted so two runs that built the same mapping in a different order still
    compare equal, and stringified so the envelope stays JSON-ready without a
    custom encoder.
    """
    return tuple((key, str(values[key])) for key in sorted(values))


@dataclass(frozen=True)
class EnvironmentEvidence:
    """The parts of the envelope that describe the machine, not the run."""

    rapidmesh_version: str
    commit_sha: str
    working_tree: str
    platform: tuple[tuple[str, str], ...]
    libraries: tuple[tuple[str, str], ...]
    threads: tuple[tuple[str, str], ...]
    versions: tuple[tuple[str, str], ...]

    @classmethod
    def collect(cls, *, qa_workers: int = DEFAULT_QA_WORKERS) -> EnvironmentEvidence:
        return cls(
            rapidmesh_version=package_version(),
            commit_sha=commit_sha(),
            working_tree=working_tree_state(),
            platform=platform_fields(),
            libraries=library_versions(),
            threads=thread_fields(resolve_qa_workers(qa_workers)),
            versions=metric_versions(),
        )


# Every field the exported metadata must carry to be usable as evidence. Kept
# as data so the test asserts the set rather than a hand-written list drifting
# from the dataclass. `lattice_source` sits on `StationQAReport` beside the
# metadata, not inside it, and is checked there.
REQUIRED_METADATA_FIELDS = (
    "envelope_version",
    "source_sha256",
    "rapidmesh_version",
    "commit_sha",
    "working_tree",
    "platform",
    "libraries",
    "settings",
    "seeds",
    "threads",
    "frame_path",
    "versions",
    "exclusions",
    "processing_seconds",
    "peak_rss_bytes",
    "streaming",
    "reverse_qa",
)


# ---------------------------------------------------------------------------
# assembling one station's envelope
# ---------------------------------------------------------------------------


def build_station_report(
    result: MeshResult, source_sha256: str, peak_rss_bytes: int | None = None
) -> StationQAReport:
    """One run's QA outputs plus everything needed to set the run up again.

    Lives here rather than on `MeshResult` because the envelope is the subject
    of PLAN.md §5 item 11 and has its own rules — what is recorded, what is
    compared, and what is honestly left unknown.
    """
    from .types import QAReportMetadata, StationQAReport

    environment = EnvironmentEvidence.collect(qa_workers=result.qa_workers)
    diagnostics = result.diagnostics
    streaming: dict[str, object] = {}
    if diagnostics is not None:
        streaming = {
        "band_rows": diagnostics.band_rows,
        "chunk_points": diagnostics.chunk_points,
        "halo": diagnostics.halo,
        "band_count": diagnostics.band_count,
        "segment_bytes": diagnostics.segment_bytes,
        "tile_bytes": diagnostics.tile_bytes,
        "observation_bytes": diagnostics.observation_bytes,
        "observation_count": diagnostics.observation_count,
        "component_count": diagnostics.component_count,
        "triangles_before_cull": diagnostics.triangles_before_cull,
        "triangles_after_cull": diagnostics.triangles_after_cull,
        "component_area_max_ratio": repr(diagnostics.max_area_ratio),
        "component_area_fallback_components": (
            diagnostics.area_fallback_components
        ),
        }
    reverse = result.reverse_qa_evidence
    reverse_fields: dict[str, object] = {}
    if reverse is not None:
        reverse_fields = {
        "metric_version": reverse.metric_version,
        "canonical_order_version": reverse.canonical_order_version,
        "seed": reverse.seed,
        "max_samples": reverse.max_samples,
        "samples_selected": reverse.samples_selected,
        "samples_measured": reverse.samples_measured,
        "samples_unmatched": reverse.samples_unmatched,
        "qa_window_rows": reverse.qa_window_rows,
        "positive_area_triangles": reverse.positive_area_triangles,
        "windows_used": reverse.windows_used,
        "largest_window_candidates": reverse.largest_window_candidates,
        }

    metadata = QAReportMetadata(
        source_sha256=source_sha256,
        rapidmesh_version=environment.rapidmesh_version,
        settings=as_pairs(result.settings),
        exclusions=(
            "no-return",
            "despeckled",
            "carved",
            "island-culled",
            "otherwise-excluded",
        ),
        processing_seconds=sum(result.timings.values()),
        peak_rss_bytes=peak_rss_bytes,
        commit_sha=environment.commit_sha,
        working_tree=environment.working_tree,
        platform=environment.platform,
        libraries=environment.libraries,
        seeds=as_pairs(result.seeds),
        threads=environment.threads,
        frame_path=result.frame_path,
        versions=environment.versions,
        streaming=as_pairs(streaming),
        reverse_qa=as_pairs(reverse_fields),
    )
    return StationQAReport(
        retained_surface=result.deviation,
        filtering_ledger=result.stats,
        mesh_to_source=result.mesh_to_source,
        metadata=metadata,
        lattice_source=result.lattice.source.value,
    )


