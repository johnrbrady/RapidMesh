"""
Ingest contract — accept a source, validate it, and say plainly what it offers.

Phase 2a's job (`PLAN.md` §6) is not to mesh LAS or LAZ. It is to stop the
pipeline lying about them. Three rules from `00-PRODUCT-DEFINITION.md` §7 and
`CLAUDE.md` §4 shape everything in this module:

* **No deferred capability may be claimed** (rule 6). A LAS file is accepted,
  validated and reported *unsupported for surface reconstruction*, with the
  missing capability named. It never borrows the structured-E57 path's
  credibility by staying quiet.
* **No hidden loss** (rule 5). Header quantisation that cannot represent a
  millimetre is reported, because `FINDING-001` is the case where exactly that
  went unreported through a whole product.
* **Never infer units or CRS.** Georeference records are passed through as
  found. A missing CRS is reported missing, never guessed.

What "unsupported" means here, precisely
----------------------------------------
LAS and LAZ have nowhere to put a sample lattice. The scanner's `rowIndex` and
`columnIndex` — the exact angular sampling that the structured-E57 reader
treats as the whole point — are gone by the time a scan has been flattened to
these formats, and no amount of care in this module brings them back. That is
why `LatticeSource.PROJECTED` exists and why reconstruction is a separate,
scheduled research package (`PLAN.md` §9, Phase 2b), not a gap to be papered
over here.

So the honest report is: the file is readable, its contract is stated, its
quantisation is measured, and **surface reconstruction is not yet supported**,
because the missing input is a sample lattice and the missing capability is the
reconstruction path that would have to infer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .las_reader import (
    AxisQuantisation,
    GeoreferenceRecords,
    LasFormatError,
    LasHeader,
    read_header,
    source_digest,
)

#: The one sentence PLAN.md §6 requires an unsupported source to say. Kept as a
#: constant so a report, a test and an operator message cannot drift apart.
RECONSTRUCTION_UNSUPPORTED = "surface reconstruction not yet supported"

#: Container labels. `unknown` is a real outcome, not an error state.
CONTAINER_LAS = "las"
CONTAINER_LAZ = "laz"
CONTAINER_E57 = "e57"
CONTAINER_UNKNOWN = "unknown"

_E57_SIGNATURE = b"ASTM-E57"


class UnsupportedSource(ValueError):
    """The path is not a container this project reads at all.

    Distinct from "read it fine, cannot mesh it": that outcome is an
    `IngestReport` with `reconstruction_supported=False`, not an exception.
    Confusing the two is how a supported-but-deferred format ends up looking
    like a broken file.
    """


@dataclass(frozen=True)
class SampleIdentity:
    """How a sample in this source is named, stably, across runs and machines.

    Required by `PLAN.md` §6 even when no mesh is produced, because Phase 5
    comparison and the disposition ledger both need to point at an individual
    observation and be believed. The scheme is deliberately boring: a point's
    identity is its **zero-based record index in the file**, anchored to the
    file's SHA-256 so an index from a different delivery cannot be mistaken for
    this one.

    `count` is the number of identities the source defines, which is the point
    count even when not one point has been read.
    """

    scheme: str
    basis_sha256: str
    count: int

    def describe(self) -> str:
        return f"{self.scheme} over {self.count:,} records, anchored to {self.basis_sha256[:12]}"


@dataclass(frozen=True)
class IngestReport:
    """What a source is, what it guarantees, and what it cannot do.

    Carries no coordinate and no file path — only counts, contract values and
    provenance — so it can be logged, exported and reviewed without moving
    client data anywhere. That is `CLAUDE.md` §4 rule 9 applied to reporting
    rather than only to the repository.
    """

    source_sha256: str
    container: str
    version: str
    point_count: int
    point_format: int
    compressed: bool
    axes: tuple[AxisQuantisation, ...]
    georeference: GeoreferenceRecords
    sample_identity: SampleIdentity
    reconstruction_supported: bool
    unsupported_reason: str
    missing_capabilities: tuple[str, ...]
    quantisation_findings: tuple[str, ...]
    point_access_available: bool
    point_access_note: str = ""
    generating_software: str = ""
    system_identifier: str = ""

    @property
    def has_quantisation_findings(self) -> bool:
        return bool(self.quantisation_findings)

    @property
    def georeferenced(self) -> bool:
        """Whether the file carries CRS records — not whether they are correct."""
        return self.georeference.present

    def summary(self) -> str:
        lines = [
            f"{self.container.upper()} {self.version}  {self.point_count:,} points  "
            f"point format {self.point_format}"
            + ("  (compressed)" if self.compressed else ""),
            "  " + "  ".join(axis.describe() for axis in self.axes),
            f"  crs: {self.georeference.describe()}",
            f"  identity: {self.sample_identity.describe()}",
        ]
        for finding in self.quantisation_findings:
            lines.append(f"  QUANTISATION: {finding}")
        if not self.point_access_available and self.point_access_note:
            lines.append(f"  POINTS: {self.point_access_note}")
        if not self.reconstruction_supported:
            lines.append(f"  UNSUPPORTED: {self.unsupported_reason}")
            for capability in self.missing_capabilities:
                lines.append(f"    missing: {capability}")
        return "\n".join(lines)


def detect_container(path: str | Path) -> str:
    """Identify the container by its magic bytes, not by its extension.

    A `.las` file that is really a LAZ, or an E57 renamed by a delivery script,
    is a routine delivery problem. Trusting the extension is how it becomes a
    silent one.
    """
    source = Path(path)
    try:
        with source.open("rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise UnsupportedSource(f"cannot read {source.name}: {exc}") from exc
    if head[:4] == b"LASF":
        header = read_header(source)
        return CONTAINER_LAZ if header.compressed else CONTAINER_LAS
    if head[:8] == _E57_SIGNATURE:
        return CONTAINER_E57
    return CONTAINER_UNKNOWN


def quantisation_findings(header: LasHeader) -> tuple[str, ...]:
    """Everything wrong with this file's coordinate contract, in plain words.

    Empty means the contract is sound at the millimetre the downstream chain
    preserves. It is not a statement about survey accuracy, which no header can
    support.
    """
    findings: list[str] = []
    for axis in header.axes:
        if not axis.valid:
            findings.append(
                f"{axis.axis} scale is {axis.scale!r}, which is not a usable "
                "positive scale factor; coordinates on this axis cannot be trusted"
            )
            continue
        if axis.coarser_than_1mm:
            findings.append(
                f"{axis.axis} scale {axis.scale:g} quantises to "
                f"{axis.quantisation_step_mm:g} mm, coarser than the 1 mm floor the "
                "downstream chain preserves (see FINDING-001; Cairn wrote 0.01 "
                "before 5 Aug 2026 and 0.001 after)"
            )
        if axis.overflows_int32:
            findings.append(
                f"{axis.axis} bounds need an integer of "
                f"{axis.largest_stored_integer:.3g}, beyond LAS's int32 limit "
                "— the file's own bounds do not round-trip through its own "
                "header; it needs a per-file offset (offset_*=auto)"
            )
    return tuple(findings)


def _las_report(path: Path, container: str) -> IngestReport:
    header = read_header(path)
    digest = source_digest(path)

    missing = [
        "a sample lattice: LAS/LAZ carry no rowIndex/columnIndex and no "
        "spherical pair, so the scanner's own angular sampling is absent",
        "a selected reconstruction path for unstructured points "
        "(Phase 2b, PLAN.md §9 — not started)",
    ]
    point_access = not header.compressed
    note = ""
    if header.compressed:
        note = (
            "LASzip-compressed point records; the header is fully readable, but "
            "decompression is not available in this build. Missing capability: "
            "LASzip point decompression"
        )
        missing.append("LASzip point decompression (LAZ point records)")

    return IngestReport(
        source_sha256=digest,
        container=container,
        version=header.version,
        point_count=header.point_count,
        point_format=header.point_format,
        compressed=header.compressed,
        axes=header.axes,
        georeference=header.georeference,
        sample_identity=SampleIdentity(
            scheme="las-point-record-index",
            basis_sha256=digest,
            count=header.point_count,
        ),
        reconstruction_supported=False,
        unsupported_reason=RECONSTRUCTION_UNSUPPORTED,
        missing_capabilities=tuple(missing),
        quantisation_findings=quantisation_findings(header),
        point_access_available=point_access,
        point_access_note=note,
        generating_software=header.generating_software,
        system_identifier=header.system_identifier,
    )


def inspect_source(path: str | Path) -> IngestReport:
    """Accept a source, validate it, and report what it offers — never guess.

    LAS and LAZ produce a full report with `reconstruction_supported=False`.
    That is the Phase 2a deliverable: accepted, validated, honestly reported.

    Raises `UnsupportedSource` for a container this project does not read, and
    `LasFormatError` (from `las_reader`) for a LAS/LAZ file that is corrupt or
    of an unsupported version — a clean, named failure rather than a plausible
    result, per `PLAN.md` §5 item 13.
    """
    source = Path(path)
    if not source.exists():
        raise UnsupportedSource(f"{source.name} does not exist")

    container = detect_container(source)
    if container in (CONTAINER_LAS, CONTAINER_LAZ):
        return _las_report(source, container)
    if container == CONTAINER_E57:
        raise UnsupportedSource(
            f"{source.name} is an E57 — use rapidmesh.e57_reader, which reads its "
            "native lattice. This module is the LAS/LAZ ingest contract"
        )
    raise UnsupportedSource(
        f"{source.name} is not a recognised point-cloud container "
        "(no LASF or ASTM-E57 signature)"
    )


def describe_source(path: str | Path) -> str:
    """One operator-facing block: the report, or the reason there is not one."""
    try:
        return inspect_source(path).summary()
    except (UnsupportedSource, LasFormatError) as exc:
        return f"REJECTED: {exc}"
