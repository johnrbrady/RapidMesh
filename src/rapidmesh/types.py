"""
Core data types shared by every stage of the pipeline.

Design notes that are easy to get wrong and expensive to discover later:

* **Positions are f32 offsets from an f64 origin, everywhere.** A raw f32
  world coordinate at MGA northings (7 digits) has ~0.5 m of quantisation and
  cannot hold a millimetre. `00-PRODUCT-DEFINITION.md` §7 makes this
  non-negotiable, so no type in this module carries an absolute f32 position.

* **A `StructuredScan` stores only *valid* returns, not `rows * cols` cells.**
  A typical dome scan is 30-60 % empty (sky, no-return, out of range). Storing
  the full lattice densely would waste gigabytes at native resolution; the
  lattice position of each sample is carried explicitly in `row`/`col`.

* **`LatticeSource` records provenance and it is not cosmetic.** It is the
  difference between "these are the scanner's own sample positions" and "we
  guessed the lattice back from XYZ, the way Cairn has to". Downstream stages
  loosen their thresholds for the guessed case, and the deviation report states
  which one produced the mesh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]
    F64 = npt.NDArray[np.float64]
    I32 = npt.NDArray[np.int32]
    U8 = npt.NDArray[np.uint8]
    U16 = npt.NDArray[np.uint16]
    I64 = npt.NDArray[np.int64]
    BOOL = npt.NDArray[np.bool_]


class LatticeSource(str, Enum):
    """Where the sample lattice came from, best first.

    ROW_COL is the only one that is exactly the scanner's own sampling. The
    other two are reconstructions of decreasing fidelity, and PROJECTED is
    precisely the handicap RapidMesh exists to remove — it is what you are
    left with once a scan has been flattened to LAS/LAZ.
    """

    ROW_COL = "e57-rowcol"          # rowIndex/columnIndex — exact scanner lattice
    SPHERICAL = "e57-spherical"     # derived from sphericalAzimuth/Elevation
    PROJECTED = "e57-projected"     # re-derived from XYZ by arctan2/arcsin (lossy)
    SYNTHETIC = "synthetic"         # generated fixture with analytic ground truth


@dataclass(frozen=True)
class ScanPose:
    """Rigid transform from scanner-local coordinates to project coordinates.

    `translation` is the scanner origin and is the f64 anchor every f32 offset
    in the pipeline is relative to. `rotation` is scanner-local -> project.
    """

    translation: F64            # (3,)
    rotation: F64               # (3, 3), orthonormal

    def __post_init__(self) -> None:
        """Reject a pose that is not a finite, right-handed rigid transform."""
        import numpy as np

        translation = np.asarray(self.translation, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError("scan-pose translation must have shape (3,)")
        if rotation.shape != (3, 3):
            raise ValueError("scan-pose rotation must have shape (3, 3)")
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
            raise ValueError("scan pose must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-9, atol=1e-9):
            raise ValueError("scan-pose rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-9, atol=1e-9):
            raise ValueError("scan-pose rotation must be right-handed with determinant +1")

    def rotate_local(self, local: F32 | F64) -> F64:
        """Scanner-local vectors -> project-axis vectors, without translation."""
        import numpy as np

        out: F64 = np.asarray(local, dtype=np.float64) @ np.asarray(
            self.rotation, dtype=np.float64
        ).T
        return out

    def local_to_world(self, local: F32 | F64) -> F64:
        """(N,3) scanner-local offsets -> (N,3) absolute project coordinates."""
        import numpy as np

        out: F64 = self.rotate_local(local) + np.asarray(self.translation, dtype=np.float64)
        return out

    def world_to_local(self, world: F64) -> F64:
        """Absolute project coordinates -> scanner-local offsets."""
        import numpy as np

        shifted = np.asarray(world, dtype=np.float64) - np.asarray(
            self.translation, dtype=np.float64
        )
        out: F64 = shifted @ np.asarray(self.rotation, dtype=np.float64)
        return out


@dataclass(frozen=True)
class LatticeInfo:
    """The angular geometry of the sample lattice.

    `az_step`/`el_step` are radians per column/row. They are the honest
    per-scan replacement for Cairn's hard-coded `max(2*pi/2048, pi/1024)`,
    which is the same number for a 6 mm survey scan and a 60 mm preview scan.

    `az0`/`el0` are the angles of lattice cell (row 0, col 0), so a cell index
    can be turned back into a direction without consulting the samples.
    """

    rows: int
    cols: int
    az_step: float
    el_step: float
    az0: float
    el0: float
    source: LatticeSource

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def angular_step(self) -> float:
        """The coarser of the two steps, in radians — the resolution that
        actually limits triangle size."""
        return max(abs(self.az_step), abs(self.el_step))

    def describe(self) -> str:
        import math

        return (
            f"{self.rows} x {self.cols} lattice "
            f"({math.degrees(abs(self.el_step)):.4f} deg/row, "
            f"{math.degrees(abs(self.az_step)):.4f} deg/col, "
            f"source={self.source.value})"
        )


@dataclass
class StructuredScan:
    """One station, as the scanner sampled it.

    All per-sample arrays are the same length N and are held in row-major
    lattice order (sorted by ``row * cols + col``). `grid.ScanGrid` relies on
    that ordering to build its row index without re-sorting.
    """

    row: I32                    # (N,) lattice row of each sample
    col: I32                    # (N,) lattice column of each sample
    xyz: F32                    # (N,3) scanner-local offsets, metres
    rng: F32                    # (N,) range from scanner origin, metres
    pose: ScanPose
    lattice: LatticeInfo
    rgb: U8 | None = None       # (N,3)
    intensity: U16 | None = None  # (N,)
    station_id: str = ""
    sample_id: I64 | None = None  # stable index in the source sample stream
    source_sample_count: int | None = None
    dropped_no_return: int = 0
    dropped_other: int = 0

    def __len__(self) -> int:
        return int(self.row.shape[0])

    @property
    def fill(self) -> float:
        """Fraction of lattice cells carrying a return. Low is normal (sky and
        no-return dominate a dome scan); *very* low usually means the lattice
        was mis-detected, so this is worth logging."""
        return len(self) / max(self.lattice.cells, 1)


@dataclass(frozen=True)
class MeshData:
    """A finished mesh, still in memory.

    Positions are project-axis offsets from `origin`; add `origin` for project
    coordinates. The source pose is retained for provenance, not applied again
    by the renderer. Normals use the same project axes and are never baked into
    `rgb` — see `SPATIAL-CONTRACT.md`.
    """

    origin: F64                 # (3,) f64 anchor
    vertices: F32               # (V,3) project-axis offsets from origin
    triangles: npt.NDArray[np.uint32]  # (T,3)
    source_pose: ScanPose | None = None
    normals: F32 | None = None  # (V,3) unit
    rgb: U8 | None = None       # (V,3) unmodified scan colour
    uv: F32 | None = None       # (V,2) texture coordinates, once texturing lands
    source_sample_id: I64 | None = None  # source observation represented by each vertex

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class DeviationReport:
    """Point-to-mesh accuracy, in metres.

    This measures mesh fidelity to its stated source. It does not establish
    survey accuracy. `00-PRODUCT-DEFINITION.md` §4 states the budgets these are
    checked against; nothing here decides pass/fail, it only measures.
    """

    sampled_points: int
    rms: float
    mean: float
    p95: float
    p99_9: float
    maximum: float
    within_2mm: float           # fraction of sampled points
    within_5mm: float
    metric: str = "point-to-mesh"
    population: int = 0
    exact: bool = False
    source_of_truth: str = "source observations"

    def summary(self) -> str:
        return (
            f"{self.metric}  n={self.sampled_points}/{self.population or self.sampled_points}  "
            f"{'exact' if self.exact else 'sampled'}  rms={self.rms * 1000:.2f} mm  "
            f"p99.9={self.p99_9 * 1000:.2f} mm  max={self.maximum * 1000:.2f} mm  "
            f"<=2mm={self.within_2mm * 100:.2f}%"
        )


@dataclass(frozen=True)
class FilterStats:
    """Exclusive final dispositions plus non-exclusive processing events.

    Every source sample must finish in exactly one of ``retained`` or the
    ``dropped_*`` fields. ``restored_from_carve`` is deliberately an event,
    not a disposition: a restored sample ultimately belongs to ``retained``
    (or a later exclusion), so counting it in both columns would make an
    apparently detailed ledger that cannot balance.
    """

    input_points: int
    retained: int = 0
    dropped_no_return: int = 0
    dropped_despeckle: int = 0
    dropped_mover_carve: int = 0
    dropped_island: int = 0
    dropped_other: int = 0
    restored_from_carve: int = 0

    @property
    def total_dropped(self) -> int:
        return (
            self.dropped_no_return
            + self.dropped_despeckle
            + self.dropped_mover_carve
            + self.dropped_island
            + self.dropped_other
        )

    @property
    def accounted(self) -> int:
        return self.retained + self.total_dropped

    @property
    def balanced(self) -> bool:
        return self.accounted == self.input_points

    def require_balanced(self) -> None:
        if not self.balanced:
            raise ValueError(
                "filter ledger does not balance: "
                f"input={self.input_points}, accounted={self.accounted}"
            )

    def summary(self) -> str:
        n = max(self.input_points, 1)
        return (
            f"in={self.input_points}  retained={self.retained}  "
            f"no-return={self.dropped_no_return}  speckle={self.dropped_despeckle}  "
            f"carve={self.dropped_mover_carve}  island={self.dropped_island}  "
            f"other={self.dropped_other}  restored-event={self.restored_from_carve}  "
            f"({self.total_dropped / n * 100:.2f}% removed; "
            f"balanced={'yes' if self.balanced else 'NO'})"
        )


@dataclass(frozen=True)
class QAReportMetadata:
    """Reproduction context required around every exported QA report."""

    source_sha256: str
    rapidmesh_version: str
    settings: tuple[tuple[str, str], ...]
    exclusions: tuple[str, ...]
    processing_seconds: float
    peak_rss_bytes: int | None

    def __post_init__(self) -> None:
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class StationQAReport:
    """The three Phase 1 reports and their shared evidence envelope."""

    retained_surface: DeviationReport | None
    filtering_ledger: FilterStats
    mesh_to_source: DeviationReport | None
    metadata: QAReportMetadata
    lattice_source: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready structure without station names or coordinates."""
        out = asdict(self)
        out["metadata"]["settings"] = dict(self.metadata.settings)
        return out
