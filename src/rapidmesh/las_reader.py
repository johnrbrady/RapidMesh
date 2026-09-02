"""
LAS and LAZ public header block reader — the file's own numbers, unaltered.

Why this module exists, and why it is deliberately small
-------------------------------------------------------
LAS and LAZ carry **no sample lattice**. There is no `rowIndex`/`columnIndex`
and no spherical pair to recover one from, so nothing here can produce the
structured-E57 path's evidence. What these formats *do* carry, and what Phase
2a is about, is a quantisation contract: every coordinate is an `int32` scaled
and offset by per-axis header values, and that contract is the only thing
standing between a survey-grade import and a silently rounded one.

`FINDING-001-PDAL-QUANTISATION.md` is the reason this is read carefully rather
than trusted: Cairn shipped for a period writing `scale = 0.01` on all three
axes, snapping every imported point to a **one centimetre** grid, and nothing
in the pipeline said so. The header stated the fact plainly the whole time.

Two checks follow from that finding and both are implemented here:

1. **Scale coarser than 1 mm is reported.** Cairn's post-fix choice is 0.001,
   matching PotreeConverter's own octree floor; anything coarser is flagged.
2. **`int32` overflow is reported.** At `scale = 0.001` an MGA northing needs
   an integer that does not fit `int32` unless the file re-bases on its own
   data with `offset_*=auto`. A file whose own bounds cannot round-trip through
   its own header is not a file to trust, and the header is enough to know.

Nothing here infers a CRS. Georeferencing records are reported as *present and
raw*; mapping a GeoTIFF key set to an EPSG code is an inference, and
`00-PRODUCT-DEFINITION.md` forbids inferring units or datum.

Compressed files
----------------
A LAZ file is a LAS file whose point records are LASzip-compressed; the public
header block is byte-identical in layout, with bit 7 of the point data record
format set and a `laszip encoded` VLR present. So **headers are fully readable
for LAZ** and are read here. Decompressing the point records needs a LASzip
implementation that this project does not depend on, so point access raises
`PointDataUnavailable` by name rather than failing obscurely.

Stdlib only. No numpy at module scope, matching the rest of the package.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    I32 = npt.NDArray[np.int32]
    F64 = npt.NDArray[np.float64]

LAS_SIGNATURE = b"LASF"

#: Header sizes defined by the specification, keyed by (major, minor).
HEADER_SIZES: dict[tuple[int, int], int] = {
    (1, 0): 227, (1, 1): 227, (1, 2): 227, (1, 3): 235, (1, 4): 375,
}

#: Versions this reader accepts. 1.0/1.1 parse identically but are outside the
#: Phase 2a contract, so they are named rather than silently accepted.
SUPPORTED_VERSIONS: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4))

#: Scale factors coarser than this are reported as suspicious quantisation.
#: 1 mm is Cairn's post-fix value and PotreeConverter's octree floor; it is the
#: finest resolution the downstream chain preserves, not an accuracy claim.
FINEST_UNSUSPICIOUS_SCALE = 1e-3

#: LAS stores coordinates as signed 32-bit integers. Not configurable.
INT32_MAX = 2_147_483_647

_COMPRESSION_MASK = 0xC0
_POINT_FORMAT_MASK = 0x3F

_VLR_HEADER_SIZE = 54
_LASZIP_USER_ID = "laszip encoded"
_LASZIP_RECORD_ID = 22204
_PROJECTION_USER_ID = "LASF_Projection"
_GEO_KEY_DIRECTORY = 34735
_GEO_DOUBLE_PARAMS = 34736
_GEO_ASCII_PARAMS = 34737
_OGC_WKT = 2112
#: Global-encoding bit 4 declares the CRS is WKT rather than GeoTIFF (LAS 1.4).
_GLOBAL_ENCODING_WKT_BIT = 0b1_0000


class LasFormatError(ValueError):
    """The file is not a LAS/LAZ file this reader will parse.

    Raised for a bad signature, a truncated header, an unsupported version or a
    self-inconsistent header. Corrupt and unsupported inputs must fail cleanly
    and by name — PLAN.md §5 item 13 — not produce a plausible-looking result.
    """


class PointDataUnavailable(NotImplementedError):
    """Point records exist but cannot be read with the current dependencies.

    Raised for LAZ, whose records are LASzip-compressed. The header is still
    fully available; this names the one missing capability instead of implying
    the file is unreadable or, worse, that it is empty.
    """


def _text(raw: bytes) -> str:
    """A fixed-width LAS character field as text, null padding removed."""
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class AxisQuantisation:
    """One axis of the header's coordinate contract, and what it implies.

    `scale` and `offset` are the file's own values, never adjusted. A stored
    coordinate is exactly ``value = raw_int * scale + offset``; this class does
    not apply that transform, it reports whether the transform is trustworthy.
    """

    axis: str
    scale: float
    offset: float
    minimum: float
    maximum: float

    @property
    def valid(self) -> bool:
        """A usable scale is finite, positive and non-zero."""
        return self.scale > 0.0 and self.scale == self.scale and self.scale != float("inf")

    @property
    def coarser_than_1mm(self) -> bool:
        """True when this axis cannot represent a millimetre.

        The `FINDING-001` condition, per axis rather than for the file, because
        a converter can and does write different scales on Z.
        """
        return self.valid and self.scale > FINEST_UNSUSPICIOUS_SCALE

    @property
    def quantisation_step_mm(self) -> float:
        return self.scale * 1000.0

    @property
    def largest_stored_integer(self) -> float:
        """Largest ``|(bound - offset) / scale|`` over this axis's own bounds.

        Computed from the header's declared min/max, so it is what the file
        claims about itself — no point data is read.
        """
        if not self.valid:
            return float("inf")
        return max(
            abs((self.maximum - self.offset) / self.scale),
            abs((self.minimum - self.offset) / self.scale),
        )

    @property
    def overflows_int32(self) -> bool:
        """The file's own bounds do not fit the int32 it must store them in.

        `FINDING-001`: at scale 0.001 an MGA northing overflows int32 unless the
        file re-bases on its own data via ``offset_*=auto``. A file that fails
        this is internally inconsistent and its coordinates cannot be trusted.
        """
        return self.largest_stored_integer > INT32_MAX

    def describe(self) -> str:
        return (
            f"{self.axis}: scale={self.scale:g} ({self.quantisation_step_mm:g} mm) "
            f"offset={self.offset:g}"
        )


@dataclass(frozen=True)
class GeoreferenceRecords:
    """Which CRS records the file carries — presence and payload, never a datum.

    Deliberately not parsed into an EPSG code or a datum name. Turning a GeoTIFF
    key directory into "GDA2020 / MGA zone 55" is an inference, and
    `00-PRODUCT-DEFINITION.md` forbids inferring units or CRS. Downstream code
    and operators get the raw material and decide.
    """

    has_geokey_directory: bool = False
    has_geo_double_params: bool = False
    has_geo_ascii_params: bool = False
    wkt: str = ""
    declared_wkt_by_global_encoding: bool = False

    @property
    def present(self) -> bool:
        return bool(
            self.has_geokey_directory
            or self.has_geo_double_params
            or self.has_geo_ascii_params
            or self.wkt
        )

    def describe(self) -> str:
        if not self.present:
            return "no georeference records in the header"
        kinds = []
        if self.wkt:
            kinds.append("OGC WKT")
        if self.has_geokey_directory:
            kinds.append("GeoTIFF keys")
        if self.has_geo_double_params:
            kinds.append("geo doubles")
        if self.has_geo_ascii_params:
            kinds.append("geo ascii")
        return "carries " + ", ".join(kinds) + " (reported, not interpreted)"


@dataclass(frozen=True)
class LasHeader:
    """The LAS/LAZ public header block, as read.

    Every field is the file's own value. `point_count` resolves the LAS 1.4
    64-bit count against the legacy 32-bit field, preferring the 64-bit one
    when it is populated, because a 1.4 file with more than 2^32 points writes
    zero into the legacy field by design.
    """

    version_major: int
    version_minor: int
    point_format: int
    compressed: bool
    point_count: int
    point_record_length: int
    offset_to_point_data: int
    header_size: int
    vlr_count: int
    x: AxisQuantisation
    y: AxisQuantisation
    z: AxisQuantisation
    georeference: GeoreferenceRecords
    system_identifier: str = ""
    generating_software: str = ""
    creation_year: int = 0
    creation_day: int = 0
    global_encoding: int = 0
    file_source_id: int = 0

    @property
    def version(self) -> str:
        return f"{self.version_major}.{self.version_minor}"

    @property
    def axes(self) -> tuple[AxisQuantisation, AxisQuantisation, AxisQuantisation]:
        return (self.x, self.y, self.z)

    @property
    def suspicious_quantisation(self) -> tuple[AxisQuantisation, ...]:
        """Axes that cannot represent a millimetre. Empty is the good case."""
        return tuple(a for a in self.axes if a.coarser_than_1mm)

    @property
    def overflowing_axes(self) -> tuple[AxisQuantisation, ...]:
        """Axes whose own declared bounds do not fit int32."""
        return tuple(a for a in self.axes if a.overflows_int32)

    def describe(self) -> str:
        kind = "LAZ" if self.compressed else "LAS"
        return (
            f"{kind} {self.version}  point format {self.point_format}  "
            f"{self.point_count:,} points  "
            f"scales=({self.x.scale:g}, {self.y.scale:g}, {self.z.scale:g})"
        )


def _axis(
    name: str, scale: float, offset: float, minimum: float, maximum: float
) -> AxisQuantisation:
    return AxisQuantisation(
        axis=name, scale=scale, offset=offset, minimum=minimum, maximum=maximum
    )


def _read_vlrs(blob: bytes, start: int, count: int, limit: int) -> GeoreferenceRecords:
    """Walk the VLR chain for georeference records. Malformed chains stop early.

    A VLR chain that runs off the end of the header region is a corrupt file,
    not a reason to raise: the public header block has already been read and is
    the load-bearing part. What is found is reported; what is unreachable is
    simply absent, which `GeoreferenceRecords.present` states honestly.
    """
    geokeys = doubles = ascii_params = False
    wkt = ""
    cursor = start
    for _ in range(count):
        if cursor + _VLR_HEADER_SIZE > limit or cursor + _VLR_HEADER_SIZE > len(blob):
            break
        user_id = _text(blob[cursor + 2 : cursor + 18])
        record_id, payload_length = struct.unpack_from("<HH", blob, cursor + 18)
        payload_start = cursor + _VLR_HEADER_SIZE
        payload_end = payload_start + payload_length
        if payload_end > len(blob):
            break
        if user_id == _PROJECTION_USER_ID:
            if record_id == _GEO_KEY_DIRECTORY:
                geokeys = True
            elif record_id == _GEO_DOUBLE_PARAMS:
                doubles = True
            elif record_id == _GEO_ASCII_PARAMS:
                ascii_params = True
            elif record_id == _OGC_WKT:
                wkt = _text(blob[payload_start:payload_end])
        cursor = payload_end
    return GeoreferenceRecords(
        has_geokey_directory=geokeys,
        has_geo_double_params=doubles,
        has_geo_ascii_params=ascii_params,
        wkt=wkt,
    )


def read_header(path: str | Path) -> LasHeader:
    """Parse the public header block of a LAS or LAZ file.

    Cheap and read-only: reads the header region and the VLR chain, never a
    point record. Point this at a delivery to learn its quantisation contract
    before committing to processing.

    Raises `LasFormatError` on a bad signature, a truncated or self-inconsistent
    header, or a version outside `SUPPORTED_VERSIONS`.
    """
    source = Path(path)
    try:
        with source.open("rb") as handle:
            head = handle.read(HEADER_SIZES[(1, 4)])
    except OSError as exc:  # unreadable file, not a format problem
        raise LasFormatError(f"cannot read {source.name}: {exc}") from exc

    if len(head) < 8 or head[:4] != LAS_SIGNATURE:
        raise LasFormatError(
            f"{source.name} does not begin with the LAS signature "
            f"{LAS_SIGNATURE!r} — not a LAS or LAZ file"
        )

    version_major, version_minor = struct.unpack_from("<BB", head, 24)
    version = (version_major, version_minor)
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(f"{a}.{b}" for a, b in SUPPORTED_VERSIONS)
        raise LasFormatError(
            f"{source.name} declares LAS {version_major}.{version_minor}; "
            f"this reader accepts {supported}"
        )

    minimum_header = HEADER_SIZES[version]
    if len(head) < minimum_header:
        raise LasFormatError(
            f"{source.name} is truncated: LAS {version_major}.{version_minor} "
            f"needs a {minimum_header}-byte header, file holds {len(head)}"
        )

    file_source_id, global_encoding = struct.unpack_from("<HH", head, 4)
    system_identifier = _text(head[26:58])
    generating_software = _text(head[58:90])
    creation_day, creation_year, header_size = struct.unpack_from("<HHH", head, 90)
    offset_to_point_data, vlr_count = struct.unpack_from("<II", head, 96)
    raw_point_format, point_record_length = struct.unpack_from("<BH", head, 104)
    (legacy_point_count,) = struct.unpack_from("<I", head, 107)
    scales_and_bounds = struct.unpack_from("<12d", head, 131)

    compressed = bool(raw_point_format & _COMPRESSION_MASK)
    point_format = raw_point_format & _POINT_FORMAT_MASK

    point_count = legacy_point_count
    if version >= (1, 4):
        (wide_point_count,) = struct.unpack_from("<Q", head, 247)
        if wide_point_count:
            point_count = int(wide_point_count)

    if header_size < minimum_header:
        raise LasFormatError(
            f"{source.name} declares a {header_size}-byte header; "
            f"LAS {version_major}.{version_minor} requires at least {minimum_header}"
        )
    if offset_to_point_data < header_size:
        raise LasFormatError(
            f"{source.name} points to its point data at byte {offset_to_point_data}, "
            f"inside its own {header_size}-byte header"
        )
    if point_record_length == 0 and point_count:
        raise LasFormatError(
            f"{source.name} declares {point_count:,} points of zero bytes each"
        )

    x_scale, y_scale, z_scale, x_offset, y_offset, z_offset = scales_and_bounds[:6]
    max_x, min_x, max_y, min_y, max_z, min_z = scales_and_bounds[6:]

    georeference = GeoreferenceRecords()
    if vlr_count:
        with source.open("rb") as handle:
            handle.seek(0)
            region = handle.read(offset_to_point_data)
        georeference = _read_vlrs(region, header_size, vlr_count, offset_to_point_data)
    if global_encoding & _GLOBAL_ENCODING_WKT_BIT:
        georeference = GeoreferenceRecords(
            has_geokey_directory=georeference.has_geokey_directory,
            has_geo_double_params=georeference.has_geo_double_params,
            has_geo_ascii_params=georeference.has_geo_ascii_params,
            wkt=georeference.wkt,
            declared_wkt_by_global_encoding=True,
        )

    return LasHeader(
        version_major=version_major,
        version_minor=version_minor,
        point_format=point_format,
        compressed=compressed,
        point_count=int(point_count),
        point_record_length=int(point_record_length),
        offset_to_point_data=int(offset_to_point_data),
        header_size=int(header_size),
        vlr_count=int(vlr_count),
        x=_axis("x", x_scale, x_offset, min_x, max_x),
        y=_axis("y", y_scale, y_offset, min_y, max_y),
        z=_axis("z", z_scale, z_offset, min_z, max_z),
        georeference=georeference,
        system_identifier=system_identifier,
        generating_software=generating_software,
        creation_year=int(creation_year),
        creation_day=int(creation_day),
        global_encoding=int(global_encoding),
        file_source_id=int(file_source_id),
    )


def is_laszip_vlr_present(path: str | Path) -> bool:
    """Whether the `laszip encoded` VLR is present.

    A second, independent witness to compression: the header's format bit says
    the records are compressed, and this says the compressor left its VLR. They
    should agree, and a file where they do not is worth knowing about.
    """
    source = Path(path)
    header = read_header(source)
    if not header.vlr_count:
        return False
    with source.open("rb") as handle:
        region = handle.read(header.offset_to_point_data)
    cursor = header.header_size
    for _ in range(header.vlr_count):
        if cursor + _VLR_HEADER_SIZE > len(region):
            return False
        user_id = _text(region[cursor + 2 : cursor + 18])
        record_id, payload_length = struct.unpack_from("<HH", region, cursor + 18)
        if user_id == _LASZIP_USER_ID and record_id == _LASZIP_RECORD_ID:
            return True
        cursor += _VLR_HEADER_SIZE + payload_length
    return False


def read_raw_xyz(path: str | Path, limit: int | None = None) -> I32:
    """Point records as the **raw signed integers the file stores**, unscaled.

    Returns an ``(N, 3)`` int32 array. Deliberately not metres: applying scale
    and offset is a decision with a precision consequence, so it is made
    explicitly by `scaled_xyz` and never as a side effect of reading. X, Y and Z
    occupy the first twelve bytes of every LAS point record format 0-10, so this
    does not need to know which format it is looking at.

    Raises `PointDataUnavailable` for LAZ.
    """
    import numpy as np

    source = Path(path)
    header = read_header(source)
    if header.compressed:
        raise PointDataUnavailable(
            f"{source.name} is LAZ: point records are LASzip-compressed and this "
            "build has no LASzip decompressor. The header is fully readable; "
            "missing capability is LASzip point decompression"
        )

    count = header.point_count if limit is None else min(limit, header.point_count)
    if count <= 0:
        return np.empty((0, 3), dtype=np.int32)

    stride = header.point_record_length
    with source.open("rb") as handle:
        handle.seek(header.offset_to_point_data)
        blob = handle.read(count * stride)
    available = len(blob) // stride
    if available < count:
        raise LasFormatError(
            f"{source.name} declares {header.point_count:,} points but holds "
            f"{available:,} whole records after byte {header.offset_to_point_data}"
        )

    records = np.frombuffer(blob, dtype=np.uint8, count=count * stride)
    records = records.reshape(count, stride)[:, :12]
    out: I32 = np.ascontiguousarray(records).view(np.int32).reshape(count, 3)
    return out


def scaled_xyz(raw: I32, header: LasHeader) -> F64:
    """Apply the header contract: ``raw * scale + offset``, in f64.

    f64 because an MGA northing in f32 has ~0.5 m of quantisation against a
    25 mm tolerance — `CLAUDE.md` §8's first trap. The arithmetic is exactly the
    LAS specification's, with no re-centring, no rounding and no CRS handling.
    """
    import numpy as np

    scale = np.array([header.x.scale, header.y.scale, header.z.scale], dtype=np.float64)
    offset = np.array([header.x.offset, header.y.offset, header.z.offset], dtype=np.float64)
    out: F64 = np.asarray(raw, dtype=np.float64) * scale + offset
    return out


def source_digest(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of the whole file — the anchor for stable sample identity.

    Streamed so a multi-gigabyte delivery does not have to be resident, which
    is the same discipline the E57 path follows for the same reason.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()
