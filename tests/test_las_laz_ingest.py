"""
WP-2a.1 — the LAS/LAZ ingest contract.

Fixtures are **built here, in memory, from the specification's byte layout**,
not committed as binary blobs and never taken from the authorised sample. Each
one is a few hundred bytes. That matters for two reasons beyond repository
hygiene: a hand-built header states exactly which field is under test, and a
test that constructs a 1 cm scale factor on purpose is a far better witness to
`FINDING-001` than one that hopes to find such a file.

What is *not* claimed here: no test in this file reads a real LAZ point record.
`_build_las(compressed=True)` produces a genuine LAZ **header** — compression
bit set, `laszip encoded` VLR present — which is what the header contract and
the honest-unsupported path are about. Decompression is not implemented and is
not tested; it is reported as a named missing capability.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from rapidmesh.ingest import (
    CONTAINER_LAS,
    CONTAINER_LAZ,
    RECONSTRUCTION_UNSUPPORTED,
    UnsupportedSource,
    describe_source,
    detect_container,
    inspect_source,
    quantisation_findings,
)
from rapidmesh.las_reader import (
    INT32_MAX,
    LasFormatError,
    PointDataUnavailable,
    read_header,
    read_raw_xyz,
    scaled_xyz,
)

HEADER_SIZE = {(1, 2): 227, (1, 3): 235, (1, 4): 375}
VLR_HEADER_SIZE = 54


def _vlr(user_id: str, record_id: int, payload: bytes) -> bytes:
    """One variable length record, specification layout."""
    return (
        struct.pack("<H", 0)
        + user_id.encode("ascii").ljust(16, b"\x00")
        + struct.pack("<HH", record_id, len(payload))
        + b"test fixture".ljust(32, b"\x00")
        + payload
    )


def _build_las(
    *,
    version: tuple[int, int] = (1, 4),
    scales: tuple[float, float, float] = (0.001, 0.001, 0.001),
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    points: tuple[tuple[int, int, int], ...] = ((1000, 2000, 3000), (-1000, 5, 42)),
    compressed: bool = False,
    point_format: int = 0,
    point_record_length: int = 20,
    vlrs: tuple[bytes, ...] = (),
    global_encoding: int = 0,
    bounds: tuple[float, float, float, float, float, float] | None = None,
) -> bytes:
    """A minimal but specification-valid LAS file, as bytes.

    `bounds` is (max_x, min_x, max_y, min_y, max_z, min_z) and defaults to the
    scaled extent of `points`, so the header is self-consistent unless a test
    deliberately makes it otherwise.
    """
    header_size = HEADER_SIZE[version]
    vlr_blob = b"".join(vlrs)
    offset_to_points = header_size + len(vlr_blob)

    if bounds is None:
        xs = [p[0] * scales[0] + offsets[0] for p in points] or [0.0]
        ys = [p[1] * scales[1] + offsets[1] for p in points] or [0.0]
        zs = [p[2] * scales[2] + offsets[2] for p in points] or [0.0]
        bounds = (max(xs), min(xs), max(ys), min(ys), max(zs), min(zs))

    raw_format = point_format | (0x80 if compressed else 0x00)

    head = bytearray(header_size)
    head[0:4] = b"LASF"
    struct.pack_into("<HH", head, 4, 0, global_encoding)
    struct.pack_into("<BB", head, 24, version[0], version[1])
    head[26:58] = b"rapidmesh test".ljust(32, b"\x00")
    head[58:90] = b"rapidmesh fixture builder".ljust(32, b"\x00")
    struct.pack_into("<HHH", head, 90, 1, 2026, header_size)
    struct.pack_into("<II", head, 96, offset_to_points, len(vlrs))
    struct.pack_into("<BH", head, 104, raw_format, point_record_length)
    struct.pack_into("<I", head, 107, len(points) if version < (1, 4) else 0)
    struct.pack_into(
        "<12d", head, 131,
        scales[0], scales[1], scales[2],
        offsets[0], offsets[1], offsets[2],
        *bounds,
    )
    if version >= (1, 4):
        struct.pack_into("<Q", head, 247, len(points))

    body = bytearray()
    for x, y, z in points:
        record = bytearray(point_record_length)
        struct.pack_into("<iii", record, 0, x, y, z)
        body += record

    return bytes(head) + vlr_blob + bytes(body)


def _write(tmp_path: Path, name: str, blob: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(blob)
    return path


# --------------------------------------------------------------------------
# Header contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", [(1, 2), (1, 3), (1, 4)])
def test_reads_las_1_2_through_1_4_headers(
    tmp_path: Path, version: tuple[int, int]
) -> None:
    """PLAN.md §6 names 1.2-1.4; each has a different header size."""
    path = _write(tmp_path, "v.las", _build_las(version=version))
    header = read_header(path)
    assert header.version == f"{version[0]}.{version[1]}"
    assert header.point_count == 2
    assert not header.compressed


def test_the_fixture_is_small_enough_to_belong_in_a_repository() -> None:
    """The brief caps in-repo fixtures; these are built, but stay tiny anyway."""
    assert len(_build_las()) < 1024


def test_per_axis_scale_and_offset_are_preserved_exactly(tmp_path: Path) -> None:
    """The header contract is reported as written, not normalised.

    Different scales per axis is the case a single-scale reader silently
    corrupts, so the fixture uses three different values.
    """
    scales = (0.001, 0.002, 0.005)
    offsets = (515000.0, 5750000.0, 120.5)
    path = _write(tmp_path, "axes.las", _build_las(scales=scales, offsets=offsets))
    header = read_header(path)
    assert (header.x.scale, header.y.scale, header.z.scale) == scales
    assert (header.x.offset, header.y.offset, header.z.offset) == offsets


def test_a_non_las_file_is_rejected_by_name(tmp_path: Path) -> None:
    path = _write(tmp_path, "not.las", b"this is not a LAS file at all")
    with pytest.raises(LasFormatError, match="LAS signature"):
        read_header(path)


def test_an_unsupported_version_is_named_not_guessed(tmp_path: Path) -> None:
    blob = bytearray(_build_las(version=(1, 4)))
    blob[25] = 9  # LAS 1.9 does not exist
    path = _write(tmp_path, "future.las", bytes(blob))
    with pytest.raises(LasFormatError, match="1.9"):
        read_header(path)


def test_a_truncated_header_fails_cleanly(tmp_path: Path) -> None:
    """PLAN.md §5 item 13: corrupt input fails cleanly, never plausibly."""
    path = _write(tmp_path, "short.las", _build_las()[:200])
    with pytest.raises(LasFormatError, match="truncated"):
        read_header(path)


def test_point_data_inside_the_header_is_rejected(tmp_path: Path) -> None:
    """A self-inconsistent header is a corrupt file, not a parsing puzzle."""
    blob = bytearray(_build_las())
    struct.pack_into("<I", blob, 96, 10)  # points start inside the header
    path = _write(tmp_path, "overlap.las", bytes(blob))
    with pytest.raises(LasFormatError, match="inside its own"):
        read_header(path)


# --------------------------------------------------------------------------
# Suspicious quantisation — the FINDING-001 condition
# --------------------------------------------------------------------------


def test_one_millimetre_scale_is_not_flagged(tmp_path: Path) -> None:
    """Cairn's post-fix value must be clean, or the check cries wolf on good files."""
    path = _write(tmp_path, "fine.las", _build_las(scales=(0.001, 0.001, 0.001)))
    assert quantisation_findings(read_header(path)) == ()
    assert not inspect_source(path).has_quantisation_findings


def test_a_one_centimetre_scale_is_reported(tmp_path: Path) -> None:
    """The exact defect FINDING-001 records: PDAL's 0.01 default, unreported."""
    path = _write(tmp_path, "coarse.las", _build_las(scales=(0.01, 0.01, 0.01)))
    findings = quantisation_findings(read_header(path))
    assert len(findings) == 3, findings
    assert all("10 mm" in f for f in findings)
    assert any("FINDING-001" in f for f in findings)


def test_quantisation_is_detected_per_axis_not_per_file(tmp_path: Path) -> None:
    """A converter can write a coarse Z and fine X/Y; a file-level check misses it."""
    path = _write(tmp_path, "zcoarse.las", _build_las(scales=(0.001, 0.001, 0.01)))
    header = read_header(path)
    assert not header.x.coarser_than_1mm
    assert not header.y.coarser_than_1mm
    assert header.z.coarser_than_1mm
    assert len(quantisation_findings(header)) == 1


def test_finer_than_a_millimetre_is_accepted_silently(tmp_path: Path) -> None:
    """0.0001 is what FINDING-001 originally proposed; it is better, not worse."""
    path = _write(tmp_path, "veryfine.las", _build_las(scales=(1e-4, 1e-4, 1e-4)))
    assert quantisation_findings(read_header(path)) == ()


def test_an_invalid_scale_is_reported_rather_than_dividing_by_zero(
    tmp_path: Path
) -> None:
    path = _write(tmp_path, "zeroscale.las", _build_las(scales=(0.0, 0.001, 0.001)))
    header = read_header(path)
    assert not header.x.valid
    assert any("not a usable positive scale" in f for f in quantisation_findings(header))


def test_bounds_that_overflow_int32_are_reported(tmp_path: Path) -> None:
    """FINDING-001's other half: at 1 mm an MGA northing overflows int32.

    A northing near 5,750,000 m at scale 0.001 with a zero offset needs
    5.75e9 — well past int32 — which is precisely why offset_*=auto was
    adopted. The file declares bounds it cannot store.
    """
    path = _write(
        tmp_path, "overflow.las",
        _build_las(
            scales=(0.001, 0.001, 0.001),
            offsets=(0.0, 0.0, 0.0),
            bounds=(515000.0, 514000.0, 5750000.0, 5749000.0, 200.0, 100.0),
        ),
    )
    header = read_header(path)
    assert header.y.overflows_int32
    assert header.y.largest_stored_integer > INT32_MAX
    assert any("int32" in f for f in quantisation_findings(header))


def test_a_per_file_offset_removes_the_overflow(tmp_path: Path) -> None:
    """The same coordinates, re-based — the fix, shown working.

    Paired with the test above this is the sensitivity evidence for the
    overflow check: same bounds, same scale, only the offset differs.
    """
    path = _write(
        tmp_path, "rebased.las",
        _build_las(
            scales=(0.001, 0.001, 0.001),
            offsets=(514000.0, 5749000.0, 100.0),
            bounds=(515000.0, 514000.0, 5750000.0, 5749000.0, 200.0, 100.0),
        ),
    )
    header = read_header(path)
    assert not header.y.overflows_int32
    assert quantisation_findings(header) == ()


# --------------------------------------------------------------------------
# CRS pass-through, never inference
# --------------------------------------------------------------------------


def test_wkt_is_passed_through_verbatim(tmp_path: Path) -> None:
    wkt = b'PROJCS["fixture only, not a real CRS"]\x00'
    path = _write(
        tmp_path, "wkt.las",
        _build_las(vlrs=(_vlr("LASF_Projection", 2112, wkt),), global_encoding=0b1_0000),
    )
    header = read_header(path)
    assert header.georeference.wkt == 'PROJCS["fixture only, not a real CRS"]'
    assert header.georeference.declared_wkt_by_global_encoding


def test_geotiff_key_records_are_reported_present_but_not_interpreted(
    tmp_path: Path
) -> None:
    """Presence is a fact; 'this is MGA zone 55' would be an inference."""
    path = _write(
        tmp_path, "geokeys.las",
        _build_las(vlrs=(_vlr("LASF_Projection", 34735, b"\x00" * 8),)),
    )
    report = inspect_source(path)
    assert report.georeference.has_geokey_directory
    assert report.georeferenced
    described = report.georeference.describe()
    assert "GeoTIFF keys" in described
    assert "not interpreted" in described
    assert "EPSG" not in described


def test_a_file_with_no_crs_says_so_rather_than_assuming_one(tmp_path: Path) -> None:
    path = _write(tmp_path, "nocrs.las", _build_las())
    report = inspect_source(path)
    assert not report.georeferenced
    assert "no georeference records" in report.georeference.describe()


# --------------------------------------------------------------------------
# Honest unsupported reporting
# --------------------------------------------------------------------------


def test_las_is_accepted_validated_and_reported_unsupported(tmp_path: Path) -> None:
    """The Phase 2a deliverable in one assertion block.

    Accepted (no exception), validated (counts and contract present), and
    unsupported *by report* — not by failure.
    """
    path = _write(tmp_path, "ok.las", _build_las())
    report = inspect_source(path)
    assert report.container == CONTAINER_LAS
    assert report.point_count == 2
    assert report.reconstruction_supported is False
    assert report.unsupported_reason == RECONSTRUCTION_UNSUPPORTED


def test_the_missing_capability_is_named_not_merely_denied(tmp_path: Path) -> None:
    """CLAUDE.md §4 rule 6: name the missing data, do not just refuse."""
    path = _write(tmp_path, "named.las", _build_las())
    capabilities = inspect_source(path).missing_capabilities
    assert capabilities
    assert any("lattice" in c for c in capabilities)
    assert any("rowIndex" in c for c in capabilities)
    assert any("Phase 2b" in c for c in capabilities)


def test_the_unsupported_sentence_appears_in_the_operator_summary(
    tmp_path: Path
) -> None:
    path = _write(tmp_path, "summary.las", _build_las())
    summary = describe_source(path)
    assert RECONSTRUCTION_UNSUPPORTED in summary
    assert "missing:" in summary


def test_no_coordinate_or_path_leaks_into_the_report(tmp_path: Path) -> None:
    """The report is exportable: counts and contract, never client geometry."""
    offsets = (515000.0, 5750000.0, 120.5)
    path = _write(
        tmp_path, "leak.las",
        _build_las(scales=(0.001, 0.001, 0.001), offsets=offsets,
                   bounds=(515001.0, 515000.0, 5750001.0, 5750000.0, 121.0, 120.5)),
    )
    report = inspect_source(path)
    assert "leak.las" not in report.summary()
    assert str(tmp_path) not in report.summary()


# --------------------------------------------------------------------------
# Stable sample identity
# --------------------------------------------------------------------------


def test_sample_identity_exists_even_though_no_mesh_is_produced(
    tmp_path: Path
) -> None:
    """PLAN.md §6 requires identity from ingest, not from meshing."""
    path = _write(tmp_path, "identity.las", _build_las())
    report = inspect_source(path)
    assert report.reconstruction_supported is False
    assert report.sample_identity.scheme == "las-point-record-index"
    assert report.sample_identity.count == report.point_count
    assert len(report.sample_identity.basis_sha256) == 64


def test_sample_identity_is_stable_across_reads_and_specific_to_the_file(
    tmp_path: Path
) -> None:
    same_a = _write(tmp_path, "a.las", _build_las())
    same_b = _write(tmp_path, "b.las", _build_las())
    different = _write(tmp_path, "c.las", _build_las(points=((7, 8, 9),)))
    assert inspect_source(same_a).sample_identity == inspect_source(same_a).sample_identity
    assert (
        inspect_source(same_a).sample_identity.basis_sha256
        == inspect_source(same_b).sample_identity.basis_sha256
    )
    assert (
        inspect_source(same_a).sample_identity.basis_sha256
        != inspect_source(different).sample_identity.basis_sha256
    )


# --------------------------------------------------------------------------
# Coordinates are never silently changed
# --------------------------------------------------------------------------


def test_raw_integers_are_returned_unscaled(tmp_path: Path) -> None:
    """The file's own integers, not metres — scaling is an explicit second step."""
    points = ((1000, -2000, 3000), (7, 8, 9))
    path = _write(tmp_path, "raw.las", _build_las(points=points))
    raw = read_raw_xyz(path)
    assert raw.shape == (2, 3)
    assert [tuple(int(v) for v in row) for row in raw] == list(points)


def test_scaling_applies_exactly_the_header_contract(tmp_path: Path) -> None:
    """value == raw * scale + offset, in f64, with nothing else done to it."""
    scales = (0.001, 0.002, 0.005)
    offsets = (515000.0, 5750000.0, 120.5)
    points = ((1234, -5678, 900),)
    path = _write(
        tmp_path, "scaled.las",
        _build_las(scales=scales, offsets=offsets, points=points),
    )
    header = read_header(path)
    got = scaled_xyz(read_raw_xyz(path), header)
    expected = [points[0][i] * scales[i] + offsets[i] for i in range(3)]
    assert [float(v) for v in got[0]] == expected


def test_scaled_coordinates_are_f64_not_f32(tmp_path: Path) -> None:
    """CLAUDE.md §8: an MGA northing in f32 has ~0.5 m of quantisation."""
    path = _write(tmp_path, "f64.las", _build_las(offsets=(0.0, 5750000.0, 0.0)))
    header = read_header(path)
    assert scaled_xyz(read_raw_xyz(path), header).dtype.name == "float64"


def test_a_declared_point_count_the_file_cannot_hold_is_rejected(
    tmp_path: Path
) -> None:
    blob = bytearray(_build_las(version=(1, 4)))
    struct.pack_into("<Q", blob, 247, 5000)  # claims 5000, holds 2
    path = _write(tmp_path, "liar.las", bytes(blob))
    with pytest.raises(LasFormatError, match="whole records"):
        read_raw_xyz(path)


# --------------------------------------------------------------------------
# LAZ — header readable, points named as unavailable
# --------------------------------------------------------------------------


def test_a_laz_header_is_fully_readable(tmp_path: Path) -> None:
    """Compression changes the point records, not the public header block."""
    path = _write(tmp_path, "c.laz", _build_las(compressed=True, scales=(0.01, 0.01, 0.01)))
    header = read_header(path)
    assert header.compressed
    assert header.point_format == 0  # the compression bit is masked off
    assert header.point_count == 2
    assert len(quantisation_findings(header)) == 3  # still checkable


def test_laz_point_access_names_the_missing_capability(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.laz", _build_las(compressed=True))
    with pytest.raises(PointDataUnavailable, match="LASzip"):
        read_raw_xyz(path)
    report = inspect_source(path)
    assert report.container == CONTAINER_LAZ
    assert not report.point_access_available
    assert "LASzip point decompression" in report.point_access_note
    assert any("LASzip" in c for c in report.missing_capabilities)


def test_container_is_detected_from_magic_bytes_not_the_extension(
    tmp_path: Path
) -> None:
    """A LAZ delivered as .las is a routine problem; trusting the name hides it."""
    mislabelled = _write(tmp_path, "actually_laz.las", _build_las(compressed=True))
    assert detect_container(mislabelled) == CONTAINER_LAZ
    honest = _write(tmp_path, "plain.las", _build_las())
    assert detect_container(honest) == CONTAINER_LAS


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_an_e57_is_directed_to_the_reader_that_has_its_lattice(
    tmp_path: Path
) -> None:
    path = _write(tmp_path, "scan.e57", b"ASTM-E57" + b"\x00" * 64)
    with pytest.raises(UnsupportedSource, match="e57_reader"):
        inspect_source(path)


def test_an_unrecognised_container_is_refused_by_name(tmp_path: Path) -> None:
    path = _write(tmp_path, "mystery.bin", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(UnsupportedSource, match="not a recognised"):
        inspect_source(path)


def test_a_missing_file_is_refused_before_any_parsing(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSource, match="does not exist"):
        inspect_source(tmp_path / "absent.las")


def test_describe_source_reports_a_rejection_instead_of_raising(
    tmp_path: Path
) -> None:
    """The operator-facing helper never makes the caller handle an exception."""
    path = _write(tmp_path, "junk.las", b"not a las file")
    assert describe_source(path).startswith("REJECTED:")
