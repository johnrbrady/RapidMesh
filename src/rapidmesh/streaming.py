"""
Input streaming — DEC-009 step 2. The whole scan is never resident in Pass A.

WP-1.7 bounded Pass B and left the input as the largest remaining term: a
`StructuredScan` costs about 35 bytes per sample (row 4, col 4, xyz 12, range 4,
colour 3, stable id 8), which is roughly 438 MB at the reference station's
12.5 M valid returns — 85% of the whole 512,000,000-byte working budget before
any processing. `pass_a_sweep` took one of those and `iter_clean_bands` built a
`ScanGrid` over all of it.

The shape here instead:

    chunks  ->  iter_row_bands(halo=3)  ->  one band scan  ->  filter, triangulate,
                                            label, write segment, release

Points arrive band by band and leave when the band's segment is written. What
is held for the whole sweep is `StationMetadata` — pose, lattice, station id
and the counts — which is a fixed number of bytes regardless of station size.

Two decisions that must stay station-level, and why
---------------------------------------------------
Some facts about a scan cannot be recomputed per band without changing the
answer, and each of them fails quietly:

* **`columns_wrap`** compares `az_step * cols` against 2π. A band of a full
  sweep judged on its own is still a full sweep — the lattice is carried — so
  this one is safe, and it is safe *because* the lattice travels with the band
  rather than being re-estimated (`PHASE1-HALO-CALCULUS.md` §3).
* **The lattice itself.** `_lattice` estimates angular steps from a strided
  subsample of the whole scan. Estimated per band it would differ band to band
  and every incidence threshold would move with it.
* **The frame decision** (`_resolve_frame`) compares azimuth spread within
  lattice *columns*; a band holds a fraction of each column, so a per-band
  decision could rewrite one band's coordinates and not another's.
* **The colour width** (`_rgb` shifts 16-bit colour down by 8 based on the
  observed maximum). Per band, a band whose maximum happens to be ≤255 would
  not shift while its neighbour did.

So all four are resolved once, into `StationMetadata` and `FramePolicy`, and
then applied uniformly. `streaming_e57.e57_station_metadata` does that in
bounded passes for the production path; the fixture path carries them from the
scan it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .e57_reader import RawChunk, iter_row_bands
from .filters import BandPlan, iter_clean_bands
from .types import LatticeInfo, ScanPose, StructuredScan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from .filters import BandFilterResult
    from .grid import CoarseRangeGrid

# Field names on the wire between a producer and a band converter. Deliberately
# not E57's names: the fixture producer carries positions it already has, and
# the E57 producer converts into this vocabulary once per band.
CHUNK_FIELDS = ("rowIndex", "columnIndex", "x", "y", "z", "range", "sampleId")
COLOUR_FIELDS = ("r", "g", "b")


@dataclass(frozen=True)
class StationMetadata:
    """Everything about a station that is not a point.

    Held for the whole sweep. Fixed size — a pose, a lattice and four integers —
    so holding it says nothing about how big the station is.
    """

    pose: ScanPose
    lattice: LatticeInfo
    station_id: str = ""
    source_sample_count: int | None = None
    dropped_no_return: int = 0
    dropped_other: int = 0
    has_rgb: bool = False
    has_sample_id: bool = True

    @classmethod
    def from_scan(cls, scan: StructuredScan) -> StationMetadata:
        return cls(
            pose=scan.pose,
            lattice=scan.lattice,
            station_id=scan.station_id,
            source_sample_count=scan.source_sample_count,
            dropped_no_return=scan.dropped_no_return,
            dropped_other=scan.dropped_other,
            has_rgb=scan.rgb is not None,
            has_sample_id=scan.sample_id is not None,
        )

    def handle_scan(self) -> StructuredScan:
        """A zero-length scan carrying only this metadata.

        Pass B needs the lattice, the pose, the station id and the counts, and
        it decides whether colour and stable ids exist by asking whether those
        arrays are `None`. It never indexes them. Handing it a scan with empty
        arrays of the right presence gives it exactly what it reads and nothing
        it does not — and keeps `pass_b.py` untouched, which this package
        requires.
        """
        import numpy as np

        empty32 = np.empty(0, np.int32)
        return StructuredScan(
            row=empty32, col=empty32.copy(),
            xyz=np.empty((0, 3), np.float32), rng=np.empty(0, np.float32),
            pose=self.pose, lattice=self.lattice,
            rgb=np.empty((0, 3), np.uint8) if self.has_rgb else None,
            intensity=None, station_id=self.station_id,
            sample_id=np.empty(0, np.int64) if self.has_sample_id else None,
            source_sample_count=self.source_sample_count,
            dropped_no_return=self.dropped_no_return,
            dropped_other=self.dropped_other,
        )


# ---------------------------------------------------------------------------
# fixture producer: a resident scan, re-emitted as a chunk stream
# ---------------------------------------------------------------------------


def scan_to_chunks(
    scan: StructuredScan, chunk_points: int = 250_000
) -> Iterator[RawChunk]:
    """Re-emit a resident scan as fixed-capacity chunks, row-major.

    For fixtures and for the equivalence harness, which needs the *same* station
    down both paths. Positions are copied rather than recomputed, so a band
    rebuilt from these chunks carries the identical float32 values the scan
    held — the streamed and in-memory paths must not differ by a conversion.

    This does not save memory on its own: the caller already has the scan. It
    makes `chunk_points` a live axis, which is what lets the harness show that
    moving chunk boundaries does not move the output.
    """
    import numpy as np

    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    total = len(scan)
    ids = (
        np.arange(total, dtype=np.int64) if scan.sample_id is None else scan.sample_id
    )
    for start in range(0, total, chunk_points):
        stop = min(start + chunk_points, total)
        data: dict[str, Any] = {
            "rowIndex": scan.row[start:stop].astype(np.int64),
            "columnIndex": scan.col[start:stop].astype(np.int64),
            "x": scan.xyz[start:stop, 0].copy(),
            "y": scan.xyz[start:stop, 1].copy(),
            "z": scan.xyz[start:stop, 2].copy(),
            "range": scan.rng[start:stop].copy(),
            "sampleId": np.asarray(ids[start:stop], np.int64),
        }
        if scan.rgb is not None:
            for offset, name in enumerate(COLOUR_FIELDS):
                data[name] = scan.rgb[start:stop, offset].copy()
        yield RawChunk(offset=start, count=stop - start, data=data)


def band_to_scan(band: Any, metadata: StationMetadata) -> StructuredScan:
    """One `RawRowBand` in the wire vocabulary, as a band-local scan.

    The lattice comes from `metadata`, never from the band: re-estimating it
    here is the trap this module's docstring opens with. Absolute lattice rows
    are preserved, so a `ScanGrid` over this answers `dense_rows` in absolute
    coordinates and ownership stays keyed on `(row, col)`.
    """
    import numpy as np

    data = band.data
    rgb = None
    if metadata.has_rgb and all(name in data for name in COLOUR_FIELDS):
        rgb = np.stack([np.asarray(data[name]) for name in COLOUR_FIELDS], axis=1)
        rgb = rgb.astype(np.uint8, copy=False)
    return StructuredScan(
        row=np.asarray(data["rowIndex"], np.int32),
        col=np.asarray(data["columnIndex"], np.int32),
        xyz=np.stack(
            [np.asarray(data[name], np.float32) for name in ("x", "y", "z")], axis=1
        ),
        rng=np.asarray(data["range"], np.float32),
        pose=metadata.pose,
        lattice=metadata.lattice,
        rgb=rgb,
        intensity=None,
        station_id=metadata.station_id,
        sample_id=(
            np.asarray(data["sampleId"], np.int64)
            if metadata.has_sample_id and "sampleId" in data
            else None
        ),
        source_sample_count=metadata.source_sample_count,
        dropped_no_return=metadata.dropped_no_return,
        dropped_other=metadata.dropped_other,
    )


# ---------------------------------------------------------------------------
# the sweep driver
# ---------------------------------------------------------------------------


def iter_band_filter_results(
    chunks: Iterable[RawChunk],
    metadata: StationMetadata,
    *,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    band_rows: int = 256,
    halo: int = 3,
    converter: Callable[[Any, StationMetadata], StructuredScan] = band_to_scan,
) -> Iterator[BandFilterResult]:
    """Filtered, core-owned band output, from a chunk stream.

    The two axes stay separate, per `PHASE1-HALO-CALCULUS.md` §6: `chunk_points`
    decides how the producer slices its reads, `band_rows` decides what a band
    owns, and `iter_row_bands` reconciles them. A band is handed to the same
    `iter_clean_bands` the resident path uses, with its own plan, so there is
    one filter implementation and not two.

    Nothing accumulates. Each band's points are converted, filtered, yielded
    and dropped before the next band's chunks are read.
    """
    rows = metadata.lattice.rows
    band_scan: StructuredScan | None = None
    for band in iter_row_bands(
        iter(chunks), row_min=0, row_stop=rows, band_rows=band_rows, halo=halo
    ):
        band_scan = converter(band, metadata)
        if len(band_scan) == 0:
            band_scan = None
            continue
        plan = BandPlan(
            core_row_start=band.core_row_start,
            core_row_stop=band.core_row_stop,
            data_row_start=band.data_row_start,
            data_row_stop=band.data_row_stop,
        )
        # `iter_clean_bands` builds its grid over whatever scan it is given, so
        # a band-sized scan means a band-sized grid. The explicit plan keeps the
        # judged core exactly the one `iter_row_bands` emitted rather than a
        # second tiling computed from row zero of the band's own array
        # (`PHASE1-HALO-CALCULUS.md` §4, trap 2).
        yield from iter_clean_bands(
            band_scan, others, despeckle, band_rows=band_rows, halo=halo,
            bands=[plan],
        )
        # Released before the next band's chunks are read, so two bands are
        # never reachable at once. Without this the loop variable holds the
        # previous band alive across the next conversion — one extra band
        # resident for the whole sweep, which is what the sweep exists to avoid.
        band_scan = None


def scan_band_stream(
    scan: StructuredScan,
    *,
    chunk_points: int = 250_000,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    band_rows: int = 256,
    halo: int = 3,
) -> Iterator[BandFilterResult]:
    """The fixture path end to end: resident scan in, filtered bands out."""
    return iter_band_filter_results(
        scan_to_chunks(scan, chunk_points),
        StationMetadata.from_scan(scan),
        others=others, despeckle=despeckle, band_rows=band_rows, halo=halo,
    )
