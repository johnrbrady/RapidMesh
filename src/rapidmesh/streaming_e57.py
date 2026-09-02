"""
The E57 producer for input streaming — DEC-009 step 2.

`streaming.py` owns the shape: chunks in, one band at a time, nothing
accumulated. This module is the half that knows about E57 files, and it exists
separately because the two fail differently. The generic driver's contract is
"no band outlives the next"; this one's is "**a station-level fact is decided
once**".

Three of those facts would otherwise be recomputed per band, and each fails
without raising:

* the **lattice** — `_lattice` estimates angular steps from a strided subsample
  of the whole scan; estimated per band it differs band to band and every
  incidence threshold moves with it;
* the **frame decision** — `_resolve_frame` compares azimuth spread *within
  lattice columns*, and a band holds only a fraction of each column, so one
  band could have its coordinates rewritten and its neighbour not;
* the **colour width** — `_rgb` shifts 16-bit colour down by 8 based on the
  observed maximum, so a band whose maximum happens to fall below 256 would
  render at a different brightness from the band beside it.

`e57_station_metadata` resolves all three in bounded passes and hands back a
`StationMetadata` plus a `FramePolicy`; `e57_band_to_scan` applies them and
decides nothing itself.

**Only the `rowIndex`/`columnIndex` tier streams.** The spherical and
reprojected tiers derive their lattice from an angle histogram over the whole
scan, which a band cannot reproduce, so they are refused by name rather than
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .e57_reader import (
    FRAME_MIN_SAMPLES,
    MAX_PLAUSIBLE_RANGE_M,
    RawChunk,
    assert_plausible_range,
    decide_frame,
)
from .streaming import StationMetadata, iter_band_filter_results
from .types import LatticeInfo, LatticeSource, StructuredScan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .filters import BandFilterResult
    from .grid import CoarseRangeGrid


class UnsupportedStreamingInput(NotImplementedError):
    """This scan cannot be streamed, and the reason is named.

    Streaming needs an exact lattice that travels with the metadata. Only the
    `rowIndex`/`columnIndex` tier has one: the spherical and reprojected tiers
    derive their lattice from angle histograms over the whole scan, and a band
    cannot reproduce that without the scan. Reported rather than approximated —
    a lattice guessed per band moves every incidence threshold with it.
    """


@dataclass(frozen=True)
class FramePolicy:
    """Station-level decisions that must not be retaken per band.

    `_resolve_frame` and `_rgb` both look at the whole scan: one compares
    azimuth spread within lattice columns, the other reads the colour maximum
    to tell 8-bit from 16-bit. Recomputed per band, either can decide
    differently for two halves of one station, and neither raises when it does.
    """

    rewrite_to_local: bool
    shift_colour: bool
    #: The plausibility setting in force for this station, carried with the
    #: policy so a band cannot be built under a different limit than the one the
    #: frame was decided under.
    max_plausible_range_m: float = MAX_PLAUSIBLE_RANGE_M


def e57_station_metadata(
    path: str,
    index: int = 0,
    *,
    chunk_points: int = 250_000,
    station_id: str = "",
) -> tuple[StationMetadata, FramePolicy]:
    """Lattice, pose and the station-level decisions, in bounded passes.

    Two passes over the point stream, retaining only scalars and a **bounded
    subsample**: the same strided subsample `_lattice` and `_resolve_frame`
    take of the whole scan, capped at a few hundred thousand samples however
    large the station is. The subsample is selected by valid-ordinal, exactly
    as `idx[::idx.size // cap]` selects it out of the whole-scan valid index
    array, so the lattice this returns is the lattice `read_scan` would have
    produced — asserted by test rather than argued.
    """
    import numpy as np
    import pye57

    from .e57_reader import (
        _RGB,
        _SPHERICAL,
        _angular_steps_from_indices,
        _classify,
        _column_azimuth_spread,
        _header_fields,
        _pose_from_header,
        _positions,
        _validity,
        iter_raw_chunks,
    )

    # The file object must outlive the header: pye57's header nodes hold weak
    # references into it, and letting it fall out of scope raises bad_weak_ptr
    # on the first attribute read.
    source_file = pye57.E57(str(path))
    header = source_file.get_header(index)
    pose = _pose_from_header(header)
    fields = tuple(_header_fields(header))
    source = _classify(fields)
    if source is not LatticeSource.ROW_COL:
        raise UnsupportedStreamingInput(
            "streaming needs the rowIndex/columnIndex lattice tier; this scan is "
            + source.value
            + ". Read it with read_scan instead, or supply an export carrying "
            "the scanner's own row and column indices."
        )
    has_rgb = all(name in fields for name in _RGB)

    def chunks() -> Iterator[RawChunk]:
        return iter_raw_chunks(path, index, chunk_points=chunk_points)

    total_records = 0
    valid_total = 0
    max_row = max_col = -1
    colour_max = 0
    for chunk in chunks():
        raw = dict(chunk.data)
        _xyz, rng = _positions(raw)
        ok = _validity(raw, rng)
        total_records += int(ok.size)
        valid_total += int(np.count_nonzero(ok))
        if bool(ok.any()):
            max_row = max(max_row, int(np.asarray(raw["rowIndex"])[ok].max()))
            max_col = max(max_col, int(np.asarray(raw["columnIndex"])[ok].max()))
        if has_rgb:
            for name in _RGB:
                channel = np.asarray(raw[name])
                if channel.dtype.kind in "iu" and channel.size:
                    colour_max = max(colour_max, int(channel.max()))
    rows, cols = max_row + 1, max_col + 1

    lattice_sub, frame_sub = _collect_subsamples(chunks, valid_total, 200_000, 100_000)
    ones = np.ones(lattice_sub["xyz"].shape[0], dtype=bool)
    az_step, el_step, az0, el0 = _angular_steps_from_indices(
        lattice_sub["xyz"], lattice_sub["rng"], lattice_sub["row"],
        lattice_sub["col"], ones, rows, cols,
    )
    lattice = LatticeInfo(rows, cols, az_step, el_step, az0, el0, LatticeSource.ROW_COL)

    # F1. This block used to re-implement the frame rule *without* the
    # spherical short-circuit, so every real station — all of which are
    # spherical — took the rewrite branch and was meshed 5.8 x 10^6 m from
    # where it belonged. There is now exactly one decision function and both
    # paths call it; the streamed path differs only in feeding it a bounded
    # subsample's statistics instead of the whole scan's.
    translation = np.asarray(pose.translation, np.float64)
    rotation = np.asarray(pose.rotation, np.float64)
    posed = float(np.linalg.norm(translation)) >= 0.01 or not np.allclose(
        rotation, np.eye(3), rtol=1e-12, atol=1e-12
    )
    frame_xyz = frame_sub["xyz"]
    sample_count = int(frame_xyz.shape[0])
    if not posed or sample_count < FRAME_MIN_SAMPLES:
        evidence = decide_frame(
            has_spherical=all(name in fields for name in _SPHERICAL),
            has_row_column=True, posed=posed, sample_count=sample_count,
            as_local=float("nan"), as_world=float("nan"),
            max_range_local=float("nan"), max_range_world=float("nan"), az_step=az_step,
        )
    else:
        world_candidate = pose.world_to_local(frame_xyz)
        evidence = decide_frame(
            has_spherical=all(name in fields for name in _SPHERICAL),
            has_row_column=True,
            posed=True,
            sample_count=sample_count,
            as_local=_column_azimuth_spread(frame_xyz, frame_sub["col"]),
            as_world=_column_azimuth_spread(world_candidate, frame_sub["col"]),
            max_range_local=float(np.max(np.linalg.norm(frame_xyz, axis=1))),
            max_range_world=float(np.max(np.linalg.norm(world_candidate, axis=1))),
            az_step=az_step,
        )
    frame_path = evidence.path
    rewrite = evidence.rewrite

    name = str(getattr(header, "name", "") or "")
    del source_file
    metadata = StationMetadata(
        pose=pose,
        lattice=lattice,
        station_id=station_id or name or f"scan{index:03d}",
        # The ledger's denominator. `_build` takes `source_sample_count` as
        # every record in the stream and `dropped_no_return` as the invalid
        # ones; counted here so the streamed ledger balances against the same
        # total the resident path uses, rather than against zero.
        source_sample_count=total_records,
        dropped_no_return=total_records - valid_total,
        dropped_other=0,
        has_rgb=has_rgb,
        has_sample_id=True,
        frame_path=frame_path,
    )
    return metadata, FramePolicy(
        rewrite_to_local=rewrite, shift_colour=colour_max > 255
    )


def _collect_subsamples(
    chunks: Any, valid_total: int, lattice_cap: int, frame_cap: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The strided valid subsamples `_lattice` and `_resolve_frame` would take.

    Selected by valid-ordinal, so the result is the same set `idx[::stride]`
    picks out of the whole-scan valid index array — without ever holding that
    array. Bounded by the caps regardless of station size.
    """
    import numpy as np

    from .e57_reader import _positions, _validity

    caps = (lattice_cap, frame_cap)
    strides = [max(valid_total // cap, 1) if valid_total > cap else 1 for cap in caps]
    parts: list[dict[str, list[Any]]] = [
        {"xyz": [], "rng": [], "row": [], "col": []} for _ in caps
    ]
    seen = 0
    for chunk in chunks():
        raw = dict(chunk.data)
        xyz, rng = _positions(raw)
        ok = _validity(raw, rng)
        ordinals = seen + np.cumsum(ok) - 1
        for part, stride in zip(parts, strides, strict=True):
            pick = ok & (ordinals % stride == 0)
            if not bool(pick.any()):
                continue
            part["xyz"].append(xyz[pick])
            part["rng"].append(rng[pick])
            part["row"].append(np.asarray(raw["rowIndex"], np.int64)[pick])
            part["col"].append(np.asarray(raw["columnIndex"], np.int64)[pick])
        seen += int(np.count_nonzero(ok))

    def joined(part: dict[str, list[Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, values in part.items():
            if values:
                out[key] = np.concatenate(values)
            else:
                out[key] = np.empty((0, 3)) if key == "xyz" else np.empty(0)
        return out

    return joined(parts[0]), joined(parts[1])


def e57_chunks(
    path: str, index: int = 0, *, chunk_points: int = 250_000
) -> Iterator[RawChunk]:
    """E57 chunks in the wire vocabulary, with stable ids attached.

    The stable id is the sample's ordinal in the source stream — the identity
    `read_scan` assigns — added here because `iter_row_bands` slices every
    field generically and the band converter needs it once the halo has
    duplicated rows across bands.
    """
    import numpy as np

    from .e57_reader import iter_raw_chunks

    for chunk in iter_raw_chunks(path, index, chunk_points=chunk_points):
        data = dict(chunk.data)
        data["sampleId"] = np.arange(
            chunk.offset, chunk.offset + chunk.count, dtype=np.int64
        )
        yield RawChunk(offset=chunk.offset, count=chunk.count, data=data)


def e57_band_to_scan(
    band: Any, metadata: StationMetadata, policy: FramePolicy
) -> StructuredScan:
    """One raw E57 band as a band-local scan, under station-level decisions.

    Mirrors `_build` for a band, with the two differences that are the point:
    the lattice comes from `metadata` rather than being re-estimated, and the
    frame and colour decisions come from `policy` rather than being retaken.
    """
    import numpy as np

    from .e57_reader import _RGB, _positions, _validity

    raw = dict(band.data)
    xyz, rng = _positions(raw)
    keep = _validity(raw, rng)
    if policy.rewrite_to_local:
        xyz = metadata.pose.world_to_local(xyz)
        rng = np.linalg.norm(xyz, axis=1)
    # A2. Every band F1 produced carried ranges above 10^5 m and nothing asked.
    # Checked on the band's own valid samples, after the policy has been
    # applied, so a wrong station-level decision fails here instead of being
    # meshed into a scene kilometres wide.
    assert_plausible_range(
        rng[keep],
        where=f"e57_band_to_scan (frame: {metadata.frame_path})",
        limit=policy.max_plausible_range_m,
    )

    rgb = None
    if metadata.has_rgb and all(name in raw for name in _RGB):
        channels = []
        for name in _RGB:
            channel = np.asarray(raw[name])[keep]
            if policy.shift_colour:
                channel = channel >> 8
            channels.append(channel.astype(np.uint8))
        rgb = np.stack(channels, axis=1)

    return StructuredScan(
        row=np.asarray(raw["rowIndex"], np.int64)[keep].astype(np.int32),
        col=np.asarray(raw["columnIndex"], np.int64)[keep].astype(np.int32),
        xyz=xyz[keep].astype(np.float32),
        rng=rng[keep].astype(np.float32),
        pose=metadata.pose,
        lattice=metadata.lattice,
        rgb=rgb,
        intensity=None,
        station_id=metadata.station_id,
        sample_id=np.asarray(raw["sampleId"], np.int64)[keep],
        source_sample_count=metadata.source_sample_count,
        dropped_no_return=metadata.dropped_no_return,
        dropped_other=metadata.dropped_other,
    )


def e57_band_filter_results(
    path: str,
    metadata: StationMetadata,
    policy: FramePolicy,
    *,
    index: int = 0,
    chunk_points: int = 250_000,
    others: list[CoarseRangeGrid] | None = None,
    despeckle: bool = True,
    band_rows: int = 256,
    halo: int = 3,
) -> Iterator[BandFilterResult]:
    """The production path end to end: an E57 on disk in, filtered bands out."""
    return iter_band_filter_results(
        e57_chunks(path, index, chunk_points=chunk_points),
        metadata,
        others=others, despeckle=despeckle, band_rows=band_rows, halo=halo,
        converter=lambda band, meta: e57_band_to_scan(band, meta, policy),
        chunk_factory=lambda: e57_chunks(path, index, chunk_points=chunk_points),
    )
