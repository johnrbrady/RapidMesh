"""
Round 4c — Pass B loads only triangle-named `pos` samples.

The Class F Gate 1 FAIL was Pass B materialising the whole `pos` set while the
mesh only indexes cells that appear in a triangle. On a measured structured
station that gap was ~99% of `pos` bytes. These tests pin the filter that closes
it, including a required-red that fails if the filter is wired back to the full
rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rapidmesh.retained_io import (
    retained_scan_for_cells,
    retained_scan_from_segments,
)
from rapidmesh.segments_io import SegmentError, write_band_segments
from rapidmesh.types import (
    LatticeInfo,
    LatticeSource,
    ScanPose,
    StructuredScan,
)


def _scan(n: int, *, cols: int = 10) -> StructuredScan:
    row = np.arange(n, dtype=np.int32) // cols
    col = np.arange(n, dtype=np.int32) % cols
    xyz = np.stack(
        [col.astype(np.float32), row.astype(np.float32), np.zeros(n, np.float32)],
        axis=1,
    )
    return StructuredScan(
        row=row,
        col=col,
        xyz=xyz,
        rng=np.ones(n, np.float32),
        pose=ScanPose(
            translation=np.zeros(3, np.float64),
            rotation=np.eye(3, dtype=np.float64),
        ),
        lattice=LatticeInfo(
            rows=max(int(row.max()) + 1, 1),
            cols=cols,
            az_step=0.01,
            el_step=0.01,
            az0=0.0,
            el0=0.0,
            source=LatticeSource.SYNTHETIC,
        ),
        rgb=np.zeros((n, 3), np.uint8),
        sample_id=np.arange(n, dtype=np.int64),
        source_sample_count=n,
        dropped_no_return=0,
        dropped_other=0,
    )


def _write_pos_only(tmp_path: Path, owned: StructuredScan) -> object:
    empty = np.empty((0, 3), np.int64)
    return write_band_segments(
        tmp_path,
        core_row_start=0,
        core_row_stop=int(owned.row.max()) + 1 if len(owned) else 1,
        tri_cells=empty,
        tri_components=np.empty(0, np.int64),
        owned=owned,
        cols=owned.lattice.cols,
        dropped_despeckle=0,
        dropped_mover_carve=0,
        restored_from_carve=0,
    )


def test_retained_scan_for_cells_keeps_only_named_samples(tmp_path: Path) -> None:
    full = _scan(20, cols=5)
    segment = _write_pos_only(tmp_path, full)
    needed = np.array([0, 7, 19], dtype=np.int64)
    compact = retained_scan_for_cells(full, [segment], needed)
    assert len(compact) == 3
    cells = compact.row.astype(np.int64) * full.lattice.cols + compact.col
    assert np.array_equal(cells, needed)
    assert np.array_equal(compact.sample_id, needed)


def test_retained_scan_for_cells_refuses_a_missing_triangle_cell(
    tmp_path: Path,
) -> None:
    full = _scan(10, cols=5)
    segment = _write_pos_only(tmp_path, full)
    with pytest.raises(SegmentError, match="no pos segment carries"):
        retained_scan_for_cells(full, [segment], np.array([0, 99], dtype=np.int64))


def test_full_rebuild_still_loads_every_pos_sample(tmp_path: Path) -> None:
    """The unfiltered path stays available for diagnostics; Pass B must not use it."""
    full = _scan(15, cols=5)
    segment = _write_pos_only(tmp_path, full)
    rebuilt = retained_scan_from_segments(full, [segment])
    assert len(rebuilt) == 15


def test_pass_b_must_not_call_the_full_pos_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required-red: if Pass B wires `retained_scan_from_segments` back in, fail.

    The memory bound Round 4c claims is structural — load triangle cells only —
    and this is the tripwire for that claim. A synthetic station meshes almost
    every `pos` sample, so a size comparison would not catch the regression.
    """
    import rapidmesh.retained_io as retained_io

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "pass_b must not materialise the full pos set; use retained_scan_for_cells"
        )

    # Patched on `retained_io`, which is where the full rebuild lives after the
    # DEC-010 split. `pass_b_finalise` imports it inside the function body, so a
    # regression resolves the name at call time and trips this.
    monkeypatch.setattr(retained_io, "retained_scan_from_segments", _boom)

    from rapidmesh import synthetic
    from rapidmesh.pipeline import mesh_station_streamed

    station = synthetic.generate(
        synthetic.RoomScene(mover=True),
        rows=40,
        cols=160,
        dropout=0.01,
        range_noise=0.002,
        seed=3,
        station_id="compact",
    )
    result = mesh_station_streamed(
        station.scan, band_rows=16, chunk_points=50_000, halo=3, measure=False,
    )
    assert result.stats.balanced
    assert result.mesh is not None
    assert result.mesh.vertex_count > 0
