"""Regression tests for Phase 1's bidirectional QA and exact ledger."""

from __future__ import annotations

import hashlib

import numpy as np

from rapidmesh import qa, synthetic
from rapidmesh.e57_reader import RawChunk, iter_row_bands
from rapidmesh.grid import ScanGrid, select, sort_row_major
from rapidmesh.pipeline import mesh_station
from rapidmesh.triangulate import build_mesh
from rapidmesh.types import MeshData


def test_filter_ledger_accounts_for_every_generated_sample_once() -> None:
    syn = synthetic.generate(
        rows=80,
        cols=320,
        dropout=0.08,
        range_noise=0.0,
        seed=19,
    )

    result = mesh_station(
        syn.scan,
        despeckle=True,
        measure=False,
        min_component_area=0.005,
    )

    assert result.stats.balanced
    assert result.stats.accounted == syn.scan.lattice.cells
    assert result.stats.dropped_no_return == syn.scan.dropped_no_return
    assert result.stats.retained == result.mesh.vertex_count


def test_source_identity_survives_sort_select_and_mesh_compaction() -> None:
    syn = synthetic.generate(rows=40, cols=160, dropout=0.0, seed=3)
    scan = syn.scan
    assert scan.sample_id is not None

    reverse = np.arange(len(scan) - 1, -1, -1)
    shuffled = sort_row_major(
        type(scan)(
            row=scan.row[reverse],
            col=scan.col[reverse],
            xyz=scan.xyz[reverse],
            rng=scan.rng[reverse],
            pose=scan.pose,
            lattice=scan.lattice,
            rgb=None if scan.rgb is None else scan.rgb[reverse],
            station_id=scan.station_id,
            sample_id=scan.sample_id[reverse],
            source_sample_count=scan.source_sample_count,
            dropped_no_return=scan.dropped_no_return,
        )
    )
    keep = np.arange(len(shuffled)) % 2 == 0
    subset = select(shuffled, keep)
    grid = ScanGrid.build(subset)
    # A deliberately small triangle proves compaction copies identity from the
    # represented observations rather than inventing vertex-local identities.
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(grid.scan, tris, with_normals=False)

    assert mesh.source_sample_id is not None
    assert np.array_equal(mesh.source_sample_id, grid.scan.sample_id[:3])


def test_reverse_metric_detects_surface_invented_between_source_vertices() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float32,
    )
    mesh = MeshData(
        origin=np.zeros(3, np.float64),
        vertices=vertices,
        triangles=np.array([[0, 1, 2]], dtype=np.uint32),
    )

    retained = qa.deviation_report(mesh, vertices, max_samples=10)
    reverse = qa.mesh_to_source_report(mesh, vertices, max_samples=10, seed=7)

    assert retained.maximum == 0.0
    assert reverse.maximum > 0.25
    assert reverse.metric == "mesh-to-retained-source"
    assert not reverse.exact


def test_pipeline_reports_both_qa_directions_over_retained_observations() -> None:
    syn = synthetic.generate(
        rows=60,
        cols=240,
        dropout=0.02,
        range_noise=0.0,
        seed=11,
    )
    result = mesh_station(syn.scan, measure=True, measure_samples=50_000)

    assert result.deviation is not None
    assert result.mesh_to_source is not None
    assert result.deviation.population == result.stats.retained
    assert result.deviation.sampled_points == result.stats.retained
    assert result.mesh_to_source.population == result.mesh.triangle_count

    source_digest = hashlib.sha256(b"synthetic-fixture").hexdigest()
    report = result.evidence_report(source_digest, peak_rss_bytes=123_456)
    payload = report.to_dict()
    assert payload["metadata"]["source_sha256"] == source_digest  # type: ignore[index]
    assert payload["metadata"]["peak_rss_bytes"] == 123_456  # type: ignore[index]
    assert payload["filtering_ledger"]["retained"] == result.stats.retained  # type: ignore[index]
    assert "station_id" not in payload


def test_row_band_halos_repeat_evidence_but_core_owns_each_sample_once() -> None:
    rows = np.repeat(np.arange(12, dtype=np.uint16), 3)
    ids = np.arange(rows.size, dtype=np.int64)
    chunks = iter(
        [
            RawChunk(offset=start, count=min(10, rows.size - start), data={
                "rowIndex": rows[start : start + 10],
                "sampleId": ids[start : start + 10],
            })
            for start in range(0, rows.size, 10)
        ]
    )

    bands = list(
        iter_row_bands(chunks, row_min=0, row_stop=12, band_rows=4, halo=1)
    )
    owned: list[np.ndarray] = []
    for band in bands:
        band_rows = np.asarray(band.data["rowIndex"])
        band_ids = np.asarray(band.data["sampleId"])
        core = (band_rows >= band.core_row_start) & (band_rows < band.core_row_stop)
        owned.append(band_ids[core])

    assert [(b.core_row_start, b.core_row_stop) for b in bands] == [
        (0, 4),
        (4, 8),
        (8, 12),
    ]
    assert np.array_equal(np.concatenate(owned), ids)
    assert set(np.asarray(bands[0].data["sampleId"])) & set(
        np.asarray(bands[1].data["sampleId"])
    )


def test_row_band_assembler_rejects_non_monotonic_input() -> None:
    chunks = iter(
        [
            RawChunk(
                offset=0,
                count=3,
                data={"rowIndex": np.array([0, 2, 1], dtype=np.uint16)},
            )
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "not row-major"):
        list(iter_row_bands(chunks, row_min=0, row_stop=3))


def test_row_band_working_set_is_independent_of_scan_height() -> None:
    total_rows = 10_000
    cols = 4
    chunk_points = 137
    band_rows = 64
    halo = 1
    rows = np.repeat(np.arange(total_rows, dtype=np.uint16), cols)

    def chunks():  # type: ignore[no-untyped-def]
        for start in range(0, rows.size, chunk_points):
            part = rows[start : start + chunk_points]
            yield RawChunk(
                offset=start,
                count=part.size,
                data={"rowIndex": part, "value": part.astype(np.float32)},
            )

    bands = iter_row_bands(
        chunks(),
        row_min=0,
        row_stop=total_rows,
        band_rows=band_rows,
        halo=halo,
    )
    peak_buffered = max(band.buffered_points_before_emit for band in bands)

    # One band plus its two halo rows and at most one unread raw chunk. The
    # bound depends on width/chunk size, not the 10,000-row scan height.
    assert peak_buffered <= (band_rows + 2 * halo) * cols + chunk_points
