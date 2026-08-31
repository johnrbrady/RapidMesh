#!/usr/bin/env python3
"""
isolation_matrix.py — FINDING-003's required isolation matrix.

`FINDING-003-GEOMETRIC-TAIL.md`: at fine (0.090 deg) sampling, p99.9 against
analytic truth is 3.55 mm, which passes the 25 mm default-tolerance budget
(8.0 mm) but fails the 10 mm floor's budget (3.2 mm), by 11%. Two candidate
causes were eliminated by measurement — range noise and cross-station carving
each move the figure only ~5% — but the responsible STAGE was never
identified, and the finding is explicit that promoting a fix before that
happens is exactly the reasoning error this project's charter prohibits (a
silhouette-aware-carving suggestion was floated and retracted for precisely
this reason).

This script is the isolation matrix the finding specifies: noise swept
0/1/2/3/5 mm, carving and parallax-restore and island-culling each toggled
independently, and `max_incidence_deg` swept — all at the fine 0.090 deg
sampling, never the coarse setting used for the earlier, inconclusive runs.
For every condition it reports walls and floors SEPARATELY (not combined,
per the finding), residuals grouped by incidence angle / range / distance
from the nearest depth discontinuity, and the worst 0.1% of points by
coordinate and mesh triangle id.

It does not conclude anything. `main()` prints the finding's own decision
table against whatever the numbers turn out to say and leaves the reading to
whoever runs it — the whole point of running the matrix instead of arguing
from one number is that the stage gets IDENTIFIED, not guessed.

    python tools/isolation_matrix.py [--rows 1334] [--cols 4000] [--out out/isolation_matrix]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Default lattice: 0.090 deg on both axes (verified: az_step=0.09000,
# el_step=0.09002 deg with the fixture's default +-60 deg elevation span).
# This is the "fine sampling that produced the 3.55 mm figure" the finding
# names explicitly — the earlier bench_synthetic.py runs used a coarser
# default and are NOT what this matrix is meant to reproduce.
DEFAULT_ROWS = 1334
DEFAULT_COLS = 4000

# Face classification tolerance for the synthetic room's axis-aligned
# surfaces, metres. Loose enough to absorb range noise up to the 5 mm sigma
# case, tight enough that no wall sample is misclassified as the floor.
_FACE_TOL = 0.05

INCIDENCE_BUCKETS = [(0, 40), (40, 60), (60, 70), (70, 75), (75, 78), (78, 80), (80, 82), (82, 90)]
RANGE_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
# In lattice CELLS, via a Euclidean distance transform over the discontinuity
# mask — not metres, because "how many samples away from an edge" is the
# quantity discontinuity-handling code actually reasons about.
DISCONTINUITY_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1_000_000)]


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ap.add_argument("--out", type=str, default="out/isolation_matrix")
    args = ap.parse_args()

    from rapidmesh import synthetic

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify the lattice is genuinely the fine sampling before anything else
    # runs — this whole exercise is worthless at the wrong resolution.
    probe = synthetic.generate(rows=args.rows, cols=args.cols, range_noise=0.0,
                               dropout=0.0, seed=1)
    az_deg = np.degrees(probe.scan.lattice.az_step)
    el_deg = np.degrees(probe.scan.lattice.el_step)
    print(f"lattice check: az_step={az_deg:.5f} deg  el_step={el_deg:.5f} deg  "
          f"({args.rows} x {args.cols}, {len(probe.scan):,} samples/station)")
    if not (0.088 <= az_deg <= 0.092 and 0.088 <= el_deg <= 0.092):
        print("REFUSING TO RUN: this is not the fine 0.090 deg sampling the "
              "finding specifies. Pass --rows/--cols to hit it.", file=sys.stderr)
        return 1

    conditions = _build_matrix(args.rows, args.cols)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, cond in enumerate(conditions):
        print(f"\n[{i + 1}/{len(conditions)}] {cond['label']}")
        result = _run_one(cond, out_dir)
        results.append(result)
        w, f = result["walls"], result["floor"]
        print(f"  walls  rms={w['rms_mm']:.2f} mm  p99.9={w['p99_9_mm']:.2f} mm  max={w['max_mm']:.1f} mm  n={w['n']}")
        print(f"  floor  rms={f['rms_mm']:.2f} mm  p99.9={f['p99_9_mm']:.2f} mm  max={f['max_mm']:.1f} mm  n={f['n']}")
    print(f"\ntotal matrix time: {time.perf_counter() - t0:.1f}s")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_summary_table(results)
    _print_decision_table(results)
    print(f"\nfull results, per-condition heat maps and worst-0.1% listings: {out_dir}/")
    return 0


# --------------------------------------------------------------------------
# the matrix itself — one condition at a time from a shared baseline
# --------------------------------------------------------------------------


def _build_matrix(rows: int, cols: int) -> list[dict[str, Any]]:
    base = dict(rows=rows, cols=cols, noise=0.002, carving=True, restore=True,
                island=True, max_incidence_deg=82.0)

    conditions = []

    # Noise sweep, per the finding's specified levels. baseline (2mm) reused
    # as the "carving on, restore on, island on, default incidence" anchor
    # the other single-variable toggles are compared against.
    for noise_mm in (0, 1, 2, 3, 5):
        c = dict(base, noise=noise_mm / 1000.0)
        c["label"] = f"noise={noise_mm}mm (baseline pipeline)"
        c["key"] = f"noise_{noise_mm}mm"
        conditions.append(c)

    # Filter-stage isolation, one at a time, at the baseline 2mm noise level.
    c = dict(base, carving=False)
    c["label"] = "carving OFF (2mm noise, restore/island on)"
    c["key"] = "carving_off"
    conditions.append(c)

    c = dict(base, restore=False)
    c["label"] = "parallax restore OFF (2mm noise, carving/island on)"
    c["key"] = "restore_off"
    conditions.append(c)

    c = dict(base, island=False)
    c["label"] = "island culling OFF (2mm noise, carving/restore on)"
    c["key"] = "island_off"
    conditions.append(c)

    # max_incidence_deg sweep at the baseline pipeline.
    for deg in (70.0, 75.0, 78.0, 80.0, 82.0, 85.0):
        c = dict(base, max_incidence_deg=deg)
        c["label"] = f"max_incidence_deg={deg:.0f} (2mm noise, all filters on)"
        c["key"] = f"incidence_{int(deg)}"
        conditions.append(c)

    return conditions


def _run_one(cond: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import numpy as np

    from rapidmesh import synthetic
    from rapidmesh.filters import carve_movers, isolation_despeckle, restore_parallax_carve
    from rapidmesh.grid import CoarseRangeGrid, ScanGrid, select
    from rapidmesh.triangulate import build_mesh, cull_islands, triangulate

    rows, cols, noise = cond["rows"], cond["cols"], cond["noise"]
    scene_a = synthetic.RoomScene(scanner=(0.35, -0.2, 0.0), mover=True)
    scene_b = synthetic.RoomScene(scanner=(-2.6, 1.7, 0.0), mover=False)
    scene_c = synthetic.RoomScene(scanner=(2.7, 1.4, 0.0), mover=False)

    a = synthetic.generate(scene_a, rows=rows, cols=cols, range_noise=noise, station_id="A", seed=7)
    b = synthetic.generate(scene_b, rows=rows, cols=cols, range_noise=noise, station_id="B", seed=11)
    c = synthetic.generate(scene_c, rows=rows, cols=cols, range_noise=noise, station_id="C", seed=13)
    neighbours = [CoarseRangeGrid.build(b.scan), CoarseRangeGrid.build(c.scan)]

    grid = ScanGrid.build(a.scan)
    keep = isolation_despeckle(grid)
    stage2 = select(grid.scan, keep)

    if cond["carving"]:
        keep2 = carve_movers(stage2, neighbours, min_votes=1)
        drop2 = ~keep2
        if cond["restore"]:
            drop2 = restore_parallax_carve(ScanGrid.build(stage2), drop2)
        keep[np.nonzero(keep)[0][drop2]] = False
        stage2 = select(stage2, ~drop2)

    g2 = ScanGrid.build(stage2)
    tris = triangulate(g2, max_incidence_deg=cond["max_incidence_deg"])
    if cond["island"]:
        tris = cull_islands(g2.scan.xyz, tris)
    mesh = build_mesh(g2.scan, tris)

    used = np.zeros(len(stage2), bool)
    if len(tris):
        used[tris.ravel()] = True
    in_mesh = np.zeros(len(a.scan), bool)
    in_mesh[np.nonzero(keep)[0]] = used

    analysis = _analyze(a, mesh, in_mesh, cond)
    analysis["label"] = cond["label"]
    analysis["key"] = cond["key"]
    analysis["condition"] = {k: v for k, v in cond.items() if k not in ("rows", "cols")}
    analysis["mesh"] = {"vertices": mesh.vertex_count, "triangles": mesh.triangle_count}

    rows_, cols_, vals_ = analysis.pop("_heatmap_residuals")
    _write_heatmap(a.scan.lattice.rows, a.scan.lattice.cols, rows_, cols_, vals_,
                   out_dir / f"{cond['key']}_heatmap.png")
    return analysis


# --------------------------------------------------------------------------
# analysis: incidence, range, discontinuity distance, worst 0.1%
# --------------------------------------------------------------------------


def _classify_faces(a: Any) -> tuple[Any, Any]:
    """(face_id, normal) per ORIGINAL sample, for SURF_ROOM samples only.

    face_id: 0=floor, 1=ceiling, 2=wall (X or Y). Classified from the WORLD
    point against the scene's own axis-aligned bounds, independent of the ray
    caster's internal wall tag (which is not stored on SyntheticScan) — this
    keeps the diagnostic self-contained rather than reaching into synthetic.py
    internals.
    """
    import numpy as np

    from rapidmesh import synthetic

    scene = a.scene
    world = a.scan.pose.translation[None, :] + a.scan.xyz.astype(np.float64)
    face = np.full(len(a.scan), -1, np.int8)
    normal = np.zeros((len(a.scan), 3), np.float64)

    room = a.surface == synthetic.SURF_ROOM
    z = world[:, 2]
    is_floor = room & (np.abs(z - scene.floor_z) < _FACE_TOL)
    is_ceil = room & (np.abs(z - scene.ceil_z) < _FACE_TOL)
    is_wall = room & ~is_floor & ~is_ceil

    face[is_floor] = 0
    normal[is_floor] = (0, 0, 1)
    face[is_ceil] = 1
    normal[is_ceil] = (0, 0, -1)
    face[is_wall] = 2
    # Wall normal: whichever of +-X/+-Y the point sits closest to.
    x, y = world[:, 0], world[:, 1]
    dists = np.stack([
        np.abs(x - (-scene.half_x)), np.abs(x - scene.half_x),
        np.abs(y - (-scene.half_y)), np.abs(y - scene.half_y),
    ], axis=1)
    which = np.argmin(dists, axis=1)
    wall_normals = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)], np.float64)
    normal[is_wall] = wall_normals[which[is_wall]]

    return face, normal


def _discontinuity_cell_distance(a: Any) -> Any:
    """Per-ORIGINAL-sample distance, in lattice cells, to the nearest lattice
    quad that spans a discontinuity.

    A cell is a discontinuity if its four corners do not all share the same
    analytic surface id (including "no return" / mover as their own ids) —
    this is the ground-truth definition of an edge, independent of whatever
    the triangulator's own incidence test decides, so it can be used to check
    the triangulator against reality rather than against itself.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    from rapidmesh import synthetic

    rows, cols = a.scan.lattice.rows, a.scan.lattice.cols
    # SURF_NONE (0) already means "no return"; mover samples get their own id
    # so a mover/static boundary counts as a discontinuity too.
    label = np.zeros((rows, cols), np.int32)
    surf_id = np.where(a.is_mover, np.int32(synthetic.SURF_MOVER), a.surface.astype(np.int32))
    label[a.scan.row, a.scan.col] = surf_id + 1  # +1 so "never written" (0) reads as its own id

    # 4-neighbour comparison, row and column directions, with lattice column
    # wrap (the fixture is a full 360 deg sweep).
    diff_row = np.zeros((rows, cols), bool)
    diff_row[:-1, :] |= label[:-1, :] != label[1:, :]
    diff_row[1:, :] |= label[:-1, :] != label[1:, :]

    rolled = np.roll(label, -1, axis=1)
    diff_col = label != rolled
    diff_col |= np.roll(diff_col, 1, axis=1)

    disc = diff_row | diff_col
    # Distance transform of the INVERSE mask: distance from every cell to the
    # nearest True (discontinuity) cell.
    dist = distance_transform_edt(~disc)
    return dist[a.scan.row, a.scan.col]


def _analyze(a: Any, mesh: Any, in_mesh: Any, cond: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from rapidmesh import qa, synthetic

    static = ~a.is_mover
    face, _normal = _classify_faces(a)
    disc_dist = _discontinuity_cell_distance(a)
    ranges = a.true_range

    # Incidence angle: analytically exact for wall/floor/ceiling samples,
    # since the surface normal is known exactly (axis-aligned box). This is
    # the true incidence the ray struck the surface at, not an estimate from
    # noisy neighbouring points — which is the whole reason a synthetic
    # fixture is used for this question at all.
    room_mask = (a.surface == synthetic.SURF_ROOM) & static
    incidence_deg = np.full(len(a.scan), np.nan)
    if np.any(room_mask):
        _f, n = _classify_faces(a)
        cos_th = np.abs(np.einsum("ij,ij->i", a.direction[room_mask], n[room_mask]))
        incidence_deg[room_mask] = np.degrees(np.arccos(np.clip(cos_th, 0.0, 1.0)))

    def bucket_report(mask: Any) -> dict[str, Any]:
        sel = mask & in_mesh
        n = int(sel.sum())
        if n == 0:
            return {"n": 0, "rms_mm": 0.0, "p99_9_mm": 0.0, "max_mm": 0.0}
        pts = (a.direction[sel] * ranges[sel][:, None]).astype(np.float32)
        d = qa.distances(mesh, pts, max_samples=200_000)
        return {
            "n": n,
            "rms_mm": float(np.sqrt(np.mean(d * d)) * 1000),
            "p99_9_mm": float(np.percentile(d, 99.9) * 1000) if d.size else 0.0,
            "max_mm": float(d.max() * 1000) if d.size else 0.0,
        }

    walls = bucket_report(room_mask & (face == 2))
    floor = bucket_report(room_mask & (face == 0))
    overall = bucket_report(static)

    # Grouped breakdowns, computed on the walls+floor population only (the
    # planar, analytically-clean surfaces the tail was traced to).
    plane_mask = room_mask & in_mesh
    plane_idx = np.nonzero(plane_mask)[0]
    by_incidence: dict[str, Any] = {}
    by_range: dict[str, Any] = {}
    by_discontinuity: dict[str, Any] = {}
    worst: list[dict[str, Any]] = []
    heatmap_rows = heatmap_cols = heatmap_vals = np.empty(0, np.float32)

    if plane_idx.size:
        pts = (a.direction[plane_idx] * ranges[plane_idx][:, None]).astype(np.float32)
        tri_id, d = _distances_with_triangle(mesh, pts)
        heatmap_rows = a.scan.row[plane_idx]
        heatmap_cols = a.scan.col[plane_idx]
        heatmap_vals = (d * 1000).astype(np.float32)  # mm, for the heat map's own scale

        for lo, hi in INCIDENCE_BUCKETS:
            sel = (incidence_deg[plane_idx] >= lo) & (incidence_deg[plane_idx] < hi)
            by_incidence[f"{lo}-{hi}deg"] = _stats(d[sel])
        for lo, hi in RANGE_BUCKETS:
            sel = (ranges[plane_idx] >= lo) & (ranges[plane_idx] < hi)
            by_range[f"{lo}-{hi}m"] = _stats(d[sel])
        for lo, hi in DISCONTINUITY_BUCKETS:
            sel = (disc_dist[plane_idx] >= lo) & (disc_dist[plane_idx] < hi)
            by_discontinuity[f"{lo}-{hi}cells"] = _stats(d[sel])

        n_worst = max(1, int(len(d) * 0.001))
        worst_local = np.argsort(d)[-n_worst:][::-1]
        world = a.scan.pose.translation[None, :] + a.scan.xyz[plane_idx].astype(np.float64)
        for li in worst_local:
            gi = int(plane_idx[li])
            worst.append({
                "sample_row": int(a.scan.row[gi]), "sample_col": int(a.scan.col[gi]),
                "world_xyz": [round(float(v), 5) for v in world[li]],
                "residual_mm": round(float(d[li]) * 1000, 3),
                "incidence_deg": round(float(incidence_deg[plane_idx[li]]), 2),
                "range_m": round(float(ranges[gi]), 3),
                "discontinuity_cells": round(float(disc_dist[gi]), 2),
                "nearest_triangle_id": int(tri_id[li]),
                "face": "floor" if face[gi] == 0 else ("ceiling" if face[gi] == 1 else "wall"),
            })

    return {
        "overall": overall, "walls": walls, "floor": floor,
        "by_incidence": by_incidence, "by_range": by_range,
        "by_discontinuity_cells": by_discontinuity,
        "worst_0_1pct": worst,
        "_heatmap_residuals": (heatmap_rows, heatmap_cols, heatmap_vals),
    }


def _stats(d: Any) -> dict[str, Any]:
    import numpy as np

    if d.size == 0:
        return {"n": 0, "rms_mm": 0.0, "p99_9_mm": 0.0, "max_mm": 0.0}
    return {
        "n": int(d.size),
        "rms_mm": float(np.sqrt(np.mean(d * d)) * 1000),
        "p99_9_mm": float(np.percentile(d, 99.9) * 1000),
        "max_mm": float(d.max() * 1000),
    }


def _distances_with_triangle(mesh: Any, points: Any) -> tuple[Any, Any]:
    """Like `qa.distances`, but also returns the nearest triangle id per
    point. Kept out of qa.py deliberately: this is diagnostic-only, and the
    product QA module's public contract and tests stay untouched.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from rapidmesh.qa import _point_triangle_distance
    from rapidmesh.qa_reference import _vertex_triangle_map

    verts = mesh.vertices.astype(np.float64)
    tris = mesh.triangles.astype(np.int64)
    q64 = points.astype(np.float64)

    tree = cKDTree(verts)
    _, nearest = tree.query(q64, k=2, workers=-1)
    nearest = np.atleast_2d(nearest)

    start, incident = _vertex_triangle_map(tris, verts.shape[0])
    best = np.full(q64.shape[0], np.inf)
    best_tri = np.full(q64.shape[0], -1, np.int64)
    for col in range(nearest.shape[1]):
        v = nearest[:, col]
        cnt = (start[v + 1] - start[v]).astype(np.int64)
        if not cnt.sum():
            continue
        owner = np.repeat(np.arange(q64.shape[0], dtype=np.int64), cnt)
        pos = np.arange(owner.size, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tri_ids = incident[np.repeat(start[v], cnt) + pos]
        t = tris[tri_ids]
        d = _point_triangle_distance(q64[owner], verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]])
        better = d < best[owner]
        idx = owner[better]
        best[idx] = d[better]
        best_tri[idx] = tri_ids[better]
    return best_tri, best


# --------------------------------------------------------------------------
# heat map — dependency-free PNG (no matplotlib in this project's deps)
# --------------------------------------------------------------------------


def _write_heatmap(
    rows: int, cols: int, sample_rows: Any, sample_cols: Any, residual_mm: Any, path: Path
) -> None:
    """Lattice-space (row, col) heat map of residual magnitude for the
    walls+floor population — the natural 'image' for a structured scan, since
    row/col already IS the sensor's own raster. Blue->red, grey where no
    sample was retained at that cell.

    Takes the per-sample (row, col, residual) arrays directly — computed once
    in `_analyze` alongside the grouped statistics — rather than recomputing
    point-to-mesh distances a second time here.
    """
    import numpy as np

    if sample_rows.size == 0:
        return
    img = np.full((rows, cols), -1.0, np.float32)
    img[sample_rows, sample_cols] = residual_mm

    valid = img >= 0
    if not np.any(valid):
        return
    vmax = max(float(np.percentile(img[valid], 99.5)), 1e-6)
    _write_png_heatmap(img, valid, vmax, path)


def _write_png_heatmap(img: Any, valid: Any, vmax: float, path: Path) -> None:
    import numpy as np

    h, w = img.shape
    t = np.clip(img / vmax, 0.0, 1.0)
    # Blue (low) -> yellow -> red (high), grey (no data).
    r = np.clip(1.5 * t, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(t - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - t), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0.5
    rgb8 = (rgb * 255).astype(np.uint8)

    # Downscale row-wise if very tall, so the PNG stays a reasonable size —
    # this is a diagnostic image, not a data product.
    max_h = 2000
    if h > max_h:
        step = h // max_h
        rgb8 = rgb8[::step]
        h = rgb8.shape[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, rgb8)


def _write_png(path: Path, rgb: Any) -> None:
    """Minimal stdlib-only PNG writer (no PIL/matplotlib dependency)."""
    import struct

    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = bytearray()
    for row in rgb:
        raw += b"\x00" + row.tobytes()
    idat = zlib.compress(bytes(raw), level=6)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY — walls and floor reported separately, per FINDING-003")
    print("=" * 100)
    print(f"{'condition':<42}{'walls rms':>10}{'walls p99.9':>13}{'floor rms':>11}{'floor p99.9':>13}")
    for r in results:
        w, f = r["walls"], r["floor"]
        print(f"{r['label']:<42}{w['rms_mm']:>9.2f}m{w['p99_9_mm']:>12.2f}m"
              f"{f['rms_mm']:>10.2f}m{f['p99_9_mm']:>12.2f}m")
    print("\n(budget: 25mm tolerance -> 8.0mm p99.9 budget; 10mm floor -> 3.2mm p99.9 budget)")


def _print_decision_table(results: list[dict[str, Any]]) -> None:
    """Prints the finding's own if/then table alongside what THIS run
    actually measured for each row's condition — it does not pick a winner.
    Reading the conclusion off this table, rather than deciding it here, is
    the entire point of running the matrix."""
    print("\n" + "=" * 100)
    print("DECISION TABLE (FINDING-003) — read against the numbers above, not asserted here")
    print("=" * 100)
    by_key = {r["key"]: r for r in results}
    carving_on = next((r for r in results if r["key"] == "noise_2mm"), None)
    carving_off = by_key.get("carving_off")
    if carving_on and carving_off:
        print("'Tail disappears with carving off, clusters at silhouettes' -> promote silhouette-aware carving:")
        print(f"    carving on  p99.9(walls)={carving_on['walls']['p99_9_mm']:.2f}mm  "
              f"p99.9(floor)={carving_on['floor']['p99_9_mm']:.2f}mm")
        print(f"    carving off p99.9(walls)={carving_off['walls']['p99_9_mm']:.2f}mm  "
              f"p99.9(floor)={carving_off['floor']['p99_9_mm']:.2f}mm")
    incidence_rows = [r for r in results if r["key"].startswith("incidence_")]
    if incidence_rows:
        print("\n'Tracks max_incidence_deg' -> prioritise incidence-aware triangulation:")
        for r in sorted(incidence_rows, key=lambda r: r["condition"]["max_incidence_deg"]):
            print(f"    max_incidence_deg={r['condition']['max_incidence_deg']:.0f}  "
                  f"p99.9(walls)={r['walls']['p99_9_mm']:.2f}mm  p99.9(floor)={r['floor']['p99_9_mm']:.2f}mm")
    print("\n'Worst triangles join different analytic surfaces' -> fix boundary connectivity:")
    print("    see each condition's *_heatmap.png and results.json worst_0_1pct[].discontinuity_cells")
    print("\n'Geometry valid but metric misattributes it' -> fix the truth metric, not the mesher:")
    print("    check results.json worst_0_1pct[] entries for points far from ANY discontinuity")
    print("    (large discontinuity_cells) that still show a large residual — that pattern is not")
    print("    explained by triangulation at all and points at the measurement, not the mesh.")


if __name__ == "__main__":
    raise SystemExit(main())
