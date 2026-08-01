#!/usr/bin/env python3
"""
bench_synthetic.py — score the pipeline against analytic ground truth.

Two stations of the same room. Station A is scanned while someone walks across
it; station B is scanned after they have left. That is the ordinary situation
on site, and it is the only situation in which mover removal is soundly
solvable (see `filters.py` module docstring).

Reports four numbers that matter, none of which can be obtained from real data:

  accuracy      mesh vs the ANALYTIC surface, noise removed. The honest figure.
  fidelity      mesh vs the observed points. What a client report can quote.
  recall        fraction of the walking person's samples removed.
  false pos     fraction of genuine surface samples removed. The one to watch.

    python tools/bench_synthetic.py [--rows 600] [--cols 2400]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    import numpy as np

    from rapidmesh import qa, synthetic
    from rapidmesh.filters import carve_movers, isolation_despeckle, restore_parallax_carve
    from rapidmesh.grid import CoarseRangeGrid, ScanGrid, select
    from rapidmesh.triangulate import build_mesh, cull_islands, triangulate

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--cols", type=int, default=2400)
    ap.add_argument("--noise", type=float, default=0.002, help="range sigma, metres")
    args = ap.parse_args()

    scene_a = synthetic.RoomScene(scanner=(0.35, -0.2, 0.0), mover=True)
    # Stations B and C: same room, different setups, person has walked out.
    # Two neighbours rather than one so the min_votes trade can be measured
    # instead of argued about.
    scene_b = synthetic.RoomScene(scanner=(-2.6, 1.7, 0.0), mover=False)
    scene_c = synthetic.RoomScene(scanner=(2.7, 1.4, 0.0), mover=False)

    print(f"generating three stations at {args.rows} x {args.cols} ...")
    t0 = time.perf_counter()
    a = synthetic.generate(scene_a, rows=args.rows, cols=args.cols,
                           range_noise=args.noise, station_id="A", seed=7)
    b = synthetic.generate(scene_b, rows=args.rows, cols=args.cols,
                           range_noise=args.noise, station_id="B", seed=11)
    c = synthetic.generate(scene_c, rows=args.rows, cols=args.cols,
                           range_noise=args.noise, station_id="C", seed=13)
    print(f"  {len(a.scan):,} + {len(b.scan):,} + {len(c.scan):,} samples "
          f"in {time.perf_counter() - t0:.1f}s")
    print(f"  station A contains {a.mover_count:,} mover samples "
          f"({100 * a.mover_count / len(a.scan):.2f}% of the scan)")

    neighbours = [CoarseRangeGrid.build(b.scan), CoarseRangeGrid.build(c.scan)]

    for label, others, votes, restore in (
        ("no carving", [], 1, False),
        ("carving, 1 vote (Cairn's rule)", neighbours, 1, False),
        ("carving, 2 votes", neighbours, 2, False),
        ("carving, 1 vote + parallax restore", neighbours, 1, True),
    ):
        print(f"\n=== {label} " + "=" * max(46 - len(label), 3))
        t0 = time.perf_counter()

        grid = ScanGrid.build(a.scan)
        keep = isolation_despeckle(grid)
        n_speckle = int((~keep).sum())

        stage2 = select(grid.scan, keep)
        if others:
            keep2 = carve_movers(stage2, others, min_votes=votes)
            drop2 = ~keep2
            if restore:
                drop2 = restore_parallax_carve(ScanGrid.build(stage2), drop2)
            n_carve = int(drop2.sum())
            keep[np.nonzero(keep)[0][drop2]] = False
            stage2 = select(stage2, ~drop2)
        else:
            n_carve = 0

        g2 = ScanGrid.build(stage2)
        tris = triangulate(g2)
        tris = cull_islands(g2.scan.xyz, tris)
        mesh = build_mesh(g2.scan, tris)

        # Which ORIGINAL samples made it into the mesh: survived filtering and
        # ended up in at least one triangle.
        used = np.zeros(len(stage2), bool)
        if len(tris):
            used[tris.ravel()] = True
        in_mesh = np.zeros(len(a.scan), bool)
        in_mesh[np.nonzero(keep)[0]] = used

        elapsed = time.perf_counter() - t0

        # Accuracy against analytic truth, static surfaces only. Mover samples
        # are excluded because the truth behind a mover is a surface we never
        # observed, and a correct pipeline leaves a hole there rather than
        # inventing geometry.
        static = ~a.is_mover
        truth_pts = (a.direction[static] * a.true_range[static][:, None]).astype(np.float32)
        acc = qa.deviation_report(mesh, truth_pts, max_samples=250_000)
        # Fidelity is measured against the points the pipeline RETAINED, not
        # every input point. Including deliberately-removed movers would score
        # correct filtering as error, which is exactly backwards.
        fid = qa.deviation_report(mesh, a.scan.xyz[in_mesh], max_samples=250_000)
        score = qa.score_mover_filter(a.is_mover, in_mesh)

        print(f"  mesh        {mesh.vertex_count:,} verts  {mesh.triangle_count:,} tris  ({elapsed:.1f}s)")
        print(f"  removed     speckle={n_speckle:,}  carve={n_carve:,}")
        print(f"  accuracy    rms={acc.rms * 1000:.2f} mm  p99.9={acc.p99_9 * 1000:.2f} mm  max={acc.maximum * 1000:.1f} mm")
        print(f"  fidelity    rms={fid.rms * 1000:.2f} mm  p99.9={fid.p99_9 * 1000:.2f} mm  (vs retained points)")
        print(f"  recall      {score['recall'] * 100:.1f}% of {int(score['mover_samples']):,} mover samples removed")
        print(f"  false pos   {score['false_positive'] * 100:.3f}% of static surface samples removed")
        print(f"  ghost       {int(score['kept_movers']):,} mover samples still in the mesh")

        # Per-feature breakdown. One RMS for a whole scene hides the case the
        # brief actually worries about: a filter that is fine on walls and
        # quietly deletes handrails. Survival is the number to read.
        print("  per-feature   samples   survived   accuracy rms   max")
        names = {
            synthetic.SURF_ROOM: "walls/floor",
            synthetic.SURF_BEYOND_DOOR: "through door",
            synthetic.SURF_COLUMN: "column 300mm",
            synthetic.SURF_RAIL: "handrail 60mm",
        }
        for sid, name in names.items():
            sel = (a.surface == sid) & static
            n = int(sel.sum())
            if not n:
                continue
            surv = int((sel & in_mesh).sum())
            pts = (a.direction[sel] * a.true_range[sel][:, None]).astype(np.float32)
            d = qa.distances(mesh, pts, max_samples=60_000)
            rms = float(np.sqrt(np.mean(d * d))) * 1000 if d.size else 0.0
            mx = float(d.max()) * 1000 if d.size else 0.0
            print(f"    {name:<14}{n:>8,}   {surv / n * 100:>7.2f}%   {rms:>10.2f} mm  {mx:>6.1f} mm")

    print("\nbudgets (00-PRODUCT-DEFINITION.md §4): rms <= 2 mm, p99.9 <= 8 mm,")
    print("recall >= 95%, false positives <= 0.1%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
