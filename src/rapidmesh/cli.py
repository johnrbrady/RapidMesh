"""
Command line entry point.

    rapidmesh probe   <file.e57|folder>       what lattice will we actually get
    rapidmesh mesh    <file.e57> [--out DIR]  mesh every station in a file
    rapidmesh bench                           score against analytic truth

`probe` first, always. It answers the one question that decides how good the
output can be — did this export carry the scanner's structured grid — in
seconds and without reading a point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rapidmesh", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="report the lattice tier of every scan in a file")
    p.add_argument("path", type=Path)

    m = sub.add_parser("mesh", help="mesh every station in an E57")
    m.add_argument("path", type=Path)
    m.add_argument("--out", type=Path, default=Path("out"))
    m.add_argument("--max-points", type=int, default=None,
                   help="stride the lattice down to roughly this many samples")
    m.add_argument("--no-carve", action="store_true",
                   help="skip cross-station mover removal")

    sub.add_parser("bench", help="score the pipeline against synthetic ground truth")

    args = ap.parse_args(argv)

    if args.cmd == "probe":
        return _probe(args.path)
    if args.cmd == "mesh":
        return _mesh(args.path, args.out, args.max_points, not args.no_carve)
    return _bench()


def _probe(path: Path) -> int:
    from . import e57_reader

    if not e57_reader.deps_ok():
        print("pye57 is not installed — pip install 'rapidmesh[e57]'", file=sys.stderr)
        return 2

    files = sorted(path.glob("*.e57")) if path.is_dir() else [path]
    if not files:
        print(f"no E57 files under {path}", file=sys.stderr)
        return 2

    tiers: dict[str, int] = {}
    for f in files:
        print(f"\n{f.name}")
        try:
            for scan in e57_reader.probe(f):
                print("  " + scan.describe())
                tiers[scan.source.value] = tiers.get(scan.source.value, 0) + 1
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the sweep
            print(f"  could not read: {exc}")

    print("\nlattice tiers found:")
    for k, v in sorted(tiers.items()):
        print(f"  {k:<18} {v} scan(s)")
    if tiers.get("e57-projected"):
        print(
            "\n  WARNING: 'e57-projected' means the export did not carry the scanner's\n"
            "  structured grid, so the lattice has to be guessed back from XYZ. That is\n"
            "  the handicap RapidMesh exists to remove. Re-export from the scanner\n"
            "  software with structured/gridded output enabled if the option exists."
        )
    return 0


def _mesh(path: Path, out: Path, max_points: int | None, carve: bool) -> int:
    from . import e57_reader
    from .grid import CoarseRangeGrid
    from .pipeline import mesh_station

    if not e57_reader.deps_ok():
        print("pye57 is not installed — pip install 'rapidmesh[e57]'", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    probes = e57_reader.probe(path)
    print(f"{path.name}: {len(probes)} scan(s)")

    scans = []
    for pr in probes:
        print(f"  reading [{pr.index}] {pr.name} ({pr.point_count:,} pts, {pr.source.value}) ...")
        scans.append(e57_reader.read_scan(path, pr.index, max_points=max_points))

    for i, scan in enumerate(scans):
        others = []
        if carve and len(scans) > 1:
            import numpy as np

            o = scan.pose.translation
            ranked = sorted(
                (s for j, s in enumerate(scans) if j != i),
                key=lambda s: float(np.linalg.norm(s.pose.translation - o)),
            )
            others = [CoarseRangeGrid.build(s) for s in ranked[:2]]

        res = mesh_station(scan, others=others)
        print("\n" + res.summary())

    print(f"\n(meshes are in memory only — the RMX container is not built yet, "
          f"see ARCHITECTURE.md. Nothing written to {out}.)")
    return 0


def _bench() -> int:
    import subprocess

    script = Path(__file__).resolve().parents[2] / "tools" / "bench_synthetic.py"
    return subprocess.call([sys.executable, str(script)])


if __name__ == "__main__":
    raise SystemExit(main())
