#!/usr/bin/env python3
"""
check_laz_precision.py — is your LAZ quantised, and by how much?

Run this against any LAZ that Cairn produced from an E57. It reads only the
header plus a small sample of points, never writes, and is safe to point at
live client data.

    python tools/check_laz_precision.py "D:\\CairnData\\<pid>\\raw\\<sid>_e57.laz"
    python tools/check_laz_precision.py "D:\\CairnData\\<pid>\\raw"      # folder

What it is looking for
----------------------
LAS/LAZ does not store coordinates as floats. It stores int32s, and recovers
the real value as ``offset + stored * scale``. The **scale is the resolution of
the file** — nothing finer than one scale unit can be represented, full stop.

PDAL's LAS writer defaults ``scale_x/y/z`` to **0.01**, i.e. one centimetre
(https://pdal.io/en/latest/stages/writers.las.html — "Scale to be divided from
the X, Y and Z nominal values ... [Default: .01]"). A `pdal translate` with no
writer options therefore snaps every point in a survey to a 1 cm lattice, and
`backend/converter.py` in Cairn issues exactly that command:

    _run([pdal, "translate", str(input_path), str(out_path)], ...)

If this script reports scale 0.01, that is what happened, and every downstream
consumer — Potree, the mesher, measurement — inherits it.

The lattice test
----------------
Header scale says what the file *can* hold. The residual test says what
actually happened: if the coordinates are all exact multiples of the scale with
no remainder, the data was quantised on the way in. A file whose points sit on
a 1 cm lattice did not measure to 1 cm; it was rounded to it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check(path: Path, sample: int = 200_000) -> int:
    """Print a verdict for one file. Returns 0 = fine, 1 = quantised."""
    import laspy
    import numpy as np

    with laspy.open(str(path)) as fh:
        hdr = fh.header
        scales = np.asarray(hdr.scales, float)
        offsets = np.asarray(hdr.offsets, float)
        count = hdr.point_count
        pts = fh.read_points(min(sample, count))

    xyz = np.stack([np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)], axis=1)

    print(f"\n{path.name}")
    print(f"  points        {count:,}")
    print(f"  scale         x={scales[0]:g}  y={scales[1]:g}  z={scales[2]:g}")
    print(f"  offset        x={offsets[0]:.3f}  y={offsets[1]:.3f}  z={offsets[2]:.3f}")
    print(f"  point format  {hdr.point_format.id}  (rgb={'yes' if 'red' in hdr.point_format.dimension_names else 'NO'})")

    worst = float(scales.max())
    print(f"  resolution    {worst * 1000:.2f} mm  <- nothing finer than this survives")

    # Residual: how far each coordinate sits from the nearest scale multiple.
    # Genuinely quantised data has a residual of ~0 by construction.
    resid = np.abs(xyz - offsets - np.round((xyz - offsets) / scales) * scales)
    on_lattice = float((resid.max(axis=1) < scales.min() * 1e-6).mean())
    print(f"  on-lattice    {on_lattice * 100:.1f}% of sampled points sit exactly on the scale grid")

    if worst > 0.001:
        mm = worst * 1000
        print(f"  VERDICT       QUANTISED to {mm:.1f} mm.")
        print("                A Trimble X7 is specified around 2 mm range accuracy, so this")
        print(f"                file is roughly {mm / 2:.0f}x coarser than the instrument that made it.")
        print("                Fix in backend/converter.py e57_to_laz():")
        print('                  pdal translate in.e57 out.laz \\')
        print("                    --writers.las.scale_x=0.0001 \\")
        print("                    --writers.las.scale_y=0.0001 \\")
        print("                    --writers.las.scale_z=0.0001 \\")
        print("                    --writers.las.offset_x=auto \\")
        print("                    --writers.las.offset_y=auto \\")
        print("                    --writers.las.offset_z=auto")
        print("                offset=auto is REQUIRED, not optional: with scale 0.0001 and")
        print("                offset 0, an MGA northing of 6,900,000 needs 69,000,000,000")
        print("                stored units and overflows the int32 the format uses.")
        return 1

    print("  VERDICT       fine — sub-millimetre resolution retained.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    files = sorted(target.glob("*.la[sz]")) if target.is_dir() else [target]
    if not files:
        print(f"no LAS/LAZ files under {target}")
        return 2
    bad = 0
    for f in files:
        try:
            bad += check(f, )
        except Exception as exc:  # noqa: BLE001 — one unreadable file must not stop the sweep
            print(f"\n{f.name}\n  could not read: {exc}")
    print(f"\n{bad} of {len(files)} file(s) quantised coarser than 1 mm.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
