# Finding 001 — Cairn is quantising every scan to 1 cm before Potree ever sees it

**Date:** 31 July 2026
**Severity:** High. Affects the point cloud, the mesh, and every measurement.
**Effort to fix:** One line in `backend/converter.py`.
**Status:** Verified against PDAL's documentation. **Needs confirming against a
real Cairn LAZ** with `tools/check_laz_precision.py` before acting.

---

## The question that led here

> "Point cloud data is not great — can we use the full resolution of the
> imported scan and give that to Potree?"

Yes. But the loss is not in Potree, and the fix is upstream of it.

---

## Potree is not the problem

PotreeConverter 2.x is **lossless**. It does not thin the cloud; it distributes
every point across octree levels, coarse ones for distant viewing and the full
set in the leaves. Cairn invokes it with no flags at all
(`backend/converter.py`):

```python
_run([converter, str(input_path), "-o", str(output_dir)], "PotreeConverter")
```

That is fine. Nothing is discarded there.

The **viewer** point budget (`frontend/src/appearance.ts`, the Low/Medium/High
control) does cap how many points are *drawn*, but that is a render setting, not
data loss — turning it up recovers the detail, if the detail is still in the
file.

The detail is not still in the file.

---

## Where the loss actually happens

Cairn's E57 import is a two-step: PDAL converts E57 to LAZ, then PotreeConverter
builds the octree from that LAZ.

```python
def e57_to_laz(input_path, out_path):
    _run([pdal, "translate", str(input_path), str(out_path)], "PDAL (E57 -> LAZ)", ...)
```

No writer options. And LAS/LAZ **does not store coordinates as floats** — it
stores int32s and reconstructs the real value as `offset + stored * scale`. The
scale *is* the resolution of the file. Nothing finer than one scale unit can
exist in it.

PDAL's LAS writer default:

> **scale_x, scale_y, scale_z** — Scale to be divided from the X, Y and Z
> nominal values, respectively, after the offset has been applied. …
> **[Default: .01]**
>
> — https://pdal.io/en/latest/stages/writers.las.html

**0.01 metres. One centimetre.** Every point in every Cairn project imported
from E57 is snapped to a 1 cm lattice at the moment of import.

A Trimble X7 is specified at roughly 2 mm range accuracy. Cairn is currently
storing its output about five times coarser than the instrument measured it,
and there is no way to recover it afterwards — the LAZ is the only copy the
pipeline consumes.

---

## What it explains

| Symptom | Cause |
| --- | --- |
| Point cloud looks blocky or banded, especially on flat surfaces at grazing angles | Points snapped to a 1 cm grid form visible terraces |
| Fine detail (mortar lines, reveals, edges) looks mushy | Detail below 1 cm was rounded away at import |
| Mesh quality is disappointing | `mesher.py` reads the same LAZ, so it inherits the quantisation |
| Measurements disagree slightly with the source scan | Every endpoint moved by up to 5 mm |

It also puts a hard floor under RapidMesh: a 2 mm RMS target
(`00-PRODUCT-DEFINITION.md` §4) is **arithmetically impossible** from a 1 cm
quantised source. This is independent confirmation that RapidMesh reading the
E57 directly, and never touching the LAZ, is the right architecture — but Cairn
should be fixed regardless, because the point cloud has the same problem and
Cairn ships today.

---

## Confirm it first

Do not take this on the documentation alone. Run this against any LAZ Cairn
produced from an E57:

```
python tools/check_laz_precision.py "D:\CairnData\<pid>\raw\<sid>_e57.laz"
```

It prints the header scale and, more tellingly, what fraction of sampled points
sit *exactly* on the scale grid. Genuinely quantised data reads 100 %, because
by construction it cannot be anywhere else. Sub-millimetre data reads about
20 %, which is what chance alone produces.

Expected output on a current Cairn file:

```
  scale         x=0.01  y=0.01  z=0.01
  resolution    10.00 mm  <- nothing finer than this survives
  on-lattice    100.0% of sampled points sit exactly on the scale grid
  VERDICT       QUANTISED to 10.0 mm.
```

---

## The fix

In `backend/converter.py`, `e57_to_laz()`:

```python
_run([
    pdal, "translate", str(input_path), str(out_path),
    "--writers.las.scale_x=0.0001",
    "--writers.las.scale_y=0.0001",
    "--writers.las.scale_z=0.0001",
    "--writers.las.offset_x=auto",
    "--writers.las.offset_y=auto",
    "--writers.las.offset_z=auto",
], "PDAL (E57 -> LAZ)", env=_pdal_env(pdal))
```

**`offset=auto` is required, not cosmetic.** PDAL's offset default is 0, and
`written = (nominal - offset) / scale`. With scale 0.0001 and offset 0, an MGA
northing of 6,900,000 needs 69,000,000,000 stored units, which overflows the
int32 the format uses — PDAL will error or wrap. Setting offset to `auto` makes
it the minimum of each dimension, so only the *extent* of the scan has to fit,
which for a 100 m station is 1,000,000 units. Comfortable.

0.0001 m is 0.1 mm — twenty times finer than the instrument, which is the right
side of the line to be on. It costs nothing: LAZ compression is on the
differences between neighbouring points, not their absolute magnitude, so file
size barely moves.

### Also worth setting while you are in there

- `--writers.las.a_srs=<EPSG>` if the project has a known CRS. Currently
  unset, so the LAZ declares no coordinate system.
- Check `dataformat_id`. PDAL defaults to 7 (colour + time) which is correct,
  but worth confirming RGB survived — `check_laz_precision.py` reports it.

### Existing projects

The fix only affects new imports. Anything already converted is quantised
permanently in its LAZ. Re-running conversion from the retained raw E57
recovers it — Cairn keeps the raw file (`projects/<pid>/raw/<sid>.e57`), so
this is a reimport, not a re-survey. Worth doing for any project where
measurement accuracy matters.

---

## What this does not fix

The mesh still bins to 2048 × 1024 regardless
(`REVIEW-CAIRN-MESHING.md` §3.1), so `mesher.py` output stays capped even with
a clean LAZ. Fixing the quantisation raises the floor; only RapidMesh raises
the ceiling.
