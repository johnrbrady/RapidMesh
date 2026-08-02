# Finding 001 — Cairn is likely quantising E57 imports to 1 cm

**Date:** 31 July 2026. **Wording revised 2 Aug 2026** per
`RAPIDMESH-REVIEW-FINDINGS.md` §6.1: unconfirmed against a real file, so
neither the title nor the body may state it as settled.
**Severity:** High if confirmed. Would affect the point cloud, the mesh, and
every measurement.
**Effort to fix:** One line in `backend/converter.py`.
**Status:** Derived from PDAL's documented default and not yet run against a
real Cairn LAZ. **Confirmation required** with `tools/check_laz_precision.py`
before any of this is acted on, restated as settled, or used to justify a
code change. The command and its actual output belong in this document as
evidence the moment that run happens — until then this is a documented
hypothesis, not a finding.

---

## The question that led here

> "Point cloud data is not great — can we use the full resolution of the
> imported scan and give that to Potree?"

Yes. But the loss is not in Potree, and the fix is upstream of it.

---

## Potree is not the problem

The current PotreeConverter invocation does not appear to deliberately thin
the point count. It distributes points across octree levels, coarse ones for
distant viewing and the full set in the leaves, rather than discarding any —
but output counts, attributes and coordinate precision through this specific
invocation still require verification before "lossless" is stated as fact
(`RAPIDMESH-REVIEW-FINDINGS.md` §6.2). Cairn invokes it with no flags at all
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

**0.01 metres. One centimetre.** If this default is genuinely in force, every
point in every Cairn project imported from E57 would be snapped to a 1 cm
lattice at the moment of import — but that is exactly what §6.1 says is not
yet confirmed against a real Cairn LAZ, and the sentence must be read that
way until it is.

The 30 structured sample stations declare no `sensorVendor`
(`docs/DATA-INVENTORY.md` §1.6), so the instrument, its specification and its
calibration state are not recoverable from these files. Do not name a
scanner make or model here; the point stands without one. If a source
instrument's specified range accuracy is known for a given project, comparing
it against this quantisation step is the right check to make, on that
project's own evidence.

---

## What it explains

| Symptom | Cause |
| --- | --- |
| Point cloud looks blocky or banded, especially on flat surfaces at grazing angles | Points snapped to a 1 cm grid form visible terraces |
| Fine detail (mortar lines, reveals, edges) looks mushy | Detail below 1 cm was rounded away at import |
| Mesh quality is disappointing | `mesher.py` reads the same LAZ, so it inherits the quantisation |
| Measurements disagree slightly with the source scan | Every endpoint moved by up to 5 mm |

**Corrected argument** (`RAPIDMESH-REVIEW-FINDINGS.md` §7 — the original
"arithmetically impossible" framing here was wrong and is retracted). A
1 cm-quantised LAZ cannot reliably preserve 2 mm geometric fidelity relative
to the original E57 or the underlying real surface, even though a mesh may
still report a low deviation when measured against the same quantised LAZ. A
mesh built from quantised points can fit those same quantised points to well
under 2 mm — it would only be inaccurate relative to the E57, the real
surface, or the scanner's own unquantised coordinates, not relative to the
degraded LAZ it was actually built from. This is exactly why the QA report
must state which dataset it is treating as the source of truth
(`FINDING-002-QA-DEFINITION.md` makes the identical point about a different
metric).

**Measurement impact, stated precisely.** Each measurement endpoint may move
independently by up to 5 mm under 1 cm quantisation. For a distance measured
between two independently displaced endpoints, the worst-case total error
approaches 10 mm, not 5 mm.

This is a reason to prefer RapidMesh reading the E57 directly and never
touching the LAZ — but that architectural preference does not depend on this
finding being confirmed, and Cairn's own quantisation should be checked and
fixed on its own merits regardless, because the point cloud Cairn ships today
would have the identical problem if the default is genuinely in force.

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

0.0001 m is 0.1 mm — twenty times finer than a typical terrestrial scanner's
range accuracy, which is the right side of the line to be on. It is expected
to have a small file-size impact, which must be confirmed on representative
scans before this is called free (`RAPIDMESH-REVIEW-FINDINGS.md` §8). LAZ
compresses the differences between neighbouring points, not their absolute
magnitude, so the effect should be small — but "should be" is not a
measurement, and §8's own list of representative scans to test against
(internal room, façade, large external station, RGB, MGA coordinates,
multi-station project) is the way to make it one.

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
