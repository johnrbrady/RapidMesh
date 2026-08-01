# Finding 002 — the deviation report measures the wrong thing, and on real data it measures nothing

**Date:** 2 August 2026
**Severity:** High. It is the product's headline number.
**Effort to fix:** Moderate. Three separate reports, per
`RAPIDMESH-REVIEW-FINDINGS.md` §1.
**Status:** **Confirmed by measurement** on real client data, first run.

---

## What was run

RapidMesh had never been run on real client data (`README.md`, "Status"). It
has now.

```
python -m pytest                                    21 passed
tools/bench_synthetic.py --rows 300 --cols 1200     reproduces documented behaviour
```

Then against `H:\Sample\Structured\02516.182_6.e57`, a real terrestrial
station, 2,951,950 points, lattice 2746 × 1075:

```
station    scan000
lattice    1075 x 2746 (0.1311 deg/row, 0.1311 deg/col, source=e57-rowcol)
filters    in=2936521  no-return=0  speckle=2084  sweep=0  carve=0
           island=12905  (0.51% removed)
mesh       2,921,532 verts  5,760,953 tris
deviation  n=300000  rms=50.84 mm  p99.9=165.07 mm  max=8724.90 mm  <=2mm=99.63%
time       10.38s  (clean=0.18  triangulate=2.82  cull=1.89  assemble=1.85  measure=3.64)
```

**RMS 50.84 mm against a 2 mm commitment** (`00-PRODUCT-DEFINITION.md` §4).

---

## First hypothesis: bridging. Wrong.

The obvious explanation for an 8.7 m maximum is triangles spanning open space.
Measured, over all 5,760,953 triangles:

```
max edge   p50=0.007  p90=0.017  p99=0.036  p99.9=0.065  max=0.51 m
  edges > 0.05 m :   15,215  (0.264%)
  edges > 0.10 m :      489  (0.008%)
  edges > 0.25 m :       61  (0.001%)
  edges > 0.50 m :        1  (0.000%)
  edges > 1.00 m :        0  (0.000%)
```

One triangle edge over half a metre, in 5.76 million. **The discontinuity
cutting is working.** Bridging is not the cause.

---

## Actual cause: the report includes points the pipeline deliberately removed

Raw point-to-mesh distances, 300,000 samples:

```
p50      0.000 mm          > 2 mm    :  1,120  (0.373%)
p90      0.000 mm          > 8 mm    :    926  (0.309%)
p99      0.000 mm          > 25 mm   :    758  (0.253%)
p99.5    0.000 mm          > 100 mm  :    446  (0.149%)
p99.6    0.728 mm          > 1000 mm :     65  (0.022%)
p99.9  165.065 mm
p100  8724.899 mm          islands culled: 12,905 of 2,934,437 (0.440%)
```

Excluding only the 0.022% beyond one metre, **RMS falls from 50.84 mm to
10.34 mm.**

`pipeline.py:109`:

```python
dev = deviation_report(mesh, grid.scan.xyz, max_samples=measure_samples)
```

`grid.scan` is the *cleaned* scan, so despeckle and carve removals are already
excluded. **Island-culled samples are not.** They remain in `grid.scan.xyz`,
are absent from the mesh by design, and are then measured against it. 12,905
of them, 0.440%, matching the 0.373% of sampled points beyond 2 mm.

The mesher is not failing. The measurement is.

This is exactly what `RAPIDMESH-REVIEW-FINDINGS.md` §1 predicted in writing on
1 August:

> A correctly removed person will naturally be far from the resulting mesh.
> Including those points in the primary deviation result would make correct
> filtering appear to be poor meshing.

---

## The deeper problem: on real data the number is degenerate

Look again at the percentiles. **p50, p90, p99 and p99.5 are all exactly
0.000 mm.**

Every retained sample becomes a mesh vertex. There is no decimation yet, so
the mesh interpolates its own input exactly. Point-to-mesh distance is
therefore trivially zero for 99.5% of points, and the metric is measuring
whether points are vertices of a mesh built from those points. They are.

On synthetic fixtures this does not show, because there the comparison is
`qa.truth_report` against **analytic** geometry — noise-free ground truth the
mesh never saw. That is a real measurement and the 1.26 mm / 1.00 mm RMS
figures in `README.md` are legitimate. Real data has no analytic truth, so the
only number available is the degenerate one.

**Consequence:** until decimation exists, the only QA direction capable of
detecting anything on real data is **mesh-to-source** — sampling the finished
triangles and measuring back to the source points. That is
`RAPIDMESH-REVIEW-FINDINGS.md` §1 item 3, and it is not implemented.

---

## What must change

Three separate reports, never merged, per the review:

1. **Retained-surface fidelity.** Point-to-mesh over samples that are actually
   represented in the mesh. Excludes despeckle, carve **and island** removals.
2. **Filtering and coverage ledger.** Counts for retained, despeckled, carved,
   island-culled, restored, no-return, otherwise excluded. Every input sample
   accounted for exactly once.
3. **Mesh-to-source deviation.** Sample the triangles, measure back to source
   points. This is the only measure that detects invented surface, and it is
   the one that will matter on real data before decimation lands.

Additionally: the summary line must state which dataset is being treated as
the source of truth, and whether the figure is exact or sampled.

---

## Effect on the quality commitments

`00-PRODUCT-DEFINITION.md` §4 lists 2 mm RMS and 8 mm p99.9 as commitments.
Those numbers have only ever been demonstrated against synthetic analytic
truth. **There is currently no real-data fidelity figure at all**, and there
cannot be one until item 1 above exists.

The commitments themselves are not in doubt. The evidence for them is
narrower than the document implies, and the wording must say so until a
real-data measurement replaces it.

---

## Secondary finding from the same run: memory

Reading alone, no meshing:

| Station | Points | Lattice | Read | Peak RSS |
|---|---|---|---|---|
| `02516.182_6.e57` | 2,951,950 | 2746 × 1075 | 1.1 s | 356 MB |
| `02516.182_1.e57` | 14,548,765 | 6095 × 2387 | 6.9 s | **1,418 MB** |

Full pipeline on the medium station peaked at **1,747 MB**.

`pye57.read_scan_raw` materialises the whole scan, as `e57_reader.py`'s own
docstring warns. This confirms `RAPIDMESH-REVIEW-FINDINGS.md` §4: chunked
reading is a near-term gate, not a later task. The largest sample station is
14.5 M points; a 100 M-point station would be roughly 7× worse.

---

## Positive findings from the same run

Recorded so the finding is not read as wholly negative.

- **The reader works on real client data.** All 30 structured stations
  classify as `e57-rowcol`, the exact tier. The NavVis export correctly
  degrades to `e57-projected`.
- **The measured angular step matches the file header exactly.** The reader
  reports 0.0591° / 0.0589° on the high-resolution station; the same figure
  derived independently from the E57 XML footer is 0.0591°. Two independent
  routes, same answer.
- **The invalid-return question is answered.** `MIN_RANGE = 0.3` drops 14.2%
  of cells on the high-resolution outdoor station and 0.5% on the medium
  enclosed one — exactly the sky-versus-enclosed difference expected. This
  closes an open item that was previously a guess.
- **The core premise holds.** Native lattice at 0.0591° against Cairn's fixed
  2048 × 1024 grid at 0.1758° is a measured 2.98× finer angular sampling on
  the high-resolution stations.
