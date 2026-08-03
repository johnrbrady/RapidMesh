# Finding 003 — the p99.9 tail is range noise, propagated correctly

**Date:** 2 August 2026. **Resolved 2 August 2026** — see §"The isolation
matrix, run" below. The two candidate causes eliminated by the original
measurement (noise, carving) were eliminated at the WRONG sampling
resolution, which is exactly the kind of error `docs/HANDOVER.md` §10 rule 2
exists to prevent ("do not attribute a cause without isolating the stage").
Re-run at the correct fine 0.090° sampling the matrix specifies, the picture
inverts: **noise is not eliminated — it is confirmed as the whole story**, and
none of carving, parallax restoration, island culling, or `max_incidence_deg`
in the range that matters explain any part of the tail.
**Severity:** Downgraded from "cause unknown" to "understood for this planar
fixture." The acceptance-test failure itself is unchanged and still real — see
§"What this does and does not mean" below.
**Status:** **Isolation matrix run in full, per the specification this
document laid out.** Every branch of the decision table below is checked
against real numbers, not asserted.

---

## The statement of the problem, in the exact wording to use

> **Passes the 25 mm default-tolerance budget and fails the 10 mm
> minimum-tolerance budget.**

Derived budget (`docs/adr/ADR-002` Decision 5): mesh error ≤ 10% of tolerance.

| Tolerance | p99.9 budget | Best synthetic p99.9 | Verdict |
|---|---|---|---|
| 25 mm (default) | 8.0 mm | 3.55 mm | **Pass**, with margin |
| 10 mm (floor) | 3.2 mm | 3.55 mm | **Fail**, by 11% |

Do not widen the 3.2 mm budget quietly. Do not assume one synthetic noise
setting establishes the appropriate minimum tolerance for every project.

---

## Measurement 1 — range noise is not the cause

**Superseded — see "The isolation matrix, run" below.** This measurement was
honestly run and is left in place as the record of what was known before the
matrix, not deleted. It was run at `--rows 400 --cols 1600` — coarser than
this document's own "fine 0.090°" definition — and at that resolution the
conclusion below was correct for what it measured. Rerun at the sampling this
document actually specifies, with zero noise, the result is not "13.45 mm
regardless of noise" — it is **0.00 mm**. The coarse-sampling run was
measuring binning/resolution error that swamped the noise signal, not
evidence that noise is irrelevant. Do not cite the table immediately below as
current evidence about the cause; cite the isolation matrix section instead.

`tools/bench_synthetic.py --rows 400 --cols 1600`, three-station room fixture,
carving with one vote plus parallax restore. Only `--noise` varied.

| Range sigma | RMS | p99.9 | Max |
|---|---|---|---|
| **0.000 mm** | **0.64 mm** | **13.45 mm** | 53.7 mm |
| 2.0 mm | 1.45 mm | 14.17 mm | 48.9 mm |

Removing observation noise entirely halves RMS and moves p99.9 by roughly 5%
*at this coarser sampling*. **This does not hold at the fine 0.090° sampling
— see below.**

## Measurement 2 — cross-station carving is not the cause

Same fixture, zero noise, carving disabled:

```
=== no carving ====================================
  mesh        632,912 verts  1,246,127 tris
  removed     speckle=726  carve=0
  accuracy    rms=0.52 mm  p99.9=12.89 mm  max=48.9 mm
  per-feature   samples   survived   accuracy rms   max
    walls/floor    608,762     99.88%         0.57 mm    52.1 mm
    through door     6,238    100.00%         0.00 mm     0.0 mm
    column 300mm     9,272    100.00%         0.00 mm     0.0 mm
    handrail 60mm      776    100.00%         0.00 mm     0.0 mm
```

With carving **off** and noise **zero**, p99.9 is 12.89 mm. Enabling carving
moves it to 13.52 mm — a contribution of about 0.6 mm, roughly 5% of the tail.

**Carving is not the primary cause, and silhouette-aware carving would not fix
this.** An earlier suggestion in session discussion that silhouette-aware
carving be promoted on the strength of this tail is **retracted**. It was
inferred from a per-feature line without isolating the stage, which is exactly
the reasoning the project charter prohibits.

## What the numbers do establish

- **Corrected below — see "The isolation matrix, run".** At this section's
  coarse sampling the tail did look geometric rather than stochastic, because
  binning error was large enough to hide the noise signal underneath it. At
  the fine sampling this document specifies, the opposite holds: the tail
  tracks injected noise almost exactly and vanishes at zero noise. This
  bullet is left for the historical record, not as a current conclusion.
- It is confined entirely to **walls and floor**. Door openings, the 300 mm
  column and the 60 mm handrail all report 0.00 mm RMS and 0.0 mm max at zero
  noise with carving off. **This part still holds** at fine sampling too.
- Roughly **0.12% of wall/floor samples do not survive** even with carving
  disabled. Speckle removal accounts for 726 of 633,643 samples, or 0.115%,
  so essentially all of that loss is despeckle.
- Sampling resolution is the dominant term overall. From `README.md`, p99.9
  falls 19.18 → 14.0 → 4.83 → 4.49 → 3.55 mm as sampling goes 0.300° → 0.090°.

A mathematically exact planar wall or floor should produce samples lying on
that plane, and triangles joining same-plane samples should also lie on it. A
52.1 mm error at zero noise was therefore anomalous and needed a stage
identified, not a plausible story attached — **it has one now: this was the
coarse-sampling binning/resolution error described above, at a sampling this
document's own title has always specified should be finer.** See "The
isolation matrix, run" below, where the same zero-noise case at the correct
resolution reads 0.00 mm.

---

## The isolation matrix, run

`tools/isolation_matrix.py`. Full command, environment and output committed
as evidence: `python tools/isolation_matrix.py --out out/isolation_matrix`,
1,363 s wall-clock, three-station room fixture, **verified at 0.09000°/
0.09002° az/el step (1334 × 4000, 5,336,000 samples/station)** before a single
condition ran — the script refuses to run at any other resolution, so this
cannot silently repeat the earlier mistake of measuring at the wrong scale.
`out/isolation_matrix/` (gitignored, 155 MB: `results.json` plus one heat map
PNG per condition) holds the full per-sample record; the numbers below are
the ones that decide the finding.

**Walls and floor, reported separately, at every condition:**

| Condition | walls RMS | walls p99.9 | floor RMS | floor p99.9 |
|---|---|---|---|---|
| noise = 0 mm | **0.00 mm** | **0.00 mm** | **0.00 mm** | **0.00 mm** |
| noise = 1 mm | 0.75 mm | 2.23 mm | 0.50 mm | 1.43 mm |
| noise = 2 mm (baseline) | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |
| noise = 3 mm | 1.60 mm | 4.40 mm | 0.99 mm | 3.12 mm |
| noise = 5 mm | 2.06 mm | 5.99 mm | 1.31 mm | 4.40 mm |
| carving OFF (2 mm noise) | 1.25 mm | 3.46 mm | 0.78 mm | 2.38 mm |
| parallax restore OFF (2 mm noise) | 1.25 mm | 3.45 mm | 0.78 mm | 2.39 mm |
| island culling OFF (2 mm noise) | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |
| `max_incidence_deg` = 70° | 1.25 mm | 3.46 mm | 0.92 mm | **3.99 mm** |
| `max_incidence_deg` = 75° | 1.26 mm | 3.47 mm | 0.79 mm | 2.70 mm |
| `max_incidence_deg` = 78° | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |
| `max_incidence_deg` = 80° (≈ default) | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |
| `max_incidence_deg` = 82° (default) | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |
| `max_incidence_deg` = 85° | 1.26 mm | 3.47 mm | 0.78 mm | 2.37 mm |

**The headline result for this fixture: at zero noise, both walls and floor read exactly
0.00 mm — RMS, p99.9 and max.** Not small; zero. The earlier "12.89 mm at zero
noise" finding (Measurement 1, above) was real and honestly measured, but at
a coarser sampling than this document's own title names as the one that
matters. At the correct fine sampling, RapidMesh's own geometric contribution
to this fixture is not merely small — it is not measurably present at all.

**The tail tracks noise almost exactly, and nothing else moves it.** p99.9
rises from 0.00 → 2.23 → 3.47 → 4.40 → 5.99 mm as injected noise rises from
0 → 1 → 2 → 3 → 5 mm — monotonic, and the only variable in the whole matrix
that produces a change of this size. Carving, parallax restore and island
culling each move the number by ≤ 0.02 mm off baseline — noise, not signal.
`max_incidence_deg` in the range that matters (78–85°) is **bit-for-bit
identical** across five separate runs; the only incidence setting that moved
anything was 70°, which made the floor tail *worse* (3.99 mm, and 51,495 fewer
floor samples survived) by cutting real, valid geometry — the opposite of
what "incidence causes the tail" would predict.

**Grouped breakdowns settle the remaining two candidates decisively**
(`noise_2mm` condition, full tables in `out/isolation_matrix/results.json`):

- **By distance from the nearest depth discontinuity:** RMS and p99.9 are
  statistically flat from 0–1 cells (at a discontinuity: 1.20 mm RMS, 4.22 mm
  p99.9) all the way to 10+ cells (far from any: 1.15 mm RMS, 4.15 mm p99.9).
  If triangles joining different analytic surfaces, or filtering altering
  coverage near an edge, were the cause, samples at a discontinuity would read
  measurably worse than samples far from one. They do not.
- **By incidence angle:** no monotonic trend toward the 82° limit (0–40°:
  4.23 mm p99.9; 40–60°: 3.98 mm; 60–70°: 4.29 mm; 70–75°: 4.56 mm — bouncing,
  not climbing). Nothing survives past 78° in this fixture's actual geometry.
- **Worst 0.1% of points, individually:** `discontinuity_cells` among the
  worst-case points ranges from 0.0 to 16.6, mean 4.1 — scattered across the
  full range, not clustered at edges. Incidence angle among the worst points
  averages 40.4°, nowhere near the limit. This is the signature of noise
  landing where it lands, not a systematic geometric failure mode.

## Decision table, applied

| If | Then | What was measured |
|---|---|---|
| Tail disappears with carving off, clusters at silhouettes | Promote silhouette-aware carving | **No.** Carving off changes walls/floor by ≤ 0.02 mm |
| Tracks `max_incidence_deg` | Prioritise incidence-aware triangulation | **No.** Identical across 78–85°; 70° makes it *worse* by cutting valid geometry |
| Worst triangles join different analytic surfaces | Fix boundary connectivity | **No.** Residuals are flat across all discontinuity-distance buckets |
| Geometry is valid but the metric misattributes it | Fix the truth metric, not the mesher | **Closest, with a correction below** |

None of the first three hold. The fourth is the nearest fit, but "the metric
misattributes it" is not quite the right description of what was found, and
stating it precisely matters:

**The metric has no bug.** `qa.truth_report` is doing exactly what it is
documented to do: comparing the finished mesh against noise-free analytic
geometry. What was wrong was not the metric — it was reading its number at
coarse sampling as evidence about RapidMesh's own error, when at coarse
sampling the geometric (binning/resolution) error and the noise-propagation
error were both present and not separated. Running at the fine sampling this
document's own title names, with noise as an explicit, independent variable,
separates them cleanly: **RapidMesh's own contribution is ≈ 0 mm; the entire
reported tail is the propagated effect of the fixture's assumed instrument
noise**, damped by triangle-vertex averaging exactly as `ARCHITECTURE.md`'s
measured-performance section already predicted ("a triangle's surface
averages three noisy vertices and cancels part of the error").

## What this does and does not mean

**Does not mean:** that the acceptance test now passes. It does not.
`ADR-002` Decision 6's own stated rationale for the 10 mm floor lists
"scanner range noise at working distance" as one of several things
competing for that budget alongside RapidMesh's own error — this finding is
the first measurement that actually separates the two, and confirms the
budget's own reasoning was correct: at a 2 mm-sigma instrument, propagated
noise alone consumes more than the 3.2 mm mesher-only sub-budget, before
RapidMesh's own geometry contributes anything at all.

**Does mean:** this matrix found no mesher defect in the tested planar fixture.
The tested filter stages and triangulation parameters do not materially explain
its tail. Further
engineering effort aimed at "closing the gap" between 3.47 mm and 3.2 mm by
tuning carving, restoration, island culling or incidence handling would be
chasing noise, not signal — the isolation matrix is the evidence for that,
not an assertion.

**What would actually move this number:** a smaller assumed noise sigma (the
real instrument's actual specification, not the fixture's placeholder 2 mm —
see the next section), or accepting that the 10 mm floor and a ~2 mm
instrument are close to the physical limit for the mesher-only sub-budget
regardless of code quality, which is a product/scope question, not an
engineering one.

---

## Noise is a test parameter, not a fact about the reference data

The 2 mm range sigma in the fixture is a **documented test assumption**. It is
not a measured property of the 02516.182 stations. Those files declare no
`sensorVendor`, so the instrument, its specification and its calibration state
are all unknown (`docs/DATA-INVENTORY.md` §1.6, open item 2).

The fixture keeps 2 mm for continuity, but as one case in a parameterised sweep
— initially 0, 1, 2, 3 and 5 mm — with systematic bias and grazing-angle cases
added where practical.

### Four-layer test model

| Layer | Measures | Answers |
|---|---|---|
| 1. Noise-free analytic fixture | Pure reconstruction and topology error against exact geometry | Is the mesher itself correct? |
| 2. Noisy analytic fixture | Delivered mesh against known geometry after realistic observation noise | Does the end-to-end result meet the budget? |
| 3. Decimation contribution | Each LOD against retained observations **and** the undecimated reference surface | How much error did simplification add? |
| 4. Real-data report | Retained-surface and mesh-to-source behaviour | What happens on real scans, without pretending analytic truth exists |

**Acceptance stays on layer 2**, noise included. Layers 1 and 3 are diagnostic:
together they say whether a failure came from RapidMesh, from the simulated
observations, or from simplification.

---

## Evidence required from project records — phase 2

Until these are obtained from the field or processing records, the instrument
and the registration quality of 02516.182 are unknown, and no statement about
minimum defensible tolerance for that dataset can be made.

- scanner make, model and serial number
- scan resolution and quality settings
- current calibration information, if available
- registration software and method
- station-to-station residuals
- target and checkpoint residuals
- control-network accuracy
- maximum and RMS registration errors
- any excluded or weakly constrained stations
