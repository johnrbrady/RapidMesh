# Finding 003 — the p99.9 tail is geometric, and the cause is not yet identified

**Date:** 2 August 2026
**Severity:** Medium. It fails a stated acceptance test at the tolerance floor.
**Status:** **Two causes eliminated by measurement. Cause not yet identified.**
An isolation matrix is specified below and is a phase 1 deliverable.

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

`tools/bench_synthetic.py --rows 400 --cols 1600`, three-station room fixture,
carving with one vote plus parallax restore. Only `--noise` varied.

| Range sigma | RMS | p99.9 | Max |
|---|---|---|---|
| **0.000 mm** | **0.64 mm** | **13.45 mm** | 53.7 mm |
| 2.0 mm | 1.45 mm | 14.17 mm | 48.9 mm |

Removing observation noise entirely halves RMS and moves p99.9 by roughly 5%.
**With a mathematically perfect instrument the tail is still there.**

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

- The tail is **geometric, not stochastic**.
- It is confined entirely to **walls and floor**. Door openings, the 300 mm
  column and the 60 mm handrail all report 0.00 mm RMS and 0.0 mm max at zero
  noise with carving off.
- Roughly **0.12% of wall/floor samples do not survive** even with carving
  disabled. Speckle removal accounts for 726 of 633,643 samples, or 0.115%,
  so essentially all of that loss is despeckle.
- Sampling resolution is the dominant term overall. From `README.md`, p99.9
  falls 19.18 → 14.0 → 4.83 → 4.49 → 3.55 mm as sampling goes 0.300° → 0.090°.

A mathematically exact planar wall or floor should produce samples lying on
that plane, and triangles joining same-plane samples should also lie on it. A
52.1 mm error at zero noise is therefore anomalous and needs a stage
identified, not a plausible story attached.

---

## Remaining candidates — none eliminated

1. Triangles joining **different** analytic surfaces near an edge or a
   wall/floor intersection.
2. Filtering (despeckle) or island culling altering coverage near a
   discontinuity.
3. An incidence threshold (`max_incidence_deg`) producing gaps or invalid
   reconstruction at grazing angles.
4. Island culling or parallax restoration side effects.
5. **Incorrect attribution inside the truth-report calculation itself.**
   Not a remote possibility: `FINDING-002-QA-DEFINITION.md` already documents a
   QA attribution defect in the sibling metric.

---

## Required isolation matrix — phase 1 deliverable

Run at the **fine 0.090° sampling** that produced the 3.55 mm figure, not at
the coarse setting used above.

**Vary, one at a time:**

- noise: 0 mm (and the parameterised sweep, see below)
- carving: enabled / disabled
- parallax restoration: enabled / disabled
- island culling: enabled / disabled
- `max_incidence_deg`: controlled sweep

**Report:**

- walls and floors **separately**, not combined
- residuals grouped by incidence angle, by range, and by distance from the
  nearest depth discontinuity
- coordinates and triangle IDs for the worst 0.1% of results
- a visual heat map of those residuals

**The result decides the fix, and nothing is reprioritised before it:**

| If | Then |
|---|---|
| The tail disappears with carving disabled and clusters at silhouettes | Promote silhouette-aware carving |
| It tracks `max_incidence_deg` | Prioritise incidence-aware triangulation |
| The worst triangles join different analytic surfaces | Fix boundary connectivity and discontinuity handling |
| The geometry is valid but the metric misattributes it | Fix the truth metric, not the mesher |

Measurements 1 and 2 above have already eliminated noise and carving as the
*primary* cause. They have not identified the responsible stage.

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
