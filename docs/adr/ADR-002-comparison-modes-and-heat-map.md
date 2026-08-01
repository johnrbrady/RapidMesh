# ADR-002 — Comparison modes, heat map states and colour target

**Status:** Accepted
**Date:** 1 Aug 2026
**Supersedes:** the nine-state categorical heat map in the original brief

---

## Context

The original specification called for nine heat map states, signed front/behind
banding, sign-confidence flags and trajectory-confidence flags. The surveyor
workflow it is meant to serve is simpler: set a tolerance, typically 25 mm, and
see what passes.

## Decision 1 — Two colours, one threshold, plus grey

| State | Colour | Condition |
|---|---|---|
| Within tolerance | Green | \|deviation\| ≤ tolerance |
| Outside tolerance | Red | \|deviation\| > tolerance |
| No corresponding geometry | Red | Where the surface was observable |
| **No scan coverage** | **Grey** | Occluded, or outside scan range or field of view |

Tolerance defaults to 25 mm, symmetric, surveyor-editable, **clamped at a
10 mm minimum**. See Decision 6.

Signed distance is still computed where topology and normals make sign
reliable, stored with a confidence flag, and shown on click. **It does not
drive colour.**

## Decision 2 — Grey is separate from red

"Not present" covers three physically different situations:

| Situation | Meaning | Red? |
|---|---|---|
| Model surface, deviation 60 mm | Modelling or set-out error | Yes |
| Scan surface, no model | Built but not modelled | Yes |
| Model surface, scanner never saw it | Occlusion | **No** |

If unobserved surface renders red, an unscanned ceiling void reads as a
modelling failure and the report overstates model error. That is not defensible
to a client.

The renderer offers a toggle to draw grey as red for surveyors wanting a strict
two-colour output. **The JSON report always separates the two.**

## Decision 3 — Colour target is user-switchable

| Target | Powered by | Answers |
|---|---|---|
| Model coloured | Mode D | "What is wrong with my model?" Element-level statistics per wall, slab, column |
| Scan mesh coloured | Mode A | "What is on site that the model does not have?" |

Both ship together. **Modes A and D are therefore both phase 3 deliverables**,
not sequential phases. This is the main scheduling consequence of this ADR.

## Decision 4 — Mode ordering

| Mode | Status | Reason |
|---|---|---|
| A — nearest surface | Phase 3, first | The only mode testable end to end on current data |
| D — model-to-scan coverage | Phase 3, with A | Required for the grey state and for colouring the model |
| B — scanner-ray | Built phase 3, validated later | Nearly free on structured data, but the available IFC pairs with the NavVis site, not with 02516.182 |
| C — observation-ray | Deferred indefinitely | Requires per-point sensor poses. No NavVis export examined provides them |

An earlier draft of this plan put mode B first on cost grounds. That was
reversed once the model was inspected and found to pair with the NavVis
dataset (`../DATA-INVENTORY.md` §3.3). Recorded here so the reversal is not
re-litigated.

## Decision 5 — Mesh error budget is derived from tolerance

RapidMesh's own surface error must never be able to flip a cell from green to
red on its own. Budget: **10% of tolerance**.

| Tolerance | Mesh RMS budget | Mesh p99.9 budget |
|---|---|---|
| **10 mm (floor)** | **1.0 mm** | **3.2 mm** |
| **25 mm (default)** | **2.5 mm** | **8.0 mm** |
| 50 mm | 5.0 mm | 16.0 mm |

This replaces the previously asserted flat 2 mm RMS, which had no derivation
and did not rescale.

## Decision 6 — Tolerance floor of 10 mm

The tolerance control clamps at **10 mm minimum**, default 25 mm, no upper
bound.

**Rationale.** Below 10 mm the heat map stops measuring the model and starts
measuring everything else in the chain:

| Error source at the 10 mm scale | Typical contribution |
|---|---|
| Registration residual between stations, or SLAM drift | Often several mm |
| Scanner range noise at working distance | mm-scale |
| RapidMesh surface error | 1.0 mm RMS at this budget |
| Model tessellation of curved surfaces | Bounded by IFC precision, 0.01 in the sample model |

A surveyor who dials in 3 mm gets a red screen that says nothing useful about
the model and cannot be defended to a client. The floor stops the product
generating an indefensible result.

**Consequence, and it is the useful part.** The floor makes the mesher's
specification finite. **1.0 mm RMS and 3.2 mm p99.9 are the tightest figures
RapidMesh will ever be asked to meet**, at any tolerance any user can select.
That is a fixed acceptance test rather than an open-ended accuracy chase.

Implemented as a single constant, tested, and stated in the legend of every
heat map view alongside the selected tolerance.

## Consequences

- The comparison engine, heat map and reporting layer all get simpler.
- Coverage analysis (mode D) becomes mandatory rather than a nice-to-have,
  because grey cannot be computed without it.
- Every heat map view must carry a legend stating tolerance, units, mode,
  colour target, source dataset, model identifier, and percentage per state.
  Colour alone is never the only channel.
