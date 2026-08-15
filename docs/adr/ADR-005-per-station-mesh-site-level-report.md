# ADR-005 — Per-station meshes, site-level comparison and reporting

**Status:** Accepted for evidence geometry and site-level QC; combined-project
display choice reopened by `ADR-008`
**Date:** 2 August 2026
**Amends:** `00-PRODUCT-DEFINITION.md` §5, which places station merging out of
scope. That remains true for **geometry**. It is now explicitly false for
**comparison and reporting**.

---

## Context

The original design meshes each station independently and reports per station.
Merging stations into one continuous surface is deliberately out of scope, to
avoid registration and seam-blending problems. That reasoning is sound and is
not being overturned.

But the product's commercial sentence is site-level:

> "Across 165 modelled walls, 91.4% of scanned surface sits within ±25 mm of
> the model. These 14 elements are outside tolerance, worst case 87 mm. 6.2%
> of modelled surface was not observed and is excluded from these figures."

Two things make that unwritable from independent per-station reports.

**A wall seen by five stations gets five deviation values**, and nothing
decides which is authoritative. Element-level statistics become ambiguous
exactly where coverage is best.

**Per-station coverage analysis is meaningless.** Every station fails to see
most of the site, so mode D run per station reports almost all model surface
as unobserved. The grey state — the one that stops an unscanned ceiling being
reported as a modelling error (`ADR-002` Decision 2) — cannot be computed
correctly at station scope. Grey means *no station saw it*, which is by
definition a site-level question.

The 30 sample stations sit inside roughly a 55 × 25 m footprint and total
~227.7 M points (`DATA-INVENTORY.md` §1). Overlap is not incidental here; it
is the dominant characteristic of the dataset.

## Decision

**Meshing stays per station, unchanged. Comparison and reporting move to site
level.**

| Layer | Scope | Rationale |
|---|---|---|
| Ingestion, filtering, triangulation, QA, decimation, LOD | **Per station** | Unchanged. Native lattice, bounded memory, independent failure |
| Model import and BVH | Site | One model, one tree |
| Comparison modes A, B, D | **Site** — all stations as one evidence set | Required for correct grey and unambiguous element statistics |
| Heat map and report | Site, with per-station attribution retained | The billable deliverable |

For each model surface sample, the supporting observation is selected across
**all** stations by a documented, testable rule. The report records which
station supplied each measurement, so auditability survives.

The selection rule is deferred to phase 3 measurement. Candidates: nearest
observation, best incidence angle, shortest range, or a combination. Whichever
is chosen is stated in the report, and the alternatives must be measurable
against each other on the sample site.

## Rejected: fuse all stations into one surface

Considered and rejected for now, on one decisive ground.

**Fusion makes registration error indistinguishable from model error.** If
merging stations smears a 15 mm station-to-station residual into the surface,
RapidMesh reports its own registration as the client's set-out error, against a
25 mm tolerance. That is the precise failure the "no silent alignment" rule
(`00-PRODUCT-DEFINITION.md`) exists to prevent, arriving by a different door.

Supporting reasons:

- Fusion requires TSDF, Poisson or similar, which **discards the native
  lattice** — trading away the project's entire competitive premise to solve a
  display problem.
- Scanner-ray comparison (mode B) is nearly free on structured data because
  the ray and range are the stored fields. Fusion destroys that unless
  per-point provenance is carried through, which is significant extra work.
- Whole-site memory operation, against a pipeline whose current problem is
  already memory (`FINDING-002` secondary finding).

**Revisit condition:** measured station-to-station registration residuals for
the sample site. These exist in the field registration software and are not
present in the E57s. If residuals are small against the working tolerance,
fusion becomes defensible as a **display-only** output that never produces
numbers. It would need its own ADR.

## Consequences

**Positive**

- The commercial sentence becomes writable.
- Grey coverage is correct by construction.
- Element-level statistics are unambiguous: one number per wall.
- The native lattice, mode B and bounded per-station memory all survive.
- No fusion code, no seam blending, no normal reconciliation.

**Negative**

- No single seamless walkable surface. The browser experience is station-based:
  one station at a time, or a best-station-per-region selection so overlapping
  shells are never drawn simultaneously. This is how surveyors already work,
  but it is not a walkthrough.
- Total package size scales with station count rather than site area, because
  overlapping geometry is shipped rather than deduplicated. On a 30-station
  site inside 55 × 25 m that is a real cost, and it works against the
  package-size target. Must be measured in phase 4 and may force a display-only
  fused LOD later.
- The observation-selection rule becomes a documented product behaviour with
  its own test, not an implementation detail.

**Neutral**

- Per-station QA reports remain, unchanged and still useful. The site report is
  additional, not a replacement.
