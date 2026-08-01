# ADR-003 — NavVis reconstruction approach

**Status:** Proposed. Decision deferred to phase 5 measurement.
**Date:** 1 Aug 2026

---

## Context

The reference NavVis export (`../DATA-INVENTORY.md` §2) is a single
unstructured cloud of 56,950,017 points carrying **precomputed per-point
normals** via the libE57 `NOR` extension, plus RGB and intensity.

It carries **no** trajectory, timestamps, sensor poses, sensor-head
identifiers, acquisition sessions or panoramic imagery.

## What this rules out immediately

The original brief specified a trajectory-aware pipeline: trajectory-distance
segmentation, time-window partitioning, scanner-pose continuity, view-direction
consistency filtering, and reconstruction driven by observation direction.

**None of it is possible on this data.** Any approach requiring an observation
origin, a view direction or a timestamp is eliminated before evaluation
begins:

- trajectory-aware range projection — eliminated
- local range images per sweep — eliminated
- TSDF fusion driven by sensor rays — eliminated
- timestamp-based transient filtering — eliminated

## What the data does favour

Precomputed normals are a significant advantage. Normal estimation is the
slowest and least reliable stage of point-based reconstruction, and it is
already done by the vendor.

Surviving candidates, all of which consume oriented points:

| Candidate | Strength | Risk |
|---|---|---|
| **Screened Poisson** | Watertight, robust to noise, consumes oriented points natively, well understood | Bridges openings; smooths sharp edges; tends to close doorways and windows |
| **Surfel / splat** | Cheap, no topology to get wrong, never invents surface | Not a mesh; poor for model comparison and for LOD chains |
| **Ball pivoting** | Never invents surface; preserves openings | Fragile with varying density; leaves holes; slow |
| **Region-growing planar + freeform residual** | Excellent on the flat surfaces that dominate a service station forecourt; sharp edges preserved | Complex; poor on curved and organic geometry |

## Proposed decision

Benchmark **screened Poisson** and **region-growing planar plus freeform
residual** against each other on the reference dataset. Do not choose before
measuring.

Bridging is the decisive criterion. A watertight Poisson surface that closes a
doorway is worse than useless for this product: it manufactures geometry that
the comparison engine then reports as agreeing with, or disagreeing with, a
model. Poisson must be evaluated with an aggressive density-based trim and
that trim must be measured, not tuned by eye.

## Evaluation criteria, in priority order

1. **Mesh-to-source deviation** — detects invented surface. Decisive.
2. **Source-to-mesh deviation** — fidelity.
3. **Opening preservation** — doors, windows, gaps. Measured on synthetic
   fixtures with analytic ground truth.
4. **Thin-object survival** — pipes, handrails, bollards, canopy members.
5. **Storey separation** — floors and ceilings must not merge.
6. **Tileability and seam behaviour** at octree boundaries.
7. Peak memory per tile against the 512 MB budget.
8. Processing time.
9. Topological validity: winding, manifoldness, inverted triangles.
10. Determinism.

Visual smoothness is **not** a criterion. A smooth but geometrically wrong
surface is a failure.

## Not claimed

- **Mover removal.** Reliable transient detection needs timestamps or repeated
  observation with view direction. Neither exists here. Phase 5 makes no
  mover-removal claim, and the inspection report says so explicitly.
- **Absolute survey accuracy.** RapidMesh measures fidelity to the supplied
  registered cloud. SLAM drift in the source is neither detected nor corrected,
  and every NavVis report must state this.

## Prerequisite

Supplied normals must be validated for orientation consistency before they are
trusted. Where they fail, re-estimate and **report the count**. Silently
re-estimating vendor normals would hide a source-data quality signal worth
knowing about.

## Scope note — this ADR covers path B1 only

A raw NavVis `rec-v4` recording is now available (`../DATA-INVENTORY.md` §2A)
and does contain trajectory, per-sweep timestamps and full sensor calibration.
That is **path B2**, scheduled for phase 6, and it reopens the
trajectory-aware candidates eliminated above.

This ADR governs **path B1**, reconstruction from a registered NavVis E57
export with points and normals and nothing else. B1 remains the phase 5
deliverable, and its constraints are unchanged: the registered export still
has no rays, no timestamps and no poses regardless of what the raw recording
for a different project contains.

When B2 is scheduled, write a separate ADR for its reconstruction approach.
Do not amend this one. The two paths consume different data and will likely
reach different answers, which is the reason they are separate pipelines.
