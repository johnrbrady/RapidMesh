# ADR-007 — The comparison engine consumes observations, never a display mesh

**Status:** Accepted
**Date:** 2 August 2026

---

## Context

`00-PRODUCT-DEFINITION.md` §7 states: *the scan is evidence; the model is what
is being checked.* Numerical results trace to original scan observations, never
to a decimated display mesh.

Today that rule is **unenforceable and untestable**, because the two are
numerically identical. There is no decimation, so every retained observation
becomes a mesh vertex. Comparing mesh-to-IFC and observations-to-IFC returns
the same numbers. This is the same degeneracy `FINDING-002-QA-DEFINITION.md`
identified in the deviation metric.

A comparison engine built against the mesh would therefore look correct, pass
every test, and become silently wrong the day decimation lands. That is the
worst possible failure shape: correct-looking, well-tested, and wrong later.

## Decision

**The comparison engine's input type is an observation set. It never takes a
mesh as its numerical source.**

```
ObservationSet
    positions      f64, localised to a site origin
    ray_origin     optional, per observation or per station
    ray_direction  optional
    station_id     provenance, per observation
    sample_id      stable identity for the filtering ledger
    flags          retained / excluded, with reason
```

| Rule | |
|---|---|
| Modes A, B and D all consume `ObservationSet` | Not a mesh |
| Deviation is computed **per observation** | Not per vertex |
| The mesh re-enters only downstream, for rendering | Deviation values are mapped onto mesh vertices or textures purely for display |
| Decimation changes the display, never the numbers | See the invariance test below |
| Mode B's ray is the stored spherical measurement | azimuth, elevation and range from the E57, not reconstructed from a mesh normal |

This makes "the scan is evidence" an **enforced interface** rather than a
documentation rule.

## The invariance test — this is the point of the ADR

A design rule that nothing enforces will eventually be violated by someone in a
hurry. So it is a test, not a paragraph:

```
1. Run the comparison on a station. Record the report.
2. Decimate the mesh to LOD3.
3. Re-run the comparison.
4. Assert the numerical report is identical.
```

Identical **excluding runtime metadata and generated identifiers** — timestamps,
durations, peak memory, run UUIDs. Every deviation value, every percentage,
every element attribution, every count must match exactly.

This test fails loudly the moment anyone couples the engine to the mesh
representation. It is a phase 3a deliverable and must be in place before the
comparison engine grows.

## Consequences

**Positive**

- Mode B stays possible. The stored spherical triple *is* the ray; a fused or
  decimated surface destroys that, and an observation-set interface preserves
  it structurally.
- Decimation can be tuned aggressively for size without any risk to reported
  accuracy, because the two are not connected.
- The filtering ledger and the comparison engine share the same sample identity,
  so exclusions reconcile across both.
- Phase 3a can be built now against the full-resolution mesh's underlying
  observations, and remains valid after phase 3b changes the mesh entirely.

**Negative**

- Two representations to keep in step: observations for numbers, mesh for
  display. The join is the deviation-to-vertex mapping, which needs its own
  correctness test at the point where a decimated vertex no longer corresponds
  to a single observation.
- Observation sets are larger than meshes at equal coverage, so site-level
  comparison memory is bounded by observations rather than by decimated
  geometry. Must be measured at phase 5.

**Neutral**

- Nothing about per-station meshing changes (`ADR-005`), and nothing about
  streamed assembly changes (`ADR-006`).

## Related

`ADR-005` puts comparison at site level. `ADR-006` makes the pipeline streamed
and tiled. This ADR ensures that neither of those, nor decimation, can alter a
reported number.
