# ADR-008 — Cairn-aligned multiresolution surface first

**Status:** Accepted  
**Date:** 15 August 2026  
**Decision owner:** John Brady

## Context

The previous roadmap put an early scan-to-model comparison slice ahead of
decimation, LODs, the output container and browser alignment. A new engineering
brief made the primary product test explicit: Points → RapidMesh → Points must
remain spatially fixed, and overlay must expose any disagreement.

The plan review also found that the current in-memory mesh dropped non-identity
scanner rotation. Continuing the old sequence could therefore build comparison
and format work on an incomplete spatial contract.

## Decision

1. Resume RapidMesh as a separate project while Cairn remains read-only.
2. Prioritise the Cairn-aligned multiresolution surface, output contract and
   engineering alignment view.
3. Required point-cloud inputs are E57, LAS and LAZ. Their ingestion and
   reconstruction paths may differ where their evidence differs.
4. Move scan-to-model comparison later. It is surveyor/admin QC only and is
   absent from all client-facing interfaces and outputs.
5. Investigate combined-project display before choosing coordinated
   per-station geometry, a fused display-only surface, or both.
6. Integration into Cairn requires separate explicit approval and cannot make
   Cairn V1 depend on RapidMesh.

## Consequences

- `SPATIAL-CONTRACT.md` becomes the coordinate and transform authority.
- Transform correctness and alignment tests precede further geometry work.
- QA and bounded streaming remain the next core engine gate.
- Error-bounded decimation, LODs, RMX and the engineering alignment view move
  ahead of comparison.
- No claim of LAS/LAZ or browser support is made until its gate passes.

