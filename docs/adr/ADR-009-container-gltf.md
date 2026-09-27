# ADR-009 — The streamable container is glTF 2.0

**Status:** Accepted  
**Date:** 27 September 2026  
**Decision owner:** John Brady (ruled by the Lead under his authorisation of
27 September 2026; DEC-005, recorded as DEC-026)

## Context

DEC-005 required the container to be decided by benchmark: standard payloads
against a bespoke binary, on real reference data, judged on size, browser decode
time and implementation effort, with bespoke chosen only if the standard route
failed a stated gate. Round 15 (WP-4.1) ran that benchmark on two stations, one
from each reference class, with Gate C (the DEC-004 structural requirements)
stated in advance and each of its clauses proven red against a writer broken on
that clause alone.

What it measured, on identical lossless geometry:

- **Gate C** — both candidates pass all three clauses on both stations.
- **Size** — container against container the two differ by 0.02%. The bespoke
  format's only lead comes from a byte filter plus DEFLATE, and against glTF
  served gzipped that lead is 1.30× on the hardest station and 1.17× on the
  control, against a decision threshold of 2×.
- **Decode** — in a real browser the compressed bespoke tier decodes 10.95×
  slower than glTF on the hardest station and 8.62× slower on the control. Core
  glTF accessors need no decode step.
- **Effort** — a bespoke format means two implementations of a private
  specification kept in step, roughly three times the JavaScript, and no
  validator; one alignment defect found while building it is one glTF's own
  rules prevent.

## Decision

1. The client container is **glTF 2.0 binary (GLB)**. Round 15 benchmarked one
   GLB per tier with each tile a separate mesh and byte range. **The delivery
   unit is not settled by this ADR:** an HTTP range request against a
   gzip-encoded response addresses the compressed bytes, so "one file per tier,
   tiles by range" and decision 3 do not compose. The writer already produces a
   self-attributing one-tile GLB, and Round 16 measures that shape, reporting
   per-tile gzip size beside Round 15's per-tier figure.
2. Tier identity travels in the `RM_tier` extension, listed in
   **`extensionsRequired`**. A stock loader therefore refuses a RapidMesh tile;
   that is intended, because an optional marking would let a stock loader open a
   QC tier while ignoring what it is. The progressive loader is RapidMesh's own.
3. Tiles are **served gzip-compressed**, which every browser inflates natively
   with no hosted decoder.
4. No client tier carries per-vertex source identity. The writer refuses one
   that does (`container.check_tier`); DEC-004 is enforced where the bytes are
   written, not where they are read.
5. `EXT_meshopt_compression` is **deferred** until size becomes the binding
   constraint. Adopting it means vendoring a decoder and proving a reference
   round trip, as a bounded package of its own.
6. The bespoke candidate is not a product container. `container_binary.py`
   stays in the tree only as the benchmark's comparison arm until a later
   package removes it.

## Consequences

- The retired name "RMX" is superseded by this ADR. The remaining references in
  the documentation are owed a single docs slice rather than piecemeal edits.
- The size figures behind this decision were measured on tiles carrying
  positions and indices only. They are re-measured in the round that adds
  per-vertex normals; the decode conclusion, which the decision turns on, is not
  exposed to that.
- The evidence rests on two stations, one browser engine and one machine. That
  could move the size margin; it cannot plausibly invert an 8.6–11× decode ratio.
- No first-paint, frame-rate, GPU-memory or fidelity figure follows from this
  decision. A decode time is not a first-paint time.
