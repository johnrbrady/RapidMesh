# RapidMesh / Cairn spatial contract

**Status:** Authoritative technical contract  
**Owner:** John Brady  
**Established:** 15 August 2026

This document defines how RapidMesh geometry occupies the same space as the
source point data Cairn displays. `00-PRODUCT-DEFINITION.md` decides what the
product is; this document is the authority for coordinate frames, transforms,
precision and alignment behaviour.

The acceptance statement is:

> Switching Points → RapidMesh → Points, or overlaying both, must not reveal a
> translation, rotation, reflection, scale change, coordinate drift, tile seam
> or LOD-dependent systematic movement.

RapidMesh is developed outside Cairn. This contract does not authorise changes
to Cairn or make RapidMesh a Cairn V1 dependency.

---

## 1. Scope and current capability

Required source formats are the point-cloud formats Cairn accepts: **E57, LAS
and LAZ**.

| Capability | Required product state | Current implementation |
|---|---|---|
| Structured E57 | Preserve native lattice, pose and source observations | Implemented; Phase 0c spatial regressions pass |
| LAS / LAZ | Preserve header quantisation and reconstruct without claiming a native lattice | Not implemented |
| Multiresolution mesh | Error-bounded LODs in one project frame | Not implemented |
| RapidMesh output container | Versioned, tiled, progressively streamable | Not implemented |
| Cairn alignment view | Points, RapidMesh, overlay, wireframe and deviation modes | Not implemented; no Cairn changes are currently authorised |

An unavailable row is reported as unsupported. It is never claimed on the
strength of a planned phase.

---

## 2. Coordinate frames

RapidMesh uses the following named frames. Code, reports and format fields must
use these names rather than an ambiguous `world` or `local`.

### 2.1 Source coordinate frame

Coordinates and metadata exactly as supplied by the source file. RapidMesh
does not infer a CRS, datum, vertical datum, geoid model or unit.

### 2.2 Scanner-local frame

For a structured E57 station, the scanner is the origin and source observations
are scanner-relative. Let `s` be a row-vector position in metres.

### 2.3 Project frame

The coordinate frame Cairn uses for the project, as supplied. RapidMesh does
not reproject or adjust it. For an E57 station:

```text
p = s @ R.T + t
```

where:

- `p` is a float64 project-frame position;
- `R` is the float64 3×3 scanner-local-to-project rotation;
- `t` is the float64 project-frame scanner origin;
- `R` is finite, orthonormal and right-handed with determinant +1.

E57 quaternions are interpreted as `[w, x, y, z]`. Quaternion-to-matrix tests
must independently cover 90° and 180° rotations about every axis. A reflection
or non-rigid matrix is invalid input, not a rotation to be repaired silently.

### 2.4 Mesh-origin frame

Stored vertex positions are float32 **project-axis offsets** from a float64
origin `o`:

```text
v = float32(p - o)
p_reconstructed = float64(v) + o
```

For the current per-station E57 mesh, `o = t`, therefore:

```text
v = float32(s @ R.T)
```

The scanner rotation is applied exactly once during mesh assembly. The source
pose remains output provenance, but a renderer does not apply it a second time.

Future tiles may introduce a float64 tile origin. Their stored positions remain
project-axis offsets and reconstruct by addition; a tile must not introduce an
independent rotation or scale.

### 2.5 Render frame

Cairn may rebase the entire scene for GPU precision, but it must apply the same
scene transform to points, RapidMesh geometry, models, section boxes, picks and
measurements. RapidMesh does not supply a second alignment transform for the
renderer.

---

## 3. Source-format rules

### Structured E57

1. Read source coordinates and pose as float64.
2. Validate the pose before narrowing any coordinate.
3. Keep scanner-local observations for lattice reconstruction and provenance.
4. Apply the full rigid pose when producing project-axis mesh offsets.
5. Rotate normals by `R`; never translate them.
6. If a non-conforming exporter stores project coordinates alongside a pose,
   conversion back to scanner-local is `(p - t) @ R`, not translation-only.
7. Any heuristic that detects that exporter behaviour must fail closed when it
   cannot discriminate decisively and must record which path was taken.

### LAS and LAZ

1. Reconstruct each coordinate from its integer, per-axis header scale and
   offset in float64.
2. Preserve and report every source scale independently. Do not normalise it.
3. Treat supplied coordinates as source/project-frame coordinates; LAS/LAZ do
   not provide an E57 native scanner lattice or reliable scanner pose.
4. Do not infer units or CRS. Missing units require an explicit operator value
   before processing; unknown CRS remains unknown and is carried through.
5. Choose and record a deterministic float64 mesh/tile origin before narrowing
   to float32. The origin-selection algorithm must be versioned and tested on
   positive, negative and far-from-zero coordinates.
6. The LAS/LAZ reconstruction algorithm is a separate path. It must not claim
   native-lattice fidelity and must be benchmarked against original evidence
   where that evidence is available.

---

## 4. Units, CRS and precision

- Internal geometry is metres.
- Unit conversion happens once, in float64, at ingestion and is recorded.
- RapidMesh performs no survey-coordinate reprojection.
- CRS declarations and their provenance pass through unchanged.
- Source quantisation, conversion quantisation, float32 rebasing error and LOD
  error are separate reported terms.
- Casting an absolute project coordinate to float32 is forbidden.
- The localising subtraction or rotation occurs in float64 before a float32
  cast.
- For every mesh or tile, compute the worst float32 storage quantisation from
  its maximum absolute offset. The storage contribution must not exceed
  **0.1 mm per axis**; otherwise choose a nearer origin or smaller tile.
- All transformations are deterministic. Reprocessing identical input with the
  same version and settings produces equivalent coordinates and topology.

---

## 5. Geometry and LOD invariants

The original observations are authoritative. Reconstruction, filtering and
simplification never adjust registration or rescale the project.

Every LOD:

- uses the same project axes, units and handedness;
- reconstructs through the same float64 origin chain;
- records its geometric error against the highest retained evidence;
- has no systematic translation, rotation or scale term;
- preserves shared tile boundaries without cracks or T-junctions;
- meets an explicit local maximum as well as percentile error budget;
- is checked at edges, openings, thin objects and tile seams rather than only
  by a site-wide average.

Cross-fading may hide a pop in triangle density. It must never hide a geometric
misregistration that exceeds the LOD's recorded error budget.

The meaning of combined-project display remains open for investigation. No
fused site surface may become numerical evidence or conceal per-station
registration disagreement.

---

## 6. Required provenance

Every mesh output and QA record carries, at minimum:

- source identity by safe internal identifier and cryptographic hash;
- source format and format-specific quantisation;
- declared unit and its provenance;
- declared CRS and its provenance, including unknown;
- source scanner pose where present;
- every applied matrix, translation, origin and scale;
- RapidMesh and output-format versions;
- filter, reconstruction, tiling and LOD settings;
- achieved precision and geometric-error figures;
- explicit warnings for missing or ambiguous metadata.

Client-facing artefacts must not expose protected source identifiers or project
content. Scan-to-model comparison is surveyor/admin QC only.

---

## 7. Alignment verification

### Automated transform tests

The suite must detect:

- translation on every axis;
- 90° and 180° rotations on every axis;
- quaternion component-order mistakes;
- row/column-major matrix mistakes;
- reflection and handedness errors;
- unit and scale errors;
- positive, negative, local and far-from-zero coordinates;
- float32 narrowing before rebasing;
- tile-origin and LOD movement;
- normal rotation errors;
- forward/inverse round-trip failure.

At least one expected answer for each transform family is calculated
independently of the production transform helper.

### Cairn integration view matrix

Before integration is accepted, Points, RapidMesh and their overlay are checked
in perspective, orthographic, top, front, side and oblique views; close-up and
overview distances; section-box boundaries; tile boundaries; every LOD
transition; individual-scan and combined-project modes; and overlapping scans.

### Engineering alignment view

The development view provides:

- point cloud only;
- RapidMesh only;
- rapid switching;
- adjustable transparency overlay, including 50/50;
- optional wireframe;
- optional point-to-mesh/deviation colouring;
- active LOD, origins and transform provenance.

This is an engineering and surveyor/admin QC capability. It is not a client
control unless John makes a later explicit product decision.

---

## 8. Prohibited shortcuts

- No silent ICP, fitting, registration, datum choice or unit inference.
- No appearance-driven vertex movement without a measured displacement budget.
- No mesh fallback for a Cairn measurement; Cairn measurements remain against
  point observations under Cairn's product authority.
- No client-side comparison report or QC control.
- No Cairn code change until integration is explicitly approved.
