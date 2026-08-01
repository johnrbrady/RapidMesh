# RAPIDMESH: PRODUCT DEFINITION

> **READ THIS FILE BEFORE ANY OTHER FILE IN THIS REPOSITORY.**
>
> Highest authority document. Where any other file disagrees with this one,
> this one wins and the other file is wrong.
>
> Owner: John Brady. Established 31 July 2026.

---

## 1. The product, in one sentence

**RapidMesh converts a single terrestrial laser scan station into a
survey-grade, streamable, textured mesh at the scanner's own sampling
resolution, and reports how far that mesh deviates from the source points.**

It is a library and a command-line tool. It is not a viewer, not a server, and
not a Cairn feature. It becomes a Cairn feature later, once it is proven.

---

## 2. Relationship to Cairn

Separate repository. Separate package. **Zero imports from Cairn, in either
direction, until integration is explicitly approved.**

Cairn's roadmap continues unaffected. Cairn keeps shipping `backend/mesher.py`
until RapidMesh beats it on measured numbers, not on opinion.

When RapidMesh wins, integration is three call-site swaps —
`conversion.run_conversion`, `routers/models.py build_meshes`, and an RMX
loader alongside the existing CMH1 loader in `frontend/src/scans.ts`. Old
projects keep their `.cmh` files and keep working. See
`REVIEW-CAIRN-MESHING.md` §7.

---

## 3. The one design decision everything else follows from

**Mesh at the scanner's native lattice, then decimate to a measured tolerance.
Never decimate first.**

Cairn's mesher bins to a fixed 2048×1024 grid before it makes a single
triangle, which is roughly a 1/100 area decimation on a Trimble X7 at high
resolution, applied blindly and before any geometry is understood. Every
quality complaint downstream traces back to that line.

RapidMesh inverts the order:

1. Read the scanner's real sample lattice out of the structured E57.
2. Triangulate every sample.
3. Remove what is provably not surface.
4. *Then* simplify, driven by a geometric error budget, so flat surfaces get
   cheap and detailed surfaces stay expensive.
5. Measure and report what the simplification cost.

Step 5 is not optional. It is the product's differentiator.

---

## 4. Quality bar, stated as numbers

These are commitments, not aspirations. Each needs a test that fails if it
regresses.

| Property | Target | How it is measured |
| --- | --- | --- |
| Geometric deviation, RMS | ≤ 2 mm | Point-to-mesh distance, every source point, `qa.deviation_report` |
| Geometric deviation, 99.9th percentile | ≤ 8 mm | Same |
| Geometric deviation, max | Reported, never hidden | Same |
| Real detail preserved | Door frames, handrails, window reveals survive filtering | Synthetic fixtures with known edges |
| Mover removal | ≥ 95 % of transient points dropped | Synthetic scan with a scripted moving object |
| False positives from mover removal | ≤ 0.1 % of static surface points dropped | Same fixture |
| Time to first paint, browser | ≤ 1 s on a 10 Mbit link | LOD0 tier size budget |
| Size per station | Beat Cairn's 13–20 MB at equal or better deviation | Direct comparison on the same scans |

**"Survey-grade" means the deviation report ships with the mesh.** If we cannot
state the number, we do not use the word.

---

## 5. Scope

### In scope

- Structured E57 as the primary input. This is the format that carries the
  scanner lattice.
- Terrestrial static scanners: Trimble X7 / X9, Faro Focus.
- Per-station meshing. One station in, one mesh out.
- Cross-station data used **only** for occlusion carving (removing transients).
  Never merged into the output geometry.
- Texture baked from the scan's own RGB or its panorama.
- LOD chain produced by decimation, not by re-binning.
- A deviation report per station.

### Explicitly out of scope

- **Merging stations into one continuous walkable mesh.** Per-station isolation
  is a deliberate design choice inherited from the brief. It avoids
  registration and seam-blending problems entirely. The trade-off — no
  cross-station continuity — is accepted.
- Mobile/SLAM scanners (NavVis, handheld). Different sampling model. Later, if
  at all.
- Rendering. RapidMesh produces files; something else displays them.
- Point cloud storage or conversion. That is Potree's job inside Cairn.

---

## 6. What "better than TurboMesh" means concretely

TurboMesh's published claim is per-station multi-resolution meshing for fast
browser streaming. That is table stakes and Cairn already does a version of it.

RapidMesh beats it on two axes that nobody in this market currently claims:

1. **Native-lattice input.** Every browser-streaming mesh product decimates
   early because it is cheaper. Decimating *after* triangulation, against a
   measured error budget, is strictly more accurate at the same output size.
2. **A published deviation figure per scan.** No competitor ships this. For a
   product sold to surveying firms, a number beats an adjective.

---

## 7. Non-negotiables

- **Millimetre precision at MGA coordinates.** Positions are stored as f32
  offsets from an f64 origin. A raw f32 world coordinate at 6-digit northings
  cannot hold a millimetre and must never appear in the format.
- **No baked lighting in colour data.** Normals go in the file; lighting is the
  renderer's job. Cairn multiplies a lambert term into the RGB and discards the
  normals — that contaminates survey colour data and cannot be undone.
- **A scan that fails to mesh is not a failed scan.** Meshing is an overlay.
  Every failure is caught, reported and skipped, never fatal.
- **No client data in the repository.** Test fixtures are synthetic or
  explicitly cleared.
