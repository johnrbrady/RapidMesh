# RAPIDMESH: PRODUCT DEFINITION

> **READ THIS FILE BEFORE ANY OTHER FILE IN THIS REPOSITORY.**
>
> Highest authority document. Where any other file disagrees with this one,
> this one wins and the other file is wrong.
>
> Owner: John Brady. Established 31 July 2026. Revised 2 August 2026.

**Revision note (2 Aug 2026).** Three approved changes and one correction:
NavVis mobile scan data is now in scope (§5); model-to-scan comparison is now
the primary commercial deliverable (§3A); comparison and reporting move to
site level while meshing stays per station (§5, `docs/adr/ADR-005`); and the
quality claims in §4 are narrowed to what has actually been measured
(`FINDING-002-QA-DEFINITION.md`).

---

## 1. The product, in one sentence

**RapidMesh converts terrestrial laser scan data into a survey-fidelity,
streamable, textured mesh at the scanner's own sampling resolution, and
measures an imported BIM model against the original scan evidence, reporting
both how far the mesh deviates from the source points and how far the model
deviates from what was measured.**

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

**One exception, and it is not integration.** `CAIRN-MESH-MEMORY-ISSUE.md`
documents a live production fault: `backend/mesher.py` reads whole clouds
in-process and OOM-killed the uvicorn worker on a real 1.77 GB client file,
taking the whole application down. Fixing that is remediation of a shipping
defect, not a step toward integration, and it is the only permitted change to
Cairn before integration is approved.

---

## 3. The one design decision everything else follows from

**Mesh at the scanner's native lattice, then decimate to a measured tolerance.
Never decimate first.**

Cairn's mesher bins to a fixed 2048×1024 grid before it makes a single
triangle, applied blindly and before any geometry is understood. Every quality
complaint downstream traces back to that line.

RapidMesh inverts the order:

1. Read and classify every native-lattice sample from the structured E57.
2. Triangulate the retained samples.
3. Remove invalid fragments.
4. *Then* simplify, driven by a geometric error budget, so flat surfaces get
   cheap and detailed surfaces stay expensive.
5. Measure and report what the simplification cost.

Step 5 is not optional. It is the product's differentiator.

**Measured, on the real sample data** (`docs/DATA-INVENTORY.md` §1.7):

| | Cells | Horizontal step | Spacing @ 10 m |
|---|---|---|---|
| Cairn fixed grid 2048 × 1024 | 2.10 M | 0.1758° | 30.7 mm |
| Native, medium stations (18 of 30) | 2.95 M | 0.1311° | 22.9 mm |
| Native, high-res stations (12 of 30) | 14.55 M | 0.0591° | 10.3 mm |

**6.94× the cells on high-resolution stations, 1.41× on medium.** State it
that way. A blanket "6.9× more detail" is not supportable on this dataset. See
`docs/adr/ADR-001`.

---

## 3A. The commercial deliverable

The mesh is what makes the product usable. **The comparison report is what
makes it billable.** A surveyor must be able to hand a client:

> "Across 165 modelled walls, 91.4% of scanned surface sits within ±25 mm of
> the model. These 14 elements are outside tolerance, worst case 87 mm. 6.2%
> of modelled surface was not observed and is excluded from these figures."

Everything in §4 exists to make that sentence true and defensible.

### The heat map

One tolerance, set by the surveyor.

| | Value |
|---|---|
| Default | **25 mm** |
| Minimum settable | **10 mm** — the control clamps, and will not accept less |
| Maximum | Unbounded |

| State | Colour | Condition |
|---|---|---|
| Within tolerance | **Green** | \|deviation\| ≤ tolerance |
| Outside tolerance | **Red** | \|deviation\| > tolerance |
| No corresponding geometry | **Red** | Where the surface was observable |
| **No scan coverage** | **Grey** | Occluded, or outside scan range or field of view |

Grey is a first-class state in both renderer and report. An unscanned ceiling
void is not a modelling error. A toggle renders grey as red for surveyors who
want strict two-colour output; the JSON always separates them.

Colour target is user-switchable: colour the model (mode D) or colour the scan
mesh (mode A). Both ship together.

Every view carries a legend: tolerance, units, comparison mode, colour target,
source dataset, model identifier, and percentage in each state. Colour is never
the only channel.

Full reasoning in `docs/adr/ADR-002`.

### Three accuracies, never conflated

| Quantity | Meaning | RapidMesh measures it? |
|---|---|---|
| Mesh fidelity | How well the surface represents the supplied points | **Yes** |
| Survey accuracy | How well the points represent the real world | **No** |
| Model agreement | How well the BIM model matches the supplied points | **Yes** |

Reports state which of these they are reporting. See §6.

---

## 4. Quality bar, stated as numbers

These are commitments. Each needs a test that fails if it regresses.

### 4.1 Mesh fidelity, derived from tolerance

RapidMesh's own error must never be able to flip a heat map cell from green to
red on its own. Budget: **10% of the surveyor's tolerance.**

| Tolerance | RMS budget | p99.9 budget |
|---|---|---|
| **10 mm — the floor, and therefore the hardest case** | **1.0 mm** | **3.2 mm** |
| 25 mm — default | 2.5 mm | 8.0 mm |
| 50 mm | 5.0 mm | 16.0 mm |

Because the tolerance control clamps at 10 mm, **1.0 mm RMS and 3.2 mm p99.9
are the tightest figures RapidMesh will ever be asked to meet.** That is the
mesher's fixed acceptance test, not an open-ended accuracy chase.

Maximum deviation is always reported and never capped.

### 4.2 Everything else

| Property | Target | How it is measured |
|---|---|---|
| Real detail preserved | Door frames, handrails, window reveals survive filtering | Synthetic fixtures with known edges |
| Mover removal | ≥ 95% of transient points dropped | Synthetic scan with a scripted moving object |
| False positives from mover removal | ≤ 0.1% of static surface points dropped | Same fixture |
| Peak RSS per station | ≤ 512 MB | Measured on the largest real sample station |
| Time to first paint, browser | ≤ 1 s on a 10 Mbit link | LOD0 tier size budget, test contract per `RAPIDMESH-REVIEW-FINDINGS.md` §11 |
| Size per site | Beat Cairn at equal or better deviation | Same scans, same components counted |

### 4.3 State of the evidence — read this before quoting any number above

**Measured against synthetic analytic ground truth:** 1.00 mm RMS / 3.55 mm
p99.9 at 0.090° sampling; mover recall 97.0%; false positives 0.015%. These
are legitimate. `qa.truth_report` compares against noise-free geometry the
mesh never saw.

**Measured on real client data:** nothing yet, and not through neglect.
`FINDING-002-QA-DEFINITION.md` shows why: with no decimation, every retained
sample becomes a mesh vertex, so point-to-mesh distance is trivially zero for
99.5% of points. The metric currently answers "are these points vertices of a
mesh built from these points". Real data has no analytic truth to substitute.

**Measured memory:** 1,418 MB peak reading one 14.5 M-point station; 1,747 MB
for the full pipeline on a 2.95 M-point station. The 512 MB target is **not
currently met**.

Until §5's QA rework lands, **no real-data fidelity claim may be made.** The
targets stand; the evidence for them is narrower than this document previously
implied.

---

## 5. Scope

### 5.1 Scan families

| | Pipeline A — **TLS, primary** | Pipeline B — NavVis |
|---|---|---|
| Input | Structured terrestrial E57 | B1 registered E57 · B2 raw `rec-v4` |
| Method | Native lattice | B1 points + supplied normals · B2 accumulate from raw sweeps |

**TLS is the primary family.** It is built first, carries the competitive
claim, and accounts for 30 of the 31 sample scan files.

NavVis was previously out of scope. That is **superseded by approved product
change, 2 Aug 2026.** NavVis gets its own ingestion and reconstruction path and
shares no assumption that is true only of static structured scans. See
`docs/adr/ADR-003` and `docs/adr/ADR-004`.

### 5.2 In scope

- Structured E57 as the primary input. It carries the scanner lattice.
- Terrestrial static scanners. **Note:** the 30 sample stations declare no
  `sensorVendor`, so make and model are not recoverable from them. Do not
  assert Trimble or Faro on the basis of this dataset.
- **Per-station meshing.** One station in, one mesh out.
- Cross-station data used for occlusion carving (removing transients). Never
  merged into the output geometry.
- **Site-level comparison and reporting** across all stations as one evidence
  set. New, per `docs/adr/ADR-005`.
- IFC2X3 / IFC4 model import, comparison modes A, B and D, tolerance heat map.
- Texture baked from the scan's own RGB or its panorama. **Optional:** 4 of 30
  sample stations carry no RGB, and `images2D` is empty on all 31 sample E57s.
- LOD chain produced by decimation, not by re-binning.
- A deviation report per station **and** a comparison report per site.

### 5.3 Explicitly out of scope

- **Merging stations into one continuous walkable mesh.** Per-station isolation
  is deliberate: it avoids registration and seam-blending entirely, and fusing
  would make registration error indistinguishable from model error. The
  trade-off — no seamless site surface — is accepted. Comparison and reporting
  are site-level regardless; only the geometry stays per station. Revisit
  condition and full reasoning in `docs/adr/ADR-005`.
- Rendering. RapidMesh produces files; something else displays them.
- Point cloud storage or conversion. That is Potree's job inside Cairn.
- Registration or re-registration. RapidMesh consumes registered data and never
  modifies it.
- Automatic model alignment (ICP) as a default. See §7.

### 5.4 Deferred, with named unblocking conditions

No deferred capability may be claimed anywhere in the code, CLI or
documentation. If the source data is absent, report it unsupported and name
what is missing.

| Deferred | Status | Unblocked by |
|---|---|---|
| NavVis trajectory, mode C, transient filtering, multi-sensor-head | Scheduled | Data present in the raw recording |
| Georeferencing of raw NavVis output | Blocked | Surveyed coordinates for anchors TDS4–TDS10; `anchor_poses.txt` is empty |
| Panoramas, texture from imagery | Blocked | Stitched panoramas. Raw recording has 196 unstitched DNG, `processed_panoramas: 0` |
| Scanner-ray comparison (mode B) validation | Blocked | A model for site 02516.182 |
| Fused site surface for display | Blocked | Measured station-to-station registration residuals |
| Scanner make/model detection | Blocked | Files that declare `sensorVendor` |

---

## 6. QA — three reports, never merged

Per `RAPIDMESH-REVIEW-FINDINGS.md` §1 and `FINDING-002-QA-DEFINITION.md`.

1. **Retained-surface fidelity.** Point-to-mesh over samples actually
   represented in the mesh. Excludes despeckle, carve **and island** removals.
   The current code includes island-culled points, which is the defect
   `FINDING-002` records.
2. **Filtering and coverage ledger.** Retained, despeckled, carved,
   island-culled, restored, no-return, otherwise excluded. Every input sample
   accounted for exactly once.
3. **Mesh-to-source deviation.** Sample the finished triangles, measure back to
   source points. The only measure that detects invented surface, and the only
   one capable of saying anything on real data before decimation exists.

Separately, **scan-to-model deviation** (§3A) is a different report with a
different subject and is never combined with the three above.

Every report carries the reproducibility fields listed in
`RAPIDMESH-REVIEW-FINDINGS.md` §9, and states which dataset is the source of
truth and whether figures are exact or sampled.

---

## 7. Non-negotiables

- **Measure before claiming.** No quality statement without a number and the
  command that produced it.
- **Millimetre precision at MGA coordinates.** Positions are stored as f32
  offsets from an f64 origin. A raw f32 world coordinate at 6-digit northings
  cannot hold a millimetre and must never appear in the format. The sample IFC
  places at northing 5,896,381,927 mm, where an f32 ULP is **512 mm**.
- **The scan is evidence; the model is what is being checked.** Numerical
  results trace to original scan observations, never to a decimated display
  mesh unless the report says so and states the simplification error.
- **No silent alignment.** ICP never runs by default. It would conceal the
  set-out errors the product exists to find. If requested explicitly, the
  original transform is preserved, the proposed transform reported, and
  statistics given before and after.
- **No hidden loss.** Every excluded sample is counted and categorised. Every
  threshold has a physical unit, a documented default, a test, and a line in
  the report.
- **No baked lighting in colour data.** Normals go in the file; lighting is the
  renderer's job. Cairn multiplies a lambert term into RGB and discards the
  normals, which contaminates survey colour data irreversibly.
- **A scan that fails to mesh is not a failed scan.** Meshing, texture and
  comparison fail independently and always leave the point cloud usable.
- **Meshing runs out-of-process** with its own memory limit. An OOM kills its
  own job and nothing else.
- **No client data in the repository. Ever.** Fixtures are synthetic or
  explicitly cleared. Metadata, counts and hashes only.
- **Python first.** Rust or C++ only for hotspots proven by measurement.
- **Clients cannot export the source point cloud.** Deviation attributes ride
  on the mesh; source points do not leave the server. Reports export freely.

---

## 8. Phases

Each phase produces a number before the next begins.

| Phase | Deliverable | Gate |
|---|---|---|
| **0** | Cairn memory fix: subprocess isolation, `RLIMIT_AS`, remove the redundant f64 copy, pre-flight ceiling, all three call sites | 1.77 GB file fails its own job only; `/api/health` answers 200 throughout |
| **1** | QA rework (§6) + chunked E57 reading | Real-data retained-surface and mesh-to-source figures exist; ≤ 512 MB on the 14.5 M-point station |
| **2** | Cairn baseline + first real comparison | Measured RapidMesh vs `mesher.py` on the same file, same metric |
| **3** | Error-bounded decimation, LOD chain | Every LOD within its stated budget; §4.1 met on real data |
| **4** | IFC import, modes A + D, site-level heat map and report | Heat map matches its own JSON; grey correct on known-occluded surfaces |
| **5** | RMX container, browser first paint, texture | First paint ≤ 1 s; size beats Cairn at equal fidelity |
| **6** | NavVis B1, then B2 | Per `docs/adr/ADR-003`, `ADR-004` |
| **7** | Cairn integration | Legacy projects open; RapidMesh disableable; export restrictions enforced |

Phase ordering changed on 2 Aug 2026. QA and memory moved ahead of decimation
because `FINDING-002` showed there is currently no real-data number to
decimate against.

---

## 9. Document map

| Document | Role |
|---|---|
| `00-PRODUCT-DEFINITION.md` | This file. Highest authority |
| `ARCHITECTURE.md` | Technical design |
| `docs/DATA-INVENTORY.md` | Measured facts about the sample data. Never guess where this has a number |
| `REVIEW-CAIRN-MESHING.md` | What Cairn's mesher does, and the real gaps |
| `RAPIDMESH-REVIEW-FINDINGS.md` | External review, 1 Aug 2026. Requirements, not commentary |
| `FINDING-001-PDAL-QUANTISATION.md` | Cairn likely quantising E57 imports to 1 cm |
| `FINDING-002-QA-DEFINITION.md` | The deviation report measures the wrong thing |
| `CAIRN-MESH-MEMORY-ISSUE.md` | The production defect phase 0 fixes |
| `docs/adr/ADR-001` … `ADR-005` | Decisions taken, with reasoning |
| `docs/HANDOVER.md` | Cold-start brief for an implementation session |
| `CLAUDE.md` | Charter. Read every session |

Where a document and the code disagree, **report the conflict and stop.** Do
not resolve it silently.
