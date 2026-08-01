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
| **Processing working memory** | **≤ 512 MB** | Resident during processing, excluding incrementally written output. Measured on the largest real sample station |
| **Peak process RSS** | **≤ 1.5 GB**, enforced by `RLIMIT_AS` | The hard ceiling. Sized to sit alongside the web server on a 4 GB host |
| Time to first paint, browser | ≤ 1 s on a 10 Mbit link | LOD0 tier size budget, test contract per `RAPIDMESH-REVIEW-FINDINGS.md` §11 |
| Size per site | Beat Cairn at equal or better deviation | Same scans, same components counted |

The single flat "512 MB peak RSS" figure previously stated here was wrong. A
finished full-resolution mesh for one high-resolution station is **628.9 MB of
output alone**, before working memory. See `docs/adr/ADR-006`.

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
for the full pipeline on a 2.95 M-point station. Neither budget is currently
met.

**Known gap against the acceptance test.** The best synthetic result is
1.00 mm RMS / **3.55 mm p99.9**. The derived budget at the 10 mm tolerance
floor is 1.0 mm RMS / **3.2 mm p99.9**. RMS passes; **p99.9 misses by 11%**.

Stated precisely, because the distinction matters: at the 25 mm default
tolerance the p99.9 budget is 8.0 mm and 3.55 mm passes with margin. The miss
is at the 10 mm floor only, and the synthetic fixture carries 2 mm range sigma,
so part of that tail is instrument noise rather than mesh error. It is still a
miss against a stated acceptance test and is tracked as open item 9 in
`CLAUDE.md` §9. Resolution options, in order of preference: improve the mesher;
separate instrument noise from mesh error in the metric; or raise the tolerance
floor. Do not quietly widen the budget.

Until §6's QA rework lands, **no real-data fidelity claim may be made.** The
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

Every phase ends with a **measured result and a pass/fail gate**, not with code
being finished.

| Phase | Deliverable | Gate — all must pass |
|---|---|---|
| **0a** | **Endpoint safety, today.** Disable or authorise-gate `build_meshes` and `build_vantage_meshes` at the server, not the frontend. Automatic meshing stays off | Both routes return 403/404 to an unauthorised direct call, proven by test. No route can reach `mesher.py` from an ordinary request |
| **0b** | Job isolation. Child process, per-workload `RLIMIT_AS`, pre-flight ceiling, no retry loop | 1.77 GB file fails its own job; `/api/health` 200 throughout; other projects viewable; scan preserved; explicit failure status; `dmesg` shows the child killed, not uvicorn |
| **1** | QA rework (§6) + streamed/chunked processing | Real-data retained-surface, ledger and mesh-to-source figures exist for both sample resolutions; ledger balances to the sample; working memory ≤ 512 MB, RSS ≤ 1.5 GB on the 14.5 M-point station |
| **2** | Real-data baseline: Cairn vs RapidMesh, same file, same metric. **Confirm or refute `FINDING-001` first** | Both numbers published with the commands that produced them; LAZ quantisation status stated |
| **3a** | **Comparison vertical slice.** Mode A only, one station against the IFC, JSON out, no heat map, no site aggregation | A signed deviation number per model element, traceable to scan points, with the georeferencing checks passing on the real IFC |
| **3b** | Error-bounded decimation + LOD chain + tiled incremental writing | Every LOD within its stated deviation budget on real data; §4.1 met; memory budgets held with output streamed, per `docs/adr/ADR-006` |
| **4** | RMX container, browser first paint, progressive refinement, texture | First paint ≤ 1 s at 10 Mbit under the §11 test contract; 30-station site navigable at a stated frame rate; package size beats Cairn at equal or better measured fidelity |
| **5** | Site-level comparison: modes B and D, heat map both targets, full report | Heat map matches its own JSON exactly; grey correct on known-occluded surfaces; element attribution correct; denominators documented |
| **6** | NavVis B1, then B2 | Per `docs/adr/ADR-003`, `ADR-004` |
| **7** | Cairn integration | Legacy projects open; RapidMesh disableable; export restrictions enforced; comparison UI performs |

**Two ordering decisions worth defending.**

Phase 0 splits. Isolation is real engineering and takes time; the two live
routes are a production availability risk *today*. Disabling them is hours of
work and does not depend on isolation being finished.

Phase 3a exists because the comparison report is the commercial deliverable and
would otherwise sit behind decimation, RMX and browser streaming — three
substantial phases — before anyone learns whether it works. A thin slice is
cheap (BVH plus nearest-surface distance) and can run against the existing
full-resolution mesh. It is far better to discover an IFC georeferencing or
correspondence problem at phase 3 than at phase 5.

---

## 8A. Parity targets — what "TurboMesh or better" means

RapidMesh is the processing engine. **The customer experience is RapidMesh
operating through Cairn**: upload, processing, browser streaming, navigation,
measurement, model comparison and reporting.

Cintoo and TurboMesh are a **public capability benchmark only**. Nothing is
copied, examined or reverse-engineered. We compete against published outcomes
using our own architecture and our own measurements.

**We may state that RapidMesh is intended to compete with or exceed TurboMesh.
We may not state that it does until benchmark evidence exists.**

### Parity — the table stakes

| Target | Measured how |
|---|---|
| Per-station multiresolution meshes | LOD chain with per-level deviation budgets |
| Fast first paint, then progressive refinement | ≤ 1 s at 10 Mbit under the §11 test contract |
| Smooth navigation across a many-station project | Stated frame rate and GPU memory on the 30-station sample site, on named hardware |
| Fine features survive: handrails, door frames, window reveals, pipes, openings | Synthetic fixtures with analytic edges, plus named real features |
| Scan colour and texture where the source supports it | 4 of 30 sample stations have no RGB; `images2D` empty on all 31. Report the limitation, do not fabricate |
| Measurements traceable to the original scan | Never to a decimated display mesh unless stated with its simplification error |
| Model overlay and scan-to-model comparison | §3A |
| Structured TLS and NavVis through appropriate pipelines | §5.1 |
| Processing cannot take Cairn offline | Phase 0 gate |

### Differentiators — where we intend to be ahead

1. Published **mesh-to-source** QA, not just point-to-mesh.
2. Error-bounded decimation with the achieved error measured, not assumed.
3. A filtering and coverage ledger accounting for every input sample exactly
   once.
4. Site-level, element-attributed model comparison.
5. Auditable green / red / **grey** tolerance reporting, with grey never
   silently folded into red.
6. Complete provenance and reproducibility on every report.
7. No silent ICP. Set-out errors are never concealed by alignment.
8. Native structured-lattice processing before any simplification.
9. Source point clouds protected from client export while authorised mesh
   outputs and reports remain available.

These are the claims to defend. Each needs a benchmark row in §8B.

---

## 8B. Controlled benchmark suite

One suite, identical source datasets, results committed to `benchmarks/`.
Scheduled by the phase that can first produce each figure — a browser metric
cannot exist before phase 4.

| Metric | First available |
|---|---|
| Source size, point count, lattice tier | Now |
| Processing time, peak RSS, working memory | Phase 1 |
| RMS, p99.9, max — retained-surface **and** mesh-to-source | Phase 1 |
| Filtering ledger completeness | Phase 1 |
| Cairn vs RapidMesh on the same file | Phase 2 |
| Output size, compression ratio | Phase 3b |
| Fine-feature survival; opening and occlusion preservation | Phase 3b |
| Time to first paint; time to usable detail | Phase 4 |
| Browser frame rate, GPU memory | Phase 4 |
| Measurement differences against the original E57 | Phase 4 |
| Scan-to-model classification correctness | Phase 5 |
| Behaviour on unsupported, corrupt and exceptionally large files | Continuous from phase 1 |

Every row records the command that produced it and the commit it ran at.

---

## 8C. Comparison engine — what must be defined, and when

The comparison report is the commercial differentiator, so its behaviour is
product definition rather than implementation detail. But specifying all of it
before writing any of it is documentation instead of code, which §7 forbids.
Split accordingly.

**Decide before phase 3a, because these change the architecture:**

1. **Site-level observation selection** across stations — nearest, best
   incidence angle, shortest range, or a combination. Documented, testable,
   recorded in the report.
2. **Grey classification** — the visibility and occlusion test that separates
   "not observed" from "wrong". Grey is never automatically red.
3. **Denominators and exclusions** — what "91.4% of scanned surface" is a
   percentage *of*, and which samples are excluded from it.
4. **Attribution** — element, storey and category, from IFC `GlobalId`.

**Decide during phase 5, with measurement, because a guess now would be
arbitrary:**

model and scan coordinate validity checks · normal-angle acceptance ·
overlapping and contradictory observations · ambiguous correspondence ·
model-surface sampling density · weighting by area versus scan samples versus
elements · signed-deviation confidence · behaviour when registration quality is
unknown · behaviour when the model and scan disagree grossly.

Each is answered in the report it appears in, with the rule stated.

---

## 8D. Unified site mesh — deferred, not cancelled

Per-station geometry is the correct first release: it preserves the original
evidence and cannot hide registration error (`docs/adr/ADR-005`).

A seamless display surface is part of the expected experience of comparable
products and stays on the roadmap. It remains deferred until station-to-station
registration residuals can be measured and reported.

**Condition, non-negotiable:** a future unified surface is **display only**. It
never produces numbers, and it never replaces or obscures the per-station
evidence that measurements are traced to.

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
