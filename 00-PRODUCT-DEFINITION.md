# RAPIDMESH: PRODUCT DEFINITION

> **READ THIS FILE BEFORE ANY OTHER FILE IN THIS REPOSITORY.**
>
> Highest authority document. Where any other file disagrees with this one,
> this one wins and the other file is wrong.
>
> Owner: John Brady. Established 31 July 2026. Revised 15 August 2026.

**Revision note (15 Aug 2026).** RapidMesh resumed as a separate project with a
new primary outcome: a Cairn-aligned multiresolution surface that can be
switched and overlaid against the original point cloud without spatial
movement. E57, LAS and LAZ are required inputs. Scan-to-model comparison moves
later and is surveyor/admin QC only, never a client-facing capability.
Combined-project representation is open for investigation. See
`SPATIAL-CONTRACT.md` and `docs/adr/ADR-008`.

**Superseded revision note (2 Aug 2026).** Three approved changes and one correction:
NavVis mobile scan data is now in scope (§5); model-to-scan comparison is now
the primary commercial deliverable (§3A); comparison and reporting move to
site level while meshing stays per station (§5, `docs/adr/ADR-005`); and the
quality claims in §4 are narrowed to what has actually been measured
(`FINDING-002-QA-DEFINITION.md`).

---

## 1. The product, in one sentence

**RapidMesh converts Cairn-supported E57, LAS and LAZ point data into a
survey-fidelity, streamable multiresolution surface that remains spatially
locked to the original point cloud; it later provides private surveyor/admin
QC comparison against imported models.**

It is a library and a command-line tool. It is not a server and is not yet a
Cairn feature. Integration requires proof plus separate explicit approval.

---

## 2. Relationship to Cairn

Separate repository. Separate package. **Zero imports from Cairn, in either
direction, until integration is explicitly approved.**

Cairn is read-only reference material during RapidMesh development. RapidMesh
does not alter Cairn's V1 scope and Cairn must not depend on RapidMesh until
integration is separately approved. The integration shape is re-established
from Cairn's then-current architecture rather than assumed from historical
call sites.

**One exception, and it is not integration.** `CAIRN-MESH-MEMORY-ISSUE.md`
documents a live production fault: `backend/mesher.py` reads whole clouds
in-process and OOM-killed the uvicorn worker on a real 1.77 GB client file,
taking the whole application down. Fixing that is remediation of a shipping
defect, not a step toward integration, and it is the only permitted change to
Cairn before integration is approved.

---

## 3. The design decisions everything else follows from

**Spatial truth first.** The complete source-to-project transform is explicit,
reversible and tested before reconstruction or simplification. The governing
coordinate rules are in `SPATIAL-CONTRACT.md`.

**Mesh at the scanner's native lattice, then decimate to a measured tolerance.
Never decimate first — where a native lattice exists.**

Structured E57 carries that lattice. LAS and LAZ do not, so they use a separate
evidence-preserving reconstruction path and must never be described as native-
lattice input.

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

## 3A. Private surveyor/admin QC

Scan-to-model comparison is a later QC capability for a surveying firm's
surveyors and administrators. It is not shown or delivered to client accounts
or anonymous share holders. The result remains traceable to source observations
and never upgrades a RapidMesh display surface into survey evidence.

Everything in §4 exists to make QC numbers defensible when this later phase is
built. It does not move ahead of the aligned multiresolution surface.

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

**Split reporting — DEC-006.** *Project-lead decision, 15 August 2026
(project-lead decisions register): the 10 mm-floor acceptance test is
split-reported. Mesher acceptance applies to RapidMesh's own contribution,
measured by the noise-isolated method; the propagated instrument-noise term is
reported separately beside it; the end-to-end figure is still reported and is
re-baselined when the real instrument sigma arrives.*

**The budgets in the table above do not change.** What the decision fixes is
*which measured quantity* is tested against them, and that all three quantities
are published together — never one standing in for another.

| Reported quantity | Measured how | Tested against the budgets above |
|---|---|---|
| **RapidMesh's own contribution** | Noise-isolated, by the `FINDING-003-GEOMETRIC-TAIL.md` isolation-matrix method | **Yes. This is the mesher's acceptance test** |
| **Propagated observation noise** | The same sweep at the source's declared — or, failing that, explicitly assumed — range sigma | No. Reported separately, beside it, with the sigma named |
| **End-to-end deviation** | Both terms together, as the surveyor encounters them | No, but always reported, and never omitted because it is the larger number |

Two things this is not. It is **not** a widening: no budget moves, and a
mesher-only figure over 3.2 mm still fails (§4.3 records exactly that failure
and its wording is unchanged). It is **not** a claim that the end-to-end figure
is acceptable — only that it is a different quantity, and attributing it wholly
to the mesher was the misattribution `FINDING-003` corrected.

**Pending evidence:** the end-to-end figure's re-baseline is blocked on the real
instrument sigma for the reference dataset, which has not been obtained
(`CLAUDE.md` §9 item 11). Until it arrives, every end-to-end figure states the
sigma it assumed and that the assumption is a test parameter, not a measured
property of any real instrument.

### 4.1a Minimum defensible tolerance — computed, not asserted

The 10 mm floor is a **global product minimum**, not a claim that 10 mm is
achievable on every project. Different surveys have different instruments,
control and registration quality.

RapidMesh computes a **project-specific minimum defensible tolerance** when the
inputs exist. Contributing terms, each reported separately with its confidence
basis:

- the surveyor's requested tolerance
- RapidMesh's measured contribution (reconstruction, and decimation where
  applicable)
- declared instrument uncertainty
- registration and control/georeferencing uncertainty
- comparison sampling contribution
- whether the selected tolerance is supported by the available evidence

**DEC-006 governs the second term here as well** *(project-lead decision,
15 August 2026, project-lead decisions register: mesher acceptance is the
noise-isolated own-contribution figure, the propagated-noise term is reported
separately beside it, and the end-to-end figure is still reported and
re-baselined on the real sigma).* RapidMesh's contribution enters this
computation **noise-isolated**. Propagated observation noise belongs to the
instrument-uncertainty term and is never folded into RapidMesh's, in either
direction — doing so would either flatter the mesher or charge it for the
instrument, and the split exists to make both impossible.

Where the instrument sigma is undeclared it is a **missing input** under the
rule below, not a number to assume. For the reference dataset that is exactly
where it stands today, and it is the same pending evidence the §4.1 end-to-end
re-baseline waits on.

**Do not simply add RMS, p99.9, registration residuals and survey accuracy.**
They are different statistics at different confidence levels. The combination
method must be defined, stated in the report, and applied consistently.

Where any required term is missing, the report states:

> **Minimum defensible tolerance cannot be computed from the available project
> evidence.**

and names which inputs are absent. It does not guess, and it does not fall back
to the global floor as though it were project-specific.

Phase 2 determines whether 10 mm is supportable **for the reference dataset**.
If it is not, the report says so. That result does not raise the global floor.

### 4.2 Everything else

| Property | Target | How it is measured |
|---|---|---|
| Real detail preserved | Door frames, handrails, window reveals survive filtering | Synthetic fixtures with known edges |
| Mover removal | ≥ 95% of transient points dropped | Synthetic scan with a scripted moving object |
| False positives from mover removal | ≤ 0.1% of static surface points dropped | Same fixture |
| **Processing working memory** | **≤ 512 MB** | Resident during processing, excluding incrementally written output. Measured on the largest real sample station |
| **Measured peak RSS** | **≤ 1.5 GB** | Acceptance gate, measured by an RSS watchdog. **Not** enforced by `RLIMIT_AS`, which bounds virtual address space and is a separate, calibrated fail-safe. The hard physical boundary is a job-specific cgroup limit. See `docs/adr/ADR-006` |
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

**Known gap against the acceptance test, in the exact wording to use:**

> **Passes the 25 mm default-tolerance budget and fails the 10 mm
> minimum-tolerance budget.**

Best synthetic result 1.00 mm RMS / 3.55 mm p99.9. Budget at the 25 mm default
is 8.0 mm p99.9 (pass, with margin); at the 10 mm floor it is 3.2 mm (fail, by
11%).

**Measured, not assumed:** the isolation matrix has been run
(`FINDING-003-GEOMETRIC-TAIL.md`, 2 Aug 2026), and it inverts the earlier
reading. At the fine 0.090° sampling the finding specifies, **RapidMesh's own
geometric contribution to this fixture is ≈ 0 mm** — walls and floor both read
exactly 0.00 mm RMS, p99.9 and max at zero injected noise. **The reported tail
tracks the injected noise and nothing else:** p99.9 rises 0.00 → 2.23 → 3.47 →
4.40 → 5.99 mm as range sigma rises 0 → 1 → 2 → 3 → 5 mm, while carving,
parallax restore and island culling each move it by ≤ 0.02 mm and
`max_incidence_deg` is bit-for-bit identical across 78–85°. The earlier
"12.89 mm at zero noise" figure was honestly measured but taken at a coarser
sampling, where binning error swamped the noise signal.

This is a result about **one synthetic planar fixture at an assumed 2 mm range
sigma**, not a universal zero-error claim and not a property of any real
dataset. The 2 mm sigma is a documented test parameter; the reference data's
actual instrument is unknown (`CLAUDE.md` §9 item 11). **Do not quietly widen
the 3.2 mm budget.** The verdict above is unchanged: the gap to the 10 mm floor
is real, and this finding says it is dominated by propagated observation noise
rather than by mesher defect — which is a statement about where the budget
goes, not permission to enlarge it.

Until §6's QA rework lands, **no real-data fidelity claim may be made.** The
targets stand; the evidence for them is narrower than this document previously
implied.

---

## 5. Scope

### 5.1 Scan families

| | Pipeline A — **structured TLS** | Pipeline B — **unstructured point data** | Pipeline C — NavVis |
|---|---|---|---|
| Input | Structured terrestrial E57 | LAS, LAZ and unstructured E57 | C1 registered E57 · C2 raw `rec-v4` |
| Method | Native lattice | Evidence-preserving reconstruction, to be selected by measurement | C1 points + supplied normals · C2 accumulate from raw sweeps |

**Structured TLS is the first implemented family**, but all three Cairn point-
cloud formats are required product inputs. Different evidence gets a different
front half and converges only after each format has an honest surface candidate
and provenance record.

NavVis was previously out of scope. That is **superseded by approved product
change, 2 Aug 2026.** NavVis gets its own ingestion and reconstruction path and
shares no assumption that is true only of static structured scans. See
`docs/adr/ADR-003` and `docs/adr/ADR-004`.

### 5.2 In scope

- E57, LAS and LAZ inputs. Structured E57 preserves its scanner lattice;
  unstructured inputs do not claim one.
- Terrestrial static scanners. **Note:** the 30 sample stations declare no
  `sensorVendor`, so make and model are not recoverable from them. Do not
  assert Trimble or Faro on the basis of this dataset.
- Per-station/source evidence geometry remains available even if a later
  combined display representation is selected.
- Cross-station data used for occlusion carving (removing transients). Never
  merged into the output geometry.
- Later surveyor/admin-only site-level QC comparison across all stations as one
  evidence set, per `docs/adr/ADR-005` and `ADR-008`.
- Later IFC2X3 / IFC4 model import, comparison modes A, B and D, tolerance heat map.
- Texture baked from the scan's own RGB or its panorama. **Optional:** 4 of 30
  sample stations carry no RGB, and `images2D` is empty on all 31 sample E57s.
- LOD chain produced by decimation, not by re-binning.
- A mesh-fidelity report per station; later, a private QC report per site.

### 5.3 Explicitly out of scope

- Choosing a combined-project representation before investigation. Coordinated
  per-station display, a fused display-only surface, or both remain candidates.
  A fused surface can never become numerical evidence or hide registration
  disagreement.
- A production viewer. RapidMesh produces files; the Phase 4 engineering
  alignment view is a validation harness, not a client application.
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
| Combined-project representation | **Scheduled — phase 3** (DEC-007) | Evidence comparing per-station coordination, fused display-only output, or both; registration residuals where fusion is evaluated |
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
| **0c** | Spatial contract and transform foundation | Full rigid pose, units and rebasing chain documented and tested; rotation/axis/scale/precision regressions pass; repository gates green |
| **1** | QA rework (§6) + streamed/chunked processing + **the completed `FINDING-003` isolation matrix** + clean-checkout gate. Design-first: halo calculus, two-pass island finalisation, versioned intermediate tile contract v0, determinism spec. Also the per-station observation store phase 5 later consumes | **Streamed ≡ in-memory equivalence on fixtures, exact** — mesh, ledger and both QA directions identical to `pipeline.mesh_station`; every source sample carries exactly one final disposition and the ledger balances on every station, with halo evidence never double-counted; output independent of chunk and band size per the determinism spec; no row seam, **including the azimuth wrap seam**; no full-resolution mesh in memory; working memory ≤ 512 MB and peak (`VmHWM`-class) RSS ≤ 1.5 GB on the 14.5 M-point station, asserted in tests; both QA directions on authorised real data at both reference resolutions; all 30 structured stations process, including the 4 without colour; coordinate precision at real MGA-magnitude values; existing tests green; smoke gate from a fresh clone |
| **2a** | **Ingestion contract and Cairn baseline** — on the critical path. LAS/LAZ ingestion contract, honest unsupported reporting, and a Cairn-vs-RapidMesh baseline on the same structured E57s. No reconstruction R&D here | Each required format either passes its explicit input gate or reports "surface reconstruction not yet supported" with the missing capability named; per-axis scale/offset preserved and quantisation reported; no unrecorded coordinate change; baseline rows committed to `benchmarks/` with their commands and commit ids, recording which Cairn converter version produced the LAZ |
| **3** | Error-bounded decimation + LOD chain + tiled incremental writing. **ITEM-001, the combined-project investigation, runs inside this phase** | Every LOD spatially locked and within its declared budgets on real data — **magnitude and bias**, the bias budget flat and tiny at every level; guaranteed per-tile error bounds reported beside sampled percentiles; no cracks or T-junctions; seam-local maxima pass; deterministic re-runs; fine-feature survival thresholds met; output size competitive at **equal measured fidelity** against the 2a baseline; ITEM-001 decision memo delivered |
| **4** | **Streamable container decided by benchmark**, progressive browser loader, engineering alignment view. The container and manifest make the QC-only / client-audience tier distinction structural | No visible or measurable switch displacement across the full `SPATIAL-CONTRACT.md` view matrix, at every LOD transition; large-coordinate and rotated-project cases pass; first paint ≤ 1 s under the defined 10 Mbit and stated-latency contract, per scenario; frame rate and GPU memory on named hardware **including a mid-range phone and a tablet**; progressive refinement stable with the worst streaming frame recorded; failed and missing tiles degrade safely |
| **2b** | **LAS/LAZ and unstructured-E57 reconstruction** — parallel to or after phase 4, not on the first-release critical path | Format-specific gates pass; quantisation, local spacing, method, parameters and bidirectional residuals reported; "no native lattice: surface topology inferred" stated; cross-format alignment of the same observations within quantisation plus a stated budget; unsupported inputs identified and refused cleanly |
| **5** | Surveyor/admin QC comparison: vertical slice, then modes B and D, heat map and report. **Entry criterion:** the IFC toolchain licence position answered in writing before the dependency is introduced | Observation-backed JSON and heat map agree; grey and attribution correct; decimation-invariance test in place before the engine grows; QC routes and controls absent for client audiences |
| **6** | NavVis B1. B2 deferred past the first release and re-estimated when scheduled | Per `docs/adr/ADR-003`, `ADR-004`. No mover-removal claim for B1; no invented georeferencing; map-frame output labelled as such |
| **7** | Cairn integration | Legacy projects open; RapidMesh disableable; failed jobs preserve the source project; API health under large jobs; permissions enforced server-side; QC absent and export restrictions enforced for client audiences, **tested per denial**; upgrade and rollback proven; 30-concurrent-viewer tile-serving probe passes |

**First release, and why the numbering is not the order.** The first release is
**Points ↔ RapidMesh through Cairn for structured terrestrial E57 only**
(*DEC-003, project-lead decision 15 August 2026, project-lead decisions
register*). Execution order is therefore **0a → 0b → 0c → 1 → 2a → 3 → 4 → 7**,
with **2b**, **5** and **6** after it, alongside the hardening subset a pilot
needs. Phase 2 split because the ingestion contract and the Cairn baseline are
needed early and cheaply, while reconstruction for formats with no native
lattice is open-ended research that must not sit on the critical path.
LAS and LAZ remain required product inputs throughout — until 2b they are
accepted, validated and honestly reported unsupported, never silently given
structured-E57 fidelity.

**Two ordering decisions worth defending.**

Phase 0 splits. Isolation is real engineering and takes time; the two live
routes are a production availability risk *today*. Disabling them is hours of
work and does not depend on isolation being finished.

Phase 0c was inserted because the spatial review found an incomplete rigid
transform hidden by identity-only tests. Geometry, LOD and output work must not
continue on an ambiguous coordinate contract.

Comparison moved behind the aligned multiresolution surface by John's decision
of 15 August 2026. Its observation-set architecture remains valid and prevents
later QC numbers from depending on a display mesh.

**Three decisions that shape the phases above**, each a project-lead decision
of 15 August 2026 recorded in the project-lead decisions register:

- **DEC-004** — client audiences receive decimated, error-bounded LODs only;
  the full-resolution tier is surveyor/admin QC only and per-vertex source
  identity never leaves the server. Structural in the phase 4
  container/manifest, enforced server-side with per-denial tests at phase 7.
- **DEC-005** — the streamable container is decided by benchmark, not
  preference: phase 3 builds against a draft versioned RapidMesh manifest, and
  early phase 4 benchmarks standard payloads (glTF + meshopt-class, decoders
  self-hosted) against a bespoke binary on the real reference station for size,
  browser decode time and implementation effort, committing on the numbers and
  recording the result as an ADR. A bespoke format is chosen only if the
  standard route fails a stated gate. **The name "RMX" is no longer assumed**;
  the manifest is RapidMesh's regardless of which container wins.
- **DEC-007** — a display-only site-level coarse tier is pre-approved subject
  to phase 3 measurement (§8D).

Texture baking and silhouette-aware carving carry **no phase number**: both
follow the first release, texture by sequencing and carving because
`FINDING-003` left no measured need for it. Neither may be claimed until it is
scheduled and built.

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

A **clean-checkout smoke gate** runs continuously from phase 1: the CLI, the
benchmark and the inventory tools must each execute from their documented
locations on a fresh clone. This catches missing dependencies and standard
library shadowing. It exists because `tools/inspect.py` shadowed Python's
stdlib `inspect` and broke every script run from `tools/`.

| Metric | First available |
|---|---|
| Source size, point count, lattice tier | Now |
| Layered noise sweep: 0, 1, 2, 3, 5 mm sigma, layers 1–4 | Phase 1 |
| Tile-size sweep: memory, output size, locked-boundary %, seam-local max deviation, cracks, decimation ratio, LOD0 bytes and decode time | Phase 3 |
| Processing time, peak RSS, working memory | Phase 1 |
| RMS, p99.9, max — retained-surface **and** mesh-to-source | Phase 1 |
| Filtering ledger completeness | Phase 1 |
| Cairn vs RapidMesh on the same file | Phase 2a |
| Output size, compression ratio | Phase 3 |
| Fine-feature survival; opening and occlusion preservation | Phase 3 |
| Time to first paint; time to usable detail | Phase 4 |
| Browser frame rate, GPU memory | Phase 4 |
| Measurement differences against the original E57 | Phase 4 |
| Scan-to-model classification correctness | Phase 5 |
| Behaviour on unsupported, corrupt and exceptionally large files | Continuous from phase 1 |

Every row records the command that produced it and the commit it ran at.

---

## 8C. QC comparison engine — what must be defined, and when

The comparison engine is a later surveyor/admin QC capability. Its numerical
source remains original observations under `ADR-007`; it never consumes a
display mesh and never appears for client-facing audiences.

**Decide before the phase 5 vertical slice, because these change the architecture:**

1. **Site-level observation selection** across stations — nearest, best
   incidence angle, shortest range, or a combination. Documented, testable,
   recorded in the report.
2. **Grey classification** — the visibility and occlusion test that separates
   "not observed" from "wrong". Grey is never automatically red.
3. **Denominators and exclusions** — what "91.4% of scanned surface" is a
   percentage *of*, and which samples are excluded from it.
4. **Attribution** — element, storey and category, from IFC `GlobalId`.

**Decide during phase 5 with measurement, because a guess now would be arbitrary:**

model and scan coordinate validity checks · normal-angle acceptance ·
overlapping and contradictory observations · ambiguous correspondence ·
model-surface sampling density · weighting by area versus scan samples versus
elements · signed-deviation confidence · behaviour when registration quality is
unknown · behaviour when the model and scan disagree grossly.

Each is answered in the report it appears in, with the rule stated.

---

## 8D. Combined-project representation — investigation open

Per-station geometry preserves original evidence and cannot hide registration
error (`docs/adr/ADR-005`). A seamless display surface may improve the expected
experience. The choice stays open pending investigation of alignment truth,
seams, registration uncertainty, package size, first paint and interaction
behaviour — **and that investigation is now scheduled inside phase 3** as
ITEM-001.

**DEC-007** *(project-lead decision, 15 August 2026, project-lead decisions
register)*: if phase 3's measurements show per-station LOD0s cannot meet the
1 s site first paint, a fused **display-only** site-level coarse tier is
approved product behaviour without a further decision round. What is
pre-approved is the tier, not the conclusion — the measurement still decides
whether it is built.

**Condition, non-negotiable and unchanged:** a unified surface is **display
only**, forever. It never produces numbers, and it never replaces or obscures
the per-station evidence that measurements are traced to.

---

## 9. Document map

| Document | Role |
|---|---|
| `00-PRODUCT-DEFINITION.md` | This file. Highest authority |
| `SPATIAL-CONTRACT.md` | Coordinate, transform, precision and alignment authority |
| `ARCHITECTURE.md` | Technical design |
| `docs/DATA-INVENTORY.md` | Measured facts about the sample data. Never guess where this has a number |
| `REVIEW-CAIRN-MESHING.md` | What Cairn's mesher does, and the real gaps |
| `RAPIDMESH-REVIEW-FINDINGS.md` | External review, 1 Aug 2026. Requirements, not commentary |
| `FINDING-001-PDAL-QUANTISATION.md` | Cairn was quantising E57 imports to 1 cm. Confirmed, and fixed in Cairn 4–5 Aug 2026; pre-fix imports stay quantised |
| `FINDING-002-QA-DEFINITION.md` | The deviation report measures the wrong thing |
| `FINDING-003-GEOMETRIC-TAIL.md` | On the tested fine planar fixture the p99.9 tail tracks injected noise and RapidMesh's own geometric contribution is ≈ 0 mm. Not a universal zero-error claim |
| `CAIRN-MESH-MEMORY-ISSUE.md` | The production defect phase 0 fixes |
| `docs/adr/ADR-001` … `ADR-008` | Decisions taken, with reasoning |
| `docs/HANDOVER.md` | Cold-start brief for an implementation session |
| `CLAUDE.md` | Charter. Read every session |

Where a document and the code disagree, **report the conflict and stop.** Do
not resolve it silently.
