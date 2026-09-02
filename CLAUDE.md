# CLAUDE.md — RapidMesh charter

**Read this first, every session, before doing anything.**

Short on purpose. The detail lives in the documents listed in §7. Keep this
file current: if you complete a phase or resolve a blocker, update §5 and §9 in
the same change.

Last updated: 2 September 2026 — **Gate 1 REOPENED** after the F1 frame defect
(`d681a4a`). Formal reports:
`E:\Claude Projects\Lead AI SW Eng\RapidMesh\REPORTS\2026-09-02-WP-G1r-readjudication-report.md`
(Gate 1) and `...\2026-09-02-WP-F1b-campaign-rerun-report.md` (the evidence).
The Round 7 close report is **superseded on lines 2, 5 and 6**.

> **Do not cite any campaign figure dated 31 August to 1 September.** Every one
> was measured while the streamed path rewrote each station into a frame
> ~5.8 × 10⁶ m from the scanner, meshing roughly 3% of it.

---

## 1. What we are building

RapidMesh converts Cairn-supported E57, LAS and LAZ point data into a
lightweight, streamable multiresolution surface that stays spatially locked to
the original Cairn point cloud. The primary product test is:

> Points → RapidMesh → Points, with no visible or measurable movement.

Scan-to-model comparison is later surveyor/admin QC only. It is never a client-
facing capability.

## 2. The one design decision everything follows from

**Spatial truth first; then preserve the strongest structure the source
contains; then decimate to a measured tolerance.** Structured E57 uses its
native lattice. LAS and LAZ have no such lattice and require a separate,
evidence-preserving reconstruction path. See `SPATIAL-CONTRACT.md`.

## 3. Two scan families, one back half

| | Pipeline A — structured TLS | Pipeline B — unstructured points | Pipeline C — NavVis |
|---|---|---|---|
| Input | Structured terrestrial E57 | LAS, LAZ, unstructured E57 | C1 registered E57 · C2 raw `rec-v4` |
| Front half | Native lattice | Reconstruction selected by evidence | Registered normals · raw sweeps |

Structured TLS is currently implemented. E57, LAS and LAZ are all required.
Meshing is currently per station; combined-project representation remains open
for investigation. Later QC comparison is site-level (`docs/adr/ADR-005`).

## 4. Non-negotiables

1. **Measure before claiming.** No quality statement without a number and the
   command that produced it.
2. **The scan is evidence; the model is what is being checked.**
3. **Never conflate the three accuracies:** mesh fidelity · survey accuracy ·
   model agreement. RapidMesh measures the first and third only.
4. **No silent alignment.** ICP never runs by default.
5. **No hidden loss.** Every excluded sample counted and categorised.
6. **No deferred capability may be claimed.** Report it unsupported and name
   the missing data.
7. **A scan that fails to mesh is not a failed scan.**
8. **Meshing runs out-of-process** with its own memory limit.
9. **No client data in the repository. Ever.**
10. **Python first.** Rust or C++ only for hotspots proven by measurement.
11. **Report conflicts, do not resolve them silently.**

## 5. Phase tracker — update this

Every phase ends with a **measured result and a pass/fail gate**, never with
code being finished.

**First release is Pipeline A only — DEC-003.** *Project-lead decision, 15
August 2026 (project-lead decisions register): the first release is Points ↔
RapidMesh through Cairn for structured terrestrial E57; phase 2 splits, with 2a
(ingestion contract and Cairn baseline) on the critical path and 2b (LAS/LAZ
reconstruction R&D) parallel to or after phase 4.* The table below is therefore
in **execution order, not numeric order**: 0a → 0b → 0c → 1 → 2a → 3 → 4 → 7,
with 2b, 5 and 6 following the first release. Gate content follows the
project-lead execution plan (`PLAN.md`, project-lead records, adopted 15 August
2026); the full gates are in `00-PRODUCT-DEFINITION.md` §8.

| Phase | Deliverable | Status |
|---|---|---|
| 0a | **Endpoint safety.** Disable/gate `build_meshes` + `build_vantage_meshes` at the server. Frontend removal is not protection | **DONE** 2 Aug 2026 — Cairn `e201e98` (gating) and `aac4c9c` (authorisation, found during 0b live testing; see the 0b row). Both 404 unless `CAIRN_ENABLE_MESH_ROUTES=1`; gate test proven to fail against the pre-fix code first. Merged to Cairn `master` 6 Aug 2026 in `14680f5` |
| 0b | Job isolation: child process, per-workload rlimit, ceiling, no retry loop, remove redundant f64 copy | **DONE** 2–3 Aug 2026 — Cairn `b744953` (out of process), `2ae99a4` (redundant copy), `31b245a` (failure reporting), `aac4c9c` (authorisation), `5e12425` (heavy-job admission, 3 Aug). Merged to Cairn `master` 6 Aug 2026 in `14680f5`. Closed against the real 1.77 GB Ampol file in a 3 GB-capped Docker container (see §9's former item 13, now resolved): `/api/health` 200 throughout (88/88 authenticated polls, zero anomalies), container/uvicorn never restarted, worker PID differs from uvicorn's, `RLIMIT_AS` and the RSS watchdog each independently demonstrated firing on the same file under different limit configs, source scan stayed `ready` and immediately retryable, no partial mesh output accepted, no zombie process, cancellation and timeout both proven. **Live testing found and fixed a real bug**: per-task worker failures (e.g. the `MemoryError` this exact file produces) were silently dropped by both routes — neither `meshError` nor any manifest change resulted. Fixed in `31b245a`, proven to fail against the pre-fix code first. A second finding, `aac4c9c`: the routes-enabled flag is a rollout control, not an authorisation control: both routes now also require a signed-in project admin (accounts mode only; token and open mode refused). A third, `5e12425`: only one heavy job is admitted globally |
| 0c | Spatial contract and transform foundation | **DONE** 15 Aug 2026 — `SPATIAL-CONTRACT.md`; full pose applied once to project-axis offsets and normals; inverse E57 recovery fixed; 13 spatial-contract tests, 35 total tests **at 0c close** (the suite has since grown to 448 — see below), Ruff, strict mypy and smoke pass. Header-only validation covered 312 authorised E57 files / 313 scans, including 310 non-identity poses, with zero pose-validation failures. **The 312 file / 313 scan counts are re-verified** (16 Aug 2026, `tools/e57_inventory.py` over every E57 under the authorised sample root: 312 files, 313 scans); they do not conflict with §10, which lists only the 31 files profiled in `docs/DATA-INVENTORY.md`, not the whole authorised corpus. The 310 non-identity-pose figure is carried from the 0c run and was **not** re-verified here. Cairn unchanged |
| 1 | QA rework (3 reports) + streamed/chunked processing + isolation matrix + smoke gate. Design-first items precede the code: halo calculus, two-pass island finalisation, versioned intermediate tile contract v0, determinism spec | **PARTIAL — ACTIVE. Gate 1 at `d681a4a`: 5 PASS, 2 PARTIAL, 1 FAIL; fidelity DEFER. Phase 1 is NOT complete.** Built and gated: streamed ≡ in-memory equivalence (19/19, exact), the exclusive disposition ledger (balanced, and on corrected geometry its categories sum exactly to each file's own point count — the earlier "30/30" count was F1-era and is withdrawn), chunk/band-size independence, wrap-seam geometry (12 tests), band-local geometry and QA, tile contract v0, the observation store, and the evidence envelope including the `_resolve_frame` path taken. **Memory gate FAILS on real data — line 5 is the blocker (ITEM-022).** The former "30/30 PASS, worst working 489,185,280 B (+4.46%)" claim was measured on F1 geometry and is **withdrawn**. Re-measured at `d681a4a`, `band_rows=56`, under Rule G1-M: the 14.5 M-point station the line names uses **3,194,880,000 B working (6.2× the 512 MB budget) and 3,215,560,704 B peak RSS (2.1× the 1.5 GB budget)**; worst measured 3,347,861,504 B working. Medium stations measure 1.13–2.15 GB working, and the **peak** budget fails there too wherever retention is high, so peak failure tracks meshed vertices rather than lattice class. **Budgets are not to be widened.** The real 14.5 M-point station cannot enter the repo (§4 rule 9), so the in-tree assertion is the synthetic extent ladder — a regression guard, not the certification. Smoke gate re-run **from a fresh clone** of `630008a`: install, smoke, pytest, ruff and mypy --strict all green — not repeated at `d681a4a`. **Remaining before Gate 1 closes:** (1) **line 5 — engineering**, the ADR-006 per-tile vertex lifecycle, preceded by stage attribution (ITEM-022); no further measurement can close it; (2) **lines 2 and 6 — machine time**, finishing the 30-station campaign, which needs no engineering; (3) a fidelity metric per ITEM-018. Line 6's *definition* is settled — John adopted Reading A (ITEM-019) — but the evidence first cited under it was measured on F1 geometry and is withdrawn. **No fidelity claim is licensed** — see §9 item 1 |
| 2a | **Ingestion contract and Cairn baseline** — on the critical path (DEC-003). Per-axis LAS/LAZ header scale/offset preserved and reported, suspicious quantisation detected, units/CRS never inferred; LAS/LAZ and unstructured E57 accepted, validated and honestly reported "reconstruction not yet supported"; Cairn-vs-RapidMesh baseline on the *same* structured E57s, recording which converter version produced the LAZ | **PARTIAL.** Ingestion contract **DONE** 2 Sep 2026 — `f1df59d`, WP-2a.1: per-axis LAS/LAZ header scale/offset preserved and reported, suspicious quantisation detected, units/CRS never inferred, unsupported reconstruction reported honestly with the missing capability named; 36 tests. **Cairn baseline NOT STARTED** — Round 9, and blocked behind Gate 1 |
| 3 | Error-bounded decimation, LOD chain, tiled incremental writing. **ITEM-001 (combined-project representation) is investigated inside this phase** (DEC-007). Adds the mean-signed-deviation bias metric and guaranteed per-tile error bounds; the bias budget stays flat and tiny at every LOD so switching can never reveal systematic movement | NOT STARTED |
| 4 | **Container decided by benchmark** (DEC-005), progressive browser loader, engineering alignment view. Container and manifest make the QC-only / client-audience tier distinction **structural** (DEC-004) | NOT STARTED |
| 2b | **LAS/LAZ and unstructured-E57 reconstruction R&D** — parallel to or after phase 4, not on the first-release critical path (DEC-003) | NOT STARTED |
| 5 | Surveyor/admin-only QC comparison vertical slice, then site-level modes B + D. After the first release. **Entry criterion:** the IFC toolchain licence position answered in writing before the dependency is introduced. Input is phase 1's observation store | NOT STARTED |
| 6 | NavVis B1. **B2 is deferred past the first release** and re-estimated on its own when scheduled | NOT STARTED |
| 7 | Cairn integration. Entry criteria include the DEC-004 enforcement design, with per-denial rules and regression tests on every mesh route | NOT STARTED |

**Not carrying a phase number:** texture baking and silhouette-aware carving.
Both sit after the first release — texture by DEC-003's sequencing, carving
because `FINDING-003` left no measured need for it. Neither may be claimed
anywhere until it is scheduled and built.

**DEC-004 — client audiences receive decimated, error-bounded LODs only.**
*Project-lead decision, 15 August 2026 (project-lead decisions register): the
full-resolution tier is surveyor/admin QC only, and per-vertex
`source_sample_id` — or any per-vertex source identity — never leaves the
server.* It becomes structural at phase 4 and enforced server-side at phase 7.

**Already built and working:** structured E57 reader (three lattice tiers),
band-addressable native lattice, edge-preserving despeckle, cross-station
occlusion carving with parallax restore, discontinuity-aware triangulation,
area-based island culling, oriented normals, synthetic fixtures, CLI, and the
band-streamed two-pass pipeline with its tile contract v0, observation store and
evidence envelope.
**448 tests pass** — re-run 2 September 2026 at `d681a4a`, alongside ruff,
mypy --strict and the smoke gate. The **fresh-clone** run was last performed at
`630008a` and has not been repeated at `d681a4a`.

## 6. The heat map, exactly

One tolerance. Default **25 mm**. Minimum settable **10 mm**. No upper bound.

| State | Colour |
|---|---|
| \|deviation\| ≤ tolerance | **Green** |
| \|deviation\| > tolerance, or no corresponding geometry | **Red** |
| Model surface no station saw | **Grey** |

Grey is separate from red in renderer and report. Colour target is switchable:
colour the model (mode D) or the scan mesh (mode A).

**Derived mesh budget:** RapidMesh error ≤ 10% of tolerance. Because the floor
is 10 mm, **1.0 mm RMS and 3.2 mm p99.9 are the tightest figures RapidMesh will
ever need to meet.** Fixed acceptance test. Those numbers are unchanged by
DEC-006, which fixes *which measured quantity* is tested against them —
RapidMesh's noise-isolated own contribution, with propagated noise and the
end-to-end figure reported separately beside it. See
`00-PRODUCT-DEFINITION.md` §4.1.

## 7. Documents, in authority order

| Document | Role |
|---|---|
| `00-PRODUCT-DEFINITION.md` | Highest authority |
| `SPATIAL-CONTRACT.md` | Coordinate, transform, precision and alignment authority |
| `ARCHITECTURE.md` | Technical design |
| `docs/DATA-INVENTORY.md` | Measured facts about the sample data. Never guess where this has a number |
| `RAPIDMESH-REVIEW-FINDINGS.md` | External review. Requirements, not commentary |
| `FINDING-001-PDAL-QUANTISATION.md` | Cairn was quantising imports to 1 cm. Confirmed, and fixed in Cairn 4–5 Aug 2026; pre-fix imports stay quantised. See §9 item 8 |
| `FINDING-002-QA-DEFINITION.md` | The deviation report measures the wrong thing |
| `FINDING-003-GEOMETRIC-TAIL.md` | In the tested fine planar fixture, injected noise dominates the p99.9 tail; this is not a universal zero-error claim. Isolation matrix run 2 Aug 2026 |
| `REVIEW-CAIRN-MESHING.md` | What Cairn's mesher does |
| `CAIRN-MESH-MEMORY-ISSUE.md` | The production defect phase 0 fixes |
| `docs/adr/ADR-001` … `ADR-008` | Decisions taken |
| `docs/HANDOVER.md` | Cold-start brief |

## 8. Known traps — all confirmed in the real files

| Trap | Consequence if ignored |
|---|---|
| MGA northing in f32 | ULP is 0.5 m against a 25 mm tolerance. f64 until localised |
| Do **not** narrow `mesher.py`'s f64 xyz | Deliberate. Fix memory by not materialising world cartesian |
| Sample IFC `IfcSite` points at **London** | Revit default. Georeference from the placement chain only |
| Sample IFC is in **millimetres** | E57s are metres |
| 4 of 30 stations have **no RGB** | Colour is optional |
| `images2D` empty on all 31 sample E57s | No panoramas anywhere. Texture from scan RGB only |
| Deviation report includes island-culled points | Reports 50.84 mm RMS on data that is actually fine. See `FINDING-002` |
| No decimation yet, so point-to-mesh is trivially zero | The metric proves nothing on real data until reworked |
| `read_scan_raw` materialises whole scans | 1,418 MB for one 14.5 M-point station |
| A full-res station mesh is **628.9 MB of output alone**; the site is 397 M tris / 10.21 GB | Full resolution is an intermediate, never a deliverable. Decimation is structural. See `docs/adr/ADR-006`. **DEC-004 adds an audience rule on top of the size one:** the full-resolution tier is surveyor/admin QC only, client audiences get decimated error-bounded LODs, and per-vertex source identity never leaves the server |
| Removing a frontend button does not disable a route | `build_meshes` and `build_vantage_meshes` are still callable directly |
| The 3 scan datasets do **not** pair up | See `docs/DATA-INVENTORY.md` §5 before designing any test |

## 9. Open items and blockers — update this

| # | Item | Blocks |
|---|---|---|
| 1 | **Envelope done; the blocker moved.** The evidence envelope now carries source digest, package and library versions, metric/contract versions, settings, exclusions, processing time, measured peak RSS and the `_resolve_frame` path taken (`src/rapidmesh/evidence.py`, 23 tests). **What blocks a real-data fidelity claim is now ITEM-018, not the envelope:** ITEM-015 is DIAGNOSED — forward QA reads 0.000 mm by construction on undecimated data and reverse QA is bounded below by triangle size, so **neither direction is a fidelity measurement on this data**. Lead register: `OPEN-ITEMS.md` ITEM-015 (DIAGNOSED), ITEM-018 (OPEN) | Any real-data fidelity claim — **which is therefore not licensed** |
| 2 | **Streamed path built and measured.** Band-local filtering, triangulation, QA and writing run through the two-pass streamed pipeline; `mesh_station` is the equivalence reference, not the production path. **The memory gate is NOT met.** The former "30/30 at `band_rows=56` under Rule G1-M" result was measured on F1 geometry and is withdrawn; on corrected geometry every station measured exceeds the working budget, and the high-resolution class exceeds peak RSS too — see §5 phase 1 and ITEM-022. This is now the Phase 1 blocker, and it is an **engineering** problem: no further measurement can close it | **Gate 1 line 5, and therefore Phase 1.** Also unresolved: Rule G1-M's spread S was measured on F1 geometry and needs re-measuring. The 309-file corpus is **not** this set and is unmeasured |
| 3 | Surveyor/admin QC comparison engine does not exist | Later phase 5 QC |
| 4 | No model for site 02516.182 | Mode B validation |
| 5 | Observation-selection rule for site-level comparison undecided | Phase 5 |
| 6 | NavVis raw has no point cloud, no georeferencing, unstitched panoramas | Phase 6 |
| 7 | Station-to-station registration residuals not supplied | Any future fused display surface |
| 8 | **CONFIRMED AND FIXED IN CAIRN** — no longer blocking new imports. Cairn confirmed PDAL's 0.01 default was in force and moved `e57_to_laz` to scale 0.001 with `offset_*=auto` (`backend/converter.py`; in-code comment dated 4 Aug 2026, landed in commit `59dbe02`, 5 Aug 2026). FINDING-001 proposed 0.0001; Cairn chose 0.001 to match PotreeConverter's own 1 mm octree floor, so the sidecar is no longer the limiting term. **Residual:** any project imported before that change is permanently quantised in its LAZ and needs reimport from the retained raw E57 | Phase 2 baseline interpretation **for pre-fix imports only**. A baseline drawn from a pre-fix LAZ is against a handicapped opponent and must say so; a post-fix import is not |
| 10 | Per-station output scales package size with station count, not site area | Browser first paint and package-size targets. Measure at phase 4 |
| 11 | Scanner identity and registration report for 02516.182 not obtained | Any statement about minimum defensible tolerance for the reference data. Phase 2. **Now the whole story, not one input among several**: `FINDING-003`'s isolation matrix showed RapidMesh's own geometric error is ≈0 mm and the entire reported tail is propagated instrument noise — the 2 mm sigma is a placeholder, and the real figure is exactly what this item is waiting on |
| 12 | Combination method for minimum defensible tolerance undefined | The computed project-specific floor. Different statistics, different confidence levels; do not simply add them |
| 13 | LAS/LAZ reconstruction path not selected; neither format carries E57's native lattice | Required input parity in phase 2 |
| 14 | Combined-project representation: coordinated per-station display, fused display-only surface, or both. **Scheduled — investigated inside phase 3** (DEC-007, project-lead decision 15 Aug 2026: a display-only site coarse tier is pre-approved if phase 3's measurements show per-station LOD0s cannot meet the 1 s site first paint; it never produces numbers and never replaces per-station evidence) | Phase 3 measurement, then phase 4 architecture and acceptance tests |

**Supervision-level items live in the project-lead register**, not here:
`E:\Claude Projects\Lead AI SW Eng\RapidMesh\OPEN-ITEMS.md`. Currently open
against Gate 1:

| Item | Subject | Status |
|---|---|---|
| ITEM-015 | Reverse QA reports metre-scale figures | **DIAGNOSED** 2 Sep 2026 — the metric is a triangle-size statistic, not an error figure. Not broken; misread |
| ITEM-018 | No working fidelity metric on undecimated real data | **OPEN** — blocks the Gate 1 fidelity line. Needs John's ruling plus a work package; more runs of the current metrics cannot close it |
| ITEM-019 | Gate 1 line 6's "both reference resolutions" is undefined | **RESOLVED** 2 Sep 2026 — John adopted Reading A: the two `docs/DATA-INVENTORY.md` §1.1 classes. The **definition** stands; the *evidence* first cited under it was measured on F1 geometry and is withdrawn |
| ITEM-020 | A memory test is sensitive to machine load | **OPEN** — advisory. Never run the suite concurrently with other work on the host |
| ITEM-021 | **F1** — the streamed path rewrote every real station into a frame ~5.8 × 10⁶ m from the scanner | **FIXED** in `d681a4a`; **verification incomplete**. Every station re-measured since records `spherical-is-local-by-definition`, terrestrial ranges, non-zero despeckle and FINDING-002-class triangle counts, but the 30-station campaign has not finished |
| ITEM-022 | **Gate 1 line 5 fails on real data** | **OPEN — the Phase 1 blocker.** 6.2× the working budget and 2.1× the peak RSS budget on the 14.5 M-point station. Needs the ADR-006 per-tile vertex lifecycle, preceded by stage attribution. **Budgets are not to be widened** |

## 10. Reference data

`H:\Sample` — client data, read-only, **never committed**.

```
Structured\02516.182_{1..30}.e57                        TLS, 30 stations, 5.3 GB
Navvis e57\25199_Ampol_Tallarook_250501-registered.e57  1.77 GB, 56,950,017 pts
Navvis raw\2025-09-11_01.03.56\                         3.0 GB, rec-v4
Model\25199S - Ampol Tallarook Southbound.ifc           12.2 MB, IFC2X3
```

## 11. What not to do

- Do not quote synthetic-fixture numbers as real-data results.
- Do not attribute a cause without isolating the stage — and isolate it at the
  sampling resolution the claim is about. The p99.9 tail was nearly blamed on
  grazing incidence from one per-feature line; the first measurement then
  eliminated noise and carving, but ran at a coarser sampling than the finding
  specifies, so binning error swamped the noise signal. Re-run at the specified
  fine sampling, the picture inverted: **the cause is propagated instrument
  noise, and RapidMesh's own geometric contribution is ≈ 0 mm on that fixture**
  (`FINDING-003-GEOMETRIC-TAIL.md`). The lesson stands — the right answer came
  from isolating the stage, at the right resolution, not from the first
  plausible story.
- Do not describe the fixture's 2 mm sigma as a property of 02516.182. It is a
  documented test parameter. The instrument is unknown.
- Do not say `RLIMIT_AS` enforces RSS. It bounds virtual address space.
- Do not feed a display mesh to the comparison engine (`docs/adr/ADR-007`).
- Do not claim RapidMesh already matches or beats TurboMesh. It is *intended*
  to. Benchmark evidence first (`00-PRODUCT-DEFINITION.md` §8A, §8B).
- Do not build decimation before the spatial foundation and QA rework.
- Do not materialise a full-resolution mesh in memory.
- Do not touch Cairn; it is read-only until integration is separately approved.
- Do not examine, copy or reverse-engineer Cintoo or TurboMesh.
- Do not write documentation instead of code. Every phase ends with a
  measurement, not a report about a measurement.
