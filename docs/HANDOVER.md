# HANDOVER.md — RapidMesh, cold start

Opening brief for an implementation session. Assumes no prior context.

Written 2 August 2026, after the first run of RapidMesh against real client
data.

---

## 1. Read order

1. `CLAUDE.md` — charter. Every session.
2. `00-PRODUCT-DEFINITION.md` — highest authority.
3. `ARCHITECTURE.md` — how the code is put together and why.
4. `docs/DATA-INVENTORY.md` — measured facts about the real sample files.
5. `FINDING-002-QA-DEFINITION.md` — **why phase 1 is what it is.**
6. `RAPIDMESH-REVIEW-FINDINGS.md` — external review. Requirements, not
   commentary.
7. `docs/adr/ADR-001` … `ADR-005`.
8. `REVIEW-CAIRN-MESHING.md`, `FINDING-001-PDAL-QUANTISATION.md`,
   `CAIRN-MESH-MEMORY-ISSUE.md` — Cairn context.

**Where the code disagrees with these documents, report the conflict and
stop.** Do not resolve it silently.

## 2. What this project is

Terrestrial laser scan in, lightweight streamable surface out, plus a numerical
report of how far an imported BIM model sits from the scan evidence. The mesh
makes it usable; **the comparison report is what gets sold**.

Destination: integration into Cairn 3D at phase 7, behind a feature flag,
alongside the existing loader, legacy `.cmh` assets and old projects untouched.
Until then RapidMesh neither imports Cairn nor is imported by it.

## 3. What already exists — this is not a greenfield project

~2,400 lines of working, tested Python. **21 tests pass.** The synthetic bench
reproduces its documented behaviour.

```
src/rapidmesh/   e57_reader (637) · synthetic (406) · filters (367) · grid (341)
                 triangulate (291) · qa (266) · types (227) · pipeline (152) · cli (127)
tests/           test_pipeline · test_e57_reader · conftest
tools/           bench_synthetic · check_laz_precision · e57_xml · inspect
```

**Built and working:** structured E57 reader with three lattice tiers
(`ROW_COL` exact, `SPHERICAL`, `PROJECTED` fallback), band-addressable native
lattice (CSR, memory independent of scan height), edge-preserving despeckle,
cross-station occlusion carving with parallax restore, discontinuity-aware
triangulation, area-based island culling, oriented normals, analytic synthetic
fixtures, CLI (`probe` / `mesh` / `bench`).

**Not built:** decimation, RMX container, texture, LOD chain, chunked reading,
and **the entire comparison engine** — IFC import, BVH, modes A/B/D, heat map,
site-level report. That last group is the billable half of the product.

## 4. What the first real-data run found

Run on 2 Aug 2026. Detail in `FINDING-002-QA-DEFINITION.md`.

**Good.** The reader works on real client data. All 30 structured stations
classify as `e57-rowcol`, the exact tier. The NavVis export correctly degrades
to `e57-projected`. The reader's measured angular step (0.0591°) matches the
figure derived independently from the E57 XML footer. The invalid-return
question is settled: `MIN_RANGE = 0.3` drops 14.2% on the outdoor high-res
station and 0.5% on the enclosed medium one, exactly as expected.

**Bad, and this sets phase 1.**

```
deviation  rms=50.84 mm  p99.9=165.07 mm  max=8724.90 mm  <=2mm=99.63%
```

Against a 2 mm commitment. It is **not** a meshing failure. Triangle edges max
at 0.51 m in 5.76 million, so nothing is bridging. The RMS is produced entirely
by island-culled samples being measured against a mesh that correctly excludes
them — `pipeline.py:109` passes `grid.scan.xyz`, which still contains them.

Underneath that is the deeper issue: **p50, p90, p99 and p99.5 are all exactly
0.000 mm.** With no decimation, every retained sample *is* a mesh vertex, so
point-to-mesh distance is trivially zero. The metric currently asks whether
points are vertices of a mesh built from those points.

**Memory.** 1,418 MB reading one 14.5 M-point station; 1,747 MB for the full
pipeline. Target is 512 MB.

## 5. Phase 0 — Cairn, and only this

Repo: `pointcloud-viewer`, branch `navvis-phase3-vvp-meshing`. Remediation of a
live fault, not integration. Full account in `CAIRN-MESH-MEMORY-ISSUE.md`.

**This is a production availability issue, not a performance issue.** The
kernel killed uvicorn, taking down every user's session, not just one job.

### 0a — Endpoint safety. Do this first; it is hours, not days.

Automatic meshing was removed from uploads and **stays removed**. But two
routes remain live and reachable by direct call:

- `routers/models.py build_meshes` — loads every scan in a project
  **simultaneously**
- `routers/models.py build_vantage_meshes` — may load an entire registered
  cloud

Their frontend buttons were deleted. **That is not protection.** curl, a test,
or a future UI still reaches them.

Disable or authorise-gate both **at the server**. Do not wait for isolation.

**Gate 0a:** an unauthorised direct call to either route returns 403/404,
proven by test. No ordinary request path can reach `mesher.py`.

### 0b — Job isolation

1. **Subprocess-isolate the mesh step**, matching the existing `subprocess.run`
   pattern for PDAL and PotreeConverter in `backend/converter.py`.
2. **`RLIMIT_AS` per workload, not one global value.** The three paths are
   different workloads: one station, every scan in a project, and a whole
   registered cloud. One limit either strangles the small path or fails to
   protect against the large one.
3. **Remove the redundant copy.** `load_points` ends with
   `.T.astype(np.float64)` on an array already f64. NumPy copies by default —
   **1.37 GB of pure waste** on the 56,950,017-point NavVis file. Use
   `copy=False` or drop the cast. **Do not narrow to f32**; the f64 is
   deliberate and load-bearing at MGA magnitudes.
4. **Pre-flight size/point-count ceiling** before spawning.
5. **No automatic retry loop.** A failed oversized job must not respawn itself.

**Gate 0b — all must pass:** the 1.77 GB file fails its own job only;
`/api/health` answers 200 throughout, verified by polling during the run; other
users can view other projects; the source scan remains available; the project
shows an explicit failure status; no retry loop; `dmesg` shows the child
killed, not uvicorn; a regression test proven to fail against the pre-fix code
first; full existing gate re-run (pytest, ruff, `mypy --strict`, Playwright).

Confirm every path against the current repository. Line numbers here are from
1 Aug 2026 and may have moved.

## 6. Phase 1 — start here in this repo

**Do not build decimation.** It was previously first and has been moved to
phase 3, because there is currently no trustworthy real-data metric to
decimate against.

**1a. QA rework.** Three separate reports, per `RAPIDMESH-REVIEW-FINDINGS.md`
§1:

- **Retained-surface fidelity** — point-to-mesh over samples actually
  represented in the mesh. Excludes despeckle, carve **and island** removals.
- **Filtering and coverage ledger** — retained, despeckled, carved,
  island-culled, restored, no-return, otherwise excluded. Every input sample
  accounted for exactly once.
- **Mesh-to-source deviation** — sample the finished triangles, measure back to
  source points. The only measure that detects invented surface, and the only
  one that says anything on real data before decimation exists. **Build this
  one first.**

Each report states which dataset is the source of truth and whether figures are
exact or sampled.

**1b. Streamed and chunked processing.** `pye57.read_scan_raw` materialises
whole scans. But chunked *reading* alone does not solve the problem: a finished
full-resolution mesh for the high-res station is **628.9 MB of output on its
own** (`docs/adr/ADR-006`). The whole pipeline must stream:

```
read band -> filter -> triangulate -> QA -> write tile -> release band
```

The band interface in `grid.py` is already the right shape. Two budgets, tested
separately: **working memory ≤ 512 MB** excluding incrementally written output,
and **peak RSS ≤ 1.5 GB** enforced by `RLIMIT_AS`.

**Gate 1 — all must pass:**

- [ ] Real-data retained-surface **and mesh-to-source** figures for both
      `02516.182_6.e57` and `02516.182_1.e57`
- [ ] Filtering ledger balances: every input sample in exactly one category,
      summing to the input count
- [ ] Working memory ≤ 512 MB and peak RSS ≤ 1.5 GB on the 14.5 M-point
      station, asserted in a test
- [ ] Output written incrementally; no full-resolution mesh held in memory
- [ ] All 30 structured stations process without error, both lattice
      resolutions, including the 4 without colour
- [ ] Coordinate precision test at real MGA Zone 55 values
- [ ] Every report states its source of truth, whether figures are exact or
      sampled, source hash, version, settings, exclusions, time and peak memory
- [ ] **`FINDING-003` isolation matrix run, and the geometric-tail cause
      identified by stage.** Do not reprioritise filtering or triangulation
      work before it
- [ ] Clean-checkout smoke gate: CLI, benchmark and inventory tools all run
      from their documented locations on a fresh clone
- [ ] `CLAUDE.md` §5 and §9 updated

Then stop and report the numbers.

## 6A. After that, in order

Full gates in `00-PRODUCT-DEFINITION.md` §8.

| Phase | Deliverable |
|---|---|
| **2** | Real-data baseline, Cairn vs RapidMesh, same file, same metric. **Confirm or refute `FINDING-001` first** — if Cairn quantises to 1 cm, the comparison is against a handicapped opponent and the report must say so |
| **3a** | Comparison vertical slice: mode A only, one station against the IFC, JSON out, no heat map. Cheap, and it de-risks the commercial premise early |
| **3b** | Error-bounded decimation, LOD chain, tiled incremental writing |
| **4** | RMX, browser first paint, progressive refinement, texture |
| **5** | Site-level comparison: modes B and D, heat map both targets, full report |
| **6** | NavVis B1, then B2 |
| **7** | Cairn integration |

## 7. Reference data

`H:\Sample` — **client data. Read-only. Never committed, at any time.**

```
Structured\02516.182_{1..30}.e57                        TLS, 30 stations, 5.3 GB
Navvis e57\25199_Ampol_Tallarook_250501-registered.e57  1.77 GB
Navvis raw\2025-09-11_01.03.56\                         3.0 GB, rec-v4
Model\25199S - Ampol Tallarook Southbound.ifc           12.2 MB, IFC2X3
```

**The three scan datasets do not pair up.** `docs/DATA-INVENTORY.md` §5 before
designing any test.

| Dataset | Points | Rays / trajectory | Model | Georeferenced |
|---|---|---|---|---|
| Structured 02516.182 | Yes | Implicit in spherical storage | **No** | Yes |
| NavVis E57 25199 | Yes, with normals | No | **Yes** | Yes |
| NavVis raw 25409_S | **No** | Yes | No | **No** |

Phases 0 and 1 touch only the structured set. **Do not open the raw recording
before phase 6.**

## 8. Traps, all confirmed in the real files

1. **MGA northing in f32 has a 0.5 m ULP.** The sample IFC places at
   `5896381927.` mm. Twenty times the 25 mm tolerance. f64 until localised.
2. **The sample model's `IfcSite` declares London** (51°30'23"N 0°07'37"W).
   Revit's untouched default. IFC2X3 has no `IfcMapConversion`. Georeference
   from the placement chain only.
3. **The sample IFC is in millimetres.** The E57s are in metres.
4. **4 of 30 stations have no RGB** (files 6, 7, 8, 14).
5. **`images2D` is empty on all 31 sample E57s.** No panoramas anywhere.
6. **Scanner make and model are not recoverable** from these headers. Do not
   assert Trimble or Faro on the basis of this dataset.
7. **Synthetic numbers are not real-data numbers.** `README.md`'s 1.00 mm RMS
   is against analytic truth on fixtures. There is no real-data equivalent yet.

## 9. Rules of engagement

1. **Measure before claiming.** A number and the command that produced it.
2. **New tests must be proven to fail against the old code first.**
3. **Report conflicts, do not resolve them silently.**
4. **No deferred capability may be claimed.**
5. **No client data in the repository. Ever.**
6. **Python only.** No Rust or C++ until a hotspot is proven by measurement.
7. **Do not touch Cairn beyond phase 0.**
8. **Update `CLAUDE.md` §5 and §9** in the change that completes a phase or
   resolves a blocker.
9. Commit in reviewable steps. The repo has a git baseline as of 2 Aug 2026.
