# HANDOVER.md — RapidMesh cold start

Opening brief for an implementation session. Assumes no prior context.

> **Resume notice (15 August 2026):** John explicitly resumed RapidMesh as a
> separate project. Cairn remains read-only and integration remains deferred.
> Read `SPATIAL-CONTRACT.md` and start at Phase 0c in `CLAUDE.md`; the older
> phase narrative below remains useful historical context where it does not
> conflict with `ADR-008`.

Written 2 August 2026, after the first run of RapidMesh against real client
data and the architecture review that followed.

---

## 0. Paste this to start a session

```
Read these in full before doing anything, in this order:

  CLAUDE.md
  docs/HANDOVER.md

They are the charter and the cold-start brief. The charter names the remaining
documents and the order of authority. Re-read CLAUDE.md at the start of every
session.

Cairn repo (phase 0 only):  [PATH TO pointcloud-viewer]
Reference data:             H:\Sample   read-only, never committed

RapidMesh has resumed outside Cairn. Start at Phase 0c in CLAUDE.md and keep
Cairn read-only until integration is explicitly approved.
```

---

## 1. Read order

| # | Document | Why |
|---|---|---|
| 1 | `CLAUDE.md` | Charter. Every session |
| 2 | `00-PRODUCT-DEFINITION.md` | Highest authority |
| 3 | `SPATIAL-CONTRACT.md` | Coordinate, precision and alignment authority |
| 4 | `ARCHITECTURE.md` | How the code is put together and why |
| 5 | `docs/DATA-INVENTORY.md` | Measured facts about the real sample files |
| 6 | `FINDING-002-QA-DEFINITION.md` | Why phase 1 is what it is |
| 7 | `FINDING-003-GEOMETRIC-TAIL.md` | The p99.9 isolation matrix |
| 8 | `RAPIDMESH-REVIEW-FINDINGS.md` | External review. Requirements, not commentary |
| 9 | `docs/adr/ADR-001` … `ADR-008` | Decisions taken, with reasoning |
| 10 | `REVIEW-CAIRN-MESHING.md`, `FINDING-001-PDAL-QUANTISATION.md`, `CAIRN-MESH-MEMORY-ISSUE.md` | Cairn history and reference only |

**Where the code disagrees with these documents, report the conflict and
stop.** Do not resolve it silently.

---

## 2. What this project is

E57, LAS or LAZ in; a lightweight streamable multiresolution surface out,
spatially locked to the original Cairn point cloud. Private surveyor/admin QC
comparison comes later and is never client-facing.

Two scan families. **TLS is primary** and carries the competitive claim; NavVis
is second. **Meshing is per station; comparison and reporting are per site**
(`ADR-005`).

Destination: separately approved future integration into Cairn, with old
projects preserved. Until then RapidMesh neither imports Cairn nor is imported
by it, and Cairn is read-only reference material.

Cintoo and TurboMesh are a **public capability benchmark only**. Nothing is
copied or examined. We may say RapidMesh is *intended* to compete; we may not
say it does until benchmark evidence exists.

---

## 3. What already exists — this is not a greenfield project

~2,400 lines of working, tested Python. **21 tests pass.** Git history:

```
3ceaa21  Isolate the p99.9 cause, observation-set interface, tile boundaries
da0a8fe  Incorporate review: endpoint safety, memory budget, parity targets
8a5bab4  Merge architecture session: real-data findings, NavVis, comparison
07cccfb  Baseline: RapidMesh as at 1 Aug 2026, before architecture merge
```

```
src/rapidmesh/   e57_reader (637) · synthetic (406) · filters (367) · grid (341)
                 triangulate (291) · qa (266) · types (227) · pipeline (152) · cli (127)
tests/           test_pipeline · test_e57_reader · conftest
tools/           bench_synthetic · check_laz_precision · e57_xml · e57_inventory · smoke
```

**Built and working:** structured E57 reader with three lattice tiers
(`ROW_COL` exact, `SPHERICAL`, `PROJECTED` fallback), band-addressable native
lattice (CSR, memory independent of scan height), edge-preserving despeckle,
cross-station occlusion carving with parallax restore, discontinuity-aware
triangulation, area-based island culling, oriented normals, analytic synthetic
fixtures, CLI (`probe` / `mesh` / `bench`).

**Not built:** LAS/LAZ ingestion, decimation, RMX container, texture, LOD chain,
chunked/streamed processing, the engineering alignment view, and the later
private QC comparison engine.

---

## 4. What the first real-data run found

**Good.** The reader works on real client data. All 30 structured stations
classify as `e57-rowcol`, the exact tier. The NavVis export correctly degrades
to `e57-projected`. The reader's measured angular step (0.0591°) matches the
figure derived independently from the E57 XML footer. `MIN_RANGE = 0.3` drops
14.2% of cells on the outdoor high-res station and 0.5% on the enclosed medium
one, which settles the invalid-return question.

| Station | Points | Lattice | Read | Peak RSS | Step |
|---|---|---|---|---|---|
| `02516.182_6` | 2,951,950 | 2746 × 1075 | 1.1 s | 356 MB | 0.1311° |
| `02516.182_1` | 14,548,765 | 6095 × 2387 | 6.9 s | **1,418 MB** | 0.0591° |

**Three problems, each with its own document.**

**`FINDING-002` — the QA metric is wrong, and on real data it is degenerate.**
Reported RMS is 50.84 mm against a 2 mm commitment. Not a meshing failure:
triangle edges max at 0.51 m in 5.76 million, so nothing bridges. The RMS comes
entirely from island-culled samples measured against a mesh that correctly
excludes them (`pipeline.py:109` passes `grid.scan.xyz`). Underneath that, p50
through p99.5 are all **exactly 0.000 mm**, because with no decimation every
retained sample *is* a mesh vertex.

**`ADR-006` — the memory target was not achievable.** A finished
full-resolution mesh for the high-res station is **628.9 MB of output alone**;
the whole site is **397 M triangles, 10.21 GB**. Chunked reading alone does not
fix this. Full resolution is a processing intermediate, never a delivery
format, and decimation is therefore structural rather than a tuning knob.

**`FINDING-003` — the p99.9 tail is geometric, and the cause is unknown.**
Stated exactly: **passes the 25 mm default-tolerance budget and fails the 10 mm
minimum-tolerance budget** (3.55 mm against 3.2 mm). Measured: at zero range
noise with carving disabled, p99.9 is still 12.89 mm. Noise moves it ~5%;
carving ~5%. Noise and carving are eliminated; **the responsible stage is not
identified.** The whole tail sits in walls and floor; doors, the column and the
handrail are exact.

---

## 5. Phase 0 — Cairn, and only this

Repo: `pointcloud-viewer`, branch `navvis-phase3-vvp-meshing`. Remediation of a
live fault, not integration. Full account in `CAIRN-MESH-MEMORY-ISSUE.md`.

**This is a production availability issue.** The kernel killed uvicorn, taking
down every user's session, not one job.

### 0a — Endpoint safety. Do this first; it is hours, not days.

Automatic meshing was removed from uploads and **stays removed**. Two routes
remain live and reachable by direct call:

- `routers/models.py build_meshes` — loads every scan in a project
  **simultaneously**
- `routers/models.py build_vantage_meshes` — may load an entire registered
  cloud

Their frontend buttons were deleted. **That is not protection.** curl, a test
or a future UI still reaches them.

Disable or authorise-gate both **at the server**. Do not wait for isolation.

**Gate 0a:** an unauthorised direct call to either route returns 403/404,
proven by test. No ordinary request path can reach `mesher.py`.

### 0b — Job isolation

1. **Subprocess-isolate the mesh step**, matching the existing `subprocess.run`
   pattern for PDAL and PotreeConverter in `backend/converter.py`.
2. **Three memory controls, kept distinct.** `RLIMIT_AS` bounds **virtual
   address space and does not enforce RSS** — never write that it does.

   | Control | Bounds | Failure | Role |
   |---|---|---|---|
   | `RLIMIT_AS`, calibrated | Address space | `MemoryError`, catchable | Graceful abort with a traceback and a clean failure status |
   | RSS watchdog on `/proc/self/status` `VmRSS` | Resident | Voluntary exit | The measured gate |
   | Job-specific cgroup | Resident | SIGKILL | Hard boundary. **0b stretch, not entry requirement** |

3. **Limits are per workload, not one global value.** One station, every scan in
   a project, and a whole registered cloud are three different workloads. A
   single limit either strangles the small path or fails to protect the large.
4. **Remove the redundant copy.** `load_points` ends with
   `.T.astype(np.float64)` on an array already f64. NumPy copies by default —
   **1.37 GB of pure waste** on the 56,950,017-point NavVis file. Use
   `copy=False` or drop the cast. **Do not narrow to f32**; the f64 is
   deliberate and load-bearing at MGA magnitudes.
5. **Pre-flight size / point-count ceiling** before spawning.
6. **No automatic retry loop.**

**Gate 0b — all must pass:** the 1.77 GB file fails its own job only;
`/api/health` answers 200 throughout, verified by polling during the run; other
users can view other projects; the source scan remains available; the project
shows an explicit failure status; no retry loop; `dmesg` shows the child killed,
not uvicorn; `RLIMIT_AS` plus the RSS watchdog demonstrably protect uvicorn
under the real container limit; a regression test proven to fail against the
pre-fix code first; full existing gate re-run (pytest, ruff, `mypy --strict`,
Playwright).

Confirm every path against the current repository. Line numbers in these
documents are from 1 Aug 2026 and may have moved.

---

## 6. Phase 1 — start here in this repo

**Do not build decimation.** It was first; it is now 3b, because there is no
trustworthy real-data metric to decimate against.

### 1a — QA rework

Three separate reports (`RAPIDMESH-REVIEW-FINDINGS.md` §1):

- **Retained-surface fidelity** — point-to-mesh over samples actually
  represented in the mesh. Excludes despeckle, carve **and island** removals.
- **Filtering and coverage ledger** — retained, despeckled, carved,
  island-culled, restored, no-return, otherwise excluded. Every input sample in
  exactly one category, summing to the input count.
- **Mesh-to-source deviation** — sample the finished triangles, measure back to
  source points. The only measure that detects invented surface, and the only
  one that says anything on real data before decimation exists. **Build this
  first.**

Every report states its source of truth, whether figures are exact or sampled,
source hash, version, settings, exclusions, processing time and peak memory.

### 1b — Streamed and chunked processing

`pye57.read_scan_raw` materialises whole scans. Chunked *reading* alone is not
enough: the finished mesh is 628.9 MB on its own. The pipeline must stream:

```
read band -> filter -> triangulate -> QA -> write tile -> release band
```

The band interface in `grid.py` is already the right shape. Cross-band overlap
discipline matters: triangulation overlap 1, despeckle halo 1. The wrong
overlap leaves a one-row seam, subtle enough to ship by accident.

Two budgets, tested separately: **working memory ≤ 512 MB** excluding
incrementally written output, and **measured peak RSS ≤ 1.5 GB**.

### 1c — The `FINDING-003` isolation matrix — **DONE, 2 August 2026**

`tools/isolation_matrix.py`, run at the verified fine 0.090°/0.09002° sampling
(1334 × 4000, checked at runtime — the script refuses to run at any other
resolution). All required variables swept: noise 0/1/2/3/5 mm, carving on/off,
parallax restore on/off, island culling on/off, `max_incidence_deg` sweep
70–85°. Walls and floor reported separately throughout; full breakdowns by
incidence angle, range and discontinuity-cell-distance; worst-0.1% coordinates
and triangle IDs; a heat map per condition. 155 MB of results
(`out/isolation_matrix/`, gitignored) plus the full write-up in
`FINDING-003-GEOMETRIC-TAIL.md`.

**Outcome:** none of the first three rows of the decision table held — carving
off, restore off and island off each changed walls/floor by ≤ 0.02 mm;
`max_incidence_deg` was bit-for-bit identical from 78° to 85°; residuals were
flat across every discontinuity-distance bucket. Zero noise gave **exactly**
0.00 mm on both walls and floor, and the tail tracked injected noise almost
exactly at every level tested. **RapidMesh's own geometric error is ≈ 0 mm;
the entire reported tail is correctly propagated instrument noise**, not a
mesher defect of any kind. Full reasoning and the decision table applied
against real numbers are in `FINDING-003-GEOMETRIC-TAIL.md`.

This does not close the acceptance-test gap — 3.47 mm still exceeds the
3.2 mm mesher-only budget at the assumed 2 mm noise sigma — but there is
nothing left to fix in the mesher to close it. What was previously
`CLAUDE.md` §9 open item 9 is now folded into item 11: the real instrument
noise figure is the only thing that can move this number.

### 1d — Clean-checkout smoke gate

`tools/smoke.py` exists and passes. Keep it in CI. It catches stdlib shadowing
in `tools/` and tools that will not run from their documented location. It
exists because `tools/inspect.py` shadowed stdlib `inspect` and broke every
script run from `tools/`.

### Gate 1 — all must pass

- [ ] Real-data retained-surface **and mesh-to-source** figures for both
      `02516.182_6.e57` and `02516.182_1.e57`
- [ ] Filtering ledger balances: every input sample in exactly one category
- [ ] Working memory ≤ 512 MB and measured peak RSS ≤ 1.5 GB on the
      14.5 M-point station, asserted in a test
- [ ] Output written incrementally; no full-resolution mesh held in memory
- [x] **Geometric-tail cause identified by stage** (`FINDING-003`) — **done
      2 Aug 2026**: propagated instrument noise, not a mesher defect. See 1c
      above
- [ ] All 30 structured stations process, both resolutions, including the 4
      without colour
- [ ] Coordinate precision test at real MGA Zone 55 values
- [x] `tools/smoke.py` passes from a fresh clone — verified 2 Aug 2026 after
      `pip install -e ".[e57,dev]"` (was not installed in this environment)
- [x] `CLAUDE.md` §5 and §9 updated — this pass

**Then stop and report the numbers.**

---

## 7. After that, in order

Full gates in `00-PRODUCT-DEFINITION.md` §8.

| Phase | Deliverable |
|---|---|
| **2** | E57/LAS/LAZ ingestion parity and a comparable real-data baseline |
| **3** | Error-bounded decimation, LOD chain, tiled incremental writing and tile-size benchmark |
| **4** | RMX, progressive loader, engineering alignment view and texture |
| **5** | Surveyor/admin-only QC comparison vertical slice, then site-level modes B and D |
| **6** | NavVis B1, then B2 |
| **7** | Separately approved Cairn integration |

---

## 8. Reference data

`H:\Sample` — **client data. Read-only. Never committed, at any time.**

```
Structured\02516.182_{1..30}.e57                        TLS, 30 stations, 5.3 GB
Navvis e57\25199_Ampol_Tallarook_250501-registered.e57  1.77 GB, 56,950,017 pts
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

---

## 9. Traps, all confirmed in the real files

1. **MGA northing in f32 has a 0.5 m ULP.** The sample IFC places at
   `5896381927.` mm — twenty times the 25 mm tolerance. f64 until localised.
2. **The sample model's `IfcSite` declares London** (51°30'23"N 0°07'37"W).
   Revit's untouched default. IFC2X3 has no `IfcMapConversion`. Georeference
   from the placement chain only.
3. **The sample IFC is in millimetres.** The E57s are in metres.
4. **4 of 30 stations have no RGB** (files 6, 7, 8, 14).
5. **`images2D` is empty on all 31 sample E57s.** No panoramas anywhere.
6. **Scanner make and model are not recoverable** from these headers. Do not
   assert Trimble or Faro on the basis of this dataset.
7. **The fixture's 2 mm sigma is a test parameter**, not a property of
   02516.182. The instrument is unknown.
8. **Synthetic numbers are not real-data numbers.** `README.md`'s 1.00 mm RMS
   is against analytic truth on fixtures.
9. **`RLIMIT_AS` does not enforce RSS.** Never write that it does.
10. **Never feed a display mesh to the comparison engine** (`ADR-007`).

---

## 10. Rules of engagement

1. **Measure before claiming.** A number and the command that produced it.
2. **Do not attribute a cause without isolating the stage.** The p99.9 tail was
   nearly blamed on grazing incidence from a single per-feature line.
3. **New tests must be proven to fail against the old code first.**
4. **Report conflicts, do not resolve them silently.**
5. **No deferred capability may be claimed.** Name the missing data instead.
6. **No client data in the repository. Ever.**
7. **Python only.** No Rust or C++ until a hotspot is proven by measurement.
8. **Do not touch Cairn beyond phase 0.**
9. **Do not write documentation instead of code.** Every phase ends with a
   measurement, not a report about a measurement.
10. **Update `CLAUDE.md` §5 and §9** in the change that completes a phase or
    resolves a blocker.
11. Commit in reviewable steps.

---

## 11. Blocked on John

Neither can be resolved from the files. Both gate phase 2.

- **Scanner identity for 02516.182:** make, model, serial, resolution and
  quality settings, calibration state.
- **Registration report:** software and method, station-to-station residuals,
  target and checkpoint residuals, control-network accuracy, max and RMS
  registration error, any excluded or weakly constrained stations.

Until these arrive, no minimum defensible tolerance can be stated for the
reference dataset, and the fixture noise stays a documented assumption.
