# CLAUDE.md — RapidMesh charter

**Read this first, every session, before doing anything.**

Short on purpose. The detail lives in the documents listed in §7. Keep this
file current: if you complete a phase or resolve a blocker, update §5 and §9 in
the same change.

Last updated: 2 August 2026.

---

## 1. What we are building

RapidMesh converts terrestrial laser scan data into a lightweight, streamable
surface at the scanner's own sampling resolution, and measures an imported BIM
model against the original scan evidence, producing a tolerance heat map and an
auditable numerical report.

**The mesh makes it usable. The comparison report makes it billable.**
Everything exists to make this sentence true and defensible:

> "Across 165 modelled walls, 91.4% of scanned surface sits within ±25 mm of
> the model. These 14 elements are outside tolerance, worst case 87 mm. 6.2% of
> modelled surface was not observed and is excluded from these figures."

## 2. The one design decision everything follows from

**Mesh at the scanner's native lattice, then decimate to a measured tolerance.
Never decimate first.** Cairn bins to a fixed 2048×1024 grid before it makes a
single triangle. Measured on the real sample data: native is **6.94× the cells
on high-resolution stations, 1.41× on medium**. Quote it that way; a blanket
"6.9×" is not supportable.

## 3. Two scan families, one back half

| | Pipeline A — **TLS, primary** | Pipeline B — NavVis |
|---|---|---|
| Input | Structured terrestrial E57 | B1 registered E57 · B2 raw `rec-v4` |
| Phase | 1–5 | 6 |

TLS is built first and carries the competitive claim. **Meshing is per station.
Comparison and reporting are per site** (`docs/adr/ADR-005`).

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

| Phase | Deliverable | Status |
|---|---|---|
| 0a | **Endpoint safety.** Disable/gate `build_meshes` + `build_vantage_meshes` at the server. Frontend removal is not protection | **NOT STARTED — do this first** |
| 0b | Job isolation: child process, per-workload rlimit, ceiling, no retry loop, remove redundant f64 copy | NOT STARTED |
| 1 | QA rework (3 reports) + streamed/chunked processing | NOT STARTED |
| 2 | Real-data baseline, Cairn vs RapidMesh. Confirm `FINDING-001` first | NOT STARTED |
| 3a | Comparison vertical slice: mode A, one station vs IFC, JSON only | NOT STARTED |
| 3b | Error-bounded decimation, LOD chain, tiled incremental writing | NOT STARTED |
| 4 | RMX, browser first paint, progressive refinement, texture | NOT STARTED |
| 5 | Site-level comparison: modes B + D, heat map, full report | NOT STARTED |
| 6 | NavVis B1, then B2 | NOT STARTED |
| 7 | Cairn integration | NOT STARTED |

**Already built and working:** structured E57 reader (three lattice tiers),
band-addressable native lattice, edge-preserving despeckle, cross-station
occlusion carving with parallax restore, discontinuity-aware triangulation,
area-based island culling, oriented normals, synthetic fixtures, CLI.
21 tests pass.

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
ever need to meet.** Fixed acceptance test.

## 7. Documents, in authority order

| Document | Role |
|---|---|
| `00-PRODUCT-DEFINITION.md` | Highest authority |
| `ARCHITECTURE.md` | Technical design |
| `docs/DATA-INVENTORY.md` | Measured facts about the sample data. Never guess where this has a number |
| `RAPIDMESH-REVIEW-FINDINGS.md` | External review. Requirements, not commentary |
| `FINDING-001-PDAL-QUANTISATION.md` | Cairn likely quantising imports to 1 cm |
| `FINDING-002-QA-DEFINITION.md` | The deviation report measures the wrong thing |
| `REVIEW-CAIRN-MESHING.md` | What Cairn's mesher does |
| `CAIRN-MESH-MEMORY-ISSUE.md` | The production defect phase 0 fixes |
| `docs/adr/ADR-001` … `ADR-006` | Decisions taken |
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
| A full-res station mesh is **628.9 MB of output alone**; the site is 397 M tris / 10.21 GB | Full resolution is an intermediate, never a deliverable. Decimation is structural. See `docs/adr/ADR-006` |
| Removing a frontend button does not disable a route | `build_meshes` and `build_vantage_meshes` are still callable directly |
| The 3 scan datasets do **not** pair up | See `docs/DATA-INVENTORY.md` §5 before designing any test |

## 9. Open items and blockers — update this

| # | Item | Blocks |
|---|---|---|
| 1 | QA measures island-culled points; no mesh-to-source direction | Any real-data fidelity claim |
| 2 | Chunked E57 reading not implemented | 512 MB target, production scale |
| 3 | Comparison engine does not exist | The billable deliverable |
| 4 | No model for site 02516.182 | Mode B validation |
| 5 | Observation-selection rule for site-level comparison undecided | Phase 4 |
| 6 | NavVis raw has no point cloud, no georeferencing, unstitched panoramas | Phase 6 |
| 7 | Station-to-station registration residuals not supplied | Any future fused display surface |
| 8 | FINDING-001 unconfirmed against a real Cairn LAZ | Phase 2 baseline interpretation. If Cairn quantises to 1 cm, the baseline is against a handicapped opponent and must say so |
| 9 | Synthetic p99.9 is 3.55 mm against a 3.2 mm budget at the 10 mm floor | The acceptance test at the tolerance floor. Passes at 25 mm default (budget 8.0 mm) |
| 10 | Per-station output scales package size with station count, not site area | Browser first paint and package-size targets. Measure at phase 4 |

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
- Do not claim RapidMesh already matches or beats TurboMesh. It is *intended*
  to. Benchmark evidence first (`00-PRODUCT-DEFINITION.md` §8A, §8B).
- Do not build decimation before the QA rework.
- Do not materialise a full-resolution mesh in memory.
- Do not touch Cairn beyond the phase 0 defect fix.
- Do not examine, copy or reverse-engineer Cintoo or TurboMesh.
- Do not write documentation instead of code. Every phase ends with a
  measurement, not a report about a measurement.
