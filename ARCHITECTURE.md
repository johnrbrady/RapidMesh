# RapidMesh — architecture

Read `00-PRODUCT-DEFINITION.md`, then `SPATIAL-CONTRACT.md`. This file explains
how the code is put together and why each piece is where it is.
`REVIEW-CAIRN-MESHING.md` is historical Cairn comparison evidence, not an
integration instruction.

**Revised 15 Aug 2026.** Spatially locked multiresolution output is now first;
E57, LAS and LAZ are required; private QC comparison moves later. The existing
native-lattice design remains the structured-E57 front half.

**Revised 2 Aug 2026.** The five decisions below are unchanged and remain the
core of the design. Added: real-data measurements (§Measured performance),
coordinate rules, the comparison engine, the NavVis pipelines, and process
isolation.

---

## Shape of the whole system

```
  TLS structured E57 ──► Pipeline A   native lattice
                              │
  LAS / LAZ / unstructured ─► B       evidence-preserving reconstruction
                              │
  NavVis registered/raw ────► C       supplied normals / raw sweeps
                              │
                  ┌───────────▼────────────┐
                  │  Shared back half      │
                  │  QA · decimation · LOD │
                  │  RMX                   │
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │ RMX + alignment view   │  spatial lock first
                  └───────────┬────────────┘
                              │ later
                  ┌───────────▼────────────┐
                  │ Private QC comparison  │  surveyor/admin only
                  └────────────────────────┘
```

Meshing is currently per station. The combined-project display representation
is open for investigation under `ADR-008`; later QC remains site-level and
observation-backed under `ADR-005` and `ADR-007`.

---

## The pipeline

```
  structured E57
        |
        v
  e57_reader.read_scan ........ scanner's own lattice, no binning
        |
        v
  grid.ScanGrid.build ......... CSR row index, band-addressable
        |
        v
  filters.clean ............... isolation despeckle -> occlusion carve
        |                       -> parallax restore
        v
  triangulate.triangulate ..... quads -> triangles, discontinuities cut
        |
        v
  triangulate.cull_islands .... drop fragments below an AREA threshold
        |
        v
  [decimate] .................. NOT YET BUILT — quadric, deviation-budgeted
        |
        v
  triangulate.build_mesh ...... full pose -> project-axis offsets, oriented
        |                       normals, untouched colour
        |
        v
  qa.deviation_report ......... the number that ships with the mesh
```

`pipeline.mesh_station` runs the lot. `tools/bench_synthetic.py` runs it against
analytic ground truth and prints the scorecard.

---

## Module map

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `types.py` | Dataclasses. No numpy at runtime. | — |
| `e57_reader.py` | Structured E57 -> `StructuredScan`. Lattice and full-pose handling. | pye57 |
| future format adapters | LAS/LAZ and unstructured E57 -> observation/surface input | format-specific |
| `synthetic.py` | Ray-cast fixtures with analytic ground truth. | — |
| `grid.py` | `ScanGrid` (CSR bands) and `CoarseRangeGrid` (carving). | types |
| `filters.py` | Despeckle, carve, parallax restore. | grid |
| `triangulate.py` | Quad splitting, discontinuity cut, islands, normals. | grid, filters |
| `qa.py` | Point-to-mesh distance, filter scoring. | types |
| `pipeline.py` | Orchestration, timings, stats. | all of the above |

Dependencies point one way. `types.py` imports nothing; `pipeline.py` imports
everything. Nothing imports Cairn, in either direction, until integration is
approved.

Every numpy/scipy/pye57 import is **inside a function**, never at module scope.
That is not stylistic — it is what lets `rapidmesh.deps()` report what is
missing instead of the package failing to import at all, which is the
difference between a useful error message on a client's server and a stack
trace.

---

## The five decisions worth understanding

### 1. Native lattice in, decimation last

The scanner's row/column indices are read straight out of the E57 and used as
the mesh connectivity. No binning, no projection, no resolution constant
anywhere in the code path.

Cairn does the opposite — `GRID_W, GRID_H = 2048, 1024` on line 47 of
`mesher.py`, applied before any geometry is examined — and every quality
limitation downstream follows from it. Inverting the order is the entire
premise of this project.

Consequence: decimation becomes the *only* place quality is traded for size,
so it can be governed by a measured error budget rather than a guess.

### 2. Bands, not dense grids

A native lattice is up to 20000 x 10000. A dense cell->sample index over that
is 800 MB and a dense range grid is another 800 MB, per scan.

So the lattice is never materialised. `ScanGrid` holds a CSR row pointer over
samples already sorted row-major, and any stage that needs neighbourhoods asks
for a **band** — a small dense array of a few hundred rows. Peak memory is
O(band_rows x cols), independent of scan height.

Band overlap is a parameter, not an assumption, because the two consumers need
different amounts: triangulation reads a cell and the row below it (overlap 1),
despeckle reads a 3x3 neighbourhood (halo 1, non-overlapping judged ranges). A
band boundary with the wrong overlap leaves a one-row seam of missing triangles,
which is subtle enough to ship by accident.

### 3. Filters must justify themselves

Two filters are implemented. A third was deliberately not built.

**`isolation_despeckle`** judges *support*, not deviation from a local median.
A real surface sample has neighbours on the surface; a flying point or a mixed
pixel does not. The difference from Cairn's median filter is the case that
matters: a handrail sample deviates enormously from its local median (which is
the wall three metres behind) but has excellent support from the rail samples
above and below it. Median filtering removes it; support does not.

**`carve_movers`** uses occlusion: if another station recorded a return
*beyond* a point, nothing solid was there. Geometry, not a threshold. Inherited
from Cairn, which is where the idea is best.

**Single-scan mover detection is not implemented**, and the reasoning is in the
`filters.py` docstring. A moving person and a static fence post produce range
images that are not reliably separable — both are narrow foreground objects
with clean discontinuities on every side. Any threshold that removes one
removes the other. Distinguishing them needs a second observation, full stop.

### 4. One-vote carving plus parallax restore

Carving has a failure mode Cairn does not handle: a thin object standing off a
wall is genuinely invisible from some angles, so a neighbouring station
legitimately reports the wall behind it and the object gets carved away.
Measured on the fixture, plain one-vote carving loses about 4 % of a 60 mm
handrail.

Requiring two stations to agree fixes that but drops mover recall from 97 % to
78 %. Neither is acceptable.

`restore_parallax_carve` gets both, by using the fact that the two failure
modes look identical point-by-point and completely different in a
neighbourhood:

- a **mover** is carved as a solid blob — its interior samples are surrounded
  by other carved samples;
- a **parallax loss** is scattered nicks inside an otherwise intact surface.

So carve with one vote, then restore any carved sample with five or more
surviving, range-agreeing neighbours. Applied once, never iterated —
iterating lets restoration creep inward from a mover's boundary one ring per
pass and eventually rebuilds the whole object.

Measured, at 0.18 deg sampling:

| | recall | false pos | handrail survives | column survives |
| --- | --- | --- | --- | --- |
| 1 vote (Cairn's rule) | 96.8 % | 0.069 % | 96.2 % | 99.0 % |
| 2 votes | 78.3 % | 0.038 % | 100 % | 100 % |
| **1 vote + restore** | **96.6 %** | **0.048 %** | **100 %** | **100 %** |

### 5. Thresholds are physical quantities

Cairn's edge test is `max(0.08, 6.0 * r * step)`. The 6.0 is unexplained, but it
is a grazing-angle limit in disguise: two adjacent samples on a plane at
incidence θ are separated by `r * step / cos θ`, and 1/cos 80° = 5.76.

RapidMesh takes `max_incidence_deg` as the parameter. Same arithmetic, but the
number now means something, can be justified per site, and makes a real problem
visible: a floor scanned from 1.6 m is past 85° incidence by 20 m out, which is
*beyond* Cairn's implicit limit, so distant floors disintegrate. That is a
predictable consequence of the constant, and it is not findable while the
constant is 6.0.

Same treatment elsewhere: island culling uses square metres rather than a
triangle count (150 triangles means a large object at 2048x1024 and a speck of
dust at native resolution — the same constant cannot mean both), and the carve
grid's resolution tracks the scan's own rather than sitting at a fixed
2048 x 1024.

---

## Coordinate and precision rules

`SPATIAL-CONTRACT.md` is authoritative. The implementation summary is:

| Rule | Reason |
|---|---|
| Project coordinates stay f64 until localised to a station or tile origin | Far-from-origin f32 coordinates cannot preserve millimetres |
| f32 only for offsets from a local origin | At ±100 m, f32 resolution is ~8 µm. Ample |
| Rotation/localising subtraction happens in f64 | Casting first reintroduces the error it avoids |
| Stored mesh offsets use project axes | `v = float32(local @ R.T)` and `project = origin + v`; a renderer does not apply `R` again |
| Source pose remains provenance | Losing rotation makes a locally correct mesh wrong against Cairn |
| Units are metres internally, declared explicitly | The sample IFC is in **millimetres**; the E57s are in metres |
| Georeference from the IFC placement chain only | The sample model's `IfcSite` declares 51°30'23"N 0°07'37"W — **London**, Revit's untouched default. IFC2X3 has no `IfcMapConversion` |
| RGB is optional | 4 of 30 sample stations carry none |

The same rule applies to Cairn's `mesher.py`: its f64 xyz is deliberate, and
narrowing it to halve memory reintroduces exactly this cancellation. Fix memory
by not materialising world cartesian at all.

---

## Measured performance — synthetic

Three-station room fixture, single-threaded NumPy on this machine, station A
meshed with two neighbours carving:

| sampling | samples | time | RMS vs truth | p99.9 | mover recall | false pos |
| --- | --- | --- | --- | --- | --- | --- |
| 0.300° | 380 k | 1.3 s | 1.76 mm | 19.5 mm | 96.8 % | 0.304 % |
| 0.225° | 634 k | 1.5 s | 1.46 mm | 14.0 mm | 96.6 % | 0.190 % |
| 0.180° | 990 k | 2.3 s | 1.26 mm | 4.83 mm | 97.0 % | 0.065 % |
| 0.129° | 1.94 M | 4.9 s | 1.16 mm | 4.49 mm | 97.0 % | 0.100 % |
| 0.090° | 3.96 M | 10.0 s | **1.00 mm** | **3.55 mm** | 97.0 % | **0.015 %** |

Two things to read from this.

**Accuracy improves monotonically with resolution and converges below the
instrument noise.** The fixture has 2 mm range sigma, and the mesh reaches
1.00 mm RMS against analytic truth, because a triangle's surface averages three
noisy vertices and cancels part of the error. This is the thesis of the project
in one column: more input resolution is not just more detail, it is more
accuracy.

**A Trimble X7 at high resolution samples at about 0.017°**, five times finer
than the bottom row. Real data sits well beyond the right of this table, and
Cairn's fixed grid sits at roughly 0.176° — the *third* row.

Throughput is about 400 k samples/second end to end. A 100 M-point scan is
therefore ~4 minutes, which is acceptable for a background job and is the point
at which porting the hot kernels to Rust starts to pay.

**Every figure in this section is against analytic ground truth on synthetic
fixtures.** Read the next section before quoting any of it as a real-data
result.

---

## Measured performance — real data, first run, 2 Aug 2026

`H:\Sample\Structured`, reading only:

| Station | Points | Lattice | Read | Peak RSS | Measured step |
|---|---|---|---|---|---|
| `02516.182_6.e57` | 2,951,950 | 2746 × 1075 | 1.1 s | 356 MB | 0.1311° |
| `02516.182_1.e57` | 14,548,765 | 6095 × 2387 | 6.9 s | **1,418 MB** | 0.0591° / 0.0589° |

All 30 structured stations classify as `e57-rowcol`, the exact tier. The NavVis
export correctly degrades to `e57-projected`. The reader's measured angular
step matches the figure derived independently from the E57 XML footer.

Full pipeline on the medium station: 10.4 s, **1,747 MB peak**, 2,921,532
vertices, 5,760,953 triangles.

**Two conclusions, both load-bearing.**

**Memory.** `pye57.read_scan_raw` materialises the whole scan, as
`e57_reader.py`'s own docstring warns. 1.4 GB for a 14.5 M-point station
against a 512 MB target. Chunked reading is a near-term gate, not a later item.

**The deviation number is currently meaningless on real data.** It reports
50.84 mm RMS, but 99.5% of points sit at exactly 0.000 mm and the RMS is
produced entirely by island-culled samples measured against a mesh that
correctly excludes them. Triangle edges max at 0.51 m in 5.76 million, so
bridging is not the cause. Full diagnostic, with the distribution and the
line of code responsible, in **`FINDING-002-QA-DEFINITION.md`**.

Underneath that: with no decimation, every retained sample *is* a mesh vertex,
so point-to-mesh distance is trivially zero. The metric will not mean anything
on real data until either decimation lands or the mesh-to-source direction is
implemented. The latter is cheaper and comes first.

---

## Private QC comparison engine — not built

A later surveyor/admin-only capability (`00-PRODUCT-DEFINITION.md` §3A).
None of it exists in the codebase today.

**Model import.** IFC2X3 and IFC4, tessellated to triangles in f64, localised
to a site origin, BVH over the result. Preserve `GlobalId`, element type, name,
category and storey so every result attributes back to an element. The sample
model is ~600 products and 58,382 faces; BVH cost is not a concern at that
scale.

**Scope is site-level.** All stations are one evidence set. For each model
surface sample the supporting observation is selected across every station by a
documented rule, and the report records which station supplied it. Per-station
comparison cannot compute the grey no-coverage state correctly, because every
station fails to see most of the site. See `docs/adr/ADR-005`.

| Mode | Method | Applies to | Status |
|---|---|---|---|
| **A** Nearest-surface | Point → nearest model triangle, BVH | All scan data | Primary |
| **B** Scanner-ray | azimuth/elevation *is* the ray, range *is* the measurement | Structured only | Nearly free on these files. Cannot be validated end to end until a model exists for site 02516.182 |
| **C** Observation-ray | Per-point sensor pose and direction | NavVis **B2 only** | Deferred |
| **D** Model-to-scan coverage | Sample model surface, seek supporting observations | All | Required for grey, and for colouring the model |

Sign is computed where topology and normals make it reliable, stored with a
confidence flag, shown on click. It does not drive colour.

Before any result is produced: check coordinate systems, units, transforms,
bounding-box overlap, gross misalignment, model scale, topology and normal
quality. A failed check blocks the result and names the failure. It never
silently adjusts anything.

---

## NavVis — two pipelines, not one with a flag

| | **B1 registered export** | **B2 raw recording** |
|---|---|---|
| Points | Supplied, with normals | **Must be accumulated from raw sweeps** |
| Rays, trajectory, timestamps | Absent | Present |
| Imagery | Absent | 196 unstitched DNG |
| Georeferencing | MGA Zone 55 | **Absent** |
| Modes | A, D | A, C, D |

B1 tiles by octree and reconstructs from the vendor-supplied normals, which
removes the slowest and least reliable stage of point-based reconstruction. B2
decodes 34 ROS laser bags, applies `sensor_frame.xml` extrinsics, interpolates
the SLAM trajectory, and accumulates while retaining per-point timestamp,
sensor head, origin and ray direction.

Neither is trajectory-aware on B1, because the registered export has no
trajectory. Mover removal is **not claimed** for B1: reliable transient
detection needs timestamps or repeated observation with view direction, and
neither exists there. Contrast Pipeline A, where cross-station occlusion
carving is implemented and measured.

Reasoning in `docs/adr/ADR-003` and `docs/adr/ADR-004`.

---

## Process model

The AWS incident (`CAIRN-MESH-MEMORY-ISSUE.md`) was caused by meshing running
inside the uvicorn worker. The rule that follows:

> **All meshing, reconstruction and comparison work runs in a child process
> with its own memory limit. An OOM kills that child and nothing else.**

| Control | Value |
|---|---|
| Isolation | `subprocess`, matching the existing PDAL / PotreeConverter pattern |
| Memory limit | `RLIMIT_AS` in the child, below the container ceiling. **Per workload, not one global value** |
| Pre-flight ceiling | Size and point count checked before spawning |
| Failure semantics | Job fails; scan stays converted; explicit failure status on the project; **no automatic retry loop**; `/api/health` keeps answering 200 |

**The three Cairn paths are different workloads and need different limits.**

| Path | Workload | Implication |
|---|---|---|
| `conversion.run_conversion` | One station | Smallest limit. Currently disabled and stays disabled |
| `routers/models.py build_meshes` | **Every scan in a project, loaded simultaneously** | Limit scales with project size, or the design changes to sequential |
| `routers/models.py build_vantage_meshes` | May load an entire registered cloud before creating vantage points | Largest limit. This is the path the Ampol file would exercise |

A single global limit either strangles the small path or fails to protect
against the large one.

### Endpoint safety comes before isolation

Isolation is real engineering and takes time. The two remaining routes are a
production availability risk **now**, and their frontend buttons being removed
does not protect them — they are still reachable by direct call.

**First action: disable or authorise-gate `build_meshes` and
`build_vantage_meshes` at the server**, with a test proving an unauthorised
direct call cannot reach `mesher.py`. Then build isolation, then re-enable
behind it.

---

## Streamed assembly — nothing full-resolution is materialised

Measured (`docs/adr/ADR-006`): a finished full-resolution mesh for one
high-resolution station is **628.9 MB of output alone**, and the whole sample
site is **397 M triangles, 10.21 GB**.

**A full-resolution native mesh is a processing intermediate. It is never a
deliverable.** Decimation is therefore structural, not a tuning knob.

The band interface in `grid.py` already makes memory independent of scan
height. That property extends through the whole pipeline:

```
read band -> filter -> triangulate -> QA -> decimate -> write tile -> release
```

- Triangulation, QA and decimation all become band-local. Cross-band
  correctness needs the same overlap discipline already documented above:
  triangulation overlap 1, despeckle halo 1. The wrong overlap leaves a one-row
  seam, subtle enough to ship by accident.
- Output is written incrementally as spatial tiles, never assembled then saved.
- Error-bounded decimation runs per band or per tile with documented boundary
  reconciliation. A global quadric pass over 24.5 M triangles is precisely what
  this design prevents.

Two budgets, separately tested on the largest real station:

| Budget | Value |
|---|---|
| Processing working memory, excluding incrementally written output | **≤ 512 MB** |
| Peak process RSS, measured by the RSS watchdog | **≤ 1.5 GB** |

Memory is recovered by chunking, streaming, tiling and removing verified
duplicate allocations. **Never** by narrowing world coordinates to f32.

---

## Not built yet, in priority order

**Reordered 15 Aug 2026 under ADR-008.** Spatial truth and a Cairn-aligned
multiresolution surface precede private QC comparison.

1. **Spatial contract and transform foundation.** Complete rigid pose,
   rebasing, normal transforms and independent axis/rotation/precision tests.
2. **QA rework — metric cores complete.** Retained-surface fidelity excludes
   every final removal, the exclusive filtering ledger balances to the source
   count, restoration is a separate event, and sampled **mesh-to-source
   deviation** detects invented triangle interiors. The evidence envelope
   carries source digest, version, settings, exclusions, timing, peak-memory
   status and lattice provenance. Real-data figures remain gated by streaming.
3. **Streamed E57 processing — ingestion foundation complete, geometry open.**
   libE57 now reads into fixed-capacity chunks and an explicit core/halo
   assembler produces bounded row bands. The current `mesh_station` path still
   materialises scan and mesh; band-local filtering, triangulation, QA and
   incremental output are the remaining production-memory work.
4. **LAS/LAZ and unstructured-E57 adapters.** Preserve per-axis quantisation,
   provenance and units; select reconstruction by measurement rather than
   pretending a scanner lattice survived.
5. **Quadric decimation with a deviation budget.** The largest remaining
   quality-per-byte win. Everything above produces one vertex per sample, so a
   blank wall costs as much as a cornice. Blocked behind item 1: the budget
   needs a metric that works.
6. **The RMX container and engineering alignment view.** Positions as f32
   offsets from an f64 origin (`SPATIAL-CONTRACT.md`), normals, colour, UVs, an
   LOD chain, and spatial tiling for view-dependent streaming.
7. **Texture baking.** Decouples colour resolution from triangle count, which
   is what makes aggressive decimation viable without the result looking flat.
   Note that `images2D` is empty on all 31 sample E57s, so scan RGB is the only
   available source today.
8. **Private QC comparison.** IFC import, BVH, modes A/B/D, site-level heat map
   and report for surveyors/admins only. The engine consumes observations, not
   the display mesh.
9. **Silhouette-aware carving.** Carving is least reliable exactly where the
   querying scan has a depth discontinuity, and the residual false positives in
   the synthetic table are concentrated there. Suppressing the test near
   discontinuities should take the false-positive rate down another order of
   magnitude.
10. **NavVis registered, then raw.** Per `docs/adr/ADR-003` and `ADR-004`.
11. **Rust kernels.** Only after the algorithm stops changing. The band
   interface is already the right boundary for it.
