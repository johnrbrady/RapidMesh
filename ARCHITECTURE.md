# RapidMesh — architecture

Read `00-PRODUCT-DEFINITION.md` first. This file explains how the code is put
together and why each piece is where it is. `REVIEW-CAIRN-MESHING.md` explains
what it is being built against.

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
  triangulate.build_mesh ...... compact, oriented normals, untouched colour
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
| `e57_reader.py` | Structured E57 -> `StructuredScan`. Lattice detection. | pye57 |
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

## Measured performance

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

---

## Not built yet, in priority order

1. **Quadric decimation with a deviation budget.** The largest remaining win.
   Everything above produces a mesh with one vertex per sample, so a blank wall
   costs as much as a cornice. This is where the file-size advantage lives, and
   `qa.deviation_report` already exists to govern it.
2. **The RMX container.** Positions as f32 offsets from an f64 origin
   (non-negotiable, see §7 of the product definition), normals, colour, UVs,
   an LOD chain, and spatial tiling for view-dependent streaming.
3. **Texture baking.** Decouples colour resolution from triangle count, which
   is what makes aggressive decimation viable without the result looking flat.
4. **Silhouette-aware carving.** Carving is least reliable exactly where the
   querying scan has a depth discontinuity, and the residual false positives in
   the table above are concentrated there. Suppressing the test near
   discontinuities should take the false-positive rate down another order of
   magnitude.
5. **Chunked E57 reading.** `pye57.read_scan_raw` materialises a whole scan;
   a 100 M-point scan with colour is ~2.4 GB in flight. `max_points` currently
   strides the lattice as a stopgap.
6. **Rust kernels.** Only after the algorithm stops changing. The band
   interface is already the right boundary for it.
