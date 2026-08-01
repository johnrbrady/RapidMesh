# RapidMesh

Survey-grade per-station mesh engine for terrestrial laser scans. Standalone
R&D project; folds into Cairn once it is proven.

**Mesh at the scanner's native lattice, then decimate to a measured tolerance.
Never decimate first.** Everything else follows from that.

## Where to start

| File | What it is |
| --- | --- |
| `00-PRODUCT-DEFINITION.md` | What is being built and the numbers it commits to. Read first. |
| `REVIEW-CAIRN-MESHING.md` | What Cairn's `mesher.py` already does, and the six real gaps. |
| `FINDING-001-PDAL-QUANTISATION.md` | **Cairn is snapping every scan to a 1 cm grid at import.** Affects the point cloud too, not just the mesh. One-line fix. |
| `ARCHITECTURE.md` | How the code is put together and why. |

## Status

Working end to end on synthetic fixtures. Not yet run on real client data.

```
             0.180 deg sampling     0.090 deg sampling     budget
  RMS            1.26 mm                1.00 mm            <= 2 mm
  p99.9          4.83 mm                3.55 mm            <= 8 mm
  mover recall  97.0 %                 97.0 %              >= 95 %
  false pos      0.065 %                0.015 %            <= 0.1 %
```

Deviation is measured against **analytic geometry**, not against the input
points — so it cannot be flattered by fitting the instrument's own noise. The
fixture carries 2 mm range sigma and the mesh comes out below it, because a
triangle averages three noisy vertices.

Built: native structured-E57 reader, band-addressable native lattice,
edge-preserving despeckle, cross-station occlusion carving with parallax
restore, discontinuity-aware triangulation, area-based island culling, oriented
normals, point-to-mesh accuracy reporting.

Not built: decimation, texture, the streaming container, LOD chain. See
`ARCHITECTURE.md` for the ordered list.

## Install

```
pip install -e ".[e57,dev]"
```

`numpy` and `scipy` are required. `pye57` is optional and only needed to read
real scans — the synthetic fixtures and the whole numeric core work without it.

## Use

```python
from rapidmesh import e57_reader, pipeline
from rapidmesh.grid import CoarseRangeGrid

# What lattice will this file actually give us? Read-only, seconds, no points.
for p in e57_reader.probe("station01.e57"):
    print(p.describe())

scan = e57_reader.read_scan("station01.e57", index=0)
neighbours = [CoarseRangeGrid.build(e57_reader.read_scan("station02.e57"))]

result = pipeline.mesh_station(scan, others=neighbours)
print(result.summary())
```

`probe()` is worth running across a site's raw folder before anything else. It
reports which lattice tier each scan will land in, and that single fact decides
how good the meshes can possibly be — an export made without the structured
grid falls back to reprojection, which is the handicap this project exists to
remove.

## Check and measure

```
# Score the pipeline against analytic ground truth
python tools/bench_synthetic.py --rows 500 --cols 2000

# Is an existing Cairn LAZ quantised? (see FINDING-001)
python tools/check_laz_precision.py "D:\CairnData\<pid>\raw"

pytest
```

The test suite asserts the budgets in `00-PRODUCT-DEFINITION.md` §4, not golden
values. It fails when a commitment breaks, not when the algorithm improves.
