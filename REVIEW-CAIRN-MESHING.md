# Review: the Mesh Project Brief against what Cairn actually has

**Date:** 31 July 2026
**Reviewed by:** Claude, at John's request
**Sources:** `pointcloud-viewer/backend/mesher.py`, `backend/conversion.py`,
`backend/routers/models.py`, `backend/models.py`, `frontend/src/scans.ts`,
`frontend/src/measure.ts`, `frontend/src/potree-patches.ts`

---

## 1. Correction to the brief's premise

> "Current meshing in Cairn uses Poin Tree, which is workable but not good enough."

This is not what the code does, and the distinction matters because it changes
what RapidMesh has to beat.

**Potree** is Cairn's *point cloud* renderer and its octree converter. It never
produces a mesh. It has nothing to do with meshing.

**Meshing in Cairn is `backend/mesher.py`.** Its own docstring calls it
"TurboMesh-lite v2: per-scan range-image meshing with cleanup". It is 322 lines
of NumPy/SciPy and it already implements most of what the brief proposes as new
design direction. Validated on Galvin Road, 2 July 2026: 9 scans, 360–560k verts
each, 147 MB site total against 4.9 GB of LAZ.

So the honest baseline is not "we have nothing good". It is "we have a
competent first-generation implementation with a hard quality ceiling built
into its first line". RapidMesh's job is to remove that ceiling, not to
re-invent the parts that already work.

---

## 2. What the brief proposes that Cairn already does

| Brief section | Status in Cairn | Where |
| --- | --- | --- |
| §1 Native angular grid triangulation | **Done.** Points binned into a spherical grid, neighbouring cells triangulated as quads. | `mesher.py` `load_scan_grid`, `mesh_from_grid` |
| §1 Depth-discontinuity gradient check | **Done, and better than described.** Edge length limit scales with range (`6.0 × max(r) × step`) so the threshold adapts to distance instead of being a fixed number. Plus a sliver-quality test (`twice_area / longest²  > 0.015`). | `mesher.py` `valid_triangles` |
| §2 Per-station isolation, no merging | **Done.** One `.cmh` per station, rendered per station, keyed by `cairnStationId`. | `conversion.py`, `frontend/src/scans.ts` |
| §3 Motion artefact filtering | **Done, and by a stronger method.** Two filters: a 3×3 median despeckle with a range-proportional threshold, and cross-scan **occlusion carving** — if a neighbouring station saw *through* a point's location, that point was transient. | `mesher.py` `mesh_from_grid`, movers block |
| Open item: LOD generation | **Partially done.** Two tiers: a 512×256 coarse LOD0 for first paint, plus the full mesh. Frontend streams LOD0 then swaps. | `routers/models.py` `build_meshes`, `frontend/src/scans.ts` |

The occlusion-carving approach deserves calling out. The brief proposes
detecting movers from a single scan's range spikes ("close-far-close as the
beam sweeps across"). Cairn instead uses the geometric fact that a solid
surface blocks line of sight: if station B has a clear return *beyond* where
station A recorded a point, A's point was not solid at B's capture time. That
is a much lower false-positive method than gradient thresholding, because a
real door frame is never seen through. **RapidMesh should keep this and treat
the brief's single-scan gradient method as a supplement, not a replacement** —
single-scan detection is still needed for the first scan on site, for scans
with no near neighbour, and for movers that happened to be transient in every
scan.

---

## 3. The real gaps — this is what RapidMesh is for

### 3.1 The resolution ceiling (the big one)

```python
GRID_W, GRID_H = 2048, 1024
```

That is **0.176° per bin horizontally**. A Trimble X7 at high resolution samples
at roughly **0.017°**, ten times finer in each axis. A Faro Focus at 1/1
resolution is finer still.

`load_scan_grid` keeps the **nearest return per bin** and throws the rest away:

```python
order = np.lexsort((r, bin_id))
first[1:] = bs[1:] != bs[:-1]
reps = order[first]          # one point per bin, the rest discarded
```

At a 100× cell-area ratio, a scan with 100 million points is reduced to at most
2.1 million before a single triangle is made. The brief's own stated goal —
"a mesh at the scanner's full native resolution", "tighter, more accurate
detail on walls, door frames, and fine surface changes" — is **structurally
impossible** in the current implementation. Everything downstream is meshing a
1/100-scale decimation.

This is the single highest-value thing RapidMesh changes.

### 3.2 It meshes from LAZ, so the scanner grid is already destroyed

`mesher.py` takes `laz_path` and re-derives the angular grid by projecting XYZ
back through `arctan2` / `arcsin` from the E57 pose origin. Two problems:

1. **The structured grid is already in the E57 and is being thrown away.** A
   structured E57 carries `rowIndex` / `columnIndex` (or spherical
   range/azimuth/elevation) per point — the scanner's *actual* sample lattice.
   Cairn converts E57 → LAZ first, which discards it, then guesses it back.
2. **Re-projection assumes a perfect spherical scanner model.** Real scanners
   have mirror wobble, non-ideal axis alignment and a two-face capture pattern.
   Projected bins land slightly off the true lattice, so genuine neighbours
   collide into one bin and genuine samples are dropped as duplicates.

RapidMesh reads the structured grid directly. No projection, no collisions, no
guessing.

### 3.3 No decimation — flat surfaces cost as much as detailed ones

There is no simplification step anywhere. A 6 m × 3 m blank wall gets the same
triangle density as an ornate cornice. This is backwards, and it is precisely
where TurboMesh's file-size advantage comes from: adaptive density driven by a
geometric deviation budget, dense where curvature is high and sparse where it
is not.

Cairn's LOD0 is not decimation — it re-bins the source at 512×256 and meshes
again. That is uniform downsampling, so LOD0 loses door frames and window
reveals at exactly the same rate it loses blank wall.

**Consequence:** Cairn cannot honestly state a geometric tolerance for its
output. Survey-grade means being able to say "no vertex deviates from the
source cloud by more than X mm". Nothing in the current pipeline measures that.

### 3.4 No texture — colour resolution is welded to triangle resolution

Output is `f32 pos[3V] | u8 rgb[3V]` — one RGB triple per vertex. Colour detail
therefore cannot exceed geometry detail. Decimating the geometry would destroy
the colour with it, which is likely part of why decimation was never added.

TurboMesh textures its meshes. That decouples the two: a wall can be 200
triangles and still carry a 4096px photographic texture. This is how you get
both small files and a good-looking result. It is not a cosmetic issue — it is
the thing that makes decimation viable at all.

### 3.5 Shading is baked irreversibly into vertex colours

```python
shade = (0.6 + 0.4 * np.abs(vn @ L)).reshape(-1, 1)
cols = np.clip(cols.astype(np.float32) * shade, 0, 255).astype(np.uint8)
```

A fixed light direction is multiplied into the RGB and the normals are then
thrown away — CMH1 has no normal field. So the mesh can never be relit, cannot
receive ambient occlusion, and the baked lighting is wrong from most viewing
angles. It also contaminates the colour data: a surveyor looking at what they
think is scan RGB is actually looking at RGB × a synthetic lambert term.

Normals belong in the format. Lighting belongs in the renderer.

### 3.6 Single-blob output, no streaming granularity

One `.cmh` per station, 13–20 MB, fetched whole. The two LOD tiers are separate
whole files. There is no spatial tiling, so looking at one wall downloads the
entire station including everything behind the viewer.

### 3.7 No accuracy reporting

Nothing computes mesh-to-cloud deviation. For a product sold to surveying firms
this is the credibility gap: "survey-grade" has to be a measured number in a
report, not an adjective.

### 3.8 Smaller items

- **Pole distortion.** Bins are uniform in elevation, so cells near zenith/nadir
  are extremely narrow in real angle. Wasteful and artefact-prone.
- **`min_component_tris=150` is a raw count**, not an area. On a fine grid this
  removes real small objects; on a coarse one it leaves noise.
- **Mover removal only checks the 2 nearest stations** (`near[:2]`), which is a
  speed compromise that misses movers in open areas with spread-out setups.
- **Intensity fallback discards data**: `las.intensity >> 8` collapses 16-bit
  intensity to 8-bit and copies it across all three channels.

---

## 4. What RapidMesh must inherit from Cairn

Not everything should be rebuilt. These are proven and worth carrying over:

1. **Cross-station occlusion carving** (§3 movers block) — genuinely good.
2. **Range-proportional edge thresholds** rather than a fixed metric cut.
3. **Shorter-valid-diagonal quad splitting** — `mesher.py` explicitly notes the
   fixed B–C diagonal was "a major source of visibly tangled triangles".
4. **Partial-cell triangles** — one missing bin makes a triangle, not a hole.
5. **Origin-relative f32 storage with an f64 origin** — preserves millimetres at
   MGA coordinates. Any new format must keep this property.
6. **The `r > 0.3` no-return filter** — PDAL writes E57 no-return cells as
   points at the scanner origin.
7. **Best-effort failure posture** — a scan that converts but fails to mesh is
   still a successful scan.

---

## 5. Scorecard

| Capability | Cairn today | TurboMesh | RapidMesh target |
| --- | --- | --- | --- |
| Meshing unit | Per station | Per station | Per station |
| Source grid | Re-projected from LAZ | Native | **Native structured E57** |
| Angular resolution | 2048 × 1024 fixed | Scanner native | **Scanner native, no cap** |
| Discontinuity handling | Range-scaled edge cull | Yes | Range-scaled + normal-aware |
| Noise / mover removal | Median + occlusion carve | Yes | Both + sweep-signature |
| Decimation | **None** | Adaptive | **Quadric, deviation-budgeted** |
| Texture | **None** (vertex colour) | Yes | **Atlas from scan RGB / pano** |
| Normals in output | **No** | Yes | Yes |
| LOD | 2 tiers, re-binned | Multi-res | Progressive chain from decimation |
| Streaming | Whole file | Tiled | Tiled + view-dependent |
| Stated accuracy | **None** | Not published | **Measured, per scan, in a report** |

---

## 6. Recommended position

Cairn's mesher is a good v1 that has hit its architectural ceiling. RapidMesh is
a clean-room rebuild that keeps Cairn's seven proven ideas and fixes the six
structural gaps. The two differentiators against TurboMesh are:

1. **True native-resolution input.** TurboMesh does not publish its internal
   sampling, but every browser-streaming mesh product decimates early.
   Meshing at the scanner's real lattice and decimating *afterwards, to a
   measured tolerance* is strictly better than decimating first.
2. **A published, measured accuracy figure per scan.** Nobody in this market
   ships a deviation report with the mesh. For a product sold to surveyors,
   that is a stronger claim than any rendering feature.

---

## 7. Integration back into Cairn, when proven

RapidMesh is a standalone package with no Cairn imports (`mesher.py` already
follows this discipline and it is worth keeping). Integration is then a
drop-in at three points:

- `conversion.run_conversion` — swap the `mesher.mesh_scan` call.
- `routers/models.py build_meshes` — swap the batch path.
- `frontend/src/scans.ts` `_cmhToMesh` — add an RMX loader alongside the
  existing CMH1 loader, so old projects keep working.

Nothing else in Cairn touches the mesher. The blast radius is small.
