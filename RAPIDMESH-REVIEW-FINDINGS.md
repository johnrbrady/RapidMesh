# RapidMesh Review Findings

**Date:** 1 August 2026  
**Reviewer:** ChatGPT  
**Documents reviewed:**

- `00-PRODUCT-DEFINITION.md`
- `ARCHITECTURE.md`
- `FINDING-001-PDAL-QUANTISATION.md`

---

## Overall assessment

The overall direction is strong and I would continue with this approach rather than redesign it.

The central idea is sound:

- keep RapidMesh separate from Cairn until it is proven;
- preserve the scanner's native lattice;
- delay simplification until the geometry is understood;
- measure quality rather than relying on subjective visual judgement;
- retain per-station isolation;
- make the deviation report a core product output.

The main concerns are not with the underlying meshing concept. They relate to how accuracy is defined, several claims that are currently stronger than the available evidence, the order of the remaining implementation work, and some missing production-readiness requirements.

---

## 1. Define the deviation report more precisely

This is the most important change.

The product currently promises point-to-mesh distances for every source point. However, the pipeline deliberately removes movers, isolated points and other samples classified as invalid.

A correctly removed person will naturally be far from the resulting mesh. Including those points in the primary deviation result would make correct filtering appear to be poor meshing.

The opposite problem is also possible. Excluding all difficult points without clearly reporting those exclusions could allow an overly aggressive filter to achieve excellent deviation numbers simply by deleting valid geometry.

### Recommended QA outputs

RapidMesh should produce at least three separate results:

1. **Retained-surface fidelity**

   Point-to-mesh deviation for samples classified as valid static surfaces.

2. **Coverage and filtering report**

   Report how many points were:

   - retained;
   - removed by isolation filtering;
   - removed by occlusion carving;
   - restored by parallax restoration;
   - excluded for another reason.

3. **Mesh-to-source deviation**

   Sample points across the finished mesh triangles and calculate their distance back to the source data.

The third measurement is essential. Point-to-mesh distance alone will not reliably identify triangles that incorrectly bridge across:

- door openings;
- windows;
- occlusions;
- missing areas;
- gaps between separate objects.

Every original point could be close to the mesh while the mesh still contains invented surfaces.

A deviation heatmap should also be produced alongside the numerical summary.

---

## 2. Use the term “survey-grade” carefully

The product definition currently treats a mesh as survey-grade when a deviation report ships with it.

That is useful, but the report measures **mesh fidelity to the source scan**, not the full absolute survey accuracy of the result.

It does not, by itself, account for:

- scanner calibration;
- registration uncertainty;
- control-network uncertainty;
- georeferencing error;
- systematic range bias;
- environmental conditions;
- instrument setup error.

### Recommended wording

For technical and client-facing material, consider using:

> **A survey-fidelity mesh with measured deviation from the source scan.**

The term **survey-grade** can still be used, but it should always be accompanied by a clear explanation of exactly what has and has not been measured.

---

## 3. Resolve the pipeline-order contradiction

The product definition says that every sample is triangulated before invalid surface data is removed.

The architecture shows filtering occurring before triangulation.

The architecture's order is more practical and should be retained.

### Recommended product wording

Replace the current sequence with:

> Read and classify every native-lattice sample, triangulate the retained samples, remove invalid fragments, and only then simplify.

This preserves the fundamental principle:

> No blind resolution reduction occurs before the geometry is understood.

It also avoids implying that transient and invalid points must first become triangles.

---

## 4. Promote real E57 validation and chunked reading

The product commits to structured E57 input from:

- Trimble X7;
- Trimble X9;
- Faro Focus.

The measured performance currently shown is based on synthetic fixtures, while `pye57.read_scan_raw` still materialises an entire scan in memory.

This should become an early release gate rather than a lower-priority future task.

### Required real-file validation

RapidMesh should be tested against real E57 files from every supported scanner and confirm:

- row and column indices;
- missing samples;
- duplicated indices;
- invalid-state flags;
- scan wrap-around behaviour;
- scanner origin and pose;
- RGB availability;
- panorama availability;
- unsupported or ambiguous lattice structures;
- very large station behaviour;
- partial or corrupted files.

Unsupported lattice structures should fail clearly rather than being inferred silently.

### Chunked reading

Chunked E57 reading should be moved earlier in the implementation order.

A 100-million-point scan with colour can require several gigabytes in memory before meshing begins. Production performance should not be extrapolated only from fixtures containing a few million samples.

The current throughput estimate is useful for planning, but it should not yet be treated as proof of production-scale performance.

---

## 5. Cross-station carving still depends on registration

Per-station output avoids the need to merge stations into one continuous mesh.

However, cross-station observations are still used for occlusion carving. Those neighbouring scans must be registered accurately enough for the carving comparison to be meaningful.

A small registration error near a thin object may appear similar to:

- parallax;
- a mover;
- a depth discontinuity;
- an unsupported sample.

### Add explicit carving requirements

The architecture should define:

- how neighbouring stations are selected;
- the coordinate frame expected;
- the maximum acceptable registration uncertainty;
- whether scans must come from the same survey epoch;
- how scanner pose validity is checked;
- what happens when registration quality is unknown;
- when carving is automatically disabled;
- whether a minimum overlap area is required.

The parallax restoration method is promising, but it should not be expected to compensate for poor station registration.

---

## 6. Soften claims that are not yet proven

Several statements are currently stronger than the available evidence.

### 6.1 PDAL quantisation finding

The finding correctly says that the issue still needs confirmation against a real Cairn-generated LAZ.

However, the title and parts of the body state definitively that every Cairn scan is quantised to 1 cm.

Until the real-file test is complete, use:

> **Finding 001 — Cairn is likely quantising E57 imports to 1 cm**

Once a real Cairn LAZ confirms:

- a scale of `0.01`;
- all sampled coordinates lying on that lattice;
- the expected source conversion path;

the definitive wording can be restored.

The actual command output should then be added to the finding as evidence.

### 6.2 PotreeConverter “lossless” claim

Replace:

> PotreeConverter 2.x is lossless.

with:

> The current PotreeConverter invocation does not appear to deliberately thin the point count. Output counts, attributes and coordinate precision still require verification.

The word **lossless** includes more than retaining the apparent point count.

### 6.3 Competitive claims

Statements such as:

- no competitor ships a deviation figure;
- every competing product decimates early;
- the method is strictly more accurate at the same output size;

should be treated as hypotheses unless supported by documented competitor testing.

The architecture gives RapidMesh a strong reason to perform better, but that outcome still depends on:

- the decimator;
- the error metric;
- topology handling;
- texture handling;
- noise characteristics;
- output-size accounting;
- competitor configuration.

---

## 7. Correct the quantisation argument

The current finding says that a 2 mm RMS point-to-mesh target is arithmetically impossible when the source has been quantised to 1 cm.

That is not necessarily true under the current QA definition.

A mesh constructed from 1 cm-quantised points may still fit those same quantised points with less than 2 mm point-to-mesh deviation.

It would be inaccurate relative to:

- the original E57;
- the real measured surface;
- the scanner's unquantised coordinates.

However, it could still score well when tested against the already-degraded LAZ.

### Recommended replacement

> A 1 cm-quantised LAZ cannot reliably preserve 2 mm geometric fidelity relative to the original E57 or underlying surface, even though a mesh may still report a low deviation when measured against the same quantised LAZ.

This is another reason the QA report must identify exactly which dataset is being treated as the source of truth.

### Measurement impact

The finding says that each measurement endpoint may move by up to 5 mm.

For a distance measured between two independently displaced endpoints, the worst-case total error may approach 10 mm.

That distinction should be included.

---

## 8. Do not state that finer LAZ scale “costs nothing” yet

Using the following values is technically sensible:

- `scale_x=0.0001`;
- `scale_y=0.0001`;
- `scale_z=0.0001`;
- automatic coordinate offsets.

The explanation regarding int32 overflow at MGA coordinates is also sound.

However, replace:

> It costs nothing. File size barely moves.

with:

> It is expected to have a small file-size impact, which must be confirmed on representative scans.

A finer integer scale can affect compressed residual magnitudes. The difference may be negligible, but it should be measured rather than assumed.

### Test representative datasets

Compare the old and new conversion settings using:

- an internal room scan;
- a façade scan;
- a large external station;
- a scan with RGB;
- a scan using MGA coordinates;
- a project with multiple stations.

Record:

- LAZ size;
- conversion time;
- point count;
- coordinate precision;
- RGB preservation;
- Potree conversion time;
- viewer performance;
- measurement differences.

### Describe the fix accurately

The code edit may be small, but the safe release effort is larger than one line.

It requires:

- real-file confirmation;
- regression testing;
- RGB verification;
- CRS verification;
- coordinate overflow testing;
- file-size comparison;
- conversion-time comparison;
- reimport testing for existing projects.

---

## 9. Add reproducibility information to every QA report

A report intended to support surveying decisions should be independently auditable.

Each RapidMesh report should include:

- source-file name;
- source-file hash;
- source E57 GUID, where available;
- RapidMesh version;
- source-code commit;
- RMX format version;
- scanner manufacturer and model;
- scanner serial number, where available;
- input sample count;
- retained sample count;
- rejected sample count;
- filtering parameters;
- neighbour stations used for carving;
- coordinate reference system;
- units;
- station transform;
- decimation tolerance for each LOD;
- definitions of RMS and percentile calculations;
- whether calculations are exact or sampled;
- processing duration;
- peak memory;
- warnings and unsupported fields.

This would make the deviation report a reliable engineering record rather than only a visual-quality summary.

---

## 10. Fully specify the RMX format

The proposed approach of storing positions as f32 offsets from an f64 origin is correct.

The RMX specification should also define:

- axis order;
- handedness;
- units;
- coordinate reference system handling;
- global origin;
- per-tile origin;
- endianness;
- vertex order;
- triangle winding;
- normal encoding;
- colour space;
- alpha handling;
- UV representation;
- texture encoding;
- LOD relationships;
- spatial tile hierarchy;
- checksums;
- compression;
- optional fields;
- format versioning;
- forward compatibility;
- failure behaviour for unknown fields;
- maximum supported counts and dimensions.

The format should be versioned before Cairn integration begins.

---

## 11. Tighten the streaming benchmark

The target of first paint within one second on a 10 Mbit connection is useful, but it requires a precise test contract.

At 10 Mbit/s, the theoretical maximum transfer in one second is approximately 1.25 MB before protocol overhead.

Define whether **time to first paint** includes:

- network latency;
- server response time;
- file transfer;
- decompression;
- parsing;
- mesh construction;
- GPU upload;
- shader compilation;
- initial rendering.

Also define:

- browser and version;
- representative computer hardware;
- GPU;
- cold or warm cache;
- expected server latency;
- texture inclusion;
- initial camera position;
- required visual quality for the first frame.

### Size comparison

When comparing RapidMesh with Cairn's 13–20 MB station size, ensure both totals include the same components:

- all LODs;
- vertices;
- triangle indexes;
- normals;
- colour;
- textures;
- metadata;
- spatial indexes;
- container overhead.

An untextured Cairn mesh should not be directly compared with a complete textured RapidMesh package without clearly separating the components.

---

## 12. Add more production failure modes

The non-negotiable rule that a failed mesh must not make the scan unusable is correct.

The implementation should distinguish between:

- unsupported scanner structure;
- missing lattice indices;
- invalid scanner pose;
- insufficient valid samples;
- corrupt E57 data;
- memory exhaustion;
- decimation failure;
- texture failure;
- QA calculation failure;
- RMX writing failure.

The result should state whether:

- no mesh was produced;
- a mesh was produced without texture;
- a mesh was produced without carving;
- a mesh was produced but failed the quality threshold;
- a lower-detail fallback was produced.

A failed QA threshold should not silently ship as a normal successful result.

---

## 13. Add release gates

The quality targets should become explicit release gates.

### Suggested gates

#### Gate 1 — Input integrity

- Real E57 files from each supported scanner are read correctly.
- Native row and column structure is confirmed.
- Unsupported files fail clearly.

#### Gate 2 — Geometry integrity

- Synthetic fixtures meet the defined RMS and percentile thresholds.
- Openings and gaps are not bridged.
- Thin objects survive.
- Triangle orientation and normals are valid.

#### Gate 3 — Filtering integrity

- Mover recall meets target.
- Static false-positive removal meets target.
- Registration-error scenarios are tested.
- Carving is disabled safely when confidence is insufficient.

#### Gate 4 — Scale and memory

- Production-sized stations complete within memory limits.
- Chunked reading is proven.
- Peak memory is recorded.

#### Gate 5 — Decimation

- Every LOD stays within its deviation budget.
- Important edges and silhouettes are preserved.
- No invalid topology is introduced.

#### Gate 6 — Streaming

- LOD0 meets the first-paint target.
- Full package size beats Cairn at equal or better measured fidelity.

#### Gate 7 — Reproducibility

- Every output includes a complete QA and provenance report.
- Re-running the same input and settings produces equivalent output.

---

## Recommended implementation order

I would proceed in the following order:

1. Confirm the Cairn LAZ quantisation finding using a real output file.
2. Fix Cairn's E57-to-LAZ conversion if the finding is confirmed.
3. Freeze the QA definitions, including coverage and bidirectional deviation.
4. Validate structured E57 ingestion using real X7, X9 and Faro files.
5. Establish production-scale memory behaviour and chunked reading.
6. Add registration-quality checks for cross-station carving.
7. Implement deviation-budgeted decimation.
8. Define and version the RMX container.
9. Prove first-paint and total-size targets.
10. Implement texture baking.
11. Add silhouette-aware carving.
12. Port hot kernels to Rust only after the algorithms and output format are stable.

---

## Final recommendation

Do not change the central RapidMesh concept.

The following foundations should remain:

- native-lattice input;
- no blind pre-triangulation decimation;
- per-station output;
- measured simplification;
- source colour preserved without baked lighting;
- f64 origin with f32 local offsets;
- failure isolation from Cairn;
- separate repository until measured superiority is established.

Before major implementation continues, strengthen:

- the QA definitions;
- real-scanner validation;
- registration requirements;
- production memory behaviour;
- RMX format specification;
- benchmark methodology;
- the wording of unverified claims.

The design is already stronger than most early technical specifications. The next stage should focus on converting its good engineering principles into testable, auditable and production-safe requirements.
