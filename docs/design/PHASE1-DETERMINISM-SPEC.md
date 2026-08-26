# Phase 1 design — determinism specification

**Status:** Design, not yet implemented. Documents only.
**Scope:** PLAN.md §5 design-first item 4, and the comparison contract for the
equivalence harness of PLAN.md §5 item 5.
**Date:** 17 August 2026.
**Companion documents:** `PHASE1-HALO-CALCULUS.md`,
`PHASE1-ISLANDS-FINALISATION.md`, `PHASE1-TILE-CONTRACT-V0.md`.

`SPATIAL-CONTRACT.md` §4 already requires that "reprocessing identical input
with the same version and settings produces equivalent coordinates and
topology". This document defines *equivalent* — because an undefined tolerance
makes the Gate 1 invariance requirement unfalsifiable. **Every tolerance below
carries a number and a justification.**

Measurements quoted here were taken against the repository at `56e6499` on the
pinned 3.13 venv (3.13.15), Windows, 12 logical processors, on the synthetic
fixture `synthetic.generate(rows=300, cols=1200, seed=7)`. **They are
synthetic-fixture measurements about code behaviour and are never real-data
results.** The commands are named in §9.

---

## 1. Three tiers, and the difference between them

| Tier | Name | Requirement |
|---|---|---|
| **T1** | Bitwise identical | Every compared byte equal |
| **T2** | Topologically equivalent | Same geometry as a labelled structure; array *order* may differ (§4) |
| **T3** | Numerically equal within tolerance | Differs only by floating-point reassociation, within a stated bound (§5) |

Three distinct guarantees are asserted over those tiers, and they are not the
same statement:

| Guarantee | Statement | Tier |
|---|---|---|
| **R — reproduction** | Same platform, same versions, same settings, same pinned thread count, re-run ⟹ identical output | **T1 on every field** |
| **B — band/chunk invariance** | Everything identical except `band_rows` and `chunk_points` ⟹ same output | **T1 or T2/T3 per field, per §5** |
| **E — streamed ≡ in-memory** | The streamed pipeline's mesh, ledger and both QA directions equal the version-matched `pipeline.mesh_station` reference | **exact under each field's declared predicate** (§7): T1 fields byte-equal, T2 triangle multiset equal with multiplicity, T3 normals inside the sole stated bound; no additional harness tolerance |

`band_rows` and `chunk_points` are settings for **R** and are *varied* for **B**.
Conflating the two is how a spec ends up asserting nothing: R alone is satisfied
by any deterministic implementation, including one whose answer depends on its
own band size.

Here **exact E** does not silently redefine T2 or T3 as byte equality. It means
the harness applies the one equivalence predicate authorised for that field
and nothing looser: no missing/extra triangle, no count tolerance, no
`allclose`, and no QA-number tolerance. Triangle storage order and bounded
normal reassociation are the only non-T1 cases. This reconciles PLAN.md's
"streamed ≡ in-memory, exact" gate with its separate instruction to define
topology equivalence and a numerical tolerance.

---

## 2. Conditions for T1 reproduction

All of the following must be equal, and all must be recorded in the evidence
envelope so a later re-run can be set up identically:

| Condition | Where it is recorded today | Gap |
|---|---|---|
| Platform: OS, CPU architecture | — | **not recorded** |
| RapidMesh package version and commit SHA | package version only at `QAReportMetadata.rapidmesh_version` (`types.py:308`) | **commit SHA not recorded** |
| Python, NumPy, SciPy, pye57 versions | — | **not recorded** |
| Source identity | `QAReportMetadata.source_sha256` (`types.py:307`), from `qa.sha256_file` (`qa.py:52`) | present |
| All settings, including `band_rows`, `chunk_points`, `halo`, `qa_window_rows`, `qa_workers` and sample counts | `QAReportMetadata.settings` (`types.py:309`), populated from `MeshResult.settings` (`pipeline.py:198-206`) | present container, named settings missing |
| RNG seeds | default `seed = 0` at `qa.py:71`, `qa.py:98`, `qa.py:242`; fixture seed at `synthetic.py:138` | **not recorded** |
| Requested/effective QA and numerical-library thread counts | — | **not recorded** (§3) |
| `_resolve_frame` path taken | — | **not recorded** — PLAN.md §5 item 11, closing the `SPATIAL-CONTRACT.md` §3 rule 7 conflict |
| Contract, component-area, canonical-order and QA metric versions | — | **not recorded** |

The right-hand column is the evidence-envelope work of PLAN.md §5 item 11 and
`CLAUDE.md` §9 item 1. This document specifies *what must be there*; adding the
fields is engineering, out of scope for a documents-only task.

---

## 3. Threads

### The mechanisms that could make thread count matter

- `scipy.spatial.cKDTree.query(..., workers=-1)` — hardcoded at `qa.py:158`,
  `qa.py:263` and `qa.py:289`. The parallelism is over independent queries, so
  the per-query answer should not depend on the worker count; the risk is in
  the library, not in the algebra.
- `ScanPose.rotate_local` uses `@` (`types.py:89-91`), which dispatches to the
  installed BLAS. Kernel selection — blocked vs unblocked, FMA vs separate
  multiply-add — can vary with problem size and thread count, and changes the
  last bits.
- `np.polyfit` in `_linfit` (`e57_reader.py:750`) calls LAPACK during lattice
  detection.

### What was measured

`mesh_station` on the fixture, run with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
and then unpinned on a 12-logical-processor machine:

| Compared field | 1 thread vs 12 | 12 vs 12, re-run |
|---|---|---|
| `vertices`, `triangles`, `normals`, `source_sample_id` | bitwise equal | bitwise equal |
| forward and reverse `DeviationReport` figures | bitwise equal | bitwise equal |

### What that does and does not establish

It establishes that on **this** machine, **this** BLAS and **this** fixture,
thread count did not perturb any output. It does not establish platform
independence, and larger problem sizes can change BLAS kernel dispatch.

> **Requirement, unchanged by the measurement.** Thread count is controlled
> and recorded for any run whose output is compared or committed as evidence.

The Phase 1 implementation adds `qa_workers` as an integer setting and passes
it to all three `cKDTree.query` call sites. Evidence and gate runs default to
`qa_workers = 1`; `-1` is forbidden for evidence because it means
machine-dependent "all processors". A production run may use an explicitly
configured positive count after the count has passed the determinism matrix.

KD-tree workers are only one thread pool. The child process also sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and any detected
equivalent to explicit positive values **before importing NumPy/SciPy**. The
evidence envelope records requested and effective QA/BLAS counts plus the
library versions. Runtime environment changes after numerical libraries have
initialised do not satisfy this requirement.

This is implementation inside PLAN.md §5 items 9 and 11, not a separate work
package or a new product decision.

---

## 4. Topology-equivalence rule (T2)

Needed because triangle *order* is not invariant — measured, §5(b).

> **Two meshes are topologically equivalent iff:**
>
> 1. their vertex sets are equal as sequences of `source_sample_id`
>    (`types.py:204`), in the same order; **and**
> 2. their triangles are equal as **multisets** of ordered
>    `source_sample_id` triples, taken up to cyclic rotation; **and**
> 3. no triangle appears with reversed winding in one and not the other.

Notes on each clause, from the code:

1. Vertex order is not arbitrary. `build_mesh` compacts with
   `keep_idx = np.nonzero(used)[0]` (`triangulate.py:260`) over a scan already
   in row-major lattice order (`sort_row_major`, `grid.py:59`), so vertex order
   is source order. Clause 1 is therefore a strict requirement, not a
   concession — and it held bitwise across every band size measured.
2. Cyclic rotation is permitted because it preserves orientation. Canonicalise
   `(a,b,c)` as the lexicographically smallest of `(a,b,c)`, `(b,c,a)` and
   `(c,a,b)`. Do **not** include reversed rotations. The current measurement
   established array-order differences, not a need for cyclic rotation, so
   every run also reports the count of triples that required rotation; a
   non-zero count is evidence to investigate, not silently discarded.
3. Reversal is **not** permitted: winding carries orientation, and `build_mesh`
   uses it to face normals at the scanner (`triangulate.py:286-287`).

Multiplicity is load-bearing. A set comparison would treat one occurrence and
two occurrences as equal. Comparison sorts canonical triples and compares the
full arrays, then separately reports duplicate multiplicities.

Comparison must be done on `source_sample_id` triples, never on vertex indices:
indices are positions in a compacted array (`triangulate.py:260-265`) and are
only meaningful within one run.

---

## 5. Tolerances

Each row states the tier, the number, and why that number.

### (a) Vertex positions — **tolerance 0, bitwise**

`build_mesh` applies `scan.pose.rotate_local(local_verts).astype(np.float32)`
per vertex (`triangulate.py:292`). One transform, no accumulation, no
order dependence. Any difference is a defect, not reassociation noise.

Reinforced by `SPATIAL-CONTRACT.md` §4: the float32 storage contribution must
not exceed **0.1 mm per axis**, asserted at
`tests/test_spatial_contract.py:91`. A position tolerance would sit inside a
budget that is already tight; there is no room for one and no mechanism
requiring one.

*Measured:* bitwise identical across `band_rows` ∈ {512, 256, 64} and across
thread counts 1 and 12.

### (b) Triangles — **canonical oriented multiset equality exact; order not constrained**

*Measured, and this is the load-bearing result:* changing `triangulate`'s
`band_rows` (`triangulate.py:66`, default 512) from 512 to 256 and to 64
produced the **same 701,442 triangles in a different order** — first differing
position 290,750 and 72,748 respectively.

The mechanism is in `_band_triangles`: for each band it emits all
shorter-diagonal quads, then all longer-diagonal quads, then the four
one-corner-missing classes (`triangulate.py:147-184`), each block row-major
over the whole band. Moving a band boundary re-interleaves the blocks.

**Consequence.** Global triangle order already depends on an internal band size
in the *current in-memory* path. Any quantity that depends on triangle order is
therefore already band-size dependent — see (d), (e) and (h).

### (c) `source_sample_id` — **tolerance 0, exact array equality, same order**

DEC-004 makes this a server-only field (`PHASE1-TILE-CONTRACT-V0.md` §3), and
it is the join between a vertex and its source observation
(`triangulate.py:267-271`). A permutation would silently re-attribute evidence.

*Measured:* identical across all band sizes and thread counts.

### (d) Vertex normals — **provisional ≤ 1 × 10⁻⁶ radians directed angular difference**

`build_mesh` accumulates area-weighted face normals with
`np.add.at(acc, faces[:, col], fn)` in triangle order (`triangulate.py:279-284`).
`np.add.at` is unbuffered and applies in index order, so a reordering of the
triangle array is a reassociation of an f64 sum.

**The number.** Under a *full random permutation* of all 701,442 triangles —
far more aggressive than any band reorder — 833 of 356,462 vertices changed,
with a **maximum angular difference of 3.332 × 10⁻⁸ rad** and p99.9 of
2.980 × 10⁻⁸ rad. The tolerance is set at 1 × 10⁻⁶ rad, **30× the measured
worst case**, leaving headroom for fans larger or worse-conditioned than this
fixture's.

This is a **Phase 1 invariance ceiling, not a product accuracy tolerance**, and
it remains provisional until it passes the largest real station plus a fixture
with a high-valence, near-cancelling normal fan. If either exceeds the bound,
stop and investigate the accumulation rule; do not widen the number from one
new observation.

**Why 1 × 10⁻⁶ rad cannot hide anything that matters.** It is
5.7 × 10⁻⁵ degrees — about **1/1000 of the reference lattice's own angular
sampling step of 0.0591°/column** (`docs/DATA-INVENTORY.md` line 120). No
angle-derived decision in the pipeline resolves anything near it: the
incidence limit is 82° (`pipeline.py:112`, `triangulate.py:81`) and the sliver
test is a ratio at 0.015 (`triangulate.py:65`).

*Also measured:* across `band_rows` ∈ {512, 256, 64} the normals were in fact
**bitwise identical** — the band reorder happened not to perturb any sum on
this fixture. The tolerance exists as a guarantee, not because a band reorder
was observed to need it.

**Comparison procedure.** Before calculating angle, both corresponding arrays
must be finite and every norm must be within `1 × 10⁻⁶` of one. The directed
angle is `atan2(‖a × b‖, a · b)` with no absolute value around the dot product.
A sign reversal is therefore approximately π radians and fails the angular
bound directly.

**Orientation-ambiguity monitor, and why it is separate.** `build_mesh` flips a normal on
the sign of `facing = einsum(acc, -local_verts)` (`triangulate.py:286-287`).
Near `facing ≈ 0` the sign is decidable only within rounding, and a flip is a
π-radian error that the directed comparison catches; the monitor identifies
the unstable cause before it happens.

> Report the count of vertices with `|facing| ≤ 10⁻¹² · ‖acc‖ · ‖local_verts‖`.
> The count must be **0**, or each such vertex is adjudicated individually.

10⁻¹² is ~1500× the ~3ε ≈ 6.7 × 10⁻¹⁶ relative rounding floor of a three-term
f64 dot product — above the noise, and still flagging genuine ambiguity.

### (e) Component area and island verdict — **verdict exact (tolerance 0)**

`cull_islands` computes per-triangle area in float32 and accumulates per
component with `np.bincount(comp, weights=area, minlength=ncomp)`
(`triangulate.py:225-230`), then applies `comp_area ≥ min_area` and
`comp_count ≥ min_triangles` (`triangulate.py:233`).

**Design requirement — a checked condition plus one common fallback, not a
tolerance.** Areas are evaluated by `component-area-v1` in both the in-memory
reference and streamed path, as defined in
`PHASE1-ISLANDS-FINALISATION.md` §4.1:

```
component_total_area  /  smallest_positive_triangle_area  <  2⁵³⁻²⁴ = 2²⁹ ≈ 5.37 × 10⁸
```

A float32 area carries at most 24 significant bits; added to a float64
accumulator no more than 2²⁹ times larger, the addition does not round. While
that holds, the sum is **exact under any accumulation order**, so streamed,
merged and in-memory accumulation all give the identical float64 value and
**verdicts are exact with no tolerance required**.

The condition is checked for every component of every real station. Each
station records its maximum ratio, fallback count and area-algorithm version.
At or above the bound, both execution paths use the same canonical-order
`math.fsum` result. Until the in-memory reference has adopted that versioned
fallback, encountering such a component stops the equivalence gate.

*Measured on the fixture:* total 172.638428 m² over a smallest positive
triangle of 2.400080 × 10⁻⁵ m², a ratio of 7.193 × 10⁶ — inside the bound with
~75× margin. At that ratio `np.bincount`, a strict left-to-right float64 fold,
`np.add.reduce` and `math.fsum` all returned **bitwise-identical** sums over
701,442 values, as did forward, permuted and reversed orderings, and component
areas across all three band sizes. Forcing the ratio to 1 × 10¹³ made forward
and reversed differ by 2.221 × 10⁻⁷, confirming the condition is what governs.

**The margin is a fixture property, not a station property.** Geometry, not
point count, determines the ratio, so checking only the largest station is
insufficient.

**Near-threshold diagnostic.** Report the count of components with
`|area − min_area| ≤ 10⁻⁹ m²`. This is a sensitivity monitor around the
algorithmic threshold, not a bound on physical or accumulated numerical
uncertainty. Per-triangle float32 representation errors can accumulate; one
epsilon at `min_area` does not make a component inside this window
"undecidable" or prove one outside it safe.

*Not exercised by the current fixture:* it yields 2 components of 7.75 m² and
164.88 m², both three to four orders above the threshold. A fixture built for
the near-threshold case is required before this monitor means anything.

### (f) Ledger counts — **tolerance 0, exact**

Every `FilterStats` field (`types.py:257-264`) is an integer count.
`require_balanced()` (`types.py:284`) must pass on both sides of every
comparison. `restored_from_carve` is an event, not a disposition
(`types.py:252`), and is compared as such.

### (g) Forward QA (`deviation_report`) — **bitwise equal, with a caveat**

Given identical mesh, identical retained points in identical order, and
identical seed, `distances` (`qa.py:238`) is deterministic; the subsample at
`qa.py:254-256` draws over the *points* array, whose order is row-major lattice
order and was measured invariant (see (c)).

**Caveat, stated because it would otherwise be a false reassurance.** On the
current undecimated pipeline the forward figure is *identically zero* — the
trap recorded in `CLAUDE.md` §8, "no decimation yet, so point-to-mesh is
trivially zero". The fixture run confirms it: forward RMS 0.000000000 mm. A
quantity that is always zero cannot demonstrate order-stability. Forward QA's
tier assignment is therefore **provisional until decimation exists**, and the
harness must re-establish it then.

### (h) Reverse QA (`mesh_to_source_report`) — **no tolerance is defined; the metric must be made order-independent first**

*Measured.* With identical input, identical settings and identical seed,
changing only `triangulate`'s internal `band_rows`:

| `band_rows` | p99.9 | maximum |
|---|---|---|
| 512 | 55.074164 mm | 67.616092 mm |
| 256 | 54.028991 mm | 67.616092 mm |
| 64 | 53.703413 mm | 69.333409 mm |

**Spread: 1.371 mm of p99.9 and 1.717 mm of maximum, from an internal band
size.** These are synthetic-fixture numbers on a deliberately coarse
300 × 1200 lattice; their magnitudes say nothing about RapidMesh's accuracy and
must never be quoted as results. What they establish is a property of the
*code*: this metric moves when nothing about the data or the settings moves.

**Cause.** `mesh_to_source_report` selects triangles with
`rs.choice(tri_ids, size=n, replace=True, p=weights / weights.sum())`
(`qa.py:146`), where `tri_ids` and `weights` follow the triangle array's
order (`qa.py:128`, `qa.py:145`). Inverse-CDF sampling over a reordered
probability vector selects different triangles.

**Why no tolerance is offered.** A tolerance wide enough to absorb this would
have to be of the same order as the movement, and the p99.9 figure is the one
compared against a fixed acceptance budget. A gate whose tolerance is
comparable to the quantity it tests is not a gate. Defining one here would
amount to widening an acceptance number by the back door, which is forbidden
(`PLAN.md` §3; `CLAUDE.md` §6).

> **Requirement.** Reverse-QA triangle selection must be made independent of
> triangle array order before any reverse figure is compared against a budget
> or across configurations.

#### Authorised design: `reverse-qa-v2`

1. **Triangle identity.** Convert each non-zero-area triangle to the canonical
   oriented source-id triple of §4. Reversed winding is a different identity.
2. **Bounded canonical order.** Sort each persisted band segment by that triple
   within its B10 cap, then k-way merge the sorted runs using a B13-bounded
   heap. No full-station sort is resident.
3. **Sample count.** Preserve the current rule over `T`, the count of
   positive-area triangles:

   ```text
   n = min(max_samples, max(T, min(10_000, max_samples)))
   ```

4. **Area-weighted systematic selection.** Compute total area with one strict
   left-to-right float64 recurrence over canonical order; the second pass uses
   the identical recurrence for cumulative bounds. For `j = 0 … n-1`, select the triangle
   whose cumulative half-open area interval contains
   `(j + 0.5) × total_area / n`. A large triangle may receive multiple
   samples. Intervals are `[previous, next)`, with the final interval closed at
   `total_area`. This produces exactly `n` samples and is independent of input
   array order on the pinned platform/version.
5. **Interior point.** For each selected sample, SHA-256 the domain separator
   `rapidmesh-reverse-qa-v2`, the recorded unsigned 64-bit seed, canonical
   triple and global sample ordinal `j`. Interpret the first two unsigned
   64-bit words as `u1=(x+0.5)/2^64`, `u2=(y+0.5)/2^64`; apply the existing
   square-root barycentric transform. No process-global RNG state is used.
6. **Fixed row window.** Resolve each triangle's source vertex rows through
   the server observation store. Build correspondence against retained
   observations in `[min_row-qa_window_rows, max_row+qa_window_rows]`, clipped
   to the scan, where `qa_window_rows` is a recorded semantic setting (initial
   calibration candidate 8), not the current processing-band size. Windows
   may be coalesced for performance only when the resulting candidate set is
   identical.
7. **Versioned report.** Record `metric_version=reverse-qa-v2`, seed,
   `max_samples`, actual `n`, `qa_window_rows`, canonical-order version and
   requested/effective worker counts. v1 and v2 numbers are not compared as
   the same metric.

This is bounded by persisted segments, external merge state and fixed lattice
row windows. It changes exported reverse-QA figures and therefore carries a
metric version rather than rewriting the history of existing reports.

**No acceptance tolerance is defined yet.** First, v2 must select identical
triangle identities and barycentric points and produce bitwise-identical
reports across the band/chunk/thread matrix. Then its fixed-window distances
are calibrated against the global KD result on purpose-built fixtures and
every authorised real station. The evidence states "exact within the recorded
row window" and reports any global-calibration discrepancy. Only after those
results may an acceptance tolerance be proposed; it is never chosen to absorb
internal-band movement.

### (i) Timing and peak memory — **excluded from every comparison**

`QAReportMetadata.processing_seconds` and `peak_rss_bytes` (`types.py:311-312`,
populated at `pipeline.py:96-97`) are non-deterministic by nature. The harness
compares every other envelope field and skips these two explicitly, rather than
comparing them with a wide tolerance.

---

## 6. The azimuth wrap seam

### What is covered today

`columns_wrap` (`filters.py:67`) decides whether column `cols − 1` neighbours
column `0`, by comparing `|az_step| × cols` against 2π. The **predicate** is
tested: `tests/test_pipeline.py:122`,
`test_full_sweep_wraps_partial_does_not`, asserts it is true for a full sweep
and false for a quarter-FOV lattice.

### What is not covered

**No test exercises the geometry at the seam.** The wrap is consumed in two
places:

- `_shift` rolls columns and only masks the wrapped edge when `wrap` is false
  (`filters.py:370-375`) — so despeckle and restore judge across the seam;
- `_band_triangles` forms `B` and `D` with `np.roll(idx, -1, axis=1)` and
  invalidates the final column only when `not wrap` (`triangulate.py:114-121`)
  — so triangulation bridges the seam.

Nothing asserts that the quads joining column `cols − 1` to column `0` are
produced, are produced exactly once, carry the same winding as interior quads,
or are subject to the same acceptance decision. This is precisely the "no row
seam (including the azimuth wrap seam — add the wrap-seam geometry test)"
requirement in PLAN.md §5, Gate 1.

### The gate test

> **`test_wrap_seam_quads_match_an_interior_column_pair`**

On a fixture whose geometry is rotationally symmetric about the station — a
cylinder — the seam column pair is geometrically indistinguishable from every
other adjacent pair, so any asymmetry is a bug and not a property of the scene.
Assertions:

1. Triangles exist whose vertices span column `cols − 1` and column `0`.
2. Their count equals the count for an interior column pair, exactly.
3. Their winding matches: normals face the scanner, on the same test
   `build_mesh` applies (`triangulate.py:286-287`).
4. Each seam triangle appears exactly once.
5. With a partial-FOV lattice (`columns_wrap` false) **no** triangle spans
   `cols − 1` to `0`.
6. Seam triangle count is invariant across ≥3 values of `band_rows`. Bands are
   row ranges, so the seam lies *inside* every band
   (`PHASE1-HALO-CALCULUS.md` §3) — this asserts that the two seams never
   interact.
7. The despeckle and restore keep-masks for seam-adjacent columns equal those
   for an interior column pair.

**Honesty condition on this test.** Standing rule 8 requires a new test to be
proven to fail against the pre-change code first. Reading
`triangulate.py:114-121` and `filters.py:370-375`, the wrap handling looks
correct, so this test may well pass on first run. If it does, it is a
**regression test, not a bug-fix test**, and must be reported as such — never
presented as a proven-failing test.

---

## 7. What the streamed ≡ in-memory harness must compare

PLAN.md §5 item 5: built **before** band-local geometry, on fixtures small
enough to run both ways. Built afterwards it can only confirm the code agrees
with itself.

### Fields, and the tier each is compared at

| Field | Source | Tier |
|---|---|---|
| `MeshData.origin` | `types.py:197` | **T1** — f64 bitwise |
| `MeshData.vertices` | `types.py:198` | **T1** — f32 bitwise, same order |
| `MeshData.triangles` | `types.py:199` | **T2** — exact canonical oriented multiset including multiplicity, §4 |
| `MeshData.normals` | `types.py:201` | **T3** — §5(d), provisional directed 1 × 10⁻⁶ rad + finite/unit/orientation monitor |
| `MeshData.rgb` | `types.py:202` | **T1** — exact; absent for the 4 stations without colour |
| `MeshData.source_sample_id` | `types.py:204` | **T1** — exact, same order |
| every `FilterStats` field | `types.py:257-264` | **T1** — exact; `require_balanced()` passes on both |
| every `DeviationReport` field, forward | `types.py:224-235` | **T1** — §5(g) caveat applies |
| every `DeviationReport` field and sampled-identity digest, reverse | `types.py:224-235` plus v2 evidence | **T1 after `reverse-qa-v2` passes §5(h)**; blocked until then |
| `QAReportMetadata`, all fields | `types.py:307-312` | **T1**, **except** `processing_seconds` and `peak_rss_bytes`, excluded |
| `lattice_source` | `types.py:328` | **T1** |

For T1 fields, "exact" means `np.array_equal`, never `np.allclose`. T2 uses the
complete canonical multiset comparison; T3 uses only the directed normal rule
of §5(d). A harness that falls back to `allclose` will pass real divergences.

### Configurations the harness must run

1. **In-memory reference:** the version-matched `pipeline.mesh_station` path,
   before band-local geometry is connected, with only the separately approved
   common metric changes (`component-area-v1`, `reverse-qa-v2`, controlled
   workers) applied to both paths. The harness must never compare a new
   streamed metric with an old reference metric.
2. **Streamed, ≥3 values of `band_rows`**, including one small enough to force
   at least 3 bands. A configuration with one band proves nothing — the trap in
   `PHASE1-HALO-CALCULUS.md` §1.
3. **≥2 values of `chunk_points`**, chosen so chunk boundaries do not coincide
   with band boundaries, separating the two axes.
4. A separate purpose-built halo witness run at `halo = 2`, asserted to
   **differ at a pre-identified decision**, and at 3/4, asserted to pass — see
   `PHASE1-HALO-CALCULUS.md` §9. An ordinary fixture need not diverge merely
   because it was run with halo 2.
5. Positive core sizes only. There is no halo-derived minimum `band_rows`;
   smaller values are useful stress cases when their runtime is acceptable.

### What it must report, pass or fail

The number of bands actually produced; the first differing field and index for
every mismatch; the near-threshold component count (§5(e)); the sign-flip
monitor count (§5(d)); and the full configuration matrix, so a green result is
attributable to a specific set of runs rather than to a single lucky one.

---

## 8. What the code does not settle

1. **Platform independence is untested.** §3's measurements are one machine,
   one BLAS, one fixture. Cross-platform T1 has not been demonstrated and must
   not be claimed. (WP-0 obtained cross-platform corroboration of the *gates*,
   Windows 3.13.15 and Linux 3.13.13 — that is not the same as bitwise output
   equality.)
2. **Forward QA's order-stability is unproven** while the metric is
   identically zero — §5(g).
3. **Determinism across RapidMesh versions is out of scope.** PLAN.md §13 lists
   it as a hardening item: when outputs may change, and how that is
   communicated. Nothing here promises stability across versions.
4. **NumPy's `bincount` and `add.at` ordering behaviour is undocumented.** The
   area design uses the checked condition and common versioned fallback;
   normals retain a provisional measured bound that still requires real and
   near-cancelling evidence.
5. **`np.polyfit` in lattice detection** (`e57_reader.py:750`) was not isolated;
   it ran inside every measured configuration and never differed, which is
   evidence but not isolation.
6. **The near-threshold island case and the wrap-seam geometry have no fixture
   yet.** Both are named above; neither has been exercised.

---

## 9. Falsification

**The claims.** The tier assignments in §5 are correct; the tolerances are
large enough to absorb legitimate reassociation and small enough to reject a
real error; the harness in §7 would detect a divergence.

**Every tolerance is tested in both directions.** A tolerance that never
rejects anything is not a tolerance. For each of (d) and any future T3 entry:

- **Acceptance:** the observed difference across the configuration matrix is
  within the bound.
- **Boundary tests:** inject directed perturbations just inside and outside
  the bound (for example 0.9× and 1.1×) and assert acceptance then rejection.
  A 10× case alone proves only that some check exists, not that the stated
  boundary is enforced.

| Claim | Falsifying observation | Evidence that closes it |
|---|---|---|
| §5(a) positions T1 | Any bitwise difference in `vertices` across band size, chunk size or thread count | The harness matrix with `np.array_equal` on `vertices` |
| §5(b) triangle multiset T2 | Any canonical oriented triple multiplicity differs across configurations | Full sorted-array comparison plus duplicate counts across ≥3 band sizes |
| §5(d) 10⁻⁶ rad | Any observed directed angular difference above 10⁻⁶ rad, a sign reversal, non-finite/non-unit normal, or a 1.1× perturbation that passes | Synthetic permutation, near-cancelling fixture, largest real station and 0.9×/1.1× boundary tests |
| §5(d) ambiguity monitor | A non-zero ambiguous-facing count is ignored, or a sign differs between runs | Monitor and directed-angle results reported on every run |
| §5(e) verdict exact | Areas differ by one ULP while the ratio is inside 2²⁹, or the two paths use different results at/above it | Late-merge array equality; maximum ratio/fallback count on every real station; injected above-bound case using `component-area-v1` in both paths |
| §5(e) near-threshold | Any component within 10⁻⁹ m² of `min_area` whose verdict differs between configurations | Monitor count; requires the missing near-threshold fixture |
| §5(h) reverse QA | v2 selects a different canonical triangle/sample point or produces a different report across band/chunk sizes; or fixed-window calibration is unstated | Sample-identity/point digest plus full report equality; per-station comparison against global KD result |
| §6 wrap seam | Any of the seven assertions fails | The named gate test, with its pass/fail status honestly labelled as regression or bug-fix (§6) |
| §7 harness | The purpose-built halo witness passes at 2 or fails at 3 | Witness decision and per-field diff; ordinary fixtures are not required to fail at 2 |
| §3 threads | A configured count is not honoured/recorded, or an approved count changes a T1 field | Requested/effective count evidence and the thread matrix; cross-platform T1 remains unclaimed |

**Commands that produced the measurements in this document** (read-only probes,
run against the repository at `56e6499`, written to the session scratchpad and
not committed):

```
.venv/Scripts/python.exe <scratchpad>/band_order_probe.py
.venv/Scripts/python.exe <scratchpad>/order_probe2.py
.venv/Scripts/python.exe <scratchpad>/bincount_probe.py
.venv/Scripts/python.exe <scratchpad>/exactness_probe.py
.venv/Scripts/python.exe <scratchpad>/thread_probe.py <tag>
.venv/Scripts/python.exe <scratchpad>/thread_cmp.py <scratchpad>
```

These are probes, not gates. Every measurement they produced must be
re-established by a committed test before it is relied on.
