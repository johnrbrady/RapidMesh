# Phase 1 design — two-pass island culling and finalisation

**Status:** Design, not yet implemented. Documents only.
**Scope:** PLAN.md §5 design-first item 2.
**Date:** 17 August 2026.
**Companion documents:** `PHASE1-HALO-CALCULUS.md` (band ownership),
`PHASE1-TILE-CONTRACT-V0.md` (segments and manifest),
`PHASE1-DETERMINISM-SPEC.md` (equivalence tolerances).

Every structural claim cites the function it was read from. Where the design
needs a fact the code does not settle, the gap is stated.

---

## 1. Why island culling cannot be band-local

`cull_islands` (`triangulate.py:193`) is the one stage in the chain with no
bounded read-halo (`PHASE1-HALO-CALCULUS.md` §3). It:

1. builds an edge list over every triangle in the scan — `e = np.vstack([tris[:, [0,1]], tris[:, [1,2]], tris[:, [2,0]]])` (`triangulate.py:218`);
2. builds a `coo_matrix` over **all** vertices (`triangulate.py:219-222`);
3. runs `connected_components` globally (`triangulate.py:223`);
4. computes per-triangle area (`triangulate.py:225-228`), accumulates it per
   component with `np.bincount` (`triangulate.py:230`), counts triangles per
   component (`triangulate.py:231`);
5. keeps triangles whose component satisfies **both**
   `comp_area ≥ min_area` and `comp_count ≥ min_triangles`
   (`triangulate.py:233`). Defaults: `min_area = 0.005` m²,
   `min_triangles = 8` (`triangulate.py:196-197`); `mesh_station` passes
   `min_component_area = 0.005` (`pipeline.py:113`, `pipeline.py:145-149`).

A component can span the full height of the lattice, so no halo bounds it.

**The memory cost of the current path, on the reference station.** ADR-006
records 24.48 M triangles for the high-resolution station. The edge list alone
is `3 × 24.48e6 × 2 × 8 B ≈ 1.175 GB` of int64, plus
`np.ones(len(e), np.int8) ≈ 73.4 MB` (`triangulate.py:220`), plus SciPy's
COO→CSR conversion inside `connected_components`. The global cull is therefore
already over 1,000,000,000 bytes before the mesh itself is counted — more than
twice the approved 512,000,000-byte working budget on its own. This is an
allocation count read from the code, not a measurement.

---

## 2. The shape of the answer

Two passes over the station, with the provisional geometry held on disk between
them as band-aligned segments (`PHASE1-TILE-CONTRACT-V0.md`).

```
Pass A  (streaming, one forward sweep)
    per band: filter → triangulate core-owned quads
              → label components locally
              → union across the boundary-row frontier
              → write provisional segment tagged with GLOBAL component ids
              → retire components that can no longer grow
    output: completed component table (membership + triangle counts)

Pass B  (bounded finalisation, one sequential re-read)
    evaluate component-area-v1 in canonical oriented-triangle order
    decide keep/drop per component
    finalise the ledger (dropped_island, retained-but-unmeshed)
    run both QA directions against FINAL dispositions
    emit spatial tiles atomically
    verify immutable generation; publish current.json last
```

---

## 3. Pass A — per-band labelling and the streaming frontier

### 3.1 Component connectivity must be *vertex* connectivity

`cull_islands` builds a **vertex** adjacency graph from triangle edges and runs
`connected_components` on it (`triangulate.py:218-223`); the label is per
vertex, and a triangle's component is read from its first vertex
(`triangulate.py:229`). Two triangles that share only a single vertex are
therefore in the **same** component: that vertex is adjacent to all four of the
other vertices, so the whole set is one connected component.

This matters because face-adjacency union-find — merging triangles that share
an *edge* — is the more natural streaming implementation and produces **finer**
components at vertex-only joins. It would cull differently.

> **Requirement.** The streamed union-find unions the three vertices of every
> triangle into one set. Equivalence to `cull_islands` depends on it.

Since each lattice cell holds at most one sample (`dense_rows` writes one index
per cell, `grid.py:155`), a vertex is addressable as `(row, col)`.

### 3.2 What a band labels

Per `PHASE1-HALO-CALCULUS.md` §7, a band owns the triangles whose quad top row
lies in its core `[b0, b1)`. Those triangles touch lattice rows `b0 … b1`.

Within the band, union-find runs over the band's own vertices in a band-local
index space. A local root that touches one or more incoming frontier ids reuses
the lowest of those ids as its canonical global id and records every other
incoming id as an alias of that root. A local root with no incoming frontier
id is a newly born component and receives exactly one id from a monotone
counter. Continuing a component across a band must not allocate a fresh id.

This allocation rule is part of the memory bound, not an implementation
preference. The alias map, canonical-root map, triangle counts and next-id
counter are durable component-table state and are persisted with the
provisional segments.

### 3.3 The frontier is exactly one row

Connectivity between band *k* and band *k+1* flows only through vertices they
both touch. Band *k*'s triangles reach down to row `b1`; band *k+1*'s core
starts at `b1` and its triangles reach down from there. The shared vertex set is
therefore exactly the samples in lattice row `b1`.

> **Frontier = one lattice row, `cols` wide.**

Representation: `frontier: int32[cols]`, where `frontier[c]` is the global
component id of the sample at `(b1, c)`, or `-1` where that cell has no sample
or the sample is in no owned triangle.

Band *k+1* reads `frontier[c]` for each of its row-`b1` vertices and unions its
own band-local root with that global id. Nothing else crosses the boundary.

### 3.4 Retirement — what makes the state bounded

After band *k* writes its frontier, a component that is **not referenced by any
frontier entry can never grow again**: every future band's connectivity to the
past goes through the frontier row. Its triangle count is final.

> **Retirement rule.** At the end of each band, every live component absent from
> the new frontier is finalised and removed from the live table.

Live components are therefore bounded by the number of occupied frontier
columns, which is bounded by `cols`.

### 3.5 Projected state size for the worst recorded reference lattice

The committed reference metadata records two high-resolution widths. The
worst lattice for allocation arithmetic is **6096 columns × 2387 rows =
14,551,152 source records**, with 100.0 % *source-record lattice fill*
(`docs/DATA-INVENTORY.md` §1.1/§1.4). That does not mean all records are valid
returns: invalid/no-return records are removed before `StructuredScan.xyz` is
built. The source-record count is retained here only as a conservative upper
bound on occupied component vertices.

| Structure | Sizing rule | Reference station |
|---|---|---|
| Frontier | `cols × 4 B` (int32) | 6096 × 4 = **24,384 B** |
| Live component records — parent int32, resident triangle count int32, root flag, padding | ≤ `cols` entries × 12 B | ≤ 6096 × 12 = **73,152 B** |
| Verdict/keep table over every canonical id ever allocated | 1 B per component birth | ≤ **4,850,384 B** (bound, §3.6) |
| **Pass A island state, total** | | **≤ ≈ 4.95 MB decimal**, of which <0.10 MB is live |

Against the 512,000,000-byte working budget that is under 1 %. Two caveats, stated
plainly:

- The 4.85 MB figure is a **worst-case arithmetic bound**, not a measurement.
  The realistic component count on a real station is far lower — the surface is
  dominated by one large component plus a scatter of islands — and the actual
  count is a Gate 1 measurement.
- Area accumulators are **not** in Pass A. §4 explains why they moved to
  Pass B.

The 12 B live-record estimate is for the **resident Pass A representation**,
not the persisted checkpoint schema. On the worst recorded reference lattice,
even the conservative two-triangles-per-cell bound is
`2 × (2387 - 1) × 6096 = 29,090,112` triangles, safely below signed int32's
maximum of 2,147,483,647. The Phase 1 implementation must verify that bound
from the validated lattice dimensions before selecting the resident int32
representation; it must not narrow an already accumulated count.

Persisted counts deliberately have the wider, stable representation:
`component.rmlog` stores `triangle_count_delta:u64`, and both `rmstate` live
records and `rmcomp` store `triangle_count:u64`
(`PHASE1-TILE-CONTRACT-V0.md` §3). Widening a verified resident int32 count to
u64 for a checkpoint is exact. This keeps the 12 B reference-station resident
bound valid without constraining the durable contract to that station's
lattice size.

### 3.6 The component-count bound, derived

Every component contains at least one triangle. Every triangle uses three
distinct occupied lattice cells. Every vertex carries exactly one component
label (`triangulate.py:223`), so no cell is shared between two components.
Hence

```
components  ≤  ⌊ occupied cells / 3 ⌋
            ≤  ⌊ 14,551,152 / 3 ⌋
             =  4,850,384
```

At 1 byte per verdict that is **4,850,384 B**. The bound depends on the §3.2
rule that a continuing local root reuses an incoming id. An implementation
that allocates a provisional id before every cross-band union does **not** get
this bound and must budget its aliases separately. The production design uses
the reuse rule.

### 3.7 What Pass A writes

Per band, one provisional segment (`PHASE1-TILE-CONTRACT-V0.md` §4) carrying:

- the band's core-owned triangles as `(row, col)` vertex references or stable
  `sample_id`s (`e57_reader.py:399`), never band-local array indices
  (`PHASE1-HALO-CALCULUS.md` §7);
- one **global component id** per triangle;
- the band's provisional per-sample filter dispositions — `dropped_no_return`,
  `dropped_despeckle`, `dropped_mover_carve`, `restored_from_carve`
  (`types.py:257-264`) — which *are* final at Pass A, because the filter chain
  is halo-bounded;
- the band's core row range, point count and digest.

The station also carries a versioned server-only component log/table containing
canonical id, parent/root alias, triangle count and retirement state. That
completed membership/count state is the authoritative input to Pass B. Pass B
adds area and final keep/drop verdicts to `component.rmcomp`. Durable layout
and recovery checkpoints are defined in `PHASE1-TILE-CONTRACT-V0.md` §3/§5.3.

**Pass A writes no `dropped_island` and no `retained` count.** Both are
deferred (§5).

---

## 4. Pass B — finalisation

### 4.1 Area must be accumulated in Pass B, and the exactness condition that governs it

Two separate questions. The first is *when* area can be accumulated; the second
is whether a streamed accumulation gives the same number as the in-memory one.

**When.** Area *could* be accumulated as partial sums in Pass A and combined
when roots merge, but that would introduce a second arithmetic expression and
make equality depend on merge history. Pass A therefore deliberately records
membership and integer triangle counts only. Pass B reads the completed root
table and evaluates one common, versioned area algorithm over final
components. So:

> **Pass A determines component membership and triangle counts. Pass B
> evaluates `component-area-v1` from the completed component table and applies
> the culls.**

Triangle counts are integer sums and are order-independent, so they stay in
Pass A.

**Whether the numbers match.** `cull_islands` computes per-triangle area from
`np.cross` / `np.linalg.norm` over the **float32** vertex array
(`triangulate.py:225-228`; `mesh_station` passes `grid.scan.xyz`, which is f32 —
`types.py:164`) and accumulates per component with
`np.bincount(comp, weights=area, minlength=ncomp)` (`triangulate.py:230`),
promoting the weights to float64. A streamed accumulation evaluates a different
arithmetic expression — in particular it must add two partial sums together
whenever a late merge joins two components — and floating-point addition is not
associative. The difference would land directly on `comp_area ≥ min_area`
(`triangulate.py:233`), a keep-or-delete decision about survey geometry.

The measurement found something better than a tolerance: **a condition under
which the sum is exact, and therefore order-independent.**

> #### The exactness condition
>
> A float32 value carries at most **24 significant bits**. Adding it to a
> float64 accumulator no more than `2⁵³⁻²⁴ = 2²⁹ ≈ 5.37 × 10⁸` times larger
> keeps every significant bit of both operands inside float64's 53-bit
> mantissa, so **the addition does not round**. If that holds for every
> addition, the whole sum is exact and every accumulation order — streamed,
> merged, permuted or in-memory — yields the identical float64 value.
>
> ```
> component_total_area  /  smallest_positive_triangle_area   <   2²⁹
> ```

*Measured, on the synthetic fixture at repository `56e6499`:*

| Check | Result |
|---|---|
| Total area 172.638428 m², smallest positive triangle 2.400080 × 10⁻⁵ m² | ratio **7.193 × 10⁶** — inside 2²⁹ with ~75× margin |
| `np.bincount`, a strict left-to-right float64 fold, `np.add.reduce` (pairwise) and `math.fsum` (exact) over all 701,442 areas | **all four bitwise identical** |
| Same values forward, randomly permuted, and reversed | **bitwise identical** |
| Component areas across `band_rows` ∈ {512, 256, 64} | **bitwise identical** |
| Counter-example: values forced to a ratio of 1 × 10¹³ | forward and reversed differ by 2.221 × 10⁻⁷ — the condition, when violated, does produce order dependence |

**Common summation policy — `component-area-v1`.** There is no streamed-only
fallback:

1. Per-triangle area is computed from the same float32 vertices by the same
   cross/norm expression as `cull_islands`.
2. Triangles are presented in `(final component root, canonical oriented
   source-id triple)` order, using the triple definition in
   `PHASE1-DETERMINISM-SPEC.md` §4. The server-only bounded merge view in
   `PHASE1-TILE-CONTRACT-V0.md` provides that order without holding the station.
3. For every component, obtain a stable guard total with `math.fsum` over that
   canonical stream and the smallest positive float32 triangle area. Record
   `guard_total / smallest_positive`.
4. If the ratio is below `2²⁹`, ordinary float64 accumulation is proven exact
   and must equal the current `np.bincount` result bitwise.
5. If the ratio is at or above `2²⁹`, the `math.fsum` result is the
   authoritative component area for **both** the in-memory reference and the
   streamed implementation. The evidence envelope records the fallback and
   its metric version. A streamed-only compensated result is forbidden.

Until the in-memory reference has adopted the same versioned fallback, a
ratio failure stops the equivalence gate and is reported; it is not silently
decided by only one path.

**This condition is checked on every component of every real station, not
only on the station with the most points.** Geometry determines the ratio.
The evidence envelope for each station records its maximum ratio, count of
fallback components and component-area version. The synthetic fixture's 75×
margin says nothing about a real station.

**What the fixture cannot show.** It yields two components of 7.75 m² and
164.88 m², three to four orders of magnitude above the 0.005 m² threshold, so
it exercises neither the near-threshold verdict nor a late merge. Both need
purpose-built fixtures (§7).

`PHASE1-DETERMINISM-SPEC.md` §5(e) records the resulting verdict tolerance as
zero and specifies the near-threshold monitor that keeps it honest.

### 4.2 Pass B accumulator size

| Structure | Sizing rule | Reference station, worst case |
|---|---|---|
| id → final root map | 4 B per component | ≤ 4,850,384 × 4 = **19,401,536 B** |
| area accumulator | 8 B per final root | ≤ 4,850,384 × 8 = **38,803,072 B** |
| smallest-positive-area witness (§4.1) | 4 B per final root | ≤ 4,850,384 × 4 = **19,401,536 B** |
| **Pass B component tables** | | **≤ 77,606,144 B** |

15.2 % of the approved 512,000,000-byte working budget in the conservative
source-record bound, and realistically a smaller fraction because invalid and
filtered records cannot become component vertices. Entered in the budget
table at `PHASE1-TILE-CONTRACT-V0.md` §6.

### 4.3 Ledger finalisation — two deferred dispositions

The in-memory path attributes both at the end, after culling
(`pipeline.py:174-189`):

| Disposition | Current derivation | Meaning |
|---|---|---|
| `dropped_island` | `int((used_before_cull & ~used_final).sum())` (`pipeline.py:186`) | the sample was in a triangle, and its component was culled |
| retained-but-unmeshed | folded into `dropped_other` as `int((~used_before_cull).sum())` (`pipeline.py:187`) | the sample survived filtering but landed in **no** triangle — every quad it belonged to failed the discontinuity or sliver test (`triangulate.py:123-143`) |

Both are **deferred dispositions** in the streamed design, for different
reasons:

- `dropped_island` needs the completed component table, which does not exist
  until Pass A ends.
- Retained-but-unmeshed needs certainty that *no* band produced a triangle
  containing the sample. A sample in the halo of band *k* may be a vertex of a
  triangle owned by band *k+1*. Deciding at band *k* would mis-attribute it.

> **Rule.** Pass A writes filter dispositions only. `retained`,
> `dropped_island` and the retained-but-unmeshed contribution to
> `dropped_other` are written once, in Pass B, from the finalised component
> table and the union of all owned triangles.

`FilterStats.require_balanced()` (`types.py:284`) is called once, on the Pass B
result — exactly as `mesh_station` calls it once after attribution
(`pipeline.py:190`).

**Retained-but-unmeshed keeps its current home.** It stays inside
`dropped_other`, matching `pipeline.py:187`, so the streamed ledger is
comparable field-for-field with the in-memory one. Splitting it into a named
field would be a `FilterStats` change (`types.py:246`), a behaviour change to
the exported envelope, and is not part of Phase 1. The internal diagnostic
records the sub-count so the reason is not lost. A later exported split needs
a report-schema version and separate owner authorisation.

### 4.4 QA runs against final dispositions only

The in-memory ordering is already correct and must be preserved: `mesh_station`
computes `used_final` **after** culling and measures against
`retained_offsets` built from `used_final` (`pipeline.py:157-172`). Both
directions — `deviation_report` and `mesh_to_source_report` (`qa.py:65`,
`qa.py:94`) — see only the final retained set and the final mesh.

> **Rule.** No QA sample is drawn in Pass A. Both QA directions run in Pass B,
> after culling and after ledger finalisation.

Measuring against provisional geometry would reintroduce exactly the defect
`FINDING-002` records: a deviation report that includes island-culled points
reported 50.84 mm RMS on data that was in fact fine.

### 4.5 Atomic emission and generation publication

Pass B emits spatial tiles inside a new immutable generation. Each tile is
written to a temporary file, flushed and renamed into that unpublished
generation; the generation manifest is then verified. Publication replaces
only `current.json` last. Existing published files are never overwritten, so
a reader sees either the prior complete generation or the new complete one.
The mechanism, checksums and Windows portability constraints are specified in
`PHASE1-TILE-CONTRACT-V0.md` §5.

---

## 5. Approximate per-band culling is rejected

Recorded explicitly, per PLAN.md §5 item 2.

**The proposal.** Cull inside each band from that band's own component areas,
avoiding the second pass entirely.

**Why it is rejected — two independent reasons, either sufficient.**

1. **It deletes real surface.** A component that straddles a band boundary is
   judged on a fragment of its area. A wall spanning four bands, with 0.004 m²
   in each band and 0.016 m² in total, is culled in every band by the
   `min_area = 0.005` test (`triangulate.py:196`, `triangulate.py:233`) despite
   being three times over the threshold. The same argument applies to
   `min_triangles = 8` (`triangulate.py:197`). Deleting survey geometry to save
   a bounded sequential re-read is the wrong trade in a product whose first
   non-negotiable is that a filter must be able to say why a point is not
   surface (`filters.py:5-11`).

2. **It breaks band-size invariance.** Which fragments fall in which band is a
   function of `band_rows`. The verdict, and therefore the output mesh, the
   ledger and both QA figures, would depend on the band size. That directly
   violates Gate 1's requirement that output be "independent of chunk and band
   sizes per the determinism spec" (PLAN.md §5) — a gate, not a preference. It
   would also make `dropped_island` a number with no fixed meaning, and
   `dropped_island` is an exclusive disposition in a ledger that must balance
   (`types.py:262`, `types.py:284`).

**The variant that is also rejected:** culling per band with a "large
components are exempt" heuristic. It replaces a systematic error with a
threshold, and thresholds that decide whether survey geometry is deleted are
precisely what `filters.py`'s opening rule refuses (`filters.py:5-11`). It also
does not restore band-size invariance; it only makes the dependence harder to
see.

**What is *not* rejected:** retiring components early (§3.4). That is not
approximate — a retired component's membership is provably final.

---

## 6. What the code does not settle

1. **The realistic component count is unknown.** §3.6 gives a conservative
   source-record bound under the canonical-id allocation rule; the actual
   valid-return and component counts on each real station remain Gate 1
   measurements.
2. **Whether Pass B's sequential re-read is I/O-bounded or CPU-bounded** on the
   reference station is not derivable from the code. It sets the cost of the
   two-pass structure and is a Gate 1 timing measurement.
3. **Spatial tile boundaries are not designed here.** ADR-006 Decision 2a
   specifies halo, locked boundary vertices, deterministic ownership, stitching
   and seam-local QA for *decimation* tiles at Phase 3. Phase 1 emits tiles as a
   container concern only (`PHASE1-TILE-CONTRACT-V0.md` §3); the decimation
   boundary strategy is Phase 3's.
4. **`min_area` and `min_triangles` are unchanged and unjustified by real
   data.** They are the current defaults (`triangulate.py:196-197`). This
   document preserves their behaviour exactly; it does not defend the values.
5. **Whether any real station contains a component near the 0.005 m²
   threshold** is unmeasured. The near-threshold diagnostic
   (`PHASE1-DETERMINISM-SPEC.md` §5(e)) exists to find out but is not an
   uncertainty bound. The maximum area ratio is recorded for every station,
   with the common fallback applied where required.

---

## 7. Falsification

**The claims.**
(A) Pass A's live state is O(cols) + O(live components) and fits in <0.10 MB
live / ≤4.95 MB decimal total under the canonical-id allocation rule on the
worst recorded reference lattice.
(B) While the exactness condition of §4.1 holds — component total area over
smallest positive triangle area below 2²⁹ — streamed area accumulation is
bitwise equal to `cull_islands`' result under any order, and the condition is
checked rather than assumed.
(C) Per-band culling breaks band-size invariance and the two-pass design does
not.

**What proves them wrong.**

| Claim | Falsifying observation | Evidence that closes it |
|---|---|---|
| A | The live component table exceeds `cols` entries at any point in a sweep, or measured Pass A island state exceeds the projected total. Either means the retirement rule (§3.4) is wrong — most likely because connectivity crosses a band boundary somewhere other than the frontier row | An instrumented Pass A over the reference station logging max live component count and max resident island-state bytes per band, committed with the command |
| B | Streamed component areas differ from the in-memory result by even one ULP on a fixture with a late merge while the exactness condition holds; or the two paths apply different fallbacks when it fails | A test comparing the arrays with `np.array_equal` on a late-merge fixture; per-station maximum-ratio and fallback counts for every real station; and an injected case above 2²⁹ asserting both paths use `component-area-v1` |
| C | A component straddling a band boundary, whose total area exceeds `min_area` but whose per-band fragments do not, is **culled**. Under this design it must be retained | A test with exactly that fixture, asserting retention; and `dropped_island` equal across ≥3 values of `band_rows` and ≥2 values of `chunk_points` |

**The overall gate.** All three are subsumed by the streamed ≡ in-memory
equivalence harness (PLAN.md §5 item 5): if the streamed ledger's
`dropped_island`, the triangle set and both QA reports match the in-memory
result exactly across the band-size matrix, this design is correct on that
fixture. If any of them differ, the harness's per-field diff identifies which of
A, B or C failed.

**A negative result that would still matter.** If the harness passes at every
band size *including* one where the fixture has a single band, the fixture is
not exercising the boundary and the result is uninformative — the same trap
recorded in `PHASE1-HALO-CALCULUS.md` §1. Any run must report the number of
bands actually produced.
