# Phase 1 design — halo calculus

**Status:** Design, not yet implemented. Documents only.
**Scope:** PLAN.md §5 design-first item 1.
**Date:** 17 August 2026.
**Authority:** subordinate to `00-PRODUCT-DEFINITION.md` and
`SPATIAL-CONTRACT.md`. Where this note and a repository governance document
disagree, the governance document wins and the conflict is reported.

Every structural claim below cites the function it was read from. Where the
design needs a fact the code does not settle, the gap is stated rather than
assumed.

---

## 1. The failure mode this prevents

Band-local processing is only correct if each band can reach far enough outside
its own core to reproduce the decision a full-scan run would have made. If it
cannot, the decisions near band edges diverge from the full-scan result.

The divergence has four properties that make it the worst kind of defect to
discover late:

1. **It does not raise.** A sample judged with a partially-filtered
   neighbourhood still gets a keep/drop answer; it is just the wrong one.
2. **It scales with band count**, so it is proportionally invisible on a small
   fixture and worst on the production configuration.
3. **A one-band fixture cannot show it at all.** If `band_rows` exceeds the
   fixture's row count there is exactly one band, no band edges, and the
   equivalence harness passes while production diverges.
4. **It is downstream of everything.** Filtering feeds triangulation, which
   feeds island culling, the ledger and both QA directions. A wrong keep
   decision changes the triangle set, the exclusive disposition counts
   (`FilterStats.require_balanced`, `types.py:284`) and both deviation reports.

The consequence is that the halo shortfall would be found by the Gate 1
invariance requirement — "output independent of chunk and band sizes"
(PLAN.md §5, Gate 1) — *after* band-local geometry, QA and incremental output
were all built on top of it. Sizing the halo before that work starts is the
whole point of doing this as a design-first item.

---

## 2. Definitions

For a stage `S` operating on a lattice of rows and columns:

| Term | Meaning |
|---|---|
| **read-halo** `h(S)` | Rows outside a target row that `S` must read to produce a *correct* output for that row. Stated separately for the up (lower row index) and down (higher row index) directions where they differ. |
| **write-scope** `w(S)` | The rows for which `S` is permitted to emit output — a keep decision, a triangle, a ledger entry or a QA sample. |
| **core** | The half-open row interval a band owns: `[core_row_start, core_row_stop)` on `RawRowBand` (`e57_reader.py:127`). |
| **halo rows** | Rows present in a band's `data` but outside its core. Read freely; never written for. |

The invariant that makes the whole scheme sound:

> **Read the halo, write only the core.**

`RawRowBand`'s own docstring states this rule (`e57_reader.py:130-135`), and
`tests/test_qa_ledger.py:117`
(`test_row_band_halos_repeat_evidence_but_core_owns_each_sample_once`) proves
the assembler honours it: consecutive bands share samples, and concatenating
each band's core-owned ids reproduces the input exactly once.

---

## 3. Per-stage declaration table, read from the implementations

Derived from what each function indexes, not from what its name suggests.

| Stage | Function | Read-halo, rows up | Read-halo, rows down | Column reach | Write-scope | Non-halo dependency |
|---|---|---|---|---|---|---|
| Despeckle | `isolation_despeckle` (`filters.py:81`) | 1 | 1 | ±1 column, wrapping per `columns_wrap` | keep-mask for judged rows only | none |
| Carve | `carve_movers` (`filters.py:147`) | 0 | 0 | 0 | keep-mask, per sample | **whole neighbour `CoarseRangeGrid`s** |
| Restore | `restore_parallax_carve` (`filters.py:198`) | 1 | 1 | ±1 column, wrapping | revised drop mask for judged rows only | inherits carve's |
| Triangulate | `triangulate` / `_band_triangles` (`triangulate.py:61`, `triangulate.py:97`) | 0 | 1 | +1 column, wrapping | triangles whose quad top row is in the core | none |
| Island cull | `cull_islands` (`triangulate.py:193`) | **unbounded** | **unbounded** | unbounded | global | — |
| Forward QA | `deviation_report` (`qa.py:65`) | **unbounded** | **unbounded** | unbounded | global | final mesh |
| Reverse QA | `mesh_to_source_report` (`qa.py:94`) | **unbounded** | **unbounded** | unbounded | global | final mesh + final retained set |

### Where each figure comes from

**Despeckle — 1 row up, 1 row down.** `isolation_despeckle` iterates
`_judged_bands(grid.rows, band_rows, halo=1)` and then loads
`lo, hi = max(b0 - 1, 0), min(b1 + 1, grid.rows)` (`filters.py:116-117`). The
neighbourhood is the 8-neighbourhood `_NEIGHBOURS` (`filters.py:64`), applied
by `_shift` (`filters.py:358`), whose row offsets are ±1 and whose column
offsets are ±1.

**Carve — 0 rows.** `carve_movers` (`filters.py:147`) transforms each sample to
world coordinates in chunks (`filters.py:189`) and queries each neighbour grid
with `seen_through` (`grid.py:295`). It never indexes a lattice neighbour. Its
lattice read-halo is genuinely zero.

**Carve's non-halo dependency is the important one.** The neighbour grids are
built by `CoarseRangeGrid.build` (`grid.py:202`) from a *whole* neighbouring
scan, via `pipeline.carve_grids` (`pipeline.py:224`). No halo of any size
satisfies that: it is a phase-ordering requirement, not a boundary requirement.
Every neighbour grid must be complete before this station's sweep begins. That
is what forces the cached grid artefacts of PLAN.md §5 item 7, designed in
`PHASE1-TILE-CONTRACT-V0.md` §7.

Two further facts about that dependency, both read from the code:

- `carve_grids` (`pipeline.py:224`) builds grids from the raw `StructuredScan`
  list, **not** from cleaned scans. A neighbour's grid therefore reflects its
  unfiltered returns. The streamed implementation must preserve that or carve
  results change.
- `CoarseRangeGrid.build` bakes the pose in (`grid.py:234-235`,
  `origin = scan.pose.translation`). A grid is only valid for the pose it was
  built with — the cache key consequence is in `PHASE1-TILE-CONTRACT-V0.md` §7.

**Restore — 1 row up, 1 row down, over post-carve state.** Same tiling and same
±1 load as despeckle (`filters.py:240-241`). The difference is what it reads:
`alive = present & ~dropped[safe]` (`filters.py:248`) — the *carve* state of
each neighbour, not merely its presence. Support is then counted over
`Rlive`, the range array masked to survivors (`filters.py:250`,
`filters.py:253-258`).

**Triangulate — 0 up, 1 down.** `_band_triangles` forms quads from
`A = idx[:-1, :]` and `C = idx[1:, :]` (`triangulate.py:113`,
`triangulate.py:115`); `B` and `D` are the same two rows rolled one column
(`triangulate.py:114`, `triangulate.py:116`). It reaches down one row and right
one column, never up. `ScanGrid.bands(band_rows, overlap=1)` (`grid.py:158`)
encodes exactly that one-row reach for the in-memory path.

**Island cull and both QA directions are not halo-bounded at all.**
`cull_islands` runs `connected_components` over a vertex adjacency matrix built
from every triangle in the scan (`triangulate.py:218-223`); both QA functions
build a KD-tree over the whole mesh or the whole retained set (`qa.py:157`,
`qa.py:262`). Neither can be made band-local by widening a halo. Island culling
is restructured in `PHASE1-ISLANDS-FINALISATION.md`; bounded reverse QA is
PLAN.md §5 item 9 and is not designed here.

### Columns do not need a halo

Bands are **row** ranges over the lattice, and the lattice's columns are
azimuth: `_lattice_from_angles` derives `col` from azimuth and `row` from
elevation (`e57_reader.py:651-652`), and `_angular_steps_from_indices` fits
azimuth against column and elevation against row (`e57_reader.py:716-717`).
`ScanGrid.dense_rows(r0, r1)` returns shape `(r1 - r0, cols)` — every column
(`grid.py:150`).

So a row band spans the complete azimuth range. Every column neighbour,
including the wrap neighbour between the last column and column 0, is present
inside the band. No column halo is required, at any band size.

One requirement follows: `columns_wrap` (`filters.py:67`) is a **whole-scan**
predicate — it compares `az_step * cols` against 2π. A band-local
implementation must be handed the scan-level value, never recompute it from a
band, or a band would be judged as a partial-FOV scan and the seam would be
dropped. The seam's geometry gate is specified in
`PHASE1-DETERMINISM-SPEC.md` §6.

---

## 4. Two band systems exist today, and they are not the same thing

This distinction is the most common way to get the composed number wrong, so it
is stated explicitly.

| | `_judged_bands` (`filters.py:345`) / `ScanGrid.bands` (`grid.py:158`) | `iter_row_bands` (`e57_reader.py:250`) |
|---|---|---|
| Operates on | a `StructuredScan` **already fully in memory** | a raw chunk stream from libE57 |
| Purpose | keep the dense `(rows, cols)` working array small | keep the whole scan out of memory |
| Bounds memory in | one stage's temporaries | the entire pipeline |
| Halo | `_judged_bands`: hardcoded ±1 at the call sites. `ScanGrid.bands`: `overlap` parameter, default 1 | `halo` parameter, default 1 |

The composed halo derived in §5 is a property of the **streaming** bands. It is
the number that must reach `iter_row_bands`.

### Two traps in the current code, both design-time, neither a live defect

**Trap 1 — `_judged_bands` ignores its `halo` argument.** The signature is
`_judged_bands(rows, band_rows, halo)` (`filters.py:345`) and the body is

```
step = max(band_rows, 1)
return [(b, min(b + step, rows)) for b in range(0, rows, step)]
```

`halo` is never read. The actual ±1 halo is hardcoded at both call sites
(`filters.py:117` and `filters.py:241`). Raising the composed halo therefore
**cannot** be done by passing a larger value to `_judged_bands`; the change
would be silently ineffective. The Phase 1 implementation removes this unused
parameter rather than pretending it owns the streaming halo. Streamed halo 3
is supplied explicitly to `iter_row_bands`, while band-local filtering receives
explicit data and judged-core bounds.

**Trap 2 — `_judged_bands` tiles from row 0 of whatever grid it is given.** A
band-local `ScanGrid` built over a streaming band's *data* rows has local row 0
at `data_row_start`, so `_judged_bands`' tiling would not align with the
streaming core. The band-local caller must therefore either pass the core
bounds explicitly in the band's own row space, or set `band_rows` at least as
large as the band's data-row count so exactly one judged range is produced and
then intersect that range with the core. The second is simpler and is the
recommended form.

**Required coordinate convention for the streamed consumer.** A
`RawRowBand` retains absolute lattice rows in its `rowIndex` field while its
dense working arrays are band-local. The consumer must translate the absolute
core interval into that local array exactly once:

```text
local_core_start = core_row_start - data_row_start
local_core_stop  = core_row_stop  - data_row_start
```

Filtering may read the complete local data array, but writes only this local
core slice. Triangulation additionally owns quads whose absolute top row is in
the core. No helper may independently retile the local array from zero and
thereby create a second, misaligned ownership system.

**Neither is currently a defect in shipped behaviour**, because no band-local
geometry consumer exists yet (`CLAUDE.md` §9 item 2: filtering, triangulation,
QA and writing still use the full-scan `mesh_station` path). Both become
defects the moment band-local geometry is connected.

### What the existing documents say, and why it is not the composed number

`ARCHITECTURE.md:137-141` and `ADR-006` Decision 2 both record "triangulation
overlap 1, despeckle halo 1". Read as *per-stage* declarations those are
correct and match the code. Neither states a composed figure for the chain, and
the `iter_row_bands` docstring likewise describes the stages individually —
"filtering may inspect both halo rows, triangulation may inspect the row below"
(`e57_reader.py:262-264`). This note supplies the composition. No existing
statement is contradicted; a missing one is added.

---

## 5. Composition — why the halos add

### The rule

For a serial chain `S₁ → S₂ → … → Sₙ` in which each stage consumes the previous
stage's output at the *same lattice addresses*, the raw input rows needed to
produce a correct final output at row `r` extend

```
Σ h(Sₖ)   rows beyond r, in each direction
```

**not** `max h(Sₖ)`.

Proof by induction from the end of the chain. Let `Hₖ` be the number of rows
either side of `r` over which stage `k`'s output must be correct. `Hₙ = 0`:
the last stage only has to be right at `r`. Stage `k` reads `h(Sₖ)` rows either
side of every row it must be correct at, so its *input* — stage `k-1`'s output —
must be correct over `Hₖ + h(Sₖ)` rows either side, giving
`Hₖ₋₁ = Hₖ + h(Sₖ)`. Unrolling gives `H₀ = Σₖ h(Sₖ)`. ∎

### Applied to despeckle → carve → restore

The chain is the one `clean` runs, in that order (`filters.py:294-324`).

To decide row `r` correctly, `restore_parallax_carve` reads the **post-carve**
state of rows `r-1 … r+1` (`filters.py:248`). Carve adds no lattice reach, but
carve's own input is the despeckled, compacted scan: `clean` runs despeckle,
compacts with `select` (`filters.py:303`), carves the compacted scan
(`filters.py:306`), then rebuilds a grid over it for restore
(`filters.py:317`). So the post-carve state at rows `r-1 … r+1` requires
*correct despeckle output* at rows `r-1 … r+1`, and despeckle at row `r'`
requires raw rows `r'-1 … r'+1`.

```
h(despeckle) + h(carve) + h(restore)  =  1 + 0 + 1  =  2
```

**Composed filter-chain halo: 2 rows, symmetric.** A band whose core is
`[b0, b1)` needs raw rows `b0-2 … b1+1` to produce correct final keep
decisions for its core.

The max would have given 1. A band-edge sample judged with a 1-row halo would
have its restore support counted against neighbours whose *own* despeckle
verdict was computed from an incomplete neighbourhood — so a neighbour could be
present-and-alive when the full-scan run had despeckled it away, or vice versa.
Support crossing the `min_support = 5` threshold (`filters.py:203`) in either
direction flips the restore decision, and a flipped restore decision changes
the retained set, the triangle set and the ledger.

### Adding triangulation

Triangles are owned by the band whose core contains the **top row of the quad**
(§6). A core `[b0, b1)` therefore owns quads with top rows `b0 … b1-1`, and
those quads touch lattice rows `b0 … b1` — one row below the last core row
(`triangulate.py:113-116`).

So the rows needing correct *final keep state* are `b0 … b1`, and each of those
needs raw rows ±2:

```
raw rows required  =  [b0 - 2,  b1 + 2]   inclusive
```

Relative to the core interval:

| Direction | Rows beyond the core | Composition |
|---|---|---|
| Up, beyond `b0` | **2** | 2 (filter chain) + 0 (triangulation reaches up 0) |
| Down, beyond the last core row `b1-1` | **3** | 1 (triangulation reaches down to `b1`) + 2 (filter chain around `b1`) |

The requirement is genuinely asymmetric: 2 up, 3 down.

---

## 6. Encoding it in the `iter_row_bands` call

`iter_row_bands` (`e57_reader.py:250`) takes a single symmetric `halo`. Its
three relevant expressions are:

```
data_start = max(row_min, core_start - halo)          # e57_reader.py:317
data_stop  = min(row_stop, core_stop + halo)          # e57_reader.py:318
chosen     = (row >= data_start) & (row < data_stop)  # e57_reader.py:319
```

`data_stop` is exclusive, so the last data row supplied is
`core_stop + halo - 1`.

Solving both directions against the §5 requirement:

```
first data row  b0 - halo      ≤  b0 - 2        ⟹  halo ≥ 2
last  data row  b1 + halo - 1  ≥  b1 + 2        ⟹  halo ≥ 3
```

> ### **Composed halo = 3 rows.**
>
> `iter_row_bands(chunks, row_min=…, row_stop=…, band_rows=…, halo=3)`

The symmetric parameter supplies one redundant row above the core. That is
accepted: the cost is one extra row of buffered points per band (≈0.4 % of a
256-row band) and the alternative is an asymmetric-halo change to
`iter_row_bands`, which is a `src/` change with no measured benefit.

### No change to `iter_row_bands` is required for halo 3

Both of the assembler's other halo-dependent expressions already generalise,
which was verified by reading them:

- **Readiness.** `need_row = min(core_stop + halo, row_stop) - 1`
  (`e57_reader.py:310`) is exactly the last data row for any `halo`, so the band
  is emitted at the right moment.
- **Retention.** `next_keep_from = max(row_min, core_stop - halo)`
  (`e57_reader.py:322`) equals the *next* band's `data_start` for any `halo`, so
  no needed row is discarded and no unneeded row is carried.

**No halo-derived minimum core size.** `band_rows` is the number of rows owned
by a core, not the size of the surrounding data window. A one-row core can be
correct when the assembler supplies the required clipped halo. The only
correctness constraint imposed here is the existing `band_rows > 0` check in
`iter_row_bands` (`e57_reader.py:270`). An implementation may choose a larger
operational minimum for throughput, but it must label that as a measured
performance policy rather than derive it from `halo = 3`.

**Default is not the answer.** `iter_row_bands`' default is `halo: int = 1`
(`e57_reader.py:256`). Band-local geometry must pass 3 explicitly. A default
change to 3 is rejected because this is a generic ingestion primitive and
other consumers can have different reach. Production Pipeline A call sites
are auditable only when they pass the composed value explicitly.

### The band's row range, restated

For a core `[b0, b1)` with `halo = 3`, `iter_row_bands` supplies data rows
`b0-3 … b1+2`. Of those:

| Rows | Role |
|---|---|
| `b0-3` | redundant (symmetry cost) |
| `b0-2 … b0-1` | filter-chain halo, up |
| `b0 … b1-1` | **core — the only rows written for** |
| `b1` | triangulation's down-reach row; its final keep state must be correct |
| `b1+1 … b1+2` | filter-chain halo around row `b1` |

---

## 7. Judge-once tiling and band ownership

Halo evidence is read repeatedly by design. It must never be *counted*
repeatedly. Two independent judge-once mechanisms exist and both must hold.

### Within a scan — `_judged_bands`

`_judged_bands` (`filters.py:345`) tiles `[0, rows)` in non-overlapping steps.
Its docstring states the reason directly: overlapping judged ranges would be
"harmless for a keep-mask, but it would double-count in the statistics, and the
statistics are how thresholds get tuned" (`filters.py:349-353`). Both call
sites then slice the judged sub-range back out of the padded band before
writing — `s0, s1 = b0 - lo, b1 - lo` (`filters.py:136`, `filters.py:260`).

### Across streaming bands — `RawRowBand` core ownership

`RawRowBand` carries `core_row_start` / `core_row_stop` alongside
`data_row_start` / `data_row_stop` (`e57_reader.py:127-147`) precisely so the
consumer can distinguish evidence from ownership.

### The composed rule

> **A sample's disposition is written exactly once, by the band whose core
> contains its lattice row. A triangle is written exactly once, by the band
> whose core contains its quad's top row. Halo rows produce no ledger entry, no
> triangle, no QA sample and no output record.**

Consequences, each of which is a testable obligation:

1. **The ledger stays exclusive.** `FilterStats` requires that every source
   sample finishes in exactly one field (`types.py:246-256`) and
   `require_balanced` raises otherwise (`types.py:284`). Double-counting halo
   samples would inflate `input_points`' accounted total in the direction that
   *looks* like more evidence — a ledger that fails loudly rather than
   silently, which is the correct failure but only if the rule is enforced.
2. **`restored_from_carve` stays an event, not a disposition** (`types.py:252`,
   `filters.py:320-322`). A sample restored in one band and read as halo
   evidence in the next must be counted once.
3. **Triangles are not duplicated at band seams.** The in-memory path already
   has this property: `ScanGrid.bands(overlap=1)` (`grid.py:158`) makes band
   *k* produce quads with top rows up to `end-1` and band *k+1* start at
   `end-1`, so the shared row is a top row in exactly one band. The streamed
   ownership rule reproduces it.
4. **QA measures each retained sample once.** Both QA directions run over the
   final retained set (`pipeline.py:157-172`); a duplicated sample would be
   double-weighted in the RMS.

### Ownership is defined on lattice row, never array index

`clean` compacts the scan between stages (`filters.py:303`, `filters.py:324`,
via `select`, `grid.py:90`), which renumbers every array index. A band-local
implementation will do the same within a band, so array indices are band-local
and meaningless across bands. Ownership, the frontier in
`PHASE1-ISLANDS-FINALISATION.md` and the segment layout in
`PHASE1-TILE-CONTRACT-V0.md` are therefore all keyed on `(row, col)` or on the
stable `sample_id` assigned at read time (`e57_reader.py:399`), never on a
position in a compacted array.

---

## 8. What the code does not settle

Stated rather than assumed, per the brief.

1. **The composed halo is derived, not measured.** §5 is an argument about what
   the implementations index. It is falsified or confirmed by the equivalence
   harness (§9), not by this document.
2. **Chunk boundaries are a second, independent axis.** `iter_row_bands`
   consumes `RawChunk`s of `chunk_points` (`e57_reader.py:197`) and buffers them
   until a band is ready. Nothing in the halo derivation depends on
   `chunk_points`, but that independence is an assumption about the assembler,
   not a proof; it is tested separately (`PHASE1-DETERMINISM-SPEC.md` §7).
3. **The row-major precondition is enforced but not guaranteed by the format.**
   `iter_row_bands` rejects a non-monotonic stream (`e57_reader.py:286-290`,
   tested at `tests/test_qa_ledger.py:151`). Whether every authorised
   structured E57 in the corpus is row-major has not been verified in this
   task; only the reference station's lattice dimensions were read from
   `docs/DATA-INVENTORY.md`.
4. **Non-`ROW_COL` lattices are out of this derivation's scope.**
   `iter_row_bands` requires `rowIndex` (`e57_reader.py:281`). A
   `SPHERICAL`-tier scan has its row index synthesised inside `_build`
   (`e57_reader.py:368`) after the whole scan is read, so band assembly for that
   tier is an open design question, not covered here.
5. **The halo says nothing about carve's neighbour dependency.** That is a
   phase-ordering requirement (§3) and is designed elsewhere.

---

## 9. Falsification

**The claim.** The composed read-halo for despeckle → carve → restore →
triangulate is 2 rows up and 3 rows down, encoded as `halo=3` in the
`iter_row_bands` call; at that halo, band-local output is identical to
full-scan output, and at any smaller halo it is not.

**What proves it wrong.** The streamed ≡ in-memory equivalence harness
(PLAN.md §5 item 5), run as a matrix over `halo ∈ {1, 2, 3, 4}` and multiple
positive core sizes on fixtures that exercise all three filter stages
together. At least one purpose-built **halo witness** must put a mover and a
thin restored feature across a boundary so that the missing lower support row
at `halo = 2` changes a known decision. The prediction for that witness is a
specific shape of table:

| Observation | Verdict |
|---|---|
| `halo = 3` and `halo = 4` exact; `halo = 1` and `halo = 2` differ | **Confirms** the derivation |
| `halo = 2` also exact on the purpose-built witness whose expected differing decision was independently established | Derivation is **wrong**; re-derive before implementation proceeds |
| `halo = 3` differs from the in-memory result | The halo is **larger than 3**, or the divergence has another cause (ownership, ordering, or a stage the table missed). Re-derive; do not simply raise the number |
| Every halo exact on an ordinary fixture | That fixture is not a halo falsifier. This is acceptable only if the separate purpose-built witness still rejects 1 and 2 |

**Evidence that closes it.** The completed matrix, committed with the command
that produced it, identifying the smallest halo at which every compared field
(mesh, ledger, both QA directions — enumerated in
`PHASE1-DETERMINISM-SPEC.md` §7) meets its specified equivalence tier. The
halo witness must reject 1 and 2 and accept 3 and 4. If the smallest accepted
number is not 3, this document is amended with the measured number and the
derivation error named.

**Secondary falsifier.** A test asserting that `_judged_bands`' `halo`
parameter has no effect on its output (`filters.py:345`). If that test fails,
the parameter has been wired up and §4 Trap 1 no longer holds.
