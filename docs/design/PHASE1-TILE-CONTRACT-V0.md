# Phase 1 design — versioned intermediate tile contract, v0

**Status:** Design, not yet implemented. Documents only.
**Scope:** PLAN.md §5 design-first item 3, plus the working-memory budget table
skeleton required by PLAN.md §5 item 7.
**Date:** 17 August 2026.
**Companion documents:** `PHASE1-HALO-CALCULUS.md` (ownership),
`PHASE1-ISLANDS-FINALISATION.md` (two-pass structure),
`PHASE1-DETERMINISM-SPEC.md` (what must be reproducible).

Every structural claim cites the function it was read from. Where the design
needs a fact the code does not settle, the gap is stated.

---

## 1. Two duties, one contract

This contract is not only Phase 1 plumbing. It carries two commitments made
elsewhere:

| Duty | Source | What it requires of v0 |
|---|---|---|
| **The draft manifest DEC-005 benchmarks against** | DEC-005; PLAN.md §7, §8 | Phase 3 builds against a *draft versioned RapidMesh manifest*; early Phase 4 benchmarks glTF + meshopt-class payloads against a bespoke binary and commits on the numbers. v0's manifest must therefore be **payload-agnostic**: it describes tiles, bounds, errors, versions and digests, and does not assume the encoding of the geometry it points at |
| **Phase 7's crash recovery and resumability** | PLAN.md §12, §13 (crash recovery, resumable jobs, crash-mid-write recovery, power-loss atomicity) | v0's finalise and recovery semantics are the ones Phase 7 hardens. They are designed once, here, rather than retrofitted |

A third constraint applies from the start:

**DEC-004.** Per-vertex source identity never leaves the server. `MeshData`
carries `source_sample_id` (`types.py:204`), populated by `build_mesh`
(`triangulate.py:267-271`) from the stable read-time ids (`e57_reader.py:399`).
Every provisional v0 segment is therefore server-only: triangle records also
use those ids and cannot be made client-safe merely by excluding a separate
`sid` file. Pass B creates a distinct tile-local index space and an allow-list
client manifest. Enforcement is Phase 7's; this structural redaction boundary
is v0's.

---

## 2. Write-order is not view-order

Bands are **row** ranges over the lattice, and lattice columns are azimuth:
`_lattice_from_angles` derives `col` from azimuth and `row` from elevation
(`e57_reader.py:651-652`), and `dense_rows(r0, r1)` returns shape
`(r1 - r0, cols)` — every column (`grid.py:150`).

A row band on a rotating-head station is therefore an **elevation annulus**: a
complete ring of azimuth at a band of elevations. Spatially it is not compact.
The two orders are genuinely different:

| | Write order | View order |
|---|---|---|
| Unit | band-aligned segment (an elevation ring) | spatial tile (a compact bounded region) |
| Chosen by | the ingestion sweep — `iter_row_bands` (`e57_reader.py:250`) drives it, and the stream is required to be row-major (`e57_reader.py:286-290`) | the viewer's frustum and distance |
| Why it cannot be the other one | Re-ordering to spatial tiles during the sweep would need the whole station resident, which is the thing being avoided | Serving elevation rings means every ring intersects almost every view, so first paint fetches nearly everything |

> **The contract.** The sweep writes band-aligned append-only segments. A
> bounded finalisation pass — Pass B of `PHASE1-ISLANDS-FINALISATION.md` —
> merges them into a new immutable spatial generation, verifies its manifest,
> and publishes `current.json` last.

Pass B is the natural place for the merge because it already re-reads every
segment in canonical oriented-triangle order for `component-area-v1`
(`PHASE1-ISLANDS-FINALISATION.md` §4.1). The spatial re-sort is a by-product of
a pass that has to happen anyway.

The merge is bounded, not an in-memory station sort. Pass B derives each
triangle's spatial tile id from the versioned tile-origin/extent rule, appends
the record to a bounded set of shard spools, closes least-recently-used spools
when the open-file cap is reached, then finalises one tile at a time. Canonical
triangle ordering uses per-segment bounded sorts followed by k-way external
merges. Two explicit merge views exist: `(final_component_root,
canonical_oriented_triple)` for `component-area-v1`, and
`canonical_oriented_triple` for `reverse-qa-v2` and deterministic tile
records. They may share bounded run files but must not pretend one ordering is
the other. Tile-local vertex remapping is released after that tile is written.
The spool block, merge heap and one-tile remap are B13 consumers and receive
explicit byte caps before the working-memory budget can close.

**Phase 1 does not decide the spatial tiling.** ADR-006 Decision 2a makes tile
size a measured parameter, not a chosen constant, and specifies the halo,
locked-boundary, ownership, stitching and seam-local QA rules for
*decimation* tiles at Phase 3. v0 must carry a tile index whose granularity is
a setting, not a constant, so the Phase 3 benchmark can vary it without a
format change.

---

## 3. Object model

```
station/
  generations/<generation-id>/
    run.json                    immutable run identity and settings (§5.3)
    journal.jsonl               append-only progress record (§5.3)
    state/component.rmlog       append-only union/retirement/count deltas
    state/<band>.rmstate        complete resumable component checkpoint
    component.rmcomp            completed component/root/verdict table
    seg/<kind>-<band>.rmseg     server-only band-aligned segments
    tile/<tile-id>.rmtile       final spatial tiles, written in Pass B
    manifest.json               immutable server manifest
    [client-manifest.json]      optional redacted delivery manifest; absent in Phase 1
  current.json                  atomic pointer, written LAST
```

`out/` is already in `.gitignore` (`.gitignore:26`), as are `*.rmx`
(`.gitignore:22`) and every point-cloud extension. Intermediates therefore have
a home that cannot be committed by accident. Nothing under this tree is ever
staged.

Every generation is immutable once published. A rebuild writes a new
generation id and never replaces files referenced by `current.json`.
Publication is one temp-write/fsync/`os.replace` of `current.json` after the
new generation and its manifest are complete. A reader either sees the old
complete generation or the new complete generation; it never sees an old
manifest referring to replaced tile bytes.

### Server-only provisional segment kinds

| Kind | Contents | Tier |
|---|---|---|
| `tri` | core-owned triangles as stable sample-id triples, plus the global component id per triangle (`PHASE1-ISLANDS-FINALISATION.md` §3.7) | **server only** |
| `pos` | positions keyed by stable sample id, as project-axis f32 offsets from the station's f64 origin | **server only** |
| `nrm` | normals keyed by stable sample id, in project axes | **server only** |
| `rgb` | colour keyed by stable sample id, `u8×3`; absent where the source has no colour | **server only** |
| `disp` | per-sample filter dispositions, final at Pass A | **server only** |
| `sid` | explicit per-vertex `source_sample_id` join records | **server only — DEC-004** |

No provisional segment is client-eligible. The `tri` records themselves carry
per-vertex source identity, so marking only the separate `sid` kind as
server-only would not satisfy DEC-004.

### Final tile records and redaction boundary

Pass B remaps each retained tile's vertices to dense **tile-local** indices.
Final `rmtile` triangle records reference only those local indices. A
client-eligible tile contains positions, normals, optional colour and local
triangle indices; it contains no lattice `(row, col)`, stable sample id,
component id, source digest or reversible per-vertex evidence join.

The remap is deterministic: server-side vertices are ordered by stable sample
id, assigned local indices `0…V-1`, and then the ids are discarded from the
tile. Triangles are ordered by their canonical oriented source-id triples
before conversion to local indices, with multiplicity preserved. Thus the
client payload is reproducible without containing the evidence keys that
defined its order.

The server manifest may join to the evidence envelope and source digest. The
client manifest is derived by an allow-list serializer and carries only an
opaque artefact identifier derived from the redacted payload-manifest digest
(never from the source digest), payload digest, bounds, origin,
versions and declared LOD/error metadata. It never copies `source_sha256`.
The client serializer must be tested by field denial, not by deleting fields
from a server JSON object after serialisation.

Phase 1's undecimated full-resolution generation is server-only and produces
no client manifest. `client-manifest.json` becomes valid only for a later
decimated, error-bounded client tier satisfying DEC-004; the optional path is
shown here so the redaction boundary is structural from v0, not to claim that
Phase 1 output is client-deliverable.

Positions are project-axis offsets from an f64 origin, per
`SPATIAL-CONTRACT.md` §2.4 and `MeshData` (`types.py:188-199`). `build_mesh`
applies the scanner rotation exactly once and only then narrows to f32
(`triangulate.py:290-292`). v0 stores what `build_mesh` produces; it introduces
no second transform, and a tile must never introduce an independent rotation or
scale (`SPATIAL-CONTRACT.md` §2.4).

### v0 serialisation requirements

Before implementation, each `rmseg`, `rmstate`, `rmcomp` and `rmtile` schema
is fixed in this document or a directly referenced schema file with:

- an eight-byte magic and integer contract version;
- little-endian fixed-width numeric fields (`f32`, `f64`, `u8`, `u32`,
  `u64`, `i64`) and explicit array lengths;
- record-count and byte-length limits checked before allocation;
- canonical record ordering for every server segment;
- UTF-8 JSON encoded with sorted keys, no insignificant whitespace and finite
  numbers only for `run.json`, journal records and manifests;
- SHA-256 calculated over the exact stored bytes; a JSON self-digest is over
  the canonical object with the `sha256` member omitted;
- refusal of unknown major versions and of unknown required fields.

These are contract requirements, not implementation choices left implicit in
v0. The logical field tables are fixed as follows; padding bytes, if any, are
zero and included in the digest.

| Object | Required v0 fields |
|---|---|
| Common binary header | `magic[8]`, `contract_version:u32=0`, `kind:u16`, `flags:u16`, `header_bytes:u32`, `record_count:u64`, `payload_bytes:u64` |
| `tri.rmseg` record | `sid0:i64`, `sid1:i64`, `sid2:i64`, `component_id:u32`; triples use the production winding |
| `pos.rmseg` record | `sample_id:i64`, `position:f32[3]` |
| `nrm.rmseg` record | `sample_id:i64`, `normal:f32[3]` |
| `rgb.rmseg` record | `sample_id:i64`, `rgb:u8[3]` |
| `disp.rmseg` record | `sample_id:i64`, `final_filter_disposition:u8`, `restored_event:u8` |
| `component.rmlog` record | `band_start:u32`, `operation:u8` (`birth`, `union`, `count`, `retire`), `id:u32`, `other_or_root:u32`, `triangle_count_delta:u64` |
| `rmstate` fixed fields | `core_start:u32`, `core_stop:u32`, `source_stream_offset:u64`, `next_component_id:u32`, `cols:u32`, `component_log_bytes:u64`, `component_log_sha256:u8[32]`, then `frontier:i32[cols]` and live records `(id:u32,parent:u32,triangle_count:u64,state:u8)` |
| `rmcomp` record | `id:u32`, `root:u32`, `triangle_count:u64`, `area:f64`, `smallest_positive_area:f32`, `retired:u8`, `verdict:u8`, `area_fallback:u8` |
| `rmtile` fixed/arrays | `tile_id:u64`, `origin:f64[3]`, `bounds:f64[6]`, `vertex_count:u32`, `triangle_count:u32`, `positions:f32[V,3]`, `normals:f32[V,3]`, optional `rgb:u8[V,3]`, `triangles:u32[T,3]` |

`run.json` contains the resume-identity fields listed in §5.3. The server
manifest contains generation id, server evidence join, versions, settings,
component/area summary and all tile metadata/digests. The client manifest is
an allow-listed projection containing only generation artefact id, contract
and payload versions, units, origins/bounds, LOD/error metadata and tile
paths/digests. No unknown server field flows through automatically.
`current.json` contains only `generation_id`, relative server-manifest path,
manifest byte length and SHA-256; all are verified before following the
pointer.

---

## 4. Deterministic ownership

Restated from `PHASE1-HALO-CALCULUS.md` §7 because the contract depends on it:

> A **sample**'s record is written by the band whose core contains its lattice
> row. A **triangle**'s record is written by the band whose core contains its
> quad's top row. Halo rows produce no record.

Consequences for the file format:

1. **Segments are disjoint by construction.** No de-duplication step exists,
   and none is permitted — a de-duplication step would hide an ownership bug
   rather than fail on it.
2. **Records are keyed on `(row, col)` or on the stable `sample_id`**
   (`e57_reader.py:399`), never on a position in a compacted array. `clean`
   renumbers arrays at every stage via `select` (`grid.py:90`, called at
   `filters.py:303` and `filters.py:324`), so array indices are band-local and
   meaningless across segments.
3. **Segment arrival order is band order**, but it is not a numerical or QA
   ordering contract. Pass B performs the bounded canonical oriented-triangle
   merge used by `component-area-v1` and `reverse-qa-v2`
   (`PHASE1-ISLANDS-FINALISATION.md` §4.1;
   `PHASE1-DETERMINISM-SPEC.md` §5(h)).
4. **Spatial tile ownership** in Pass B follows ADR-006 Decision 2a: a shared
   vertex or triangle belongs to exactly one tile by a rule independent of
   processing order or thread scheduling. v0 records the rule's identifier and
   version in the manifest so a reader can tell which rule produced a package.

---

## 5. Atomic finalise, checksums, version, recovery

### 5.1 Atomic write

Every file inside an unpublished generation is produced as **write to
`<name>.tmp` → flush → `os.fsync` → `os.replace` into place**. Once a
generation is published its files are immutable. `current.json` is the only
published path replaced in place, and is written last after verifying every
file referenced by the new manifest.

Three portability facts, since the workstation deployment shape is Windows
(PLAN.md §13):

- `os.replace` is atomic on Windows and POSIX **within a single volume**. The
  temporary file must be created in the destination directory, not a system
  temp directory.
- `os.rename` over an existing file raises on Windows. `os.replace` is the
  required call; `os.rename` must not be used for this.
- A rename is atomic with respect to *visibility*, not durability. Directory
  durability additionally requires an fsync of the containing directory on
  POSIX; that call is not available on Windows and the Windows behaviour is
  **not settled by anything in this repository** (§8, gap 2).

### 5.2 Checksums and version

| Field | Rule |
|---|---|
| `contract_version` | integer, `0` for v0. A reader refuses a higher major version rather than guessing |
| per-segment digest | SHA-256 of the file's bytes. `qa.sha256_file` (`qa.py:52`) already hashes a path incrementally without materialising it or exposing it, and is reused |
| per-segment length and record count | recorded, so truncation is detectable without rehashing |
| manifest self-digest | SHA-256 over the canonical UTF-8 manifest object with its `sha256` member omitted |
| `source_sha256` | station source digest and evidence join, present in `run.json` and the **server manifest only**; forbidden in `client-manifest.json` |

No station name or source path appears in a segment, tile or either manifest.
Runtime geometry artefacts necessarily carry coordinate values: project-axis
f32 offsets and the f64 origins and bounds required to reconstruct them under
`SPATIAL-CONTRACT.md`. The repository-hygiene prohibition applies to real
client coordinate values in repository reference metadata, design examples,
tests and scratch material; it does not prohibit required runtime geometry in
generated artefacts, which remain outside the repository under `out/`. A
client tile or manifest may therefore carry the origins, bounds and offsets
needed to render its geometry, but it contains no source hash, stable sample
id or other per-vertex evidence join.

### 5.3 Recovery semantics — and why a journal is required

The manifest is written last. That is the correct atomicity rule and it is
**not sufficient for resumability**: a crash mid-sweep leaves no manifest, so
without more information the only safe action is to redo the whole station.

An append-only `journal.jsonl` plus complete state checkpoints close that gap.
`run.json` is written before the first band and binds the generation to source
digest, scan index, effective pose/transform chain, RapidMesh commit and
dependency versions, contract and metric versions, complete settings, seeds
and requested/effective thread counts. A resume refuses any mismatch.

One journal line is appended and fsynced after a segment and its corresponding
state checkpoint have both been replaced successfully. Journal JSON is valid,
with half-open bounds represented as two integers:

```
{"unit":"band","core_start":256,"core_stop":512,"segment_set_sha256":"…","state":"state/00000256.rmstate","state_sha256":"…"}
```

An `rmstate` checkpoint plus the verified prefix of `component.rmlog` contains
the complete restart state after that band: frontier entries, live canonical
components, every parent/root union, triangle-count delta and retirement so
far, next component id and last consumed source-stream offset. At the end of
Pass A the log is the completed membership/count input; Pass B combines it
with area and keep/drop decisions to write `component.rmcomp`. The frontier
alone is insufficient, and copying the entire retired table into every
checkpoint is unnecessary.

Recovery rules:

1. Read the journal. Truncate it at the first line that fails to parse — a
   partial line is a crash during append and the unit it describes is not
   trusted.
2. Verify `run.json` against the requested resume and then verify every named
   segment and state file, length and digest. Any mismatch discards that unit
   and every later one — a segment's meaning depends on the component state that preceded it
   (`PHASE1-ISLANDS-FINALISATION.md` §3.3), so recovery is a **prefix**
   operation, not a set operation. Truncate `component.rmlog` to the byte
   length and digest recorded by the last verified state; later bytes are
   uncommitted debris.
3. Load the last verified complete `rmstate` and resume at the first band not
   covered by the prefix. The recorded source offset is a verification target,
   not an assumed random-access capability: reopen and seek only if the E57
   API path has proved that operation; otherwise read and discard the verified
   prefix deterministically until the offset is reached. With no verified
   state, restart Pass A from the beginning.
4. Pass B is not resumable in v0. It re-runs from the start of the verified
   segment set. It is a bounded sequential pass, so the cost is bounded; making
   it resumable is deliberately deferred.
5. The manifest is derived from the verified journal and completed component
   table, never from a directory listing. A file present but unjournalled is
   debris and ignored.
6. A completed unpublished generation may be verified and then published by
   replacing `current.json`. An already-published generation is never resumed
   or modified; a rebuild receives a new generation id.

---

## 6. Working-memory budget table (not yet closed)

One station, Pipeline A, against the approved exact limits:

- processing working memory: **≤ 512,000,000 bytes**;
- measured peak RSS: **≤ 1,500,000,000 bytes**.

The committed reference metadata has two high-resolution widths. Allocation
arithmetic uses the worst lattice, **6096 columns × 2387 rows = 14,551,152
source records**, at 100.0 % *source-record lattice fill*. That count sizes
raw ingestion and dense lattice grids; it is not the valid-return count.
`StructuredScan.xyz` contains only valid returns, and the committed ADR's
measured example contains 12,476,504 valid samples.

**Every figure below is arithmetic from the cited code, not a measurement.**
Measurement is a Gate 1 obligation (PLAN.md §5 item 10) and the column marked
*Status* says which lines are entered and which are skeleton.

| # | Consumer | Reference bound/estimate | Lifetime and overlap | Status |
|---|---|---|---|---|
| **B1** | Neighbour carve grids, resident: **64,000,000 B cap each**; **116,409,216 B** for two worst-reference grids | Pass A target sweep; coexists with B4–B8 and B10/B12/B13, but not B9/B11 finalisation if grids are released | **ENTERED — compatible in isolation** (§6.1) |
| **B2** | Current whole-scan carve-grid transform: **≥748,590,240 B** using the measured 12,476,504 valid returns | Grid preparation only; may coexist with already completed resident grids; must be eliminated | **ENTERED — current path fails** (§6.2) |
| **B3** | Current `close_dropout`: **305,574,192–421,983,408 B** on the worst reference grid | Grid preparation only; may coexist with already completed grids; must be replaced with bounded immutable-source row blocks | **ENTERED — current path fails design** (§6.3) |
| **B4** | Raw band buffer: ≈59–115 MB before chunk overshoot; current repeated concatenate can approximately double buffered storage | Pass A; overlaps B1, B5–B8, B10/B12/B13 | **ESTIMATE — dtypes/overshoot unverified** (§6.4) |
| **B5** | libE57 buffers + copied chunk: ≈74–144 MB at 1,000,000 records | Pass A ingestion; overlaps B4 | **ESTIMATE** (§6.4) |
| **B6** | Dense index: `(256 + 6) × 6096 × 4` = **6,388,608 B** | Pass A band processing | **ENTERED** |
| **B7** | Despeckle/restore band working arrays: ≈45 MB | Pass A filter stage; exact live-array schedule not yet measured | **ESTIMATE** |
| **B8** | Island Pass A state: ≤≈4.95 MB decimal | Pass A; overlaps B1/B4–B7/B10/B12/B13 | **ENTERED — conservative bound** |
| **B9** | Pass B component tables: ≤**77,606,144 B** | Pass B; carve grids and raw ingestion buffers must be released first | **ENTERED — conservative bound** |
| B10 | Band triangle output buffer before flush | Pass A | **SKELETON** |
| B11 | Bounded reverse-QA row window and KD tree | Pass B; fixed `qa_window_rows`, independent of processing bands | **SKELETON** |
| B12 | Observation-store write buffer | Pass A and/or Pass B, to be scheduled explicitly | **SKELETON** |
| B13 | Segment, external-sort, tile-partition and digest buffers | Pass A/Pass B | **SKELETON** |

**The complete working-memory budget is not closed.** B10–B13 have no byte
bounds, B4/B5 retain dtype and overshoot uncertainty, and the table contains
consumers from different lifetimes that must not simply be summed. Before Gate
1 the implementation records a phase-by-phase maximum-live allocation table
and demonstrates both exact byte limits with `VmHWM`/`PeakWorkingSetSize`.

### 6.1 The carve-grid line, derived

`CARVE_MAX_CELLS = 16_000_000` (`grid.py:56`). The grid is `float32`
(`grid.py:256`, `np.full(height * width, np.inf, np.float32)`).

```
16,000,000 cells × 4 B  =  64,000,000 B  =  64.0 MB  =  61.04 MiB
```

The constant's own comment says "16 M cells is 64 MB as float32"
(`grid.py:45-47`) — **verified**.

For the worst recorded reference lattice, no reduction applies. `build` takes
`rows = max(lattice.rows, 8) = 2387`, `cols = max(lattice.cols, 16) = 6096`
(`grid.py:230`) and only reduces if `rows × cols > max_cells`
(`grid.py:231-233`):

```
2387 × 6096  =  14,551,152 cells   <  16,000,000        → no reduction
14,551,152 × 4 B  =  58,204,608 B
```

`pipeline.carve_grids` builds `nearest = 2` grids by default
(`pipeline.py:224`), and `mesh_station` holds them for the whole run
(`pipeline.py:110`, `filters.py:305-312`):

| Configuration | Resident | % of 512,000,000 B |
|---|---|---|
| 1 grid at the `CARVE_MAX_CELLS` cap | 64,000,000 B | 12.5 % |
| 2 grids at the cap | 128,000,000 B | 25.0 % |
| 1 worst-reference grid | 58,204,608 B | 11.4 % |
| **2 worst-reference grids (the default count)** | **116,409,216 B** | **22.7 %** |
| 4 grids at the cap | 256,000,000 B | 50.0 % |

**Resident grids are compatible with the budget in isolation.** The complete
Pass A working set is not proven until B10–B13 and the live-allocation schedule
are closed. Their construction, as currently written, is not compatible.

### 6.2 Why the chunked build is mandatory, not an optimisation

`CoarseRangeGrid.build` calls `scan.pose.local_to_world(scan.xyz)` on the whole
scan (`grid.py:234`). Following that through `types.py`:

- `local_to_world` = `rotate_local(local) + translation` (`types.py:94-99`)
- `rotate_local` = `np.asarray(local, float64) @ rotation.T` (`types.py:85-92`)

`StructuredScan.xyz` contains valid returns, not all lattice records. The
committed ADR's measured high-resolution example has **12,476,504 valid
samples**. The immediate allocation lower bound for that population is:

| Array | Bytes |
|---|---|
| `scan.xyz`, f32, caller-held | 149,718,048 |
| f64 copy made by `np.asarray(..., float64)` | 299,436,096 |
| matmul or translated result, f64 | 299,436,096 |

The f64 copy and the matmul result are live simultaneously, and the caller's
f32 array is live throughout:

```
149,718,048 + 299,436,096 + 299,436,096 = 748,590,240 B
```

`from_world_points` then adds another f64 `q`, a range vector and boolean-mask
copies (`grid.py:250-253`), so the true peak is higher.

> **A single current neighbour-grid build has an immediate lower bound of
> 748,590,240 B, before later temporaries, and therefore fails the
> 512,000,000-byte working budget.**

This is the measured-in-arithmetic justification for PLAN.md §5 item 7:
chunked `local_to_world` binning, and grids cached as artefacts so the build
cost is paid once per station rather than once per neighbour-of-a-neighbour.
The chunking pattern already exists in the codebase — `carve_movers` transforms
in 2 M-sample chunks for exactly this reason (`filters.py:187-189`,
`filters.py:174-177`).

The streamed station schedule is also explicit:

1. read lightweight headers, effective poses, hashes and lattice metadata;
2. select neighbour ids from station origins without holding their point
   arrays;
3. build each required carve grid from raw chunks, one station at a time, and
   publish the cache artefact;
4. release construction buffers before loading the next grid;
5. process the target with only its selected completed grids resident.

The current `carve_grids(list[StructuredScan])` orchestration is evidence of
behaviour, not a memory-safe production interface.

### 6.3 `close_dropout` is the second transient

`close_dropout` (`grid.py:265-293`) runs 8 neighbour shifts, each allocating
`np.roll` twice (`grid.py:287`), then `np.isfinite(r)`, then `np.where(...)`,
then `np.minimum(out, ...)` (`grid.py:292`). With `out = r.copy()`
(`grid.py:282`) live throughout, the simultaneously-live grid-sized f32 arrays
are between 5 and 7 depending on when CPython frees the intermediate roll:

| Assumption | Worst reference grid | At the 16 M cap |
|---|---|---|
| 5 live f32 grids + 1 bool | **305,574,192 B** | **336,000,000 B** |
| 7 live f32 grids + 1 bool | **421,983,408 B** | **464,000,000 B** |

Called once per grid build (`grid.py:236-237`, `fill_holes=True` by default).
Either bound is most of the working budget for one grid. `close_dropout` must
become row-blocked with a one-row halo while reading an **immutable original
grid** and writing a distinct output block. A naïve in-place sweep is
forbidden: it could read values filled earlier in the same pass, propagate a
return beyond the current one-cell neighbourhood and change carving
decisions. The row-blocked result must be bitwise equal to the present
whole-grid one-ring operation.

### 6.4 The ingestion lines, and the caveat on them

`iter_row_bands` accumulates with
`np.concatenate((buffered[field], array))` per chunk per field
(`e57_reader.py:299-303`), which allocates a new array of the combined size
while the old one is live — momentarily doubling the buffer. At `halo = 3` and
`band_rows = 256` the buffer holds ≈262 rows plus the readiness overshoot:

```
262 × 6096  =  1,597,152 source records before chunk overshoot
```

**Bytes per point are not settled by this repository.** The field set is chosen
at `e57_reader.py:219-223`, and pye57's `make_buffers` dtypes are not visible
here. Two bracketing assumptions:

| Assumption | B/point | Band buffer | Peak during concatenate |
|---|---|---|---|
| Tight — f64 spherical triple, i32 row/col, u8 colour, u16 intensity | 37 | 59.1 MB | 118.2 MB |
| Conservative — every field f64 except colour | 72 | 115.0 MB | 230.0 MB |

Two design requirements follow, both cheap:

1. **Pass an explicit `fields` tuple.** With `fields=None`,
   `iter_raw_chunks` selects *every* available supported field
   (`e57_reader.py:219-223`). A file carrying both the cartesian and the
   spherical triple would read both, for no benefit — `_positions` prefers
   spherical and ignores cartesian when both are present (`e57_reader.py:422`).
2. **Replace repeated concatenate with bounded chunk slices plus one
   preallocated band destination.** Copy each selected slice once, release
   fully consumed chunks promptly, and retain only the rows needed by the next
   halo. This removes O(n²) copying. The source slices and destination can
   still overlap during emission, so that overlap remains in B4 until
   measured; a list followed by one unconstrained `concatenate` must not be
   described as eliminating the transient.

`chunk_points` defaults to 1,000,000 (`e57_reader.py:200`), giving
74–144 MB across the libE57 buffers and the per-chunk `.copy()`
(`e57_reader.py:243`). A smaller value — 250,000 — is recommended so chunk
storage is small relative to the band buffer.

---

## 7. Carve-grid artefact naming and invalidation

Designed here because the cache is what makes §6.2 affordable.

### Name

```
out/<station>/carvegrid/<source_sha256>-<scan_index>-v<grid_version>-<cells>-<params_digest>.rmgrid
```

| Component | Source | Why it is in the key |
|---|---|---|
| `source_sha256` | `qa.sha256_file` (`qa.py:52`) | the file's bytes changed ⟹ the grid is invalid. Already the join key for QA evidence (`types.py:307`) |
| `scan_index` | index within the file; `probe` enumerates multi-scan E57s (`e57_reader.py:149`) | one digest, several scans |
| `grid_version` | integer, bumped by hand | the binning rule (`_spherical_bin`, `grid.py:326`), the hole fill (`close_dropout`, `grid.py:265`) or the resolution rule (`grid.py:227-238`) changed |
| `cells` | resolved `rows × cols` after the `CARVE_MAX_CELLS` reduction | a cap change silently changes the grid; the name must change with it |
| `params_digest` | SHA-256 over canonical fixed-width bytes for `(max_cells, fill_holes, effective transform chain, lattice dimensions, grid algorithm version)` | `build` bakes the resolved pose in; external registration or frame-resolution changes must change the key |

### Invalidation

**By name only.** A cached grid is used if and only if every component of the
name matches. Nothing is updated in place, no timestamp is consulted, and no
partial match is accepted. A stale grid does not raise — it silently carves the
wrong geometry from a neighbour, which is the failure class `filters.py`'s
opening rule exists to prevent (`filters.py:5-11`) and which the
`CoarseRangeGrid.build` docstring already warns about in both directions
(`grid.py:214-225`).

The artefact repeats its full key in a header, plus a SHA-256 of the payload,
so a renamed, truncated or half-written file fails closed. Grids are written by
the same temp-then-`os.replace` discipline as §5.1.

**The effective transform chain is in the key because of a fact the code does
not settle.** If a
project can be re-registered without the source E57's bytes changing — that is,
if Cairn stores registration outside the file — then `source_sha256` alone
would be stable while the pose changed, and a cached grid would be silently
wrong. Where registration is stored is a Cairn question; Cairn is read-only
and was not inspected for this task. Including every effective matrix,
translation and frame-resolution path in `params_digest` makes the cache
correct either way. The factual Cairn question is deferred to integration and
does not block Phase 1.

---

## 8. What the code does not settle

1. **pye57 buffer dtypes**, hence bytes per point — §6.4. Bracketed, not
   measured.
2. **Windows directory-entry durability after `os.replace`** — §5.1. POSIX
   directory fsync has no Windows equivalent exposed by Python; the ordering
   guarantee NTFS actually provides is not established by anything here. This
   is a Phase 7 hardening item (power-loss atomicity, PLAN.md §13) and the
   design must not claim durability it has not demonstrated.
3. **Spatial tile granularity** — a setting, deliberately undecided. ADR-006
   Decision 2a makes it a measured parameter at Phase 3.
4. **Whether Pass B's re-read is I/O- or CPU-bound** — sets the real cost of
   the two-pass structure. Gate 1 timing measurement.
5. **Every figure in §6 is arithmetic or a bracketed estimate, not a complete
   measurement.** Gate 1 requires
   `VmHWM` / `PeakWorkingSetSize` measurement (PLAN.md §5 item 10); the RSS
   watchdog is a guard with known overshoot, not the measurement.

---

## 9. Falsification

**The claims.**
(A) Server-only band segments plus a bounded Pass B merge produce the specified
equivalent output, complete state checkpoints make Pass A resumable at band
granularity, and immutable generations make publication atomic.
(B) Resident carve grids cost 64,000,000 B each at the cap and 58,204,608 B on
the worst reference lattice; their current whole-scan construction exceeds
512,000,000 B before later temporaries.
(C) Naming a carve grid by `(source digest, scan index, grid version, cells,
params digest)` is sufficient for correct invalidation.

**What proves them wrong.**

| Claim | Falsifying observation | Evidence that closes it |
|---|---|---|
| A — correctness | The streamed ≡ in-memory harness (PLAN.md §5 item 5) shows any field mismatch attributable to segment ordering or the Pass B merge | The harness's per-field diff across the band-size matrix |
| A — recovery/publication | A resume lacks any union/root/count/next-id state; a resumed output differs from uninterrupted output; or a reader can observe an old manifest with new tile bytes | Kill at mid-segment, between segment/state replace and journal append, and mid-journal append; assert equivalent final output; rebuild into a second generation while repeatedly reading `current.json` and observe only complete old or complete new generations |
| B — resident | `CARVE_MAX_CELLS × 4 ≠ 64,000,000`, or the resolved worst-reference grid is not 14,551,152 cells | An assertion on the constant and resolved `(height, width)` for a 6096 × 2387 lattice |
| B — construction | The current whole-scan construction stays within 512,000,000 B on the measured valid-return population | Measured `PeakWorkingSetSize`; regardless of outcome, the production chunked build must close the full concurrent budget rather than inherit the arithmetic estimate |
| C | A grid cached under a matching name produces different carve results from a fresh one after only an effective transform or frame-resolution path changes | Tests mutating each effective transform/path input and asserting the key changes |

**DEC-004 gate.** A denial test enumerates every field in every client manifest
and tile and asserts the absence of stable sample ids, lattice indices,
component ids, source hashes and other reversible evidence joins. It also
asserts that no provisional segment path is reachable from a client manifest.
The test must fail against a deliberately contaminated client tile and against
a client manifest containing `source_sha256`.

**A tolerance that never rejects is not a tolerance.** Every numeric assertion
above is paired with an injected-error case: perturb the input so the quantity
moves just past the stated bound, and assert the check fires.
