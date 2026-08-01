# ADR-006 — Memory budget definition, and the full-resolution mesh as an intermediate

**Status:** Accepted
**Date:** 2 August 2026
**Corrects:** the flat "≤ 512 MB peak RSS per station" target introduced
earlier the same day. That target was derived from the range-image working set
and never carried through to the output mesh. It is not physically achievable
as stated.

---

## The arithmetic that forces this

Measured on real data (`FINDING-002-QA-DEFINITION.md`): station
`02516.182_6.e57` produced 2,921,532 vertices and 5,760,953 triangles from
2,936,521 samples — **1.9719 triangles per vertex**, and essentially one vertex
per retained sample, because there is no decimation.

Applying the measured ratio to the high-resolution station (12,476,504 valid
samples of 14,548,765 cells):

| Component | f32 / i32 | f64 / i64 |
|---|---|---|
| Positions (12.41 M × 3) | 149.0 MB | 297.9 MB |
| Normals (12.41 M × 3) | 149.0 MB | — |
| Colour (12.41 M × 3, u8) | 37.2 MB | — |
| Triangle indices (24.48 M × 3) | 293.7 MB | 587.4 MB |
| **Finished mesh, output only** | **628.9 MB** | **1,071.5 MB** |

**The output alone exceeds 512 MB before a single byte of working memory.**
Chunked E57 reading does not help if the mesh is then assembled whole in RAM.

## The larger consequence

Across the sample site — 12 high-resolution and 18 medium stations:

```
397 million triangles     10.21 GB
```

**A full-resolution native mesh is a processing intermediate. It can never be a
deliverable.** No browser will load it, no CDN should serve it, and no client
needs it.

This changes decimation's status. It was listed as "the largest remaining
quality-per-byte win" — a trade-off to be tuned. It is not a trade-off. **It is
structural.** Without it there is no shippable product at native lattice
resolution, which is the project's entire premise.

## Decision 1 — Two separate, separately-tested budgets

| Budget | Value | Meaning |
|---|---|---|
| **Processing working memory** | **≤ 512 MB** | Everything resident during a station's processing, *excluding* output written incrementally to disk |
| **Measured peak RSS** | **≤ 1.5 GB** | The acceptance gate. Sized to fit alongside the web server on a 4 GB host |

Both are asserted by tests on the largest real sample station, not on fixtures.

### Three enforcement mechanisms, all different

**`RLIMIT_AS` does not enforce an RSS ceiling.** It bounds virtual address
space, which is not physical resident memory. Documentation must never conflate
them. Large NumPy allocations are mmap'd, so address space can far exceed RSS;
calibrating the limit is empirical, not arithmetic.

| Mechanism | Bounds | Failure mode | Role |
|---|---|---|---|
| `RLIMIT_AS`, calibrated | Virtual address space | `MemoryError` — **catchable** | Fail-safe. Gives a graceful abort with a traceback, a recorded peak and a clean project failure status, rather than a silent kill |
| RSS watchdog polling `/proc/self/status` `VmRSS` | Actual resident | Voluntary termination | The measured acceptance gate, and deployment-independent |
| Job-specific cgroup limit | Actual resident | SIGKILL | The hard physical boundary |

The original incident was a **cgroup** OOM (`Memory cgroup out of memory` in
`dmesg`), so the physical boundary already exists in the deployment. It simply
was not scoped per job.

The per-job cgroup is a **phase 0b stretch, not an entry requirement**,
provided phase 0a has disabled the live endpoints and phase 0b proves that
`RLIMIT_AS` plus the RSS watchdog protects uvicorn under the actual Lightsail
container limit. Per-job cgroups on that host mean either a short-lived
container per job, which needs Docker socket access and carries its own
security question, or cgroup v2 sub-hierarchy delegation. Neither is free.

## Decision 2 — Nothing full-resolution is ever fully materialised

The band interface in `grid.py` already makes memory independent of scan
height. That property must now extend through the whole pipeline:

```
read band -> filter band -> triangulate band -> QA band -> decimate band
          -> write tile -> release band
```

Consequences:

- **Triangulation, QA and decimation all become band-local.** Cross-band
  correctness needs the same overlap discipline already documented for
  triangulation (overlap 1) and despeckle (halo 1). Getting the overlap wrong
  leaves a one-row seam, which is subtle enough to ship by accident.
- **Output is written incrementally as spatial tiles**, not assembled and then
  saved.
- **Error-bounded decimation must be able to run per band or per tile.** A
  global quadric decimator over a 24.5 M-triangle mesh is exactly what this ADR
  exists to prevent. The boundary strategy is specified below and is not
  optional.

## Decision 2a — Tile boundary strategy

Independent per-tile decimation creates cracks, T-junctions and inconsistent
LOD transitions where tiles meet. The strategy is explicit:

| Element | Requirement |
|---|---|
| **Halo** | Each tile is decimated with a halo of neighbouring geometry present, so error metrics near the edge see the real surface rather than an artificial boundary |
| **Locked boundary vertices** | Vertices on a shared tile boundary are not collapsed. Both neighbours therefore agree on the boundary ring exactly |
| **Deterministic ownership** | A shared vertex or triangle belongs to exactly **one** tile, by a rule that does not depend on processing order or thread scheduling — for example lowest tile index by (x, y, z) of the tile origin. This is a **reproducibility requirement**, not only a correctness one: non-deterministic ownership means identical inputs produce non-identical output, which breaks a non-negotiable |
| **Stitching** | Where adjacent tiles carry different LOD levels, the transition must not leave T-junctions. Either constrain neighbouring tiles to adjacent LOD levels, or emit explicit stitch geometry |
| **Cross-tile QA** | Measured **seam-locally**, in a band around each boundary. A site-wide average hides a crack completely. Report seam-local maximum deviation, not just seam-local mean |

### The cost this creates, stated plainly

**Locked boundaries never simplify.** With many tiles the output accumulates a
lattice of full-resolution seams running through the surface, which eats
directly into the size budget that decimation exists to serve.

That produces a three-way tension:

- **smaller tiles** → lower peak working memory, but more locked-boundary
  vertices and a larger share of undecimated geometry;
- **larger tiles** → fewer seams and better decimation ratio, but higher peak
  working memory, which is the constraint that forced tiling in the first
  place;
- **streaming granularity** → smaller tiles give finer view-dependent loading
  and faster first paint; larger tiles give fewer requests.

**Tile size is therefore a measured parameter, not a chosen constant.**

### Phase 3b tile-size benchmark

Run across a range of tile sizes on the largest real station and record:

- peak working memory
- total output size
- percentage of locked boundary vertices
- seam-local maximum deviation
- cracks or normal discontinuities detected
- decimation ratio
- LOD0 tile size and decode time

**Time to first paint is not measurable at 3b.** It requires RMX and the
browser, which is phase 4. LOD0 tile bytes and decode time are the honest
proxies at 3b; TTFP is confirmed at 4.

## Decision 3 — Tiled and streamed writing moves earlier

Previously: QA rework → chunked reading → decimation → comparison → RMX.

RMX and tiled writing were phase 5. They move to **phase 4, alongside
decimation**, because decimation without incremental output does not solve the
memory problem it was scheduled to solve.

## What this does not change

- The native lattice premise is untouched. Reading and triangulating at full
  resolution remains correct; only *retaining* the whole result in memory is
  prohibited.
- f64 precision rules are untouched. Memory is recovered by streaming, tiling
  and eliminating duplicate allocations, **never** by narrowing world
  coordinates to f32. See `ARCHITECTURE.md` §Coordinate and precision rules.
- Per-station scope is untouched. See `ADR-005`.

## Open consequence, not yet resolved

Per-station output means the site ships 30 overlapping meshes, and the sample
stations sit inside roughly a 55 × 25 m footprint with heavy overlap. Even
decimated, package size scales with station count rather than site area. This
works directly against the browser first-paint and package-size targets.

Measure it at phase 4. If the decimated 30-station package is uncompetitive, a
**display-only** fused surface becomes the answer, per `ADR-005`'s revisit
condition, and it must never replace or obscure the per-station evidence used
for measurement.
