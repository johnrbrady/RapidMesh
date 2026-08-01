# Cairn per-scan/vantage-point meshing — memory issue that broke the AWS pilot

_Written 1 Aug 2026 for whoever reworks `backend/mesher.py`. Standalone —
assumes no context from any other session. Repo: `pointcloud-viewer`, branch
`navvis-phase3-vvp-meshing`. The frontend entry points for this feature, and
the automatic per-scan call site that actually caused the crash, were both
deactivated the same day this was written — see "What was already done"
below. `mesher.py`'s own implementation is untouched and is what this
document is for._

---

## One paragraph

Cairn meshes every scan (and, separately, every NavVis vantage point) into a
triangulated surface — "TurboMesh-lite" — for a nicer station-view experience.
The mesh step loads the **entire point cloud into memory in one shot**, inside
the same Python process as the web server, using `laspy.open(path).read()`.
It is not subprocess-isolated the way PDAL and PotreeConverter are. On a real
1.77 GB client scan, deployed to a $24/month AWS Lightsail box (2 vCPU, 4 GB
RAM), this crashed the **entire application process** — not just the mesh job
— confirmed by the kernel's own OOM-killer log. That is the concrete problem:
it makes it impossible to size a cost-effective cloud box, because the one
feature that determines the real memory ceiling has memory behavior nobody
has characterized, and when it goes wrong it takes the whole server down with
it, not just the one conversion.

---

## The incident, first-hand

**Host:** AWS Lightsail, Sydney, Ubuntu 22.04, 2 vCPU / 4 GB RAM, `cairn`
container capped at `memory: 3g` / `cpus: 1.5` via a `docker-compose.override.yml`
(the rest of the box's ~1 GB is Caddy + OS).

**What was uploaded:** a real client E57, `25199_Ampol_Tallarook_250501-registered.e57`,
1,773,378,560 bytes (1.77 GB). For scale: the largest file previously proven
end-to-end (locally, same day) was 154 MB and peaked at 1.84 GB RAM. The Ampol
file is ~11.5× bigger by raw size.

**Timeline**, from the project manifest and container inspection:

```
manifest createdAt: 2026-08-01T04:39:22Z   (upload starts, status=converting)
memory.peak sampled ~04:46: 2.999 GiB / 3 GiB cap   (essentially at the ceiling)
manifest updatedAt: 2026-08-01T04:49:47Z   status flips to "error":
  "Conversion was interrupted by a server restart. Press Retry to convert it
   again — the raw scan is still here."
docker inspect: RestartCount=1 (container bounced), OOMKilled=false (the
  CONTAINER wasn't OOMKilled — a process INSIDE it was, which is different —
  see dmesg below)
```

**The kernel's own account, `dmesg -T`:**

```
[Sat Aug 1 04:49:44 2026] uvicorn invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[Sat Aug 1 04:49:44 2026] Memory cgroup out of memory: Killed process 11277 (uvicorn) total-vm:7012624kB, anon-rss:3105336kB, file-rss:27264kB, ... UID:10001
```

**The process the kernel killed was `uvicorn` itself** — the FastAPI/ASGI
server process — not a PotreeConverter or PDAL subprocess. `anon-rss:3105336kB`
≈ 3.1 GB, right at the cgroup's 3 GB ceiling. This is the single most important
fact in this document: **the OOM took down the whole application**, not one
job. Docker's `restart: unless-stopped` brought it back, and Cairn's own
`sweep_interrupted` correctly caught the stranded scan and marked it `error`
(raw file preserved, Retry available) rather than leaving it wedged on
`converting` forever — that safety net worked. But for however long the
restart took, every other user of the app — anyone else's session, any other
project being viewed — was also down. That is categorically worse than the
crash-recovery behavior already proven and documented for PotreeConverter
itself (`deploy/DEPLOY.md`'s "Crash recovery" section, tested 30 Jul 2026):
*"The kernel killed PotreeConverter, not the server... `/api/health` kept
answering 200."* This feature is the one place in the whole conversion
pipeline that doesn't get that isolation.

---

## Root cause — traced through the actual code, not inferred

### Ruled out: PDAL and PotreeConverter

`backend/converter.py` invokes both exclusively via `subprocess.run(...)`
(`pdal info`, `pdal translate`, the `PotreeConverter` binary — see
`converter.py:488-926`). Their memory lives in child processes and is
released the instant each call returns. Neither can explain "uvicorn itself
was killed" — confirmed by grepping the whole file for `pdal.Pipeline`/any
in-process PDAL Python binding: there is none. This was checked specifically
so the actual cause isn't assumed.

### The actual cause: `backend/mesher.py`'s `load_points`

`backend/mesher.py:103-127`:

```python
def load_points(laz_path: str | Path) -> LoadedPoints:
    """Read a cloud off disk ONCE, in world coordinates.
    ...
    """
    import numpy as np
    import laspy

    las = laspy.open(str(laz_path)).read()
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    ...
```

`laspy.open(...).read()` loads the **entire** point cloud into memory — not
chunked, not streamed. `xyz` alone is `float64` × 3 columns: for N points,
`24·N` bytes, before RGB, before any of `grid_from_points`'s several further
full-array passes (origin subtraction, `np.linalg.norm`, `np.lexsort`,
boolean masks — each spins up another array of comparable size as a
transient). This all happens **inside the uvicorn process**, in the
conversion background thread, sharing the same memory space and the same
cgroup ceiling as the web server itself.

This function is called from **three places** (mapped in full below), and
critically it is called **automatically, on every ordinary scan upload** —
not just when someone deliberately asks for a mesh.

### Why the 30 July baseline didn't show this

Per-scan meshing itself isn't new — it's been in the pipeline since 16 July
2026 (`backend/conversion.py`, commit `8e52afd6`), well before the 30 July
733 MB baseline for a ~154 MB file. So "meshing was just added" isn't the
explanation. Two live hypotheses, not confirmed either way:

1. That 30 July test file may not have carried a readable E57 scanner
   origin — meshing used to be skipped entirely when `origin` was falsy
   (`if ok and origin and mesher.deps_ok():`, formerly `conversion.py:227` —
   this guard is gone now, see "What was already done" below).
2. `mesher.py` itself changed between 30 July and 1 Aug — notably
   `load_scan_grid` was split into `load_points` + `grid_from_points`
   (commit `2179d1a`, "so a cloud is read once, not once per station") as
   part of the Phase 3 vantage-point work. That split didn't add the
   whole-file load — `load_scan_grid` always did that — but it's the most
   recent change to this exact code path and is worth checking for any
   incidental dtype/copy changes.

Whichever it is, it doesn't change the diagnosis: the mechanism is
`load_points`'s whole-cloud, in-process, un-isolated read, and it demonstrably
scales badly.

### The scaling data, all measured today

| Scan | Points | Peak RAM | Mesh output |
|---|---|---|---|
| 31 MB E57 (local) | 1,312,827 | ~1.20 GB | 89,099 verts / 114,255 tris (2.7 MB) |
| 154 MB E57 (local) | 6,464,233 | ~1.84 GB | 710,921 verts / 1,234,419 tris (25.5 MB) |
| 1.77 GB E57 (Lightsail) | unknown (never finished) | **OOM at 3.1 GB+** | — |

Two things worth noting in that table: mesh complexity grew ~9.4× (2.7→25.5 MB)
against only a ~5× file-size increase from the small to large local test —
mesh memory/output does not scale linearly with input size. And the 30 July
baseline for a similar-sized file to the "154 MB" row was 733 MB — i.e. the
memory floor for the *same size file* has already grown ~2.5× in under two
weeks for reasons not fully isolated (see previous section). Both are signs
that this code's memory behavior is not currently predictable enough to size
a box against.

---

## Why this specifically blocks using AWS efficiently

Cloud cost here is a direct function of RAM tier (see AWS Lightsail's own
pricing: 4 GB is $24/mo, 8 GB is $44/mo, 16 GB is $74–84/mo, 32 GB is
$164/mo — Sydney region, checked 1 Aug 2026). Sizing a box efficiently means
picking the smallest tier that reliably fits the workload. That requires the
workload's memory use to be **predictable and bounded**. Right now it is
neither:

- It scales worse than linearly with scan size, by measured evidence above.
- It runs in-process, so its footprint stacks directly on top of whatever
  else the server is doing at that moment (other conversions, other
  requests) rather than being a clean, separately-accountable cost.
- When it exceeds the box's ceiling, it does not fail *its own job* — it
  takes the entire application down for every user, for as long as the
  restart takes. That's a reliability cost on top of a sizing cost: even
  correctly guessing "16 GB is enough for scans up to X size" doesn't
  protect against someone uploading something bigger than X, because the
  failure mode is a full outage, not a contained error.

Contrast with PDAL/PotreeConverter, which are subprocess-isolated: an OOM
there kills the child process only, is already documented and tested
(30 Jul), and the server answers requests throughout. That is the shape this
feature needs and does not have.

---

## The three call sites — map this before touching anything

All three call into the same `mesher.py` machinery; fixing `mesher.py` alone
fixes the mechanism everywhere it's used, but each call site has its own
blast radius and triggering conditions worth knowing:

| # | Location | Trigger | Notes |
|---|---|---|---|
| 1 | *(removed)* — was `backend/conversion.py:227-247`, inside `run_conversion` | Was **automatic** — every scan upload with a readable E57 origin | **This is the one that crashed today, and it's gone now** — see "What was already done" below. `run_conversion` no longer calls `mesher` at all. Kept in this table because #2 and #3 still call the exact same `mesher.py` functions this one did. |
| 2 | `backend/routers/models.py:850` (`build_meshes`), `POST /api/projects/{pid}/meshes` | Meant to be manual — mesh every ready scan in a project at once | **Confirmed dead from the frontend**: zero references anywhere in `frontend/` (checked by grep across `.js`/`.ts`). Nothing in the UI can reach this route today. Likely the highest-risk of the three if it were ever wired up — it loads grids for **every** scan in a project into memory *simultaneously* (pass 1 in the function) before meshing any of them, specifically to support cross-scan "mover removal." Still fully live if called directly. |
| 3 | `backend/routers/models.py:699` (`build_vantage_meshes`), `POST /api/projects/{pid}/vantagepoints/meshes` | **Explicit**, user-triggered — the actual Phase 3 feature ("mesh every vantage point") | Its frontend button is also gone (see below), but **the route itself is untouched and still fully live** if called directly (curl, a future UI, a test). Reads the source cloud once (`load_points`) then re-bins per station (`grid_from_points`). Explicitly designed and documented against **the Ampol dataset's 36-station walk** as the reference case. Carries the identical OOM risk #1 did — it just hasn't been triggered on something Ampol-sized yet. |

**Net effect of today's fix: the surprise, automatic crash on ordinary uploads
is gone. #2 and #3 are not — they were always opt-in (a deliberate API call),
and remain exactly as risky as this whole document describes if anyone calls
them on a large enough scan.**

---

## What was already done

Two separate changes landed the same day this was written, as two separate
commits on `navvis-phase3-vvp-meshing`. **`backend/mesher.py` itself — the
actual algorithm this whole document is about — is untouched either way.**
That's yours.

### Commit 1 — frontend deactivated

Purely `frontend/`, trivially reversible via git history:

- The "Build vantage point meshes" button + progress/stop control
  (`frontend/src/ui.ts`'s `vantageMeshControl()`).
- The click handlers and polling for it (`frontend/cairn-viewer.js`'s
  `'vvp-mesh'`/`'vvp-mesh-cancel'` cases and `pollVantageMeshes()`).
- The "Scan meshes" layer toggle row (`frontend/src/state.ts`'s `LAYERS`
  array) and its handler.
- The API client wrappers (`buildVantageMeshes`, `vantageMeshProgress`,
  `cancelVantageMeshes`, `fetchMeshAsset`, plus two already-dead siblings
  `fetchMeshLod0`/`fetchMeshFull` in `frontend/src/api.ts`).
- `loadScanMeshes`/`_cmhToMesh` in `frontend/src/scans.ts`, deleted outright.
- The dedicated Playwright spec for the button (`vantage-meshes.spec.ts`),
  deleted (all 3 tests were specifically about the now-removed control).

**Deliberately left alone**: `frontend/src/potree-patches.ts`'s station-view
measurement raycast fallback, and `ensureStationMeasurementSurface` in
`scans.ts` (gutted to `return Promise.resolve(false)` rather than removed,
since `measure.ts` still imports it). Both become fully inert automatically —
nothing populates `state.scanMeshes` in real usage anymore — without needing
to touch the raycast code or its own test
(`e2e/tests/panorama-isolation.spec.ts:76`, which injects a mock mesh directly
and was re-verified passing after this change).

**Call sites #2 and #3 still work if called directly** (curl/Postman/tests)
— only the UI path to #3 is gone. `mesher.py` itself is exactly as it was.

Full frontend gate re-run after the change: `tsc --noEmit` clean,
`esbuild` build clean, **105/108 Playwright passing** (108 baseline − 3
deleted for the removed control = 105, all passing, nothing else broke).

### Commit 2 — the automatic call site (#1) removed

`backend/conversion.py`'s `run_conversion` no longer calls into `mesher` at
all. `mesh_info` is now unconditionally `None`; every consumer already
treats "no mesh" as a normal state (see the frontend commit above, and the
pre-existing `test_mesh_deps_not_ok_produces_ready_scan_with_no_mesh_key`).

This is a genuine, if narrow, backend fix — not just a config toggle — so it
got the same treatment as any other behavior change here: a new regression
test (`test_automatic_mesh_step_is_deactivated` in `test_main_exceptions.py`)
that sets up the exact old trigger condition (deps available, readable
origin, successful conversion) and asserts `mesher.mesh_scan` is never
reached, **proven by reverting `conversion.py` and confirming the test fails
against the old code first** — it does (`assert [1] == []` fails, `mesh_scan`
was called once). Two tests that asserted behavior of the now-removed call
site (`test_mesh_scan_exception_does_not_fail_the_scan`,
`test_run_conversion_own_mesh_step_decrypts_e57_sidecar`) were removed with
it, since they'd otherwise pass for the wrong reason — the call they exercise
no longer happens at all.

808 pytest (807 + 1 new), ruff clean, `mypy --strict` clean (34 files, both
native and `--platform linux`).

**This does not touch #2 or #3.** A large scan through `build_meshes` or
`build_vantage_meshes` — called directly, since their frontend paths are
gone — still carries the identical risk. If Ampol (or anything similarly
large) gets retried, it's the plain upload path (call site #1, now safe)
that's being exercised, not those two.

---

## Recommendations for the rework, roughly by confidence

1. **Subprocess isolation — the highest-leverage fix, and doesn't require
   touching the algorithm at all.** Run the mesh step (whichever call site)
   as its own subprocess, exactly like PDAL/PotreeConverter already are. This
   converts "a big mesh takes the whole server down" into "one mesh job
   dies, the scan still converts, everyone else keeps working" — the failure
   mode already proven and documented for PotreeConverter on 30 July. This
   alone would have prevented today's incident regardless of the actual
   memory number.

2. **A size/point-count ceiling before attempting to mesh at all.** Cheap,
   safe, and consistent with the existing philosophy already in
   `conversion.py`'s own comment: *"a scan that converts but doesn't mesh is
   still a successful scan."* Skip meshing outright above some threshold
   rather than attempting it and risking a crash.

3. **Pre-decimate before binning — worth serious consideration on the
   merits, not just as a safety net.** The mesh's own output is a **fixed
   2048×1024 angular grid** (`mesher.py`'s `GRID_W, GRID_H`) — about 2.1
   million cells, always, regardless of input size. Loading and processing
   tens of millions of raw points just to bin them down into 2.1 million
   cells is arguably wasted work well past some input density. A coarse
   pre-filter sized to the grid resolution could cut memory hard on large
   scans with little real quality loss. This is a real algorithm change, not
   a config tweak — needs care to preserve mesh quality.

4. **One that looks like a fix and isn't: don't naively cast `load_points`'s
   `xyz` to `float32`.** It's `float64` for a specific reason — world
   coordinates (UTM/MGA-style, ~6-7 digits before the decimal) lose sub-mm
   precision to floating-point cancellation if the origin subtraction happens
   in float32 first. `grid_from_points` already casts the *result* down to
   float32 once values are small (near-origin offsets), which is the correct
   point to do it. Halving the load-time array by moving that cast earlier
   would reintroduce the precision loss it was written to avoid.

Whichever combination is chosen, the three call sites in the table above
should all end up going through whatever the fixed mechanism is — a fix
applied only to call site #1 (today's crash) leaves #3 (the actual Phase 3
deliverable) carrying the identical risk on the identical reference dataset.
