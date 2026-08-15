# RapidMesh project pause handover

> **Superseded 15 August 2026.** John explicitly resumed RapidMesh as a
> separate project under `docs/adr/ADR-008`. Cairn remains read-only and
> integration remains deferred. Current work starts with
> `SPATIAL-CONTRACT.md` and Phase 0c in `CLAUDE.md`; the remainder of this file
> records the verified 3 August holding state and is not a current stop order.

**Holding date:** 3 August 2026. **Historical state:** Phase 1 was
**PARTIAL — PAUSED**. The holding instruction was to wait for Cairn stability;
the 15 August decision above supersedes it.

## Boundary and holding state

RapidMesh remains a separate library/CLI aiming for TurboMesh-class or better
results through measured, auditable engineering. There are zero Cairn imports.
Do not integrate it into Cairn, start Phase 2, or re-enable Cairn mesh routes.
Future comparison consumes retained original observations, never a decimated
display mesh. Clients must not receive point-cloud export capability.

- RapidMesh: `main`, audited from `3febe1f`; no remote.
- Cairn: `navvis-phase3-vvp-meshing`, audited from `e3e9265`, initially equal
  to `origin/navvis-phase3-vvp-meshing`.
- Phase 0 routing, accounts-mode project-admin authorisation, subprocess
  isolation, timeout, cancellation and failure reporting passed focused tests.
- The audit found admission was only per project. Cairn's existing mesh claim
  was made process-global so two per-child limits cannot exhaust one container.
- Routes remain default-off. Open and token modes remain refused.

## Facts, hypotheses and limitations

- At 0.090-degree sampling the zero-noise planar fixture reconstructs exactly.
  With 2 mm injected sigma, walls/floor p99.9 are 3.47/2.37 mm. Tested carving,
  restoration, island culling and incidence 78–85 degrees do not materially
  explain the tail. This is fixture-specific, not universal zero error.
- This passes the 25 mm default-tolerance budget and fails the 10 mm minimum
  p99.9 budget. Two millimetres is a test parameter, not scanner evidence.
- FINDING-001 remains unconfirmed. The reported Cairn LAZ was in a disposable
  Docker test volume; Docker was unavailable during this audit and no retained
  sidecar was present under the repository project directory. Run the checker
  on that exact file before changing production conversion. Header scale plus
  representative source-E57 comparison is decisive; LAS points necessarily
  lie on their own header lattice.
- The RSS watchdog starts in the child and polls worker-only `VmRSS` every
  0.25 s. A native allocation can leap past the threshold between polls, and
  native/GIL scheduling can delay the Python thread. Firing near 2.606 GB for a
  roughly 214 MB setting is overshoot, not a hard boundary. `RLIMIT_AS` limits
  virtual address space separately; the cgroup is the aggregate hard boundary.
  Per-job cgroups are deferred.
- Reported evidence says ordinary PotreeConverter processing of the 1.77 GB E57
  failed under a 3 GB cap and completed after raising it to 8 GB. Treat this as
  a Cairn release capacity limitation requiring preflight/deployment sizing and
  representative comparisons. Peak memory was not retained reliably enough to
  claim here. Do not optimise PotreeConverter in this pause task.

## Exact restart point and all-must-pass gate

Resume with Phase 1a, then 1b; do not start decimation first:

1. Separate retained-surface fidelity, exact filtering/coverage ledger and
   mesh-to-source deviation reports.
2. Stream/chunk E57 input; process bands/tiles locally and write incrementally.
3. Demonstrate <=512 MB working memory, excluding streamed output.
4. Pass report correctness, exact accounting, bidirectional fidelity, bounded
   memory, tests and documented reproduction together.

Do not begin Phase 2, comparison slices, decimation, RMX, NavVis reconstruction
or Cairn integration before that gate and explicit approval.

## Commands

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[e57,dev]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests tools
.\.venv\Scripts\python tools\smoke.py
.\.venv\Scripts\python tools\bench_synthetic.py --rows 100 --cols 400
.\.venv\Scripts\python tools\e57_inventory.py <authorised-e57>
.\.venv\Scripts\python tools\check_laz_precision.py <cairn-generated-laz>
```

The full matrix is not a routine gate. Reproduce only when needed with
`python tools/isolation_matrix.py --out out/isolation_matrix`; keep `out/` and
client data out of Git.

`H:\Sample` is read-only. Never commit client coordinates, scans, absolute
client paths, credentials or sensitive logs. Before real-data claims obtain
scanner make/model/serial, settings and calibration, registration method,
station/target/checkpoint residuals, control accuracy and excluded stations.
Back up both branches. Do not push or create a RapidMesh remote without owner
approval.
