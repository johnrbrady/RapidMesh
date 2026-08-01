# ADR-004 — Splitting NavVis into paths B1 and B2

**Status:** Accepted
**Date:** 2 Aug 2026

---

## Context

Two NavVis datasets are now available, and they are not two views of the same
thing (`../DATA-INVENTORY.md` §2, §2A, §5).

| | Registered E57 | Raw recording |
|---|---|---|
| Project | 25199 Ampol Tallarook | **25409_S** |
| Captured | 1 May 2025 | 11 Sep 2025 |
| Size | 1.77 GB | 3.0 GB |
| Points | 56,950,017, with normals | **None. Raw sweeps only** |
| Trajectory | Absent | Present, 359.1 m, ROS bags |
| Sensor calibration | Absent | Full, 8 sensors, C6.3 |
| Timestamps | Absent | Present, per sweep and per capture |
| Imagery | Absent | 196 unstitched DNG |
| Georeferencing | MGA Zone 55 | **Absent** |
| Pairs with the IFC model | **Yes** | No |

## Decision

Two separate pipelines, not one pipeline with a flag.

**B1 — registered export.** Consumes a NavVis registered E57. Points and
normals supplied. No rays, no timestamps, no poses. Delivers a surface plus
comparison modes A and D. **Phase 5.**

**B2 — raw recording.** Consumes a NavVis `rec-v4` folder. Decodes laser
bags, applies `sensor_frame.xml` extrinsics, interpolates the SLAM trajectory
and accumulates points while retaining per-point timestamp, sensor head,
observation origin and ray direction. Delivers everything B1 does plus modes
C, trajectory-aware segmentation and transient filtering. **Phase 6.**

## Why not one pipeline

Because the honest capability sets are different, and a single pipeline with
optional features invites exactly the failure this project is trying to avoid:
code that silently degrades and reports a confident number derived from
absent evidence.

A B1 dataset must never be able to produce a mode C result. Separate pipelines
make that structural rather than a runtime check someone can forget.

## Why B1 first

1. B1 is buildable now and exercises the shared back half — tiling,
   reconstruction, QA, decimation, RMX, comparison — on real data.
2. B1's dataset is the **only** one that pairs with the IFC model, so it is
   the only NavVis path with an end-to-end comparison test today.
3. B2's dataset has no point cloud to check accumulation against, and no
   georeferencing, so its output cannot be validated against anything until a
   processed export of project 25409_S is supplied.
4. B2 is substantially more work: ROS bag decoding, extrinsic chains,
   trajectory interpolation, and point accumulation that reimplements part of
   NavVis IVION.

## Consequences

**Positive**

- Mode C stops being permanently deferred and becomes phase 6 work with a
  named input.
- Transient-object filtering becomes possible with real evidence rather than
  the median-based heuristics rejected in `ADR-003`.
- Per-sensor-head quality analysis becomes possible (`laser_horiz` versus
  `laser_vert`).
- The B2 importer can be built and unit-tested against a real recording even
  though it cannot yet be validated end to end.

**Negative**

- Two ingestion paths to maintain.
- B2 output is in the SLAM map frame with no route to MGA until surveyed
  anchor coordinates or an external transform are supplied. Output must be
  labelled map-frame and must not be compared against a georeferenced model.
- ROS bag decoding adds a dependency (`rosbags`). Record it in the dependency
  and licence register.

**Neutral**

- B1 and B2 share the octree tiler, reconstruction, QA, decimation, LOD, RMX
  and comparison engine. Only ingestion and per-point attribute richness
  differ.

## Version pinning

The B2 importer is built against `rec-v4`, software release 4.1.0, calibration
C6.3 rev0, device G10. It must **validate `dataset.json` and reject unknown
layout versions loudly** rather than attempting a best-effort read. NavVis
recording layouts change between releases and a silent misread of sensor
extrinsics produces geometry that looks plausible and is wrong.

## What is still blocked after B2

- **Panoramas.** `processed_panoramas: 0`. There are 196 raw DNG frames, not
  stitched panoramas. Imagery navigation and texture baking need stitching,
  which is a separate undertaking and not in scope.
- **Georeferencing of B2 output.** `anchor_poses.txt` holds no coordinate
  rows. Anchors TDS4, TDS5, TDS6, TDS8, TDS9 and TDS10 are named but unvalued.
- **A joined NavVis comparison test.** Requires a processed export and a model
  for the same project.
