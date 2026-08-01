# ADR-001 — Native lattice over fixed angular grid

**Status:** Accepted
**Date:** 1 Aug 2026
**Supersedes:** Cairn `mesher.py`'s fixed `GRID_W × GRID_H` = 2048 × 1024

---

## Context

Cairn's existing mesher rebins every structured scan into a fixed 2048 × 1024
angular grid, roughly 2.10 M cells, regardless of the input. This is the
"TurboMesh-lite" behaviour described in `../../CAIRN-MESH-MEMORY-ISSUE.md`.

Inspection of the 30 reference stations (`../DATA-INVENTORY.md` §1) shows the
source files already contain an explicit sampling lattice, with `rowIndex` and
`columnIndex` per point and a `groupingByLine` index giving per-column byte
offsets.

## Decision

RapidMesh triangulates the scanner's **native** lattice. It never rebins to a
fixed global resolution before triangulation.

Where a file contains no reliable lattice, RapidMesh classifies it as
unstructured or ambiguous and routes it accordingly. It does not invent one.

## Measured justification

| | Cells | Horizontal step | Spacing @ 10 m |
|---|---|---|---|
| Fixed grid 2048 × 1024 | 2.10 M | 0.1758° | 30.7 mm |
| Native medium 2746 × 1075 | 2.95 M | 0.1311° | 22.9 mm |
| Native high-res 6095 × 2387 | 14.55 M | 0.0591° | 10.3 mm |

Against a 25 mm surveyor tolerance, a 30.7 mm sample spacing at 10 m is larger
than the tolerance itself. The fixed grid cannot resolve the quantity the
product is being asked to measure. That, not aesthetics, is the argument.

## The honest limitation

The gain is **6.94×** the cells on the 12 high-resolution stations and only
**1.41×** on the 18 medium stations. On 60% of this project the improvement is
modest.

This must be stated wherever the fidelity claim is made. A blanket "6.9× more
detail than the current mesher" is not supportable on this dataset. The correct
claim is "up to 6.9× on high-resolution stations, 1.4× on medium".

## Consequences

**Positive**

- Detail is bounded by the scanner, not by an arbitrary constant.
- Rebinning is eliminated, along with its aliasing. 2746 columns wrapped into
  2048 is a non-integer resample of the azimuth axis.
- Memory *improves*. The source is spherical, so the native lattice is a
  58.2 MB float32 range image at the largest sample resolution, versus
  349 MB for float64 cartesian at the same point count.
- Scanner-ray comparison becomes nearly free: azimuth and elevation are the
  ray, range is the measurement.

**Negative**

- Output triangle counts scale with input resolution, so decimation stops
  being optional and becomes load-bearing. Accepted: it is the correct place
  for the accuracy/size trade-off, made under an explicit error budget rather
  than implicitly by a constant.
- Lattice validation becomes a required stage with its own failure modes.

## Rejected alternative

**Pre-decimate to grid resolution before binning**, proposed as recommendation
3 in `../../CAIRN-MESH-MEMORY-ISSUE.md` on memory grounds.

Rejected. It was proposed to solve a memory problem that turns out to be an
artefact of materialising float64 world-coordinate cartesian arrays, not of
the point count. Reading the range image directly solves the memory problem
without discarding the resolution that is the entire competitive premise.

The recommendation was sound given what was known at the time. The spherical
storage finding removes its motivation.
