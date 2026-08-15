"""
Grid triangulation — connecting lattice neighbours, and knowing when not to.

Every cell of the lattice is a quad of four angularly adjacent samples. Almost
all of them should become two triangles. The interesting work is deciding which
should not, because a quad that spans a depth discontinuity produces the
signature artefact of this whole approach: a long stretched triangle bridging a
doorway into the room beyond.

The discontinuity test, and why it is expressed as an angle
-----------------------------------------------------------
Cairn cuts an edge when its 3D length exceeds ``max(0.08, 6.0 * r * step)``.
The 6.0 is unexplained in that file, but it is not arbitrary — it is a
grazing-angle limit in disguise, and naming it as one makes it tunable instead
of magic.

Two angularly adjacent samples on a locally planar surface at incidence angle
θ are separated in 3D by approximately::

    d ≈ r * step / cos(θ)

so a limit of ``6.0 * r * step`` is exactly the limit for ``θ ≈ 80°``
(1/cos 80° = 5.76). Expressing the parameter as `max_incidence_deg` means the
number has a physical meaning, can be reasoned about, and can be justified per
site: a floor scanned from 1.6 m height is at 85°+ incidence by 20 m out, which
is *beyond* Cairn's implicit limit and gets cut — floors disintegrating at
distance is a predictable consequence of that constant.

A noise floor is added underneath. Even at normal incidence, range noise means
adjacent samples differ; on a 2 mm-sigma instrument the limit must clear a few
sigma or the mesh perforates on flat surfaces at close range.

Normals come free here, and correctly oriented
----------------------------------------------
Generic surface reconstruction has to guess which way a surface faces. Per-
station meshing does not: **every sample was seen from the scanner**, so the
outward normal is the one pointing back towards the origin. That gives globally
consistent, correctly-oriented normals with no propagation step and no
ambiguity — a real advantage of this approach that Cairn throws away by baking
a lambert term into vertex colour and discarding the normals entirely.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .filters import columns_wrap
from .grid import ScanGrid
from .types import MeshData, StructuredScan

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    I64 = npt.NDArray[np.int64]
    F32 = npt.NDArray[np.float32]
    BOOL = npt.NDArray[np.bool_]


def triangulate(
    grid: ScanGrid,
    max_incidence_deg: float = 82.0,
    noise_floor: float = 0.012,
    min_quality: float = 0.015,
    band_rows: int = 512,
) -> I64:
    """Triangles as (T,3) indices into the scan's sample arrays.

    `min_quality` rejects slivers: it is ``2 * area / longest_edge²``, which is
    1.0 for an equilateral triangle and tends to 0 for a degenerate one. A
    sliver contributes no surface but does contribute a wild normal, so it is
    worth rejecting even when its edges individually pass.
    """
    import numpy as np

    scan = grid.scan
    wrap = columns_wrap(scan)
    verts, rng = scan.xyz, scan.rng
    step = max(abs(scan.lattice.az_step), abs(scan.lattice.el_step))
    tan_limit = math.tan(math.radians(min(max_incidence_deg, 89.5)))

    parts: list[I64] = []
    for r0, r1 in grid.bands(band_rows=band_rows, overlap=1):
        idx = grid.dense_rows(r0, r1)
        if idx.shape[0] < 2:
            continue
        parts.extend(
            _band_triangles(idx, verts, rng, wrap, step, tan_limit, noise_floor, min_quality)
        )

    if not parts:
        return np.empty((0, 3), np.int64)
    return np.concatenate(parts, axis=0)


def _band_triangles(
    idx: npt.NDArray[np.int32],
    verts: F32,
    rng: F32,
    wrap: bool,
    step: float,
    tan_limit: float,
    noise_floor: float,
    min_quality: float,
) -> list[I64]:
    """Triangles for one row band. `idx` is (h, cols), -1 where empty."""
    import numpy as np

    # Quad corners: A top-left, B top-right, C bottom-left, D bottom-right.
    # Rolling axis 1 by -1 takes the next column; for a non-wrapping scan the
    # last column has no right neighbour and is invalidated.
    A = idx[:-1, :]
    B = np.roll(idx, -1, axis=1)[:-1, :]
    C = idx[1:, :]
    D = np.roll(idx, -1, axis=1)[1:, :]
    if not wrap:
        B = B.copy()
        D = D.copy()
        B[:, -1] = -1
        D[:, -1] = -1

    def tri_ok(t: I64) -> BOOL:
        """Accept a triangle if its longest edge is inside the incidence limit
        and it is not a sliver. Testing the longest edge subsumes a per-edge
        test, so there is one threshold in one place."""
        if not len(t):
            return np.empty(0, dtype=bool)
        i, j, k = t[:, 0], t[:, 1], t[:, 2]
        e0 = verts[j] - verts[i]
        e1 = verts[k] - verts[i]
        dij = np.linalg.norm(e0, axis=1)
        dik = np.linalg.norm(e1, axis=1)
        djk = np.linalg.norm(verts[k] - verts[j], axis=1)
        longest = np.maximum.reduce([dij, djk, dik])
        limit = np.maximum(
            noise_floor,
            np.maximum.reduce([rng[i], rng[j], rng[k]]) * step * tan_limit,
        )
        twice_area = np.linalg.norm(np.cross(e0, e1), axis=1)
        quality = twice_area / np.maximum(longest * longest, 1e-12)
        out: BOOL = (longest < limit) & (quality > min_quality)
        return out

    parts: list[I64] = []

    # --- complete quads: pick the shorter diagonal that actually works ------
    # A fixed diagonal folds non-planar cells the long way. Cairn's own comment
    # records this as "a major source of visibly tangled triangles", and the
    # fix is cheap: evaluate both splits, prefer the valid one, and break ties
    # on diagonal length.
    full = (A >= 0) & (B >= 0) & (C >= 0) & (D >= 0)
    if np.any(full):
        a, b, c, d = A[full].astype(np.int64), B[full].astype(np.int64), C[full].astype(np.int64), D[full].astype(np.int64)
        bc1, bc2 = np.stack([a, c, b], 1), np.stack([b, c, d], 1)
        ad1, ad2 = np.stack([a, c, d], 1), np.stack([a, d, b], 1)
        bc_ok = tri_ok(bc1) & tri_ok(bc2)
        ad_ok = tri_ok(ad1) & tri_ok(ad2)
        len_bc = np.linalg.norm(verts[b] - verts[c], axis=1)
        len_ad = np.linalg.norm(verts[a] - verts[d], axis=1)
        use_bc = bc_ok & (~ad_ok | (len_bc <= len_ad))
        use_ad = ad_ok & ~use_bc
        for first, second, use in ((bc1, bc2, use_bc), (ad1, ad2, use_ad)):
            if np.any(use):
                pair = np.empty((int(use.sum()) * 2, 3), np.int64)
                pair[0::2], pair[1::2] = first[use], second[use]
                parts.append(pair)

    # --- one corner missing: still make a triangle, not a hole --------------
    # A single no-return cell is common (a dark surface, a grazing hit). Leaving
    # a square hole for each one perforates otherwise-perfect walls.
    for mask, i, j, k in (
        ((A >= 0) & (B >= 0) & (C >= 0) & (D < 0), A, C, B),
        ((A >= 0) & (B >= 0) & (C < 0) & (D >= 0), A, D, B),
        ((A >= 0) & (B < 0) & (C >= 0) & (D >= 0), A, C, D),
        ((A < 0) & (B >= 0) & (C >= 0) & (D >= 0), B, C, D),
    ):
        if not np.any(mask):
            continue
        cand = np.stack([i[mask].astype(np.int64), j[mask].astype(np.int64), k[mask].astype(np.int64)], 1)
        ok = tri_ok(cand)
        if np.any(ok):
            parts.append(cand[ok])

    return parts


# --------------------------------------------------------------------------
# island culling
# --------------------------------------------------------------------------


def cull_islands(
    verts: F32,
    tris: I64,
    min_area: float = 0.005,
    min_triangles: int = 8,
) -> I64:
    """Drop connected components smaller than `min_area` square metres.

    **Area, not triangle count.** Cairn culls components below 150 triangles,
    which is resolution-dependent in the worst way: at 2048x1024 that is a
    substantial object, and at native resolution it is a speck of dust — so the
    same constant means different things on different scans and cannot be
    tuned once. Square metres mean the same thing everywhere.

    `min_triangles` is a floor beneath that, catching the degenerate case of a
    handful of enormous triangles that pass the area test while representing
    nothing.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(tris) == 0:
        return tris

    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = coo_matrix(
        (np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
        shape=(len(verts), len(verts)),
    )
    ncomp, label = connected_components(adj, directed=False)

    area = 0.5 * np.linalg.norm(
        np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
        axis=1,
    )
    comp = label[tris[:, 0]]
    comp_area = np.bincount(comp, weights=area, minlength=ncomp)
    comp_count = np.bincount(comp, minlength=ncomp)

    keep = (comp_area[comp] >= min_area) & (comp_count[comp] >= min_triangles)
    out: I64 = tris[keep]
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_mesh(
    scan: StructuredScan,
    tris: I64,
    with_normals: bool = True,
) -> MeshData:
    """Compact to used vertices and assemble a `MeshData`.

    Normals are area-weighted (the cross product is not normalised before
    accumulation, so larger faces contribute more, which is what you want) and
    then flipped to face the scanner. See the module docstring: per-station
    meshing makes orientation unambiguous, and it costs one dot product.
    """
    import numpy as np

    used = np.zeros(len(scan), dtype=bool)
    if len(tris):
        used[tris.ravel()] = True
    keep_idx = np.nonzero(used)[0]
    remap = np.full(len(scan), -1, np.int64)
    remap[keep_idx] = np.arange(keep_idx.size)

    local_verts = scan.xyz[keep_idx]
    faces = remap[tris].astype(np.uint32) if len(tris) else np.empty((0, 3), np.uint32)
    rgb = None if scan.rgb is None else scan.rgb[keep_idx]
    source_sample_id = (
        keep_idx.astype(np.int64)
        if scan.sample_id is None
        else scan.sample_id[keep_idx].astype(np.int64, copy=False)
    )

    normals = None
    if with_normals and len(faces):
        fn = np.cross(
            local_verts[faces[:, 1]].astype(np.float64) - local_verts[faces[:, 0]],
            local_verts[faces[:, 2]].astype(np.float64) - local_verts[faces[:, 0]],
        )
        acc = np.zeros((local_verts.shape[0], 3), np.float64)
        for col in range(3):
            np.add.at(acc, faces[:, col], fn)
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        acc /= np.maximum(norm, 1e-12)
        # Face the scanner. `local_verts` are offsets from the scanner origin,
        # so the direction back to it is simply `-local_verts`.
        facing = np.einsum("ij,ij->i", acc, -local_verts.astype(np.float64))
        acc[facing < 0] *= -1.0
        normals = scan.pose.rotate_local(acc).astype(np.float32)

    # MeshData stores project-axis offsets from a float64 origin. Apply the
    # scanner rotation here, exactly once, and only then narrow to float32.
    project_offsets = scan.pose.rotate_local(local_verts).astype(np.float32)

    return MeshData(
        origin=scan.pose.translation.astype(np.float64),
        vertices=project_offsets,
        triangles=faces,
        source_pose=scan.pose,
        normals=normals,
        rgb=rgb,
        source_sample_id=source_sample_id,
    )
