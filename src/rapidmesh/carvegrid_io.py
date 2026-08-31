"""
The carve-grid artefact on disk — `PHASE1-TILE-CONTRACT-V0.md` §7, plus the
temp-then-replace write discipline of §5.1.

Separated from `carvegrid.py` because the two halves fail in different ways and
are audited differently. That module is numeric and its contract is "bitwise
identical to the legacy build". This one is bytes-on-disk and its contract is
"a stale, renamed, truncated or half-written grid never reaches a caller".

    out/<station>/carvegrid/<source_sha256>-<scan_index>-v<grid_version>
                            -<cells>-<params_digest>.rmgrid

**Invalidation is by name only.** A cached grid is used if and only if every
component of the name matches; nothing is updated in place, no timestamp is
consulted, and no partial match is accepted. The artefact then repeats its
full key in the header and carries a SHA-256 of the payload, so a file that
was renamed into a matching name, truncated, or left half-written fails closed
rather than carving a neighbour's geometry with the wrong evidence.

Reading is fail-closed, not fail-quiet: a missing file is an ordinary cache
miss and returns `None`, but a file that is present and disagrees with itself
raises `CarveGridArtefactError`. Rebuilding silently would hide exactly the
fault the digest exists to catch.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .carvegrid import CarveGridArtefactError, CarveGridKey
from .grid import CARVE_MAX_CELLS, CoarseRangeGrid

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    F32 = npt.NDArray[np.float32]

GRID_MAGIC = b"RAPIDGRD"
GRID_CONTRACT_VERSION = 0
GRID_KIND = 1
GRID_FLAG_FILL_HOLES = 1

# Checked before any allocation sized from a file header. 2^31 cells is 8 GB as
# f32 and is far above any grid this code produces; it exists so a corrupt
# length field cannot ask for an arbitrary allocation.
MAX_ARTEFACT_CELLS = 1 << 31

# Little-endian, no padding: magic, contract version, kind, flags, header
# bytes, record count, payload bytes, then the full key, the resolved grid and
# lattice shape, the f64 origin, and the three digests.
_HEADER = struct.Struct("<8sIHHIQQIIQIIII3d32s32s32s")

# Payload is written and hashed in row blocks so neither the writer nor the
# reader ever holds a second whole-grid copy as bytes.
PAYLOAD_BLOCK_ROWS = 256


def _pack_header(key: CarveGridKey, grid: CoarseRangeGrid, max_cells: int,
                 fill_holes: bool, lattice_rows: int, lattice_cols: int,
                 payload_sha256: bytes) -> bytes:
    cells = grid.height * grid.width
    return _HEADER.pack(
        GRID_MAGIC,
        GRID_CONTRACT_VERSION,
        GRID_KIND,
        GRID_FLAG_FILL_HOLES if fill_holes else 0,
        _HEADER.size,
        cells,
        cells * 4,
        key.grid_version,
        key.scan_index,
        max_cells,
        grid.width,
        grid.height,
        lattice_rows,
        lattice_cols,
        float(grid.origin[0]),
        float(grid.origin[1]),
        float(grid.origin[2]),
        bytes.fromhex(key.source_identity),
        bytes.fromhex(key.params_digest),
        payload_sha256,
    )


def _payload_blocks(rng: F32, block_rows: int = PAYLOAD_BLOCK_ROWS) -> list[bytes]:
    import numpy as np

    return [
        np.ascontiguousarray(rng[i : i + block_rows], dtype="<f4").tobytes()
        for i in range(0, int(rng.shape[0]), block_rows)
    ]


@dataclass(frozen=True)
class DirectoryGridStore:
    """Cached grids under one directory, named by `CarveGridKey.filename()`.

    Writes are temp-then-`os.replace` **inside the destination directory**,
    per `PHASE1-TILE-CONTRACT-V0.md` §5.1: `os.replace` is atomic within a
    volume on both Windows and POSIX, and `os.rename` over an existing file
    raises on Windows. Durability after the replace is a separate question the
    repository does not settle (§8 gap 2) and is not claimed here.
    """

    root: Path
    max_cells: int = CARVE_MAX_CELLS
    fill_holes: bool = True
    # Recorded in the header for a human reading the file, not used for
    # invalidation — the lattice is already inside `params_digest`, and the
    # resolved cell count is in the name. Zero means this writer was not told
    # them; it is never a claim that the lattice was 0 x 0.
    lattice_rows: int = 0
    lattice_cols: int = 0

    def path_for(self, key: CarveGridKey) -> Path:
        return self.root / key.filename()

    def load(self, key: CarveGridKey) -> CoarseRangeGrid | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return read_grid_artefact(path, key)

    def store(self, key: CarveGridKey, grid: CoarseRangeGrid) -> None:
        write_grid_artefact(
            self.path_for(key),
            key,
            grid,
            max_cells=self.max_cells,
            fill_holes=self.fill_holes,
            lattice_rows=self.lattice_rows,
            lattice_cols=self.lattice_cols,
        )


class MemoryGridStore:
    """Same keying as the directory store, for fixtures with no source file."""

    def __init__(self) -> None:
        self._grids: dict[CarveGridKey, CoarseRangeGrid] = {}

    def __len__(self) -> int:
        return len(self._grids)

    def load(self, key: CarveGridKey) -> CoarseRangeGrid | None:
        return self._grids.get(key)

    def store(self, key: CarveGridKey, grid: CoarseRangeGrid) -> None:
        self._grids[key] = grid


def write_grid_artefact(
    path: Path,
    key: CarveGridKey,
    grid: CoarseRangeGrid,
    *,
    max_cells: int = CARVE_MAX_CELLS,
    fill_holes: bool = True,
    lattice_rows: int = 0,
    lattice_cols: int = 0,
) -> Path:
    """Write one `.rmgrid`, temp-then-replace, header carrying the full key."""
    import hashlib
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = _payload_blocks(grid.rng)
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(block)
    header = _pack_header(
        key, grid, max_cells, fill_holes, lattice_rows, lattice_cols, digest.digest()
    )

    # The temporary must live in the destination directory: os.replace is only
    # atomic within a single volume.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        for block in blocks:
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def read_grid_artefact(path: Path, key: CarveGridKey) -> CoarseRangeGrid:
    """Read one `.rmgrid`, failing closed on any disagreement with `key`.

    A file found under a matching name whose header or payload disagrees is
    corrupt or renamed. It raises rather than returning a grid, because the
    failure this whole scheme exists to prevent is a stale grid quietly
    carving the wrong geometry out of a neighbour.
    """
    import hashlib

    import numpy as np

    with open(path, "rb") as handle:
        raw = handle.read(_HEADER.size)
        if len(raw) != _HEADER.size:
            raise CarveGridArtefactError(f"carve grid artefact is truncated: {path.name}")
        fields = _HEADER.unpack(raw)
        (
            magic, contract, kind, _flags, header_bytes, record_count, payload_bytes,
            grid_version, scan_index, _max_cells, width, height, _lat_rows, _lat_cols,
            ox, oy, oz, source_digest, params_digest, payload_sha256,
        ) = fields
        if magic != GRID_MAGIC:
            raise CarveGridArtefactError(f"not a carve grid artefact: {path.name}")
        if contract > GRID_CONTRACT_VERSION:
            raise CarveGridArtefactError(
                f"carve grid contract version {contract} is newer than {GRID_CONTRACT_VERSION}"
            )
        if kind != GRID_KIND or header_bytes != _HEADER.size:
            raise CarveGridArtefactError(f"carve grid header is malformed: {path.name}")

        # Limits checked before the allocation, not after it.
        cells = width * height
        if cells <= 0 or cells > MAX_ARTEFACT_CELLS:
            raise CarveGridArtefactError(f"carve grid cell count is out of range: {cells}")
        if record_count != cells or payload_bytes != cells * 4:
            raise CarveGridArtefactError(f"carve grid length fields disagree: {path.name}")

        header_key = CarveGridKey(
            source_identity=source_digest.hex(),
            scan_index=int(scan_index),
            grid_version=int(grid_version),
            cells=cells,
            params_digest=params_digest.hex(),
        )
        if header_key != key:
            raise CarveGridArtefactError(
                f"carve grid header key does not match its name: {path.name}"
            )
        values = np.empty(cells, dtype="<f4")
        digest = hashlib.sha256()
        view = values.view(np.uint8)
        read = 0
        while read < payload_bytes:
            block = handle.read(min(1 << 20, payload_bytes - read))
            if not block:
                raise CarveGridArtefactError(f"carve grid payload is truncated: {path.name}")
            digest.update(block)
            view[read : read + len(block)] = np.frombuffer(block, dtype=np.uint8)
            read += len(block)
        if handle.read(1):
            raise CarveGridArtefactError(f"carve grid has trailing bytes: {path.name}")
    if digest.digest() != payload_sha256:
        raise CarveGridArtefactError(f"carve grid payload digest does not match: {path.name}")

    return CoarseRangeGrid(
        origin=np.array([ox, oy, oz], dtype=np.float64),
        rng=values.astype(np.float32, copy=False).reshape(height, width),
        width=int(width),
        height=int(height),
    )


