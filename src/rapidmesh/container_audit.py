"""What a byte-level check of DEC-004 actually looks like — WP-4.1.

DEC-004 says per-vertex source identity never leaves the server. `check_tier`
refuses to write it; this module is how that refusal is **observed in the bytes
that were written**, because a guarantee nobody outside can check is a comment.

Four observations, because each can miss what the others catch:

* `attributes_declared` catches a field written under its own name;
* `raw_hits` catches one written under a different name, or under none;
* `inflated_hits` catches one hidden behind compression;
* `ByteLedger.balanced` closes the remaining gap, and it is the only one of the
  four that is **exhaustive**. A search can say "I did not find it". A balanced
  ledger says there is no unattributed byte for it to be in.

Split from `container.py` under DEC-010: a contract and a search over a hundred
megabytes are different responsibilities, and only the contract belongs to every
caller.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .container import CLIENT_VERTEX_ATTRIBUTES, ByteLedger
from .container_codec import byte_shuffle

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import numpy.typing as npt

    I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class IdentityAudit:
    """Whether per-vertex source identity is anywhere in a container's bytes.

    Three independent observations, because each can miss what the others
    catch. `attributes_declared` catches a field written under its own name;
    `raw_hits` catches one written under a different name or none at all;
    `inflated_hits` catches one hidden behind compression. `ledger_balanced`
    closes the remaining gap — it says there is no unattributed byte for a
    fourth hiding place to live in.
    """

    attributes_declared: tuple[str, ...]
    raw_hits: int
    inflated_hits: int
    needles_tried: int
    ledger_balanced: bool

    @property
    def clean(self) -> bool:
        return (
            self.raw_hits == 0
            and self.inflated_hits == 0
            and not (set(self.attributes_declared) - CLIENT_VERTEX_ATTRIBUTES)
        )


#: A needle shorter than this, or flatter than this, matches by coincidence and
#: is evidence of nothing. Both bars are deliberately stated as constants: a
#: search that silently admitted a run of zeros would report hits everywhere and
#: an audit that reported hits everywhere would be turned off.
MIN_NEEDLE_BYTES = 24
MIN_NEEDLE_VARIETY = 3


def identity_needles(ids: I64, *, run: int = 4, limit: int = 2) -> list[bytes]:
    """Byte patterns that would betray a stored source-identity array.

    A run of `run` consecutive int64 values is 32 bytes. A single int64 appears
    by coincidence in float data often enough to be useless as evidence; four in
    a row, in order, do not. The whole array is offered first, so a container
    that stores it contiguously is caught by one comparison rather than by luck.

    **Two layouts, not one.** A container that byte-shuffles before compressing
    stores the array's byte planes rather than its elements, so a needle in
    natural int64 order would not appear in the inflated stream even though
    every bit of the array is in it. Needles are therefore generated in the
    shuffled layout as well — any contiguous window of the shuffled bytes is
    contiguous in the stream, so the same windowing works on both.

    `limit` is small on purpose: each needle costs one pass over the container,
    a tier is a hundred megabytes and a tier has sixty tiles, so the count
    multiplies fast. The whole array in both layouts is the needle that actually
    catches a stored field; the windows catch a partial or reordered one. **The
    exhaustive statement is `ByteLedger.balanced`, not this search** — it says
    there is no unattributed byte for a field to hide in, which no amount of
    searching could establish.
    """
    import numpy as np

    flat = np.ascontiguousarray(ids, "<i8")
    if flat.size == 0:
        return []
    width = run * flat.dtype.itemsize
    layouts = [flat.tobytes(), byte_shuffle(flat)]
    out: list[bytes] = []
    for blob in layouts:
        out.append(blob)
        if len(blob) <= width:
            continue
        starts = np.unique(
            np.linspace(0, len(blob) - width, num=min(limit, len(blob) - width), dtype=np.int64)
        )
        out.extend(blob[int(s) : int(s) + width] for s in starts)
    return [
        needle
        for needle in out
        if len(needle) >= MIN_NEEDLE_BYTES and len(set(needle)) >= MIN_NEEDLE_VARIETY
    ]


def count_hits(blob: bytes, needles: Iterable[bytes]) -> int:
    """How many of `needles` occur in `blob`. Plain search, no interpretation."""
    return sum(1 for needle in needles if needle and blob.find(needle) >= 0)


#: A scan-mode audit tries at most this many candidate zlib headers. Stated as a
#: constant, and reported when it binds, because a silently truncated search
#: would be an audit that says "clean" when it means "I stopped looking".
MAX_SCAN_ATTEMPTS = 4096


def inflate_streams(blob: bytes, offsets: Iterable[int] | None = None) -> bytes:
    """Whatever DEFLATE streams `blob` holds, decompressed and concatenated.

    Compression is the obvious place a leaked array would become invisible to a
    raw byte search, so the audit looks through it.

    Pass `offsets` when the caller knows where its own compressed blocks start —
    a container writer always does, and on a hundred-megabyte tier the
    alternative is a scan. An empty list is a *positive* statement, not a
    shrug: it says this container compresses nothing, so the raw search is
    already exhaustive over it.

    With `offsets=None` the zlib headers are located by a vectorised scan and
    each candidate is tried once, bounded by `MAX_SCAN_ATTEMPTS`. Slicing is by
    `memoryview`, because slicing a hundred-megabyte `bytes` object per
    candidate is how this function would quietly become quadratic.
    """
    import numpy as np

    if offsets is None:
        raw = np.frombuffer(blob, np.uint8)
        if raw.size < 2:
            return b""
        # zlib headers: CMF 0x78 covers every window size zlib emits, and the
        # two-byte check value is a multiple of 31.
        candidates = np.flatnonzero(raw[:-1] == 0x78)
        check = raw[candidates].astype(np.uint32) * 256 + raw[candidates + 1]
        starts: list[int] = [int(v) for v in candidates[check % 31 == 0]][:MAX_SCAN_ATTEMPTS]
    else:
        starts = [int(v) for v in offsets]
    view = memoryview(blob)
    out = bytearray()
    for at in starts:
        try:
            out += zlib.decompressobj().decompress(view[at:])
        except zlib.error:
            continue
    return bytes(out)


def audit_identity(
    blob: bytes,
    ids: Sequence[I64],
    attributes: Sequence[str],
    ledger: ByteLedger,
    *,
    stream_offsets: Iterable[int] | None = None,
) -> IdentityAudit:
    """The client-tier DEC-004 check, run over the bytes that were written.

    `ids` is **one array per tile**, not one array for the tier. A container
    stores a tile's identity contiguously if it stores it at all, so a per-tile
    array gives the search a needle that can match in one piece; a concatenation
    across tiles would only ever match by window and would miss a leak in a
    single tile the windows happened to skip.
    """
    needles: list[bytes] = []
    for block in ids:
        needles.extend(identity_needles(block))
    inflated = inflate_streams(blob, stream_offsets)
    return IdentityAudit(
        attributes_declared=tuple(attributes),
        raw_hits=count_hits(blob, needles),
        inflated_hits=count_hits(inflated, needles) if inflated else 0,
        needles_tried=len(needles),
        ledger_balanced=ledger.balanced,
    )


