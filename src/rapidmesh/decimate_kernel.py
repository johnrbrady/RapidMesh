"""The Rust collapse kernel, behind `decimate.decimate_patch` — WP-3.4b.

This module is the whole of the Python side of the native path. It is
deliberately the only place that knows the kernel exists, so that:

* `decimate.py` keeps one dispatch branch and is otherwise untouched — it is
  the **equivalence reference** and the fallback, and a reference that had been
  restructured around its own replacement would be worth less as both;
* a tree with no `rapidmesh_kernel` built behaves exactly as it did at Round 10,
  with no import error, no warning at import time and no change in output;
* the compaction that turns sweep state into a `DecimatedPatch` is *the same
  NumPy code* on both paths, because `_KernelState` inherits `finish()` rather
  than reimplementing it. Output compaction therefore cannot be a source of
  disagreement between the two paths, and the equivalence claim is narrowed to
  the sweep itself, which is what it is about.

**What stays in NumPy, and why.** The per-vertex quadrics
(`decimate_quadrics.vertex_quadrics`) and the unique-edge list
(`decimate._unique_edges`) are computed here and handed across. Both are
vectorised, both are small — together about 6% of a tile — and the first of them
accumulates with `np.bincount`, whose summation order would have to be
reproduced exactly for a bit-identical result. Reproducing it by hand is the
one part of this port with real float-identity risk, and it buys 6% of a
speedup that has an order of magnitude in hand. So it is not taken.

**Threading.** The kernel is single-threaded. It releases the GIL while it
sweeps, which lets a caller overlap it with I/O, but it starts no threads of its
own: a threaded kernel would make determinism a separate claim needing separate
evidence, and the DEC-013 bar does not need the speed.
"""

from __future__ import annotations

import importlib
import math
import os
from array import array
from typing import TYPE_CHECKING, Any

from rapidmesh.decimate import DecimationSettings, _PatchState, _unique_edges
from rapidmesh.decimate_quadrics import vertex_quadrics

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    BOOL = npt.NDArray[np.bool_]
    F64 = npt.NDArray[np.float64]
    I64 = npt.NDArray[np.int64]

#: The kernel contract this module speaks. `rapidmesh_kernel.contract_version()`
#: must match, so that a stale `.pyd` left on `sys.path` by an earlier build is
#: refused rather than silently answering a different question.
CONTRACT_VERSION = 1

#: Environment override for the default dispatch: `python`, `rust` or `auto`.
#: An explicit `kernel=` argument to `decimate_patch` always wins over it.
ENVIRONMENT_VARIABLE = "RAPIDMESH_DECIMATE_KERNEL"

_module: Any = None
_reason: str | None = None
_probed = False


def _probe() -> None:
    """Import the extension once, and remember why if it is not there."""
    global _module, _reason, _probed
    if _probed:
        return
    _probed = True
    # `importlib` rather than a plain `import`, and not for style. A plain
    # import of an extension module makes the **type gate's result depend on
    # whether the crate happens to be built**: absent, mypy reports
    # `import-not-found`; installed without a `py.typed` marker, it reports
    # `import-untyped` instead, so a `type: ignore` written for one becomes both
    # wrong and unused under the other. A gate that passes or fails according to
    # local build state is not a gate. Resolving the name at runtime keeps
    # `mypy --strict` clean and identical either way, with no suppression.
    try:
        module = importlib.import_module("rapidmesh_kernel")
    except ImportError as error:
        _reason = f"not built or not importable: {error}"
        return
    try:
        found = int(module.contract_version())
    except Exception as error:                       # pragma: no cover - defensive
        _reason = f"present but has no usable contract_version(): {error}"
        return
    if found != CONTRACT_VERSION:
        _reason = (
            f"contract mismatch: module speaks {found}, this tree speaks "
            f"{CONTRACT_VERSION}; rebuild the crate"
        )
        return
    _module = module


def available() -> bool:
    """Whether the native sweep can be used in this interpreter."""
    _probe()
    return _module is not None


def unavailable_reason() -> str | None:
    """Why not, in words, or `None` when it is available."""
    _probe()
    return None if _module is not None else _reason


def describe() -> dict[str, Any]:
    """What a measurement record should carry about the kernel."""
    _probe()
    return {
        "available": _module is not None,
        "reason": _reason,
        "contract_version": CONTRACT_VERSION,
    }


#: What `kernel="auto"` resolves to when the environment says nothing.
#:
#: **`python`, deliberately, and not `rust`.** Whether RapidMesh adopts the
#: native sweep is the owner's decision and it has not been taken; until it is,
#: merely having built the crate must not change which code path an existing
#: caller runs. That is not a hypothetical tidiness argument — with `auto`
#: defaulting to the native sweep, Round 10's own red-case test
#: (`test_without_the_locked_join_rule_two_tiles_emit_the_same_face`, which
#: monkeypatches `_PatchState._would_join_locked` and asserts the fault
#: returns) silently stops testing anything, because a monkeypatch on the
#: Python class cannot reach the Rust one. It failed loudly rather than
#: silently, and this default is the fix.
#:
#: Flipping it to `rust` is a one-line change in whichever package acts on that
#: decision. Until then the native sweep is opt-in: pass `kernel="rust"`, or set
#: the environment variable.
DEFAULT_CHOICE = "python"


def default_choice() -> str:
    """The dispatch used when a caller does not name one."""
    choice = os.environ.get(ENVIRONMENT_VARIABLE, DEFAULT_CHOICE).strip().lower()
    return choice if choice in ("auto", "rust", "python") else DEFAULT_CHOICE


def _from_bytes(raw: bytes) -> array[float]:
    """A `bytes` blob of float64 as the `array('d')` `_PatchState` holds."""
    out = array("d")
    out.frombytes(raw)
    return out


class _KernelState(_PatchState):
    """Sweep state produced by the Rust kernel, in `_PatchState`'s shape.

    `__init__` is deliberately not called: there is no work for it to do, the
    sweep having already run. What this class exists for is `finish()`, which it
    inherits unchanged.
    """

    def __init__(self, result: dict[str, Any], count: int, locked: BOOL) -> None:
        import numpy as np

        self.count = count
        self.locked = locked.tolist()
        # Each field is given the type `_PatchState` declares for it, so that
        # the inherited `finish()` sees exactly what it sees on the Python path
        # and neither this class nor that method needs a special case.
        self.px = _from_bytes(result["pos_x"])
        self.py = _from_bytes(result["pos_y"])
        self.pz = _from_bytes(result["pos_z"])
        self.alive = np.frombuffer(result["alive"], np.uint8).astype(bool).tolist()
        self.moved = np.frombuffer(result["moved"], np.uint8).astype(bool).tolist()
        self.tri_alive = bytearray(result["triangle_alive"])
        # The one exception, and it is a deliberate one. `_PatchState` holds
        # `tv` as `list[int]` because its sweep indexes it a hundred million
        # times and a boxed-int list is faster there than a NumPy scalar read.
        # Nothing indexes it on this path — `finish()` does one
        # `np.asarray(...)` over it — so materialising 1.5 M Python ints to
        # match the annotation would cost tens of megabytes to hand straight
        # back to NumPy.
        self.tv = np.frombuffer(  # type: ignore[assignment]
            result["triangle_vertices"], np.int64
        )
        self.collapses = int(result["collapses"])
        self.rejected_link = int(result["rejected_link"])
        self.rejected_seam = int(result["rejected_seam"])
        self.rejected_turn = int(result["rejected_turn"])
        self.rejected_error = int(result["rejected_error"])
        self.max_accepted = float(result["max_accepted"])
        self.live_triangles = int(result["live_triangles"])


def sweep(
    pos64: F64, tris: I64, lock: BOOL, settings: DecimationSettings
) -> _KernelState:
    """Run one collapse sweep in Rust, returning state `finish()` can compact.

    The caller has already validated shapes; this repeats none of it.
    """
    import numpy as np

    _probe()
    if _module is None:                              # pragma: no cover - guarded
        raise RuntimeError(f"rapidmesh_kernel unavailable: {_reason}")

    count = int(pos64.shape[0])
    faces = int(tris.shape[0])
    quad = vertex_quadrics(pos64, tris, count)
    edges = _unique_edges(tris)

    # `max_error_m * max_error_m`, not `** 2`: `_PatchState.__init__` writes the
    # multiplication and the two are not required to agree in the last bit.
    error_limit = (
        math.inf
        if math.isinf(settings.max_error_m)
        else settings.max_error_m * settings.max_error_m
    )

    result = _module.decimate_sweep(
        np.ascontiguousarray(pos64[:, 0], np.float64).tobytes(),
        np.ascontiguousarray(pos64[:, 1], np.float64).tobytes(),
        np.ascontiguousarray(pos64[:, 2], np.float64).tobytes(),
        quad.tobytes(),
        np.ascontiguousarray(lock, bool).tobytes(),
        np.ascontiguousarray(tris, np.int64).tobytes(),
        np.ascontiguousarray(edges, np.int64).tobytes(),
        count,
        faces,
        -1 if settings.target_triangles is None else int(settings.target_triangles),
        error_limit,
        math.cos(math.radians(settings.max_normal_turn_deg)),
        settings.placement == "optimal",
    )
    return _KernelState(result, count, lock)
