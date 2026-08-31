"""
Peak-memory measurement — PLAN.md §5 item 10.

Gate 1 states two different budgets and they need two different instruments:

    processing working memory   <=   512,000,000 bytes
    measured peak RSS           <= 1,500,000,000 bytes

**The three quantities this module keeps apart, because conflating any two of
them produces a number that looks like evidence and is not:**

*Peak RSS* is the operating system's high-water mark for pages the process
actually had resident — `PeakWorkingSetSize` on Windows, `VmHWM` on Linux,
`ru_maxrss` on macOS. It is the quantity Gate 1 names. It counts pages that
were touched, so an allocation that is never written to does not appear in it,
and memory the allocator has freed but not returned to the OS still does.

*Allocation attribution* is `tracemalloc`: which Python-level allocations were
live, and where they came from. It is the right tool for asking *why* a stage is
large and the wrong one for asserting a budget, because it sees only what goes
through CPython's allocator. Measured on this repository's own QA path, a
`scipy.spatial.cKDTree` build over 3,000,000 points reported 24,001,673 B to
`tracemalloc` while the resident set grew 62,013,440 B — a 2.6x under-report of
the quantity the gate is about. `tests/test_peak_memory.py` asserts that gap
rather than describing it.

*Address space* is neither. `RLIMIT_AS` bounds virtual address space and does
**not** enforce RSS (`CLAUDE.md` §11); this module never reads it, never sets
it, and never reports it as a memory figure. A test asserts the module's own
source does not mention it.

Why measurement happens in a child process
------------------------------------------
A peak is a high-water mark: it only rises. Once one stage has touched 400 MB,
every later measurement in that process reports at least 400 MB, and Windows
exposes no way to reset the counter. Per-workload figures therefore come from a
**fresh child process** that runs one workload and reports its own peak before
exiting. That also matches how meshing is deployed — out of process, with its
own limit (`CLAUDE.md` §4 rule 8) — so the number measured is the number that
matters.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Gate 1 / `PHASE1-TILE-CONTRACT-V0.md` §6. Exact byte figures, not "512 MB":
# the budget is stated in decimal bytes and a MiB reading of it would quietly
# grant 7.4% more.
WORKING_MEMORY_BUDGET_BYTES = 512_000_000
PEAK_RSS_BUDGET_BYTES = 1_500_000_000

SOURCE_WINDOWS = "PeakWorkingSetSize"
SOURCE_LINUX = "VmHWM"
SOURCE_MACOS = "ru_maxrss"


class PeakMemoryUnavailable(RuntimeError):
    """The platform's peak-RSS API could not be read.

    Raised rather than returning 0. A zero would flow into a budget assertion
    and pass it, which is the failure mode this whole module exists to prevent.
    """


@dataclass(frozen=True)
class RssSample:
    """One reading of the process's resident set."""

    current_bytes: int
    peak_bytes: int
    source: str


def rss_sample() -> RssSample:
    """Current and peak RSS for this process, from the platform's own API."""
    if sys.platform == "win32":
        return _windows_sample()
    if sys.platform == "darwin":
        return _macos_sample()
    return _linux_sample()


def peak_rss_bytes() -> int:
    return rss_sample().peak_bytes


def current_rss_bytes() -> int:
    return rss_sample().current_bytes


def _windows_sample() -> RssSample:
    import ctypes
    import ctypes.wintypes as wt

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError as exc:                                  # pragma: no cover
        raise PeakMemoryUnavailable(f"psapi/kernel32 unavailable: {exc}") from exc

    psapi.GetProcessMemoryInfo.argtypes = [
        wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):                                                      # pragma: no cover
        raise PeakMemoryUnavailable(
            f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}"
        )
    peak = int(counters.PeakWorkingSetSize)
    if peak <= 0:                                           # pragma: no cover
        raise PeakMemoryUnavailable("PeakWorkingSetSize reported zero")
    return RssSample(int(counters.WorkingSetSize), peak, SOURCE_WINDOWS)


def _linux_sample() -> RssSample:                           # pragma: no cover
    """Read `/proc/self/status`. Unexercised on this workstation — see report."""
    with open("/proc/self/status", encoding="ascii") as handle:
        text = handle.read()
    current, peak = parse_proc_status(text)
    return RssSample(current, peak, SOURCE_LINUX)


def parse_proc_status(text: str) -> tuple[int, int]:
    """`(VmRSS, VmHWM)` in bytes from `/proc/<pid>/status` text.

    Split out from the file read so the parsing can be tested on a platform
    that has no `/proc`. The kernel reports these in kB; a reader that took the
    number at face value would under-report by 1024x, which is exactly the kind
    of error a budget assertion would then pass.
    """
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            parts = rest.split()
            if len(parts) != 2 or parts[1].lower() != "kb":
                raise PeakMemoryUnavailable(f"unexpected {key} line: {line!r}")
            values[key] = int(parts[0]) * 1024
    if "VmHWM" not in values:
        raise PeakMemoryUnavailable("/proc status carries no VmHWM")
    return values.get("VmRSS", 0), values["VmHWM"]


def _macos_sample() -> RssSample:                           # pragma: no cover
    # Guarded on `sys.platform` rather than by a try/except so a type checker
    # on Windows skips the block instead of failing on a module that does not
    # exist there.
    if sys.platform == "darwin":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = int(usage.ru_maxrss)             # bytes on macOS, kB on Linux
        if peak <= 0:
            raise PeakMemoryUnavailable("ru_maxrss reported zero")
        return RssSample(peak, peak, SOURCE_MACOS)
    raise PeakMemoryUnavailable("the ru_maxrss path is macOS only")


# ---------------------------------------------------------------------------
# one measured workload, in its own process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryMeasurement:
    """One workload's memory figures, with the instruments kept separate.

    `peak_rss_bytes` is the OS high-water mark and is the Gate 1 quantity.
    `working_set_delta_bytes` subtracts the interpreter-and-imports baseline
    sampled before the workload, so it attributes growth to the workload — and
    it deliberately still includes the input fixture, because in production the
    scan is resident too. Conservative in the right direction: a run that passes
    with the input counted passes without it.

    `traced_peak_bytes` is `tracemalloc` and is reported **beside** the gate
    figures, never as them.
    """

    label: str
    peak_rss_bytes: int
    baseline_rss_bytes: int
    traced_peak_bytes: int
    seconds: float
    source: str
    platform: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def working_set_delta_bytes(self) -> int:
        return max(self.peak_rss_bytes - self.baseline_rss_bytes, 0)

    @property
    def tracemalloc_shortfall_bytes(self) -> int:
        """How much resident growth `tracemalloc` did not account for."""
        return self.working_set_delta_bytes - self.traced_peak_bytes

    def within_budgets(self) -> tuple[bool, bool]:
        return (
            self.working_set_delta_bytes <= WORKING_MEMORY_BUDGET_BYTES,
            self.peak_rss_bytes <= PEAK_RSS_BUDGET_BYTES,
        )

    def describe(self) -> str:
        working, peak = self.within_budgets()
        return (
            f"{self.label:<28} peak_rss={self.peak_rss_bytes:>13,} B "
            f"({'ok' if peak else 'OVER'})  working={self.working_set_delta_bytes:>13,} B "
            f"({'ok' if working else 'OVER'})  traced={self.traced_peak_bytes:>13,} B  "
            f"{self.seconds:6.2f}s  [{self.source}]"
        )


def measure_in_child(
    module: str,
    attr: str,
    kwargs: dict[str, Any] | None = None,
    *,
    label: str | None = None,
    sys_path: Sequence[str] = (),
    timeout: float = 900.0,
) -> MemoryMeasurement:
    """Run `module.attr(**kwargs)` in a fresh interpreter and report its peak.

    A fresh process because a peak only rises: measuring two workloads in one
    process gives the second one the first one's high-water mark. The workload
    must be importable and its arguments JSON-serialisable — deliberately
    narrow, so what was measured is reproducible from the recorded spec alone.
    """
    spec = {
        "module": module,
        "attr": attr,
        "kwargs": kwargs or {},
        "sys_path": [str(p) for p in sys_path],
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import rapidmesh.memory as m; m._child_main()"],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": _package_root()},
        check=False,
    )
    if completed.returncode != 0:
        raise PeakMemoryUnavailable(
            f"measured child failed ({completed.returncode}): "
            f"{completed.stderr.strip()[-2000:]}"
        )
    payload = _last_json_line(completed.stdout)
    return MemoryMeasurement(
        label=label or f"{module}.{attr}",
        peak_rss_bytes=int(payload["peak_rss_bytes"]),
        baseline_rss_bytes=int(payload["baseline_rss_bytes"]),
        traced_peak_bytes=int(payload["traced_peak_bytes"]),
        seconds=float(payload["seconds"]),
        source=str(payload["source"]),
        platform=str(payload["platform"]),
        detail=dict(payload.get("detail") or {}),
    )


def _package_root() -> str:
    """The `src` directory this package was imported from."""
    return str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _last_json_line(text: str) -> dict[str, Any]:
    """The result line, ignoring anything the workload printed itself."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed: dict[str, Any] = json.loads(stripped)
            if "peak_rss_bytes" in parsed:
                return parsed
    raise PeakMemoryUnavailable(f"measured child produced no result line: {text[-2000:]}")


def _child_main() -> None:
    """Entry point inside the measured process. Not called directly."""
    import importlib
    import tracemalloc

    spec = json.loads(sys.stdin.read())
    for entry in spec.get("sys_path", ()):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    module = importlib.import_module(spec["module"])
    target: Callable[..., Any] = getattr(module, spec["attr"])

    # Baseline after imports, before the workload. Everything the workload
    # causes — including building its own input — is attributed to it.
    baseline = rss_sample().peak_bytes
    tracemalloc.start()
    started = time.perf_counter()
    detail = target(**spec["kwargs"])
    seconds = time.perf_counter() - started
    traced = int(tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()
    sample = rss_sample()

    print(json.dumps({
        "peak_rss_bytes": sample.peak_bytes,
        "baseline_rss_bytes": baseline,
        "traced_peak_bytes": traced,
        "seconds": seconds,
        "source": sample.source,
        "platform": f"{sys.platform}/{platform.machine()}",
        "detail": detail if isinstance(detail, dict) else {},
    }))


# ---------------------------------------------------------------------------
# self-test workloads — used to prove the instrument responds at all
# ---------------------------------------------------------------------------


def touch_bytes(total_bytes: int) -> dict[str, Any]:
    """Write to `total_bytes` of memory so the pages become resident.

    Writing matters. `numpy.zeros` reserves address space that the OS may not
    commit until it is used, so an untouched allocation moves `tracemalloc` and
    leaves peak RSS alone — which is the right behaviour for RSS and a trap for
    anyone treating the two as interchangeable.
    """
    import numpy as np

    block = np.ones(max(total_bytes // 8, 1), np.float64)
    checksum = float(block[0] + block[-1])
    del block
    return {"requested_bytes": int(total_bytes), "checksum": checksum}


def reserve_untouched_bytes(total_bytes: int) -> dict[str, Any]:
    """Allocate without writing. The contrast case for `touch_bytes`."""
    import numpy as np

    block = np.empty(max(total_bytes // 8, 1), np.float64)
    size = int(block.nbytes)
    del block
    return {"requested_bytes": int(total_bytes), "allocated_bytes": size}


def touch_mapped_bytes(total_bytes: int, page: int = 4096) -> dict[str, Any]:
    """Make pages resident **outside** CPython's allocator, via an anonymous map.

    The starkest form of the point this module is built around: `tracemalloc`
    reports a few hundred bytes for the `mmap` object while the resident set
    grows by the whole mapping. A budget asserted on allocation attribution
    would pass a run using 600 MB of real memory; the OS instrument would not.
    Nothing in the pipeline maps memory this way — this workload exists to make
    the two instruments disagree by a margin no rounding can explain.
    """
    import mmap

    mapped = mmap.mmap(-1, int(total_bytes))
    try:
        for offset in range(0, int(total_bytes), page):
            mapped[offset] = 1
        return {"requested_bytes": int(total_bytes), "pages": int(total_bytes) // page}
    finally:
        mapped.close()


def build_kdtree(points: int, seed: int = 0) -> dict[str, Any]:
    """A `cKDTree` build — the QA path's allocation, and `tracemalloc`'s blind
    spot. SciPy allocates the tree outside CPython's allocator, so this workload
    is how the module demonstrates that attribution and RSS are different
    measurements rather than two views of one."""
    import numpy as np
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    cloud = rng.random((points, 3))
    tree = cKDTree(cloud)
    distances, _ = tree.query(cloud[:1000], k=1, workers=1)
    return {"points": int(points), "mean_self_distance": float(distances.mean())}
