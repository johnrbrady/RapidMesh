"""
Peak-memory instrument and budget assertions — PLAN.md §5 item 10.

The tests that matter here are the ones that would catch a *silently broken
instrument*, because a memory gate asserted against a broken instrument passes
and means nothing:

* `test_peak_rises_when_pages_are_actually_touched` fails if the platform API
  returns 0, or returns a constant, or is never updated.
* `test_tracemalloc_under_reports_the_gate_quantity` fails if someone decides
  `tracemalloc` is close enough to stand in for resident set size. It is not,
  and the gap is measured on this repository's own QA path.
* `test_untouched_allocation_moves_tracemalloc_and_not_peak_rss` fails if the
  two instruments are treated as two views of one quantity. They disagree in
  *both* directions, and the test pins both.
* `test_the_budget_check_can_fail` runs a workload deliberately over the
  working-memory budget and requires the check to say so.

The pipeline budget assertions state their fixture size in the test itself.
None of them is the Gate 1 claim: Gate 1 names the 14.5 M-point reference
station, no real data is touched here, and `test_budgets_are_not_claimed_for_
the_reference_station` exists to keep that distinction from eroding.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from rapidmesh import memory
from rapidmesh.memory import (
    PEAK_RSS_BUDGET_BYTES,
    WORKING_MEMORY_BUDGET_BYTES,
    MemoryMeasurement,
    PeakMemoryUnavailable,
    measure_in_child,
    parse_proc_status,
    rss_sample,
)

TOOLS = str(Path(__file__).resolve().parent.parent / "tools")

# Stated fixture for the pipeline budget rows below. Small enough to keep the
# suite quick; the scaling series that says what happens at station scale is in
# `tools/measure_peak_memory.py` and the WP-1.5 report, not here.
FIXTURE_ROWS, FIXTURE_COLS = 200, 800


@pytest.fixture(scope="module")
def prepared_fixture(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One scan, saved once, loaded by every measured child.

    Generation is deliberately outside every measured run: `synthetic.generate`
    ray-casts each lattice cell and its transients would be charged to the
    pipeline, overstating every row.
    """
    path = str(tmp_path_factory.mktemp("mem") / "scan.npz")
    made = measure_in_child(
        "measure_peak_memory", "prepare_fixture",
        {"path": path, "rows": FIXTURE_ROWS, "cols": FIXTURE_COLS},
        label="prepare", sys_path=[TOOLS],
    )
    assert made.detail["samples"] > 100_000
    return path


# ---------------------------------------------------------------------------
# the instrument itself
# ---------------------------------------------------------------------------


def test_rss_sample_uses_the_documented_api_for_this_platform() -> None:
    import sys

    sample = rss_sample()
    expected = {
        "win32": memory.SOURCE_WINDOWS,
        "darwin": memory.SOURCE_MACOS,
    }.get(sys.platform, memory.SOURCE_LINUX)
    assert sample.source == expected
    assert sample.peak_bytes > 0
    assert sample.peak_bytes >= sample.current_bytes


def test_peak_never_decreases_within_a_process() -> None:
    """A high-water mark that fell would make every later reading a lie — and
    is why per-workload figures come from fresh child processes."""
    import numpy as np

    before = rss_sample().peak_bytes
    block = np.ones(8_000_000, np.float64)      # 64 MB, written
    during = rss_sample().peak_bytes
    del block
    after = rss_sample().peak_bytes
    assert during >= before
    assert after >= during


def test_peak_rises_when_pages_are_actually_touched() -> None:
    """The instrument-is-alive test.

    Fails if the platform API returns zero, returns a constant, or is never
    refreshed — each of which would make every budget assertion below vacuous.
    """
    measurement = measure_in_child(
        "rapidmesh.memory", "touch_bytes", {"total_bytes": 256_000_000},
        label="touch 256 MB",
    )
    assert measurement.peak_rss_bytes > measurement.baseline_rss_bytes
    assert measurement.working_set_delta_bytes >= 200_000_000, measurement.describe()
    assert measurement.source == rss_sample().source


def test_untouched_allocation_moves_tracemalloc_and_not_peak_rss() -> None:
    """The two instruments disagree, and this is the direction people forget.

    `numpy.empty` reserves address space the OS need not commit until it is
    written. `tracemalloc` counts the allocation in full; resident set size
    counts almost none of it. Anyone treating the two as interchangeable gets
    the wrong answer here by more than an order of magnitude.
    """
    measurement = measure_in_child(
        "rapidmesh.memory", "reserve_untouched_bytes", {"total_bytes": 256_000_000},
        label="reserve 256 MB",
    )
    assert measurement.traced_peak_bytes >= 250_000_000, measurement.describe()
    assert measurement.working_set_delta_bytes < 100_000_000, measurement.describe()


def test_tracemalloc_under_reports_the_gate_quantity() -> None:
    """And this is the direction that would make a budget assertion pass wrongly.

    `scipy.spatial.cKDTree` — the QA path's own structure — allocates outside
    CPython's allocator, so `tracemalloc` cannot see it. Resident set size can.
    A budget asserted on `tracemalloc` alone would be measuring a quantity
    smaller than the one Gate 1 names.
    """
    measurement = measure_in_child(
        "rapidmesh.memory", "build_kdtree", {"points": 2_000_000},
        label="cKDTree 2M",
    )
    assert measurement.working_set_delta_bytes > measurement.traced_peak_bytes, (
        measurement.describe()
    )
    assert measurement.tracemalloc_shortfall_bytes > 20_000_000, measurement.describe()


def test_the_module_never_reads_rlimit_as() -> None:
    """`CLAUDE.md` §11: `RLIMIT_AS` bounds virtual address space and does not
    enforce RSS. Reporting it as a memory figure is the specific error this
    project has already been burned by, so it is checked rather than trusted."""
    source = inspect.getsource(memory)
    assert "RLIMIT_AS" not in source.replace("`RLIMIT_AS`", "", 2)
    assert "getrlimit" not in source
    assert "setrlimit" not in source


# ---------------------------------------------------------------------------
# the Linux parser, on a platform without /proc
# ---------------------------------------------------------------------------


def test_parse_proc_status_converts_kilobytes_to_bytes() -> None:
    """The kernel reports kB. A reader taking the number at face value would
    under-report by 1024x and pass any budget."""
    text = "Name:\tpython\nVmRSS:\t  123456 kB\nVmHWM:\t  654321 kB\nThreads:\t8\n"
    current, peak = parse_proc_status(text)
    assert (current, peak) == (123456 * 1024, 654321 * 1024)


@pytest.mark.parametrize(
    "text",
    (
        "Name:\tpython\nVmRSS:\t 100 kB\n",                    # no VmHWM at all
        "VmHWM:\t 100\n",                                       # no unit
        "VmHWM:\t 100 pages\n",                                 # wrong unit
    ),
)
def test_parse_proc_status_refuses_what_it_cannot_read(text: str) -> None:
    with pytest.raises(PeakMemoryUnavailable):
        parse_proc_status(text)


# ---------------------------------------------------------------------------
# the child harness
# ---------------------------------------------------------------------------


def test_a_failing_workload_is_reported_not_swallowed() -> None:
    with pytest.raises(PeakMemoryUnavailable, match="measured child failed"):
        measure_in_child("rapidmesh.memory", "touch_bytes", {"wrong_kwarg": 1})


def test_a_missing_workload_is_reported() -> None:
    with pytest.raises(PeakMemoryUnavailable):
        measure_in_child("rapidmesh.memory", "no_such_workload", {})


# ---------------------------------------------------------------------------
# budgets on a stated synthetic fixture
# ---------------------------------------------------------------------------


def _measure(attr: str, prepared: str, **kwargs: Any) -> MemoryMeasurement:
    return measure_in_child(
        "measure_peak_memory", attr, {"fixture": prepared, **kwargs},
        label=attr, sys_path=[TOOLS],
    )


def test_both_budgets_hold_for_the_streamed_pipeline_on_this_fixture(
    prepared_fixture: str,
) -> None:
    """Both Gate 1 budgets, on a **stated** synthetic fixture: 200 x 800,
    about 158,000 samples. Measured by `PeakWorkingSetSize` in a fresh child,
    never by `tracemalloc`, which is reported beside it and not asserted on."""
    measurement = _measure("streamed_station", prepared_fixture, band_rows=64)
    working, peak = measurement.within_budgets()
    assert peak, measurement.describe()
    assert working, measurement.describe()
    assert measurement.source == rss_sample().source
    assert measurement.working_set_delta_bytes > 0
    assert measurement.detail["triangles"] > 100_000


def test_the_in_memory_path_is_measured_by_the_same_instrument(
    prepared_fixture: str,
) -> None:
    """Both paths must be comparable, so both are measured the same way."""
    measurement = _measure("in_memory_station", prepared_fixture)
    working, peak = measurement.within_budgets()
    assert peak and working, measurement.describe()
    assert measurement.detail["triangles"] > 100_000


def test_pass_a_is_bounded_by_band_size_not_by_station_size(
    prepared_fixture: str,
) -> None:
    """The claim item 8 rests on, measured with the gate instrument.

    Halving `band_rows` must not leave Pass A's peak unchanged; if it did, the
    banding would not be bounding anything. Pass A must also stay well under
    the whole streamed run, which is what makes Pass B the thing to fix.
    """
    wide = _measure("streamed_pass_a", prepared_fixture, band_rows=64)
    narrow = _measure("streamed_pass_a", prepared_fixture, band_rows=16)
    whole = _measure("streamed_station", prepared_fixture, band_rows=64)

    assert narrow.working_set_delta_bytes < wide.working_set_delta_bytes, (
        wide.describe(), narrow.describe()
    )
    assert wide.working_set_delta_bytes < whole.working_set_delta_bytes
    assert narrow.detail["bands"] > wide.detail["bands"]


def test_the_budget_check_can_fail(prepared_fixture: str) -> None:
    """A check that never rejects is not a check.

    A workload deliberately past the 512,000,000-byte working budget must be
    reported as over it. Without this, every green budget row above could be a
    comparison that cannot go red.
    """
    over = measure_in_child(
        "rapidmesh.memory", "touch_bytes",
        {"total_bytes": WORKING_MEMORY_BUDGET_BYTES + 200_000_000},
        label="deliberately over budget",
    )
    working, peak = over.within_budgets()
    assert not working, over.describe()
    assert peak, over.describe()        # still inside the 1.5 GB peak budget
    assert over.working_set_delta_bytes > WORKING_MEMORY_BUDGET_BYTES


def test_a_budget_asserted_on_tracemalloc_would_pass_a_run_that_is_over() -> None:
    """The failure mode named in the package objective, made concrete.

    An anonymous memory map makes 600,000,000 bytes resident while
    `tracemalloc` sees a few hundred. The OS instrument must call this over the
    working-memory budget; the attribution instrument would call it free. If
    `within_budgets` were ever rewired to `traced_peak_bytes`, this is the test
    that goes red.
    """
    over = measure_in_child(
        "rapidmesh.memory", "touch_mapped_bytes",
        {"total_bytes": WORKING_MEMORY_BUDGET_BYTES + 100_000_000},
        label="mapped, invisible to tracemalloc",
    )
    assert over.working_set_delta_bytes > WORKING_MEMORY_BUDGET_BYTES, over.describe()
    assert over.traced_peak_bytes < 10_000_000, over.describe()
    working, _peak = over.within_budgets()
    assert not working, over.describe()


def test_budgets_are_not_claimed_for_the_reference_station() -> None:
    """Gate 1's memory row names the 14.5 M-point station. Nothing in this file
    measures it, and the budget constants must stay exactly as specified so a
    later run cannot quietly be graded against a wider bar."""
    assert WORKING_MEMORY_BUDGET_BYTES == 512_000_000
    assert PEAK_RSS_BUDGET_BYTES == 1_500_000_000
    assert FIXTURE_ROWS * FIXTURE_COLS < 1_000_000     # far below 14.5 M records
