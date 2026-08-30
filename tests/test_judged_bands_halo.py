"""`_judged_bands`' ``halo`` argument controls the padded window it returns.

**This test was inverted in WP-1.2.** WP-1.1 pinned the opposite: the
signature accepted ``halo`` and the body never read it, with the +/-1 load
hardcoded at both call sites, so raising the composed halo would have been
silently ineffective (`PHASE1-HALO-CALCULUS.md` §4 Trap 1). PLAN.md §5 item 6
needs a band-local caller to set that window, so the parameter is now wired
and the characterisation is reversed.

This is exactly the secondary falsifier §9 anticipated: "a test asserting that
`_judged_bands`' halo parameter has no effect on its output. If that test
fails, the parameter has been wired up and §4 Trap 1 no longer holds." Trap 1
no longer holds. Trap 2 — that the helper tiles from row 0 of whatever grid it
is given — is untouched, and is why `iter_clean_bands` hands the kernels
explicit core bounds instead of re-tiling a band-local array.
"""

from __future__ import annotations

from rapidmesh.filters import COMPOSED_FILTER_HALO, _judged_bands


def test_judged_bands_halo_widens_the_data_window() -> None:
    rows, band_rows = 100, 10
    for halo in (0, 1, 2, 3, 99):
        plans = _judged_bands(rows, band_rows, halo=halo)
        middle = plans[5]  # a band with lattice on both sides of it
        assert middle.core_row_start == 50
        assert middle.core_row_stop == 60
        assert middle.data_row_start == max(50 - halo, 0)
        assert middle.data_row_stop == min(60 + halo, rows)

    # Distinct halos give distinct plans, which is the property WP-1.1 pinned
    # as absent.
    baseline = _judged_bands(rows, band_rows, halo=1)
    assert _judged_bands(rows, band_rows, halo=0) != baseline
    assert _judged_bands(rows, band_rows, halo=2) != baseline
    assert _judged_bands(rows, band_rows, halo=COMPOSED_FILTER_HALO) != baseline


def test_judged_cores_do_not_move_with_halo() -> None:
    """Judge-once (`PHASE1-HALO-CALCULUS.md` §7) is not negotiable.

    The halo sizes the evidence a band reads; it must not move the core
    boundaries, or two bands would own the same sample and the exclusive
    ledger of `types.FilterStats` would stop balancing the moment the halo
    changed.
    """
    rows, band_rows = 100, 10
    expected = [(b, b + band_rows) for b in range(0, rows, band_rows)]
    for halo in (0, 1, 2, 3, 99):
        plans = _judged_bands(rows, band_rows, halo=halo)
        assert [(p.core_row_start, p.core_row_stop) for p in plans] == expected


def test_data_window_is_clipped_to_the_lattice() -> None:
    """Rows never wrap: the top and bottom of a scan are zenith and nadir."""
    plans = _judged_bands(20, 10, halo=COMPOSED_FILTER_HALO)
    assert (plans[0].data_row_start, plans[0].data_row_stop) == (0, 13)
    assert (plans[-1].data_row_start, plans[-1].data_row_stop) == (7, 20)
