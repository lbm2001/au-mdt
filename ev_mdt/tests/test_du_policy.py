"""Sanity checks for the Departure Urgency policy."""


def test_baseline_e_ceil_equals_25():
    """At Baseline params, e_daily / e_daily_ref == 1.0 exactly, so e_ceil == E_CEIL_BASE."""
    from ev_mdt.params import BaselineParams
    from ev_mdt.models.common.policies import E_CEIL_BASE, _du_e_daily, _e_daily_ref

    bp = BaselineParams()
    e_daily_cur = _du_e_daily(bp)
    e_daily_ref = _e_daily_ref()

    assert e_daily_ref > 0, "e_daily_ref must be positive"
    assert abs(e_daily_cur / e_daily_ref - 1.0) < 1e-12, (
        f"e_daily_cur ({e_daily_cur}) / e_daily_ref ({e_daily_ref}) should be 1.0 at Baseline params"
    )

    for gamma in (0.0, 0.25, 0.5, 1.0):
        ratio = e_daily_cur / e_daily_ref
        e_ceil = min(bp.e_max, E_CEIL_BASE * ratio ** gamma)
        assert abs(e_ceil - E_CEIL_BASE) < 1e-10, (
            f"e_ceil ({e_ceil}) != E_CEIL_BASE ({E_CEIL_BASE}) at gamma={gamma}"
        )
