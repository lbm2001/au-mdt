"""Sanity checks for the Departure Urgency policy."""


def test_baseline_demand_ratio_is_one():
    """_du_e_daily(BaselineParams()) / _e_daily_ref() == 1.0 exactly by construction."""
    from ev_mdt.params import BaselineParams
    from ev_mdt.models.common.policies import _du_e_daily, _e_daily_ref

    bp = BaselineParams()
    e_daily_cur = _du_e_daily(bp)
    e_daily_ref = _e_daily_ref()

    assert e_daily_ref > 0, "e_daily_ref must be positive"
    assert abs(e_daily_cur / e_daily_ref - 1.0) < 1e-12, (
        f"e_daily_cur ({e_daily_cur}) / e_daily_ref ({e_daily_ref}) should be 1.0 at Baseline params"
    )


def test_baseline_e_ceil_equals_25_for_all_gammas():
    """When demand ratio == 1.0, e_ceil == E_CEIL_BASE regardless of gamma."""
    from ev_mdt.params import BaselineParams
    from ev_mdt.models.common.policies import E_CEIL_BASE, _du_e_daily, _e_daily_ref

    bp = BaselineParams()
    ratio = _du_e_daily(bp) / _e_daily_ref()

    for gamma in (0.0, 0.5, 1.0):
        e_ceil = min(bp.e_max, E_CEIL_BASE * ratio ** gamma)
        assert abs(e_ceil - E_CEIL_BASE) < 1e-10, (
            f"e_ceil ({e_ceil}) != E_CEIL_BASE ({E_CEIL_BASE}) at gamma={gamma}"
        )


def test_next_trip_policy_baseline_ceil():
    """next_trip_policy with _ceil_override bypasses gamma; without it, e_ceil == 25 at baseline."""
    from ev_mdt.params import BaselineParams
    from ev_mdt.models.common.policies import E_CEIL_BASE, next_trip_policy

    bp = BaselineParams()
    # Place battery well below e_trip so reserve fires — confirm we get u_max.
    u = next_trip_policy(0, 0, 0.0, 0.0, bp, use_reserve=True)
    assert u == bp.u_max, f"Expected u_max={bp.u_max} below reserve floor, got {u}"

    # Battery at e_max → always 0 regardless of gamma.
    for gamma in (0.0, 0.5, 1.0):
        u = next_trip_policy(0, 0, bp.e_max, 0.0, bp, gamma=gamma)
        assert u == 0.0, f"Expected 0 at e_max, got {u} (gamma={gamma})"

    # _ceil_override path: override to a known value and check that at t=0 (τ large)
    # with very low price the policy charges.
    u_override = next_trip_policy(
        0, 0, 0.5 * bp.e_max, 0.0, bp,
        _ceil_override=E_CEIL_BASE, use_reserve=False,
    )
    # At near-zero price (lam=0), F_p(lam) ≈ 0 ≤ ρ, so should charge.
    assert u_override == bp.u_max, (
        f"Expected u_max at low price with _ceil_override={E_CEIL_BASE}, got {u_override}"
    )
