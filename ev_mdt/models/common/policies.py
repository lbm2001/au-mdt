"""Shared charging policies — work with any params that has the standard fields."""
import numpy as np

from ev_mdt.models.common.model_utils import (
    expected_trip_minutes, max_minutes_to_departure, minutes_to_departure,
    price_bin, price_bin_probs,
)


def actual_charge_rate(chi: int, e: float, desired_u: float, params) -> float:
    if chi > 0 and e > params.e_min:
        return 0.0
    return float(np.clip(desired_u, 0.0, params.u_max))


def backward_induction_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, pi: np.ndarray, actions: np.ndarray, e_grid: np.ndarray,
) -> float:
    e_idx   = int(np.argmin(np.abs(e_grid - e)))
    lam_idx = price_bin(lam, params)
    return float(actions[pi[t, chi, e_idx, lam_idx]])


def maximal_charging_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_max)


def price_oriented_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, low_threshold: float, high_threshold: float,
) -> float:
    if lam <= low_threshold:
        return float(params.u_max)
    if lam <= high_threshold:
        return float(params.u_max / 2)
    return 0.0


def night_charging_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_max) if t % 1440 < 360 else 0.0


def minimum_soc_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, soc_threshold: float,
) -> float:
    return float(params.u_max) if e < soc_threshold else 0.0


def always_minimum_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_min)


def random_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, rng: np.random.Generator,
) -> float:
    return float(rng.choice([0.0, params.u_min, params.u_max / 2, params.u_max]))


def dp_heuristic_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, price_bin_probs_fn=None,
) -> float:
    """SoC-urgency heuristic: charge at u_max when F_t(lam) ≤ 1 − e/e_max.

    price_bin_probs_fn : t -> (K,) bin-probability vector for the active pricing
    world.  When None, falls back to the Gaussian-parametric distribution derived
    from `params` (correct only when that *is* the world being simulated).
    """
    if chi > 0 and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0
    thresh   = 1.0 - e / params.e_max
    probs    = price_bin_probs(t, params) if price_bin_probs_fn is None else price_bin_probs_fn(t)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K for j in range(params.K)])
    F_p      = float(probs[lam_grid <= lam].sum())
    return float(params.u_max) if F_p <= thresh else 0.0


def _departure_urgency_target(tau: float, tau_max: float, e_trip: float, e_max: float,
                              alpha: float = 1.0) -> float:
    """Time-varying target: e_trip at τ=0, e_max at τ=τ_max, shape controlled by alpha.

    alpha=1 → linear;  alpha<1 → concave (rises fast early, flattens near tau_max).
    """
    frac = min(1.0, tau / tau_max) if tau_max > 0 else 1.0
    return e_trip + (e_max - e_trip) * (frac ** alpha)


def next_trip_policy(
    t: int, chi: int, e: float, lam: float, params,
    *,
    price_bin_probs_fn=None,
    target_mode: str = "fixed",
    target_frac: float = 1.0,
    reserve_frac: float = 0.25,
    use_reserve: bool = True,
    alpha: float = 0.5,
) -> float:
    """Departure-aware urgency heuristic (Kempker-style order-statistic charging).

    target_mode:
      'fixed'  — target target_frac·e_max always
      'linear' — target interpolates linearly from e_trip (τ≈0) to target_frac·e_max (τ=τ_max)
      'power'  — same but concave: target = e_trip + (e_ceil−e_trip)·(τ/τ_max)^alpha

    target_frac — upper bound as fraction of e_max (default 1.0 = full battery).
    use_reserve — when True, charge u_max unconditionally below reserve_frac·e_max.
    alpha       — exponent for 'power' mode (ignored otherwise).
    """
    if chi > 0 and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0

    tau   = minutes_to_departure(t, params)
    e_ceil = target_frac * params.e_max

    if use_reserve and e < reserve_frac * params.e_max:
        return float(params.u_max)

    if target_mode == "fixed":
        e_target = e_ceil
    else:
        e_trip  = expected_trip_minutes(params) * params.mu * params.v * params.omega
        tau_max = max_minutes_to_departure(params)
        _alpha  = 1.0 if target_mode == "linear" else alpha
        e_target = _departure_urgency_target(tau, tau_max, e_trip, e_ceil, _alpha)

    deliverable = params.u_max * params.eta_c * params.omega * tau
    rho = min(1.0, max(0.0, e_target - e) / deliverable) if deliverable > 0 else 1.0

    probs    = price_bin_probs(t, params) if price_bin_probs_fn is None else price_bin_probs_fn(t)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K for j in range(params.K)])
    F_p      = float(probs[lam_grid <= lam].sum())

    if F_p <= rho:
        return float(params.u_max)
    if F_p <= rho + (1.0 / tau if tau > 0 else 0.0):
        return float(params.u_max / 2)
    return 0.0


def policy_registry(params, pbp_fn, *, pi, actions, e_grid,
                    low_threshold=None, high_threshold=None, soc_threshold=None,
                    du_target_mode="fixed", du_target_frac=1.0,
                    du_reserve_frac=0.25, du_use_reserve=True, du_alpha=0.5):
    """Ordered ``(name, policy_fn, kwargs)`` for every benchmark policy (POLICY_ORDER).

    Single source of truth for the policy set compared everywhere (sensitivity
    sweeps, baseline-model exports, the Policy-Rollout page). Thresholds default
    to the canonical values: low/high = params.price_night/price_evening,
    soc = params.e_max * 0.25.

    du_* kwargs control the Departure Urgency policy variant shown in all pages,
    mirroring the sliders on the Settings page.
    """
    low  = params.price_night   if low_threshold  is None else low_threshold
    high = params.price_evening if high_threshold is None else high_threshold
    soc  = params.e_max * 0.25  if soc_threshold  is None else soc_threshold
    du_kw = dict(
        price_bin_probs_fn=pbp_fn,
        target_mode=du_target_mode,
        target_frac=du_target_frac,
        reserve_frac=du_reserve_frac,
        use_reserve=du_use_reserve,
        alpha=du_alpha,
    )
    entries = [
        ("Backward Induction",    backward_induction_policy, dict(pi=pi, actions=actions, e_grid=e_grid)),
        ("Battery Level Urgency", dp_heuristic_policy,       dict(price_bin_probs_fn=pbp_fn)),
        ("Departure Urgency",     next_trip_policy,          du_kw),
    ]
    entries += [
        ("Price-Oriented",        price_oriented_policy,     dict(low_threshold=low, high_threshold=high)),
        ("Night Charging",        night_charging_policy,     {}),
        ("Always-Maximum",        maximal_charging_policy,   {}),
        ("Minimum Battery Level", minimum_soc_policy,        dict(soc_threshold=soc)),
        ("Always-Minimum",        always_minimum_policy,     {}),
    ]
    return entries
