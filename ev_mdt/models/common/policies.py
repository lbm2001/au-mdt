"""Shared charging policies — work with any params that has the standard fields."""
import numpy as np

from ev_mdt.models.common.model_utils import price_bin, price_bin_probs


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


def policy_registry(params, pbp_fn, *, pi, actions, e_grid,
                    low_threshold=None, high_threshold=None, soc_threshold=None):
    """Ordered ``(name, policy_fn, kwargs)`` for every benchmark policy (POLICY_ORDER).

    Single source of truth for the policy set compared everywhere (sensitivity
    sweeps, baseline-model exports, the Policy-Rollout page). Thresholds default
    to the canonical values: low/high = params.price_night/price_evening,
    soc = params.e_max * 0.25.
    """
    low  = params.price_night   if low_threshold  is None else low_threshold
    high = params.price_evening if high_threshold is None else high_threshold
    soc  = params.e_max * 0.25  if soc_threshold  is None else soc_threshold
    return [
        ("Backward Induction",    backward_induction_policy, dict(pi=pi, actions=actions, e_grid=e_grid)),
        ("DP-Heuristic",          dp_heuristic_policy,       dict(price_bin_probs_fn=pbp_fn)),
        ("Price-Oriented",        price_oriented_policy,     dict(low_threshold=low, high_threshold=high)),
        ("Night Charging",        night_charging_policy,     {}),
        ("Always-Maximum",        maximal_charging_policy,   {}),
        ("Minimum Battery Level", minimum_soc_policy,        dict(soc_threshold=soc)),
        ("Always-Minimum",        always_minimum_policy,     {}),
    ]
