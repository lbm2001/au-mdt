"""Shared charging policies — work with any params that has the standard fields."""
import numpy as np

from ev_mdt.models.common.model_utils import (
    minutes_to_departure, price_bin, price_bin_probs,
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


def next_trip_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, price_bin_probs_fn=None, reserve_frac=0.25, target_frac=0.60, kappa=float("inf"),
) -> float:
    """Departure-aware urgency heuristic (Kempker-style order-statistic charging).

    Two regimes, reflecting that the stranding penalty (ω·φ ≈ €16.7/min) dwarfs any
    energy price, so safety must dominate price-picking:

    1. Mandatory reserve. Below e_reserve = reserve_frac·e_max, charge u_max
       regardless of price — the buffer the cost asymmetry demands.
    2. Opportunistic top-up. Between e_reserve and e_target = target_frac·e_max,
       charge when the price is cheap *relative to urgency*, with urgency rising as
       departure nears.  The decision compares the price percentile F_t(λ) to the
       urgency ratio ρ_t = (e_target − e) / E_deliver, E_deliver = u_max·η_c·ω·τ(t):

           bang-bang (default, κ=∞):  u_t = u_max if F_t(λ) ≤ ρ_t else 0
           soft (finite κ):           u_t = u_max · σ( κ · (ρ_t − F_t(λ)) )

       The bang-bang rule is optimal under the *linear* charging cost (Kempker Thm 14.1:
       the optimum fills the cheapest slots up to the requirement) and is what the
       benchmark uses by default.  The logistic σ is its κ<∞ generalization, kept for
       robustness to noisy τ/F_t and for charge-rate-dependent (convex) costs such as
       V2G/degradation, where interior rates become genuinely optimal; it costs a little
       here because it smears energy into pricier minutes.  Either way the map keys on
       the price-vs-urgency *gap*, not ρ alone, so a relaxed state still charges full in
       a genuinely cheap hour rather than dribbling.

    Driving and full-battery guards mirror the other benchmarks; downstream clipping
    to [0, u_max] still applies.
    """
    if chi > 0 and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0

    # 1. Mandatory safety reserve — price-independent.
    e_reserve = reserve_frac * params.e_max
    if e < e_reserve:
        return float(params.u_max)

    # 2. Opportunistic region [e_reserve, e_target]: logistic price-vs-urgency map.
    e_target = max(e_reserve, target_frac * params.e_max)
    slots = minutes_to_departure(t, params)                       # τ(t), exact forward expectation
    deliverable = params.u_max * params.eta_c * params.omega * slots
    rho = max(0.0, e_target - e) / deliverable if deliverable > 0 else np.inf

    # F_t(λ): price percentile from the precomputed bin CDF (as in dp_heuristic_policy)
    probs    = price_bin_probs(t, params) if price_bin_probs_fn is None else price_bin_probs_fn(t)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K for j in range(params.K)])
    F_p      = float(probs[lam_grid <= lam].sum())

    gap = rho - F_p
    if not np.isfinite(kappa):                                     # bang-bang (κ→∞ limit)
        # Charge iff price percentile ≤ urgency (inclusive, as in dp_heuristic): at the
        # cheapest bin F_p=0, so free/negative prices (clipped to 0, ~15% of samples)
        # charge at full rate even when already at target.
        return float(params.u_max) if gap >= 0.0 else 0.0
    sigma = 1.0 / (1.0 + np.exp(-kappa * np.clip(gap, -50.0, 50.0)))
    return float(params.u_max * sigma)


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
        ("Next-Trip Urgency",     next_trip_policy,          dict(price_bin_probs_fn=pbp_fn)),
        ("Price-Oriented",        price_oriented_policy,     dict(low_threshold=low, high_threshold=high)),
        ("Night Charging",        night_charging_policy,     {}),
        ("Always-Maximum",        maximal_charging_policy,   {}),
        ("Minimum Battery Level", minimum_soc_policy,        dict(soc_threshold=soc)),
        ("Always-Minimum",        always_minimum_policy,     {}),
    ]
