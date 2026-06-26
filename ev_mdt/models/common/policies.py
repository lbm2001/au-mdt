"""Shared charging policies — work with any params that has the standard fields."""
import numpy as np

from functools import lru_cache

from ev_mdt.models.common.model_utils import (
    expected_trip_minutes, mean_minutes_to_departure,
    minutes_to_departure, price_bin, price_bin_probs,
)

# Empirically-swept baseline-optimal DU ceiling (kWh). Set from `target-sweep` at default
# Baseline params; carried as a module-level constant so the policy is param-free.
E_CEIL_BASE: float = 29.5

# Best ceiling-scaling exponent γ per mobility model, from `gamma-sweep` (N=1000).
# At Baseline the demand ratio is 1, so γ is irrelevant there (e_ceil ≡ E_CEIL_BASE);
# γ only shapes the ceiling for the NegBin models via (e_daily / e_daily_ref) ** γ.
DU_GAMMA_BASELINE:       float = 0.1
DU_GAMMA_NEGBIN_FIXED:   float = 0.3 
DU_GAMMA_NEGBIN_POISSON: float = 0.5


def du_gamma_for_params(params) -> float:
    """Resolve the per-model DU γ from the params type (NegBin fixed vs Poisson vs Baseline)."""
    if not hasattr(params, "q"):
        return DU_GAMMA_BASELINE
    return DU_GAMMA_NEGBIN_POISSON if params.lambda_k is not None else DU_GAMMA_NEGBIN_FIXED


def _du_e_daily(params) -> float:
    """Expected daily energy demand (kWh) for the given mobility params.

    E[N_trips/day] = 1440 / (mean_τ + E[T_trip])
    e_daily        = E[N_trips] × E[T_trip] × μ × v × ω
    """
    e_trip_min = expected_trip_minutes(params)
    mean_tau   = mean_minutes_to_departure(params)
    n_trips    = 1440.0 / (mean_tau + e_trip_min) if (mean_tau + e_trip_min) > 0 else 0.0
    # μ·v·ω cancels in e_daily/e_daily_ref; kept here so the return value is interpretable as kWh.
    return n_trips * e_trip_min * params.mu * params.v * params.omega


@lru_cache(maxsize=1)
def _e_daily_ref() -> float:
    """e_daily for the canonical Baseline params — computed once and cached."""
    from ev_mdt.params import BaselineParams
    return _du_e_daily(BaselineParams())


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
    *,
    price_bin_probs_fn=None,
    gamma: float = 0.5,
    use_reserve: bool = True,
    _ceil_override: float | None = None,
) -> float:
    """Departure-aware urgency heuristic.

    e_trip  = E[T_trip] · μ · v · ω  — expected energy the next trip consumes (kWh).
    tau     = E[minutes to next departure]  — from backward-recurrence hazard.

    Demand-scaled ceiling (anchored to the empirically-swept baseline optimum):
        e_daily = E[N_trips] × e_trip    (active params)
        e_ceil  = min(e_max, E_CEIL_BASE × (e_daily / e_daily_ref) ** gamma)

    Reserve: if use_reserve and e < e_trip, charge at u_max regardless of price.

    Urgency ratio: ρ = (e_ceil − e) / deliverable, where
        deliverable = u_max · η_c · ω · τ  (max kWh chargeable before departure).
    ρ rises naturally as τ shrinks — no separate target interpolation needed.

    Charge u_max if F_t(λ) ≤ ρ, u_max/2 in marginal band (±1/τ), else 0.
    """
    if chi > 0 and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0

    e_trip = expected_trip_minutes(params) * params.mu * params.v * params.omega

    if use_reserve and e < e_trip:
        return float(params.u_max)

    if _ceil_override is not None:
        e_ceil = min(params.e_max, float(_ceil_override))
    else:
        e_daily = _du_e_daily(params)
        ref     = _e_daily_ref()
        ratio   = e_daily / ref if ref > 0 else 1.0
        e_ceil  = min(params.e_max, E_CEIL_BASE * ratio ** gamma)

    tau         = minutes_to_departure(t, params)
    deliverable = params.u_max * params.eta_c * params.omega * tau
    rho = min(1.0, max(0.0, e_ceil - e) / deliverable) if deliverable > 0 else 1.0

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
                    du_gamma=None, du_use_reserve=True):
    """Ordered ``(name, policy_fn, kwargs)`` for every benchmark policy (POLICY_ORDER).

    Single source of truth for the policy set compared everywhere (sensitivity
    sweeps, baseline-model exports, the Policy-Rollout page). Thresholds default
    to the canonical values: low/high = params.price_night/price_evening,
    soc = params.e_max * 0.25.

    du_* kwargs control the Departure Urgency policy shown in all pages.
    ``du_gamma=None`` auto-selects the per-model γ via ``du_gamma_for_params``.
    """
    low  = params.price_night   if low_threshold  is None else low_threshold
    high = params.price_evening if high_threshold is None else high_threshold
    soc  = params.e_max * 0.25  if soc_threshold  is None else soc_threshold
    du_kw = dict(
        price_bin_probs_fn=pbp_fn,
        gamma=du_gamma_for_params(params) if du_gamma is None else du_gamma,
        use_reserve=du_use_reserve,
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
