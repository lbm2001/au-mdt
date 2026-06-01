import numpy as np

from .model import DRIVING, PARKED, consumption, price_bin, price_bin_probs, transition_probs
from .params import BaselineParams


def actual_charge_rate(chi: int, e: float, desired_u: float, params: BaselineParams) -> float:
    """Apply the model's no-charging-while-driving rule to a desired charge rate."""
    if chi == DRIVING and e > params.e_min:
        return 0.0
    return float(np.clip(desired_u, 0.0, params.u_max))


def backward_induction_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
    *,
    pi: np.ndarray,
    actions: np.ndarray,
    e_grid: np.ndarray,
) -> float:
    """Look up the desired charge rate from the solved backward-induction policy."""
    e_idx = int(np.argmin(np.abs(e_grid - e)))
    lam_idx = price_bin(lam, params)
    a_idx = pi[t, chi, e_idx, lam_idx]
    return float(actions[a_idx])


def maximal_charging_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
) -> float:
    """Always request the maximum charge rate."""
    return float(params.u_max)


def price_oriented_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
    *,
    low_threshold: float,
    high_threshold: float,
) -> float:
    """Request more charging at lower prices using a two-threshold rule."""
    if lam <= low_threshold:
        return float(params.u_max)
    if lam <= high_threshold:
        return float(params.u_max / 2)
    return 0.0


def night_charging_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
) -> float:
    """Charge at u_max between 00:00–06:00, nothing otherwise."""
    return float(params.u_max) if t % 1440 < 360 else 0.0


def minimum_soc_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
    *,
    soc_threshold: float,
) -> float:
    """Charge at u_max whenever battery is below soc_threshold (kWh), else stop."""
    return float(params.u_max) if e < soc_threshold else 0.0


def dp_heuristic_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
) -> float:
    """SoC-urgency heuristic: charge at u_max when F_t(lam) ≤ 1 − e/e_max.

    The urgency threshold 1 − e/e_max maps battery state directly onto the price
    CDF: an empty battery accepts any price, a full battery charges only at the
    cheapest prices.
    """
    if chi == DRIVING and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0

    thresh = 1.0 - e / params.e_max

    probs    = price_bin_probs(t, params)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K
                         for j in range(params.K)])
    F_p = float(probs[lam_grid <= lam].sum())

    return float(params.u_max) if F_p <= thresh else 0.0


def expected_parking_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
) -> float:
    """Textbook three-band rule with rem = expected parked minutes per day at time t.

    rem is derived from the stationary parked fraction π_P(t) = p_DP / (p_PD + p_DP),
    giving rem = π_P(t) × 1440. During high-departure periods (large p_PD) rem shrinks
    and thresholds rise — the policy becomes more willing to charge at moderate prices.
    """
    if chi == DRIVING and e > params.e_min:
        return 0.0
    x = params.e_max - e
    if x <= 0:
        return 0.0

    energy_per_step = params.u_max * params.omega * params.eta_c
    k = int(x // energy_per_step)

    p_PD, p_DP = transition_probs(t, params)
    denom = p_PD + p_DP
    pi_P  = p_DP / denom if denom > 0 else 0.5
    rem   = max(int(pi_P * 1440), k + 1)

    probs    = price_bin_probs(t, params)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K
                         for j in range(params.K)])
    F_p = float(probs[lam_grid <= lam].sum())

    thresh_k  = k / rem
    thresh_k1 = (k + 1) / rem

    if F_p <= thresh_k:
        u = params.u_max
    elif F_p <= thresh_k1:
        u = (x - k * energy_per_step) / (params.omega * params.eta_c)
    else:
        u = 0.0
    return float(np.clip(u, 0.0, params.u_max))


def always_minimum_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
) -> float:
    """Always charge at u_min when parked."""
    return float(params.u_min)


def random_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
    *,
    rng: np.random.Generator,
) -> float:
    """Pick a random action uniformly from {0, u_min, u_max/2, u_max}."""
    return float(rng.choice([0.0, params.u_min, params.u_max / 2, params.u_max]))
