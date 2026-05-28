import numpy as np

from .model import DRIVING, PARKED, consumption, price_bin, transition_probs
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


def plan_perfect_foresight(
    scenario: dict,
    e0: float,
    chi0: int,
    params: BaselineParams,
) -> np.ndarray:
    """Pre-plan a charging schedule given full knowledge of prices and mobility.

    Greedily assigns charging to the cheapest parked minutes, using just enough
    energy to cover total driving consumption. Returns a u_plan array of length horizon.
    """
    horizon = len(scenario["lam_path"])
    lam_path = scenario["lam_path"]

    # Simulate mobility trajectory
    chi = int(chi0)
    chi_traj = np.zeros(horizon, dtype=int)
    for t in range(horizon):
        chi_traj[t] = chi
        p_PD, p_DP = transition_probs(t, params)
        draw = float(scenario["mobility_draws"][t])
        if chi == PARKED:
            chi = DRIVING if draw < p_PD else PARKED
        else:
            chi = PARKED if draw < p_DP else DRIVING

    # Total energy consumed while driving
    drive_steps = int((chi_traj == DRIVING).sum())
    total_consumption = drive_steps * consumption(DRIVING, params)

    # Net charging energy required beyond initial battery
    energy_needed = max(0.0, (total_consumption - (float(e0) - params.e_min)) / params.eta_c)

    # Sort parked minutes by price (cheapest first), assign u_max greedily
    parked_idx = np.where(chi_traj == PARKED)[0]
    charge_slots: set[int] = set()
    if len(parked_idx) > 0:
        order = parked_idx[np.argsort(lam_path[parked_idx])]
        accumulated = 0.0
        for t_c in order:
            if accumulated >= energy_needed:
                break
            charge_slots.add(int(t_c))
            accumulated += params.u_max * params.omega

    # Forward simulate to enforce battery bounds
    u_plan = np.zeros(horizon)
    e = float(e0)
    for t in range(horizon):
        if t in charge_slots:
            space = params.e_max - e
            u = min(params.u_max, space / (params.eta_c * params.omega)) if space > 1e-9 else 0.0
            u_plan[t] = u
        cons = consumption(chi_traj[t], params)
        e = float(np.clip(e + params.eta_c * params.omega * u_plan[t] - cons,
                          params.e_min, params.e_max))

    return u_plan


def perfect_foresight_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: BaselineParams,
    *,
    u_plan: np.ndarray,
) -> float:
    """Look up the pre-planned charge rate for time step t."""
    return float(u_plan[t])
