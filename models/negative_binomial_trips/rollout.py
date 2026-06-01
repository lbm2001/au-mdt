import numpy as np

from .model import PARKED, consumption, is_driving, mean_price, p_pd, transition_matrix
from .params import NegBinParams


def actual_charge_rate(chi: int, e: float, desired_u: float, params: NegBinParams) -> float:
    if is_driving(chi) and e > params.e_min:
        return 0.0
    return float(np.clip(desired_u, 0.0, params.u_max))


def generate_rollout_scenario(
    params: NegBinParams,
    seed: int,
    horizon: int = 2880,
) -> dict[str, np.ndarray]:
    """Generate sampled prices and mobility draws shared across policy rollouts."""
    rng = np.random.default_rng(int(seed))
    lam_path       = np.zeros(horizon)
    mobility_draws = np.zeros(horizon)
    for t in range(horizon):
        lam_path[t]       = float(np.maximum(0.0, rng.normal(mean_price(t, params), params.sigma_lambda)))
        mobility_draws[t] = rng.random()
    return {"lam_path": lam_path, "mobility_draws": mobility_draws}


def _next_chi(chi: int, draw: float, t: int, params: NegBinParams) -> int:
    """Advance the mobility state by one step."""
    if chi == PARKED:
        return 1 if draw < p_pd(t, params) else PARKED
    elif chi < params.k:                        # D_1 .. D_{k-1}
        return chi + 1 if draw < params.q else chi
    else:                                        # D_k
        return PARKED if draw < params.q else chi


def simulate_policy_rollout(
    policy_fn,
    scenario: dict[str, np.ndarray],
    e0: float,
    chi0: int,
    params: NegBinParams,
    **policy_kwargs,
) -> dict[str, np.ndarray | float]:
    """Replay a shared scenario under one policy."""
    horizon = len(scenario["lam_path"])
    e   = float(e0)
    chi = int(chi0)

    e_traj   = np.zeros(horizon)
    chi_traj = np.zeros(horizon, dtype=int)
    u_traj   = np.zeros(horizon)
    lam_traj = np.asarray(scenario["lam_path"], dtype=float)
    cost_traj = np.zeros(horizon)

    for t in range(horizon):
        e_traj[t]   = e
        chi_traj[t] = chi
        lam = float(lam_traj[t])

        desired_u = policy_fn(t=t, chi=chi, e=e, lam=lam, params=params, **policy_kwargs)
        u_a = actual_charge_rate(chi, e, desired_u, params)
        u_traj[t] = u_a

        cost = lam * params.omega * u_a
        if is_driving(chi) and e <= params.e_min:
            cost += params.omega * params.phi
        cost_traj[t] = cost

        cons = consumption(chi, params)
        e = float(np.clip(e + params.eta_c * params.omega * u_a - cons, params.e_min, params.e_max))

        chi = _next_chi(chi, float(scenario["mobility_draws"][t]), t, params)

    return {
        "e_traj":    e_traj,
        "chi_traj":  chi_traj,
        "u_traj":    u_traj,
        "lam_traj":  lam_traj,
        "cost_traj": cost_traj,
        "final_e":   e,
    }


def rollout_metrics(
    rollout: dict[str, np.ndarray | float],
    params: NegBinParams,
) -> dict[str, float]:
    chi_traj  = rollout["chi_traj"]
    e_traj    = rollout["e_traj"]
    u_traj    = rollout["u_traj"]
    cost_traj = rollout["cost_traj"]
    parked    = chi_traj == PARKED
    penalty   = np.array([is_driving(int(c)) for c in chi_traj]) & (e_traj <= params.e_min)
    return {
        "Total cost (€)":                    float(cost_traj.sum()),
        "Energy charged (kWh)":              float((u_traj * params.omega).sum()),
        "Penalty minutes":                   int(penalty.sum()),
        "Final battery (kWh)":               float(rollout["final_e"]),
        "Mean charge rate while parked (kW)": float(u_traj[parked].mean()) if parked.any() else 0.0,
    }
