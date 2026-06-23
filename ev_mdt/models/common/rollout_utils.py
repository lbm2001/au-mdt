"""Rollout utilities shared by all EV charging MDP models."""
from typing import Callable

import numpy as np

from ev_mdt.models.common.model_utils import consumption, mean_price
from ev_mdt.models.common.policies import actual_charge_rate

PARKED = 0


def generate_rollout_scenario(
    params,
    seed: int,
    horizon: int = 2880,
) -> dict[str, np.ndarray]:
    """Generate sampled prices and mobility draws shared across policy rollouts.

    Always includes phase_draws; models that don't need them simply ignore the key.
    """
    rng = np.random.default_rng(int(seed))
    lam_path       = np.zeros(horizon)
    mobility_draws = np.zeros(horizon)
    phase_draws    = np.zeros(horizon)
    for t in range(horizon):
        lam_path[t]       = float(np.maximum(0.0, rng.normal(mean_price(t, params), params.sigma_lambda)))
        mobility_draws[t] = rng.random()
        phase_draws[t]    = rng.random()
    return {"lam_path": lam_path, "mobility_draws": mobility_draws, "phase_draws": phase_draws}


def simulate_policy_rollout(
    policy_fn,
    scenario: dict[str, np.ndarray],
    e0: float,
    chi0: int,
    params,
    next_state_fn: Callable,
    **policy_kwargs,
) -> dict[str, np.ndarray | float]:
    """Replay a shared scenario under one policy.

    next_state_fn(chi, scenario, t, params) -> int
        Model-specific mobility transition; receives the full scenario dict so it
        can access both mobility_draws and phase_draws.
    """
    horizon = len(scenario["lam_path"])
    e   = float(e0)
    chi = int(chi0)

    e_traj    = np.zeros(horizon)
    chi_traj  = np.zeros(horizon, dtype=int)
    u_traj    = np.zeros(horizon)
    lam_traj  = np.asarray(scenario["lam_path"], dtype=float)
    cost_traj = np.zeros(horizon)

    for t in range(horizon):
        e_traj[t]   = e
        chi_traj[t] = chi
        lam = float(lam_traj[t])

        desired_u = policy_fn(t=t, chi=chi, e=e, lam=lam, params=params, **policy_kwargs)
        u_a = actual_charge_rate(chi, e, desired_u, params)
        u_traj[t] = u_a

        cost = lam * params.omega * u_a
        if chi > 0 and e <= params.e_min:
            cost += params.omega * params.phi
        cost_traj[t] = cost

        cons = consumption(chi, params)
        e = float(np.clip(e + params.eta_c * params.omega * u_a - cons, params.e_min, params.e_max))

        chi = next_state_fn(chi, scenario, t, params)

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
    params,
) -> dict[str, float]:
    chi_traj  = rollout["chi_traj"]
    e_traj    = rollout["e_traj"]
    u_traj    = rollout["u_traj"]
    cost_traj = rollout["cost_traj"]
    parked    = chi_traj == PARKED
    penalty   = (chi_traj > 0) & (e_traj <= params.e_min)
    return {
        "Total cost (€)":                    float(cost_traj.sum()),
        "Energy charged (kWh)":              float((u_traj * params.omega).sum()),
        "Penalty minutes":                   int(penalty.sum()),
        "Final battery (kWh)":               float(rollout["final_e"]),
        "Mean charge rate while parked (kW)": float(u_traj[parked].mean()) if parked.any() else 0.0,
    }
