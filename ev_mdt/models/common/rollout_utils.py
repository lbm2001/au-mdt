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
    sampler=None,
    season: str = "winter",
    is_weekend: bool = False,
) -> dict[str, np.ndarray]:
    """Generate sampled prices and mobility draws shared across policy rollouts.

    Prices are drawn from the Gaussian-parametric marginal (``sampler=None``) or,
    when a fitted price ``sampler`` is supplied, from that sampler at the given
    ``season``/``is_weekend`` context — so a policy is evaluated in the same price
    world it was solved in.

    Always includes phase_draws; models that don't need them simply ignore the key.
    """
    rng = np.random.default_rng(int(seed))
    lam_path       = np.zeros(horizon)
    mobility_draws = np.zeros(horizon)
    phase_draws    = np.zeros(horizon)
    dow = 5 if is_weekend else 0
    for t in range(horizon):
        if sampler is None:
            lam_path[t] = float(max(0.0, rng.normal(mean_price(t, params), params.sigma_lambda)))
        else:
            lam_path[t] = float(max(0.0, sampler.sample(dow, (t // 60) % 24, season, rng=rng)))
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

    e_traj        = np.zeros(horizon)
    chi_traj      = np.zeros(horizon, dtype=int)
    u_traj        = np.zeros(horizon)
    lam_traj      = np.asarray(scenario["lam_path"], dtype=float)
    cost_traj     = np.zeros(horizon)
    charge_cost_traj  = np.zeros(horizon)
    penalty_cost_traj = np.zeros(horizon)

    for t in range(horizon):
        e_traj[t]   = e
        chi_traj[t] = chi
        lam = float(lam_traj[t])

        desired_u = policy_fn(t=t, chi=chi, e=e, lam=lam, params=params, **policy_kwargs)
        u_a = actual_charge_rate(chi, e, desired_u, params)
        u_traj[t] = u_a

        charge_cost = lam * params.omega * u_a
        pen_cost = 0.0
        if chi > 0 and e <= params.e_min:
            pen_cost = params.omega * params.phi
        charge_cost_traj[t] = charge_cost
        penalty_cost_traj[t] = pen_cost
        cost_traj[t] = charge_cost + pen_cost

        cons = consumption(chi, params)
        e = float(np.clip(e + params.eta_c * params.omega * u_a - cons, params.e_min, params.e_max))

        chi = next_state_fn(chi, scenario, t, params)

    return {
        "e_traj":    e_traj,
        "chi_traj":  chi_traj,
        "u_traj":    u_traj,
        "lam_traj":  lam_traj,
        "cost_traj":         cost_traj,
        "charge_cost_traj":  charge_cost_traj,
        "penalty_cost_traj": penalty_cost_traj,
        "final_e":   e,
    }


def run_policies(registry, scenarios, e0s, chi0, params, rollout_fn, progress=None):
    """Run every policy in ``registry`` over each (scenario, e0); return raw rollouts.

    registry   : list of (name, policy_fn, kwargs), e.g. from policies.policy_registry.
    e0s        : per-scenario initial battery, aligned with ``scenarios``.
    chi0       : initial mobility state (shared across policies within a scenario).
    rollout_fn : model-specific simulate_policy_rollout.
    progress   : optional no-arg callable invoked once per scenario (progress bar).

    Returns {policy_name: [raw rollout dict, ...]} in registry order.
    """
    out: dict[str, list] = {name: [] for name, _, _ in registry}
    for sc, e0 in zip(scenarios, e0s):
        for name, fn, kw in registry:
            out[name].append(rollout_fn(fn, sc, float(e0), int(chi0), params, **kw))
        if progress is not None:
            progress()
    return out


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
        "Total cost (€)":                     float(cost_traj.sum()),
        "Charging cost (€)":                  float(rollout["charge_cost_traj"].sum()),
        "Penalty cost (€)":                   float(rollout["penalty_cost_traj"].sum()),
        "Energy charged (kWh)":               float((u_traj * params.omega).sum()),
        "Penalty minutes":                    int(penalty.sum()),
        "Final battery (kWh)":                float(rollout["final_e"]),
        "Mean charge rate while parked (kW)": float(u_traj[parked].mean()) if parked.any() else 0.0,
    }
