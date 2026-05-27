"""
Standalone script: grid sensitivity of the backward-induction policy.

Runs backward induction with and without linear interpolation for several
values of N_e, then evaluates each solved policy on the same set of rollout
scenarios. Prints a table suitable for copy-pasting into the LaTeX template.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.baseline import (
    BaselineParams, transition_probs, consumption, price_bin_probs,
)
from models.baseline.policies import backward_induction_policy
from models.baseline.rollout import generate_rollout_scenario, simulate_policy_rollout

# ── Config ────────────────────────────────────────────────────────────────────

N_E_VALUES  = [25, 50, 100, 200, 500, 1000, 10_000]
N_SCENARIOS = 1000
SEED        = 0

params = BaselineParams()

# ── Solvers ───────────────────────────────────────────────────────────────────

def _solve(params, N_e: int, interpolate: bool):
    K        = params.K
    e_grid   = np.linspace(params.e_min, params.e_max, N_e)
    actions  = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    n_a      = len(actions)
    lam_grid = np.array([(k + 0.5) * params.lambda_max / K for k in range(K)])
    T        = 1440

    V  = np.zeros((T + 1, 2, N_e, K))
    pi = np.zeros((T,     2, N_e, K), dtype=int)

    all_p_next = np.array([price_bin_probs(t + 1, params) for t in range(T)])

    E = e_grid[:, np.newaxis]
    A = actions[np.newaxis, :]

    PARKED, DRIVING = 0, 1

    for t in range(T - 1, -1, -1):
        p_PD, p_DP = transition_probs(t, params)
        P = np.array([[1 - p_PD, p_PD],
                      [p_DP,     1 - p_DP]])

        V_bar = V[t + 1] @ all_p_next[t]   # (2, N_e)

        for chi in range(2):
            is_driving = (chi == DRIVING)

            if is_driving:
                u_a = np.where(E > params.e_min, 0.0, A)
            else:
                u_a = np.broadcast_to(A, (N_e, n_a)).copy()

            r = -(u_a[:, :, np.newaxis]
                  * params.omega
                  * lam_grid[np.newaxis, np.newaxis, :])

            if is_driving:
                penalty = np.where(E > params.e_min, 0.0, params.omega * params.phi)
                r -= penalty[:, :, np.newaxis]

            cons   = consumption(chi, params)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - cons,
                params.e_min, params.e_max,
            )

            if interpolate:
                e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
                e_lo = np.floor(e_next_f).astype(int)
                e_hi = np.minimum(e_lo + 1, N_e - 1)
                w_hi = e_next_f - e_lo
                w_lo = 1.0 - w_hi
                EV_next = (P[chi, PARKED]  * (w_lo * V_bar[PARKED,  e_lo] + w_hi * V_bar[PARKED,  e_hi])
                         + P[chi, DRIVING] * (w_lo * V_bar[DRIVING, e_lo] + w_hi * V_bar[DRIVING, e_hi]))
            else:
                e_idx = np.clip(
                    np.round((e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)).astype(int),
                    0, N_e - 1,
                )
                EV_next = (P[chi, PARKED]  * V_bar[PARKED,  e_idx]
                         + P[chi, DRIVING] * V_bar[DRIVING, e_idx])

            Q = r + params.beta * EV_next[:, :, np.newaxis]

            pi[t, chi] = np.argmax(Q, axis=1)
            V[t, chi]  = np.max(Q,   axis=1)

    return pi, actions, e_grid


def _evaluate(pi, actions, e_grid, scenarios, desc=""):
    costs, penalties = [], []
    for scenario in tqdm(scenarios, desc=desc, leave=False):
        rollout = simulate_policy_rollout(
            backward_induction_policy,
            scenario,
            e0=params.e_max / 2,
            chi0=0,
            params=params,
            pi=pi,
            actions=actions,
            e_grid=e_grid,
        )
        costs.append(rollout["cost_traj"].sum())
        penalties.append(int(((rollout["chi_traj"] == 1) & (rollout["e_traj"] <= params.e_min)).sum()))
    return float(np.mean(costs)), float(np.mean(penalties))


# ── Run ───────────────────────────────────────────────────────────────────────

rng       = np.random.default_rng(SEED)
scenarios = [generate_rollout_scenario(params, int(rng.integers(0, 100_000)))
             for _ in range(N_SCENARIOS)]

n_runs = len(N_E_VALUES) * 2
print(f"Grid values     : {N_E_VALUES}")
print(f"Scenarios       : {N_SCENARIOS}")
print(f"Total rollouts  : {n_runs * N_SCENARIOS:,}  ({n_runs} solves × {N_SCENARIOS} scenarios)")
print()

t_start = time.perf_counter()
results = []
for N_e in tqdm(N_E_VALUES, desc="Grid sizes"):
    pi_nn, act_nn, eg_nn = _solve(params, N_e, interpolate=False)
    cost_nn, pen_nn = _evaluate(pi_nn, act_nn, eg_nn, scenarios, desc=f"N_e={N_e} no-interp")

    pi_li, act_li, eg_li = _solve(params, N_e, interpolate=True)
    cost_li, pen_li = _evaluate(pi_li, act_li, eg_li, scenarios, desc=f"N_e={N_e} lin-interp")

    results.append((N_e, cost_nn, pen_nn, cost_li, pen_li))

ref_nn = results[-1][1]
ref_li = results[-1][3]

hdr = f"{'N_e':>6}  {'No interp':>14}  {'pen.min':>8}  {'eps_nn':>8}  {'Lin. interp':>14}  {'pen.min':>8}  {'eps_li':>8}"
print(hdr)
print("-" * len(hdr))

rows = []
for N_e, cost_nn, pen_nn, cost_li, pen_li in results:
    eps_nn = (cost_nn - ref_nn) / ref_nn * 100
    eps_li = (cost_li - ref_li) / ref_li * 100
    print(f"  {N_e:>6}  {cost_nn:>14.4f}  {pen_nn:>8.1f}  {eps_nn:>7.1f}%  {cost_li:>14.4f}  {pen_li:>8.1f}  {eps_li:>7.1f}%")
    rows.append(dict(N_e=N_e, cost_nn=cost_nn, pen_nn=pen_nn, eps_nn=eps_nn,
                     cost_li=cost_li, pen_li=pen_li, eps_li=eps_li))

elapsed = time.perf_counter() - t_start
print()
print(f"Total time: {elapsed:.1f}s")

out = Path(__file__).parent / "grid_sensitivity.parquet"
pd.DataFrame(rows).to_parquet(out, index=False)
print(f"Saved: {out}")
