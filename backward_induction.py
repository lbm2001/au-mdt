import numpy as np
from typing import Callable

PARKED  = 0
DRIVING = 1


def backward_induction(
    params,
    mean_price_fn: Callable[[int], float],
    transition_probs_fn: Callable[[int], tuple],
    consumption_fn: Callable[[int], float],
    T: int = 1440,
    N_e: int = 100,
):
    """
    Solve the EV charging MDP via backward induction (Table 1 pseudocode).

    The price λ_t enters only the immediate reward and not the state transition,
    so the expectation over λ_t collapses to using the time-of-day mean λ̄_t.
    The battery transition is deterministic given u_a:
        e_{t+1} = clip(e_t + η_c · ω · u_a - consumption_fn(chi), e_min, e_max).

    Parameters
    ----------
    params              : MDPParams  (from baseline_mdp)
    mean_price_fn       : t -> λ̄_t  (expected electricity price at minute t)
    transition_probs_fn : t -> (p_PD, p_DP)  (driving-state transition probs)
    consumption_fn      : chi -> energy consumed per minute in state chi (kWh/min)
    T                   : time horizon in minutes (default: 1440 = one full day)
    N_e                 : number of battery-energy grid points

    Returns
    -------
    V       : ndarray (T+1, 2, N_e)  – optimal value function; V[T] = 0
    pi      : ndarray (T,   2, N_e)  – index into `actions` for optimal u at each state
    actions : ndarray (4,)            – feasible charge rates [kW]
    e_grid  : ndarray (N_e,)          – discretised energy levels [kWh]
    """
    e_grid  = np.linspace(params.e_min, params.e_max, N_e)
    actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    n_a     = len(actions)

    # Terminal value V_T = R_T = 0
    V  = np.zeros((T + 1, 2, N_e))
    pi = np.zeros((T,     2, N_e), dtype=int)

    # Broadcast-ready views for vectorised operations over the energy grid
    E = e_grid[:, np.newaxis]   # (N_e, 1)
    A = actions[np.newaxis, :]  # (1,  n_a)

    for t in range(T - 1, -1, -1):
        lam_bar    = mean_price_fn(t)
        p_PD, p_DP = transition_probs_fn(t)
        # P[chi, chi_next]: 2×2 transition matrix
        P = np.array([[1 - p_PD, p_PD    ],
                      [p_DP,     1 - p_DP]])

        for chi in range(2):
            # ── actual charging action u_a : shape (N_e, n_a) ──────────────
            # No charging while driving unless battery is at minimum
            if chi == DRIVING:
                u_a = np.where(E > params.e_min, 0.0, A)
            else:
                u_a = np.broadcast_to(A, (N_e, n_a)).copy()

            # ── expected immediate reward E_λ[R_t(s, u)] ───────────────────
            # lam_bar is in €/MWh; divide by 1000 to match energy units in kWh
            r = -(lam_bar / 1000 * params.omega * u_a)
            if chi == DRIVING:
                r = r - np.where(E <= params.e_min, params.omega * params.phi, 0.0)

            # ── deterministic next battery level, snapped to nearest grid ───
            consumption = consumption_fn(chi)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - consumption,
                params.e_min, params.e_max,
            )
            # Grid is uniform so index is exact (no search needed)
            e_next_idx = np.round(
                (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            ).astype(int)

            # ── expected continuation value over next driving state ──────────
            V_next = (P[chi, PARKED]  * V[t + 1, PARKED,  e_next_idx]
                    + P[chi, DRIVING] * V[t + 1, DRIVING, e_next_idx])

            # ── Bellman backup ───────────────────────────────────────────────
            Q = r + params.beta * V_next   # (N_e, n_a)

            pi[t, chi] = np.argmax(Q, axis=1)
            V[t,  chi] = np.max(Q,   axis=1)

    return V, pi, actions, e_grid
