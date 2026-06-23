import numpy as np
from typing import Callable

from ev_mdt.params import N_e as _N_E, T_hours as _T_HOURS

PARKED  = 0
DRIVING = 1


def backward_induction(
    params,
    transition_probs_fn: Callable[[int], tuple],
    consumption_fn: Callable[[int], float],
    price_bin_probs_fn: Callable[[int], np.ndarray],
    T: int = _T_HOURS * 60,
    N_e: int = _N_E,
    N_a: int | None = None,
):
    """Solve the EV charging MDP via backward induction with price as a state variable.

    The discretised price bin λ̂_t is part of the state.  Because the price at t+1
    is drawn i.i.d. from the time-dependent distribution (independent of the current
    price), the expected continuation value factors as

        E[V_{t+1}(χ', e', λ̂')] = Σ_k  p_{t+1}(k) · V_{t+1}(χ', e', k)

    so the K-dimensional sum is computed once per (t, χ) rather than once per
    (t, χ, e, λ̂).

    Parameters
    ----------
    params               : BaselineParams — must include K, lambda_max, and the
                           standard battery/cost fields.
    transition_probs_fn  : t -> (p_PD, p_DP)
    consumption_fn       : chi -> kWh/min
    price_bin_probs_fn   : t -> (K,) ndarray — bin probabilities for the price at t
    T                    : time horizon in minutes
    N_e                  : battery grid points
    N_a                  : number of non-zero charge rates; None (default) uses the original
                           [0, u_min, u_max/2, u_max]; an integer N_a uses
                           [0] + linspace(u_min, u_max, N_a)

    Returns
    -------
    V        : ndarray (T+1, 2, N_e, K)
    pi       : ndarray (T,   2, N_e, K)  — index into `actions`
    actions  : ndarray (N_a+1,)
    e_grid   : ndarray (N_e,)
    lam_grid : ndarray (K,)              — bin-centre prices (€/kWh)
    """
    K       = params.K
    e_grid  = np.linspace(params.e_min, params.e_max, N_e)
    if N_a is None:
        actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    else:
        actions = np.concatenate([[0.0], np.linspace(params.u_min, params.u_max, N_a)])
    n_a     = len(actions)

    lam_grid = np.array([(k + 0.5) * params.lambda_max / K for k in range(K)])

    V  = np.zeros((T + 1, 2, N_e, K))
    pi = np.zeros((T,     2, N_e, K), dtype=int)

    all_p_next = np.array([price_bin_probs_fn(t + 1) for t in range(T)])  # (T, K)

    E = e_grid[:, np.newaxis]   # (N_e, 1)
    A = actions[np.newaxis, :]  # (1,  n_a)

    for t in range(T - 1, -1, -1):
        p_PD, p_DP = transition_probs_fn(t)
        P = np.array([[1 - p_PD, p_PD],
                      [p_DP,     1 - p_DP]])

        V_bar = V[t + 1] @ all_p_next[t]  # (2, N_e, K) @ (K,) -> (2, N_e)

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
                penalty = np.where(E > params.e_min, 0.0,
                                   params.omega * params.phi)
                r -= penalty[:, :, np.newaxis]

            cons = consumption_fn(chi)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - cons,
                params.e_min, params.e_max,
            )
            e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            e_lo = np.floor(e_next_f).astype(int)
            e_hi = np.minimum(e_lo + 1, N_e - 1)
            w_hi = e_next_f - e_lo
            w_lo = 1.0 - w_hi

            EV_next = (P[chi, PARKED]  * (w_lo * V_bar[PARKED,  e_lo] + w_hi * V_bar[PARKED,  e_hi])
                     + P[chi, DRIVING] * (w_lo * V_bar[DRIVING, e_lo] + w_hi * V_bar[DRIVING, e_hi]))

            Q = r + params.beta * EV_next[:, :, np.newaxis]

            pi[t, chi] = np.argmax(Q, axis=1)
            V[t, chi]  = np.max(Q,   axis=1)

    return V, pi, actions, e_grid, lam_grid
