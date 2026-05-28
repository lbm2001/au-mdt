import numpy as np
from typing import Callable

PARKED  = 0
DRIVING = 1


def backward_induction(
    params,
    transition_probs_fn: Callable[[int], tuple],
    consumption_fn: Callable[[int], float],
    price_bin_probs_fn: Callable[[int], np.ndarray],
    T: int = 2880,
    N_e: int = 100,
):
    """
    Solve the EV charging MDP via backward induction with price as a state variable.

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

    Returns
    -------
    V        : ndarray (T+1, 2, N_e, K)
    pi       : ndarray (T,   2, N_e, K)  — index into `actions`
    actions  : ndarray (4,)
    e_grid   : ndarray (N_e,)
    lam_grid : ndarray (K,)              — bin-centre prices (€/kWh)
    """
    K       = params.K
    e_grid  = np.linspace(params.e_min, params.e_max, N_e)
    actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    n_a     = len(actions)

    # Bin-centre prices (€/kWh), used for the immediate charging cost
    lam_grid = np.array([(k + 0.5) * params.lambda_max / K for k in range(K)])

    V  = np.zeros((T + 1, 2, N_e, K))
    pi = np.zeros((T,     2, N_e, K), dtype=int)

    # Precompute next-step bin probabilities for every t.
    # all_p_next[t] is the distribution of λ̂_{t+1}, used when computing V[t].
    all_p_next = np.array([price_bin_probs_fn(t + 1) for t in range(T)])  # (T, K)

    E = e_grid[:, np.newaxis]   # (N_e, 1)
    A = actions[np.newaxis, :]  # (1,  n_a)

    for t in range(T - 1, -1, -1):
        p_PD, p_DP = transition_probs_fn(t)
        P = np.array([[1 - p_PD, p_PD],
                      [p_DP,     1 - p_DP]])

        # Price-averaged continuation value: (2, N_e, K) @ (K,) -> (2, N_e)
        V_bar = V[t + 1] @ all_p_next[t]

        for chi in range(2):
            is_driving = (chi == DRIVING)

            # ── actual charging action u_a : (N_e, n_a) ────────────────────
            if is_driving:
                u_a = np.where(E > params.e_min, 0.0, A)
            else:
                u_a = np.broadcast_to(A, (N_e, n_a)).copy()

            # ── immediate reward: (N_e, n_a, K) ────────────────────────────
            # r[e, u, k] = -lam_grid[k] * omega * u_a[e, u]
            r = -(u_a[:, :, np.newaxis]
                  * params.omega
                  * lam_grid[np.newaxis, np.newaxis, :])

            if is_driving:
                # Penalty for driving with empty battery (price-independent)
                penalty = np.where(E > params.e_min, 0.0,
                                   params.omega * params.phi)  # (N_e, 1)
                r -= penalty[:, :, np.newaxis]

            # ── next battery level with linear interpolation ────────────────
            consumption = consumption_fn(chi)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - consumption,
                params.e_min, params.e_max,
            )
            e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            e_lo = np.floor(e_next_f).astype(int)
            e_hi = np.minimum(e_lo + 1, N_e - 1)
            w_hi = e_next_f - e_lo
            w_lo = 1.0 - w_hi

            # ── expected continuation over next mobility state: (N_e, n_a) ─
            # EV_next is independent of current price bin k
            EV_next = (P[chi, PARKED]  * (w_lo * V_bar[PARKED,  e_lo] + w_hi * V_bar[PARKED,  e_hi])
                     + P[chi, DRIVING] * (w_lo * V_bar[DRIVING, e_lo] + w_hi * V_bar[DRIVING, e_hi]))

            # ── Bellman backup: (N_e, n_a, K) ──────────────────────────────
            Q = r + params.beta * EV_next[:, :, np.newaxis]

            pi[t, chi] = np.argmax(Q, axis=1)  # (N_e, K)
            V[t, chi]  = np.max(Q,   axis=1)   # (N_e, K)

    return V, pi, actions, e_grid, lam_grid
