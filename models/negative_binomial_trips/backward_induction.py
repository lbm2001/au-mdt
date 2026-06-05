import numpy as np
from typing import Callable

from models.model_utils import consumption, price_bin_probs
from models.solver_config import N_e as _N_E, T_hours as _T_HOURS
from .model import is_driving, transition_matrix
from .params import NegBinParams


def backward_induction(
    params: NegBinParams,
    price_bin_probs_fn: Callable[[int], np.ndarray] | None = None,
    T: int = _T_HOURS * 60,
    N_e: int = _N_E,
    N_a: int | None = None,
):
    """Solve the NegBin EV charging MDP via backward induction.

    State space: χ ∈ {0=P, 1=D_1, …, k=D_k}.  The (k+1)×(k+1) transition
    matrix replaces the baseline 2×2 matrix; everything else is identical to
    the baseline backward induction.

    Returns
    -------
    V        : ndarray (T+1, k+1, N_e, K)
    pi       : ndarray (T,   k+1, N_e, K)  — index into `actions`
    actions  : ndarray (N_a+1,)
    e_grid   : ndarray (N_e,)
    lam_grid : ndarray (K,)
    """
    K      = params.K
    n_chi  = params.k + 1
    e_grid = np.linspace(params.e_min, params.e_max, N_e)
    if N_a is None:
        actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    else:
        actions = np.concatenate([[0.0], np.linspace(params.u_min, params.u_max, N_a)])
    n_a    = len(actions)

    lam_grid = np.array([(j + 0.5) * params.lambda_max / K for j in range(K)])

    V  = np.zeros((T + 1, n_chi, N_e, K))
    pi = np.zeros((T,     n_chi, N_e, K), dtype=int)

    _pbp = price_bin_probs_fn if price_bin_probs_fn is not None else (lambda t: price_bin_probs(t, params))
    all_p_next = np.array([_pbp(t + 1) for t in range(T)])  # (T, K)

    E = e_grid[:, np.newaxis]    # (N_e, 1)
    A = actions[np.newaxis, :]   # (1, n_a)

    for t in range(T - 1, -1, -1):
        P = transition_matrix(t, params)   # (n_chi, n_chi)

        # Price-averaged continuation value: (n_chi, N_e)
        V_bar = V[t + 1] @ all_p_next[t]   # (n_chi, N_e, K) @ (K,) -> (n_chi, N_e)

        for chi in range(n_chi):
            driving = is_driving(chi)

            # ── actual charging actions: (N_e, n_a) ────────────────────────
            if driving:
                u_a = np.where(E > params.e_min, 0.0, A)
            else:
                u_a = np.broadcast_to(A, (N_e, n_a)).copy()

            # ── immediate reward: (N_e, n_a, K) ────────────────────────────
            r = -(u_a[:, :, np.newaxis]
                  * params.omega
                  * lam_grid[np.newaxis, np.newaxis, :])

            if driving:
                penalty = np.where(E > params.e_min, 0.0, params.omega * params.phi)
                r -= penalty[:, :, np.newaxis]

            # ── next battery level: (N_e, n_a) ─────────────────────────────
            cons   = consumption(chi, params)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - cons,
                params.e_min, params.e_max,
            )
            e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            e_lo  = np.floor(e_next_f).astype(int)
            e_hi  = np.minimum(e_lo + 1, N_e - 1)
            w_hi  = e_next_f - e_lo
            w_lo  = 1.0 - w_hi

            # ── expected continuation over next mobility state: (N_e, n_a) ─
            EV_next = np.zeros((N_e, n_a))
            for chi_next in range(n_chi):
                p = P[chi, chi_next]
                if p == 0.0:
                    continue
                EV_next += p * (w_lo * V_bar[chi_next, e_lo]
                              + w_hi * V_bar[chi_next, e_hi])

            # ── Bellman backup: (N_e, n_a, K) ──────────────────────────────
            Q = r + params.beta * EV_next[:, :, np.newaxis]

            pi[t, chi] = np.argmax(Q, axis=1)
            V[t, chi]  = np.max(Q,   axis=1)

    return V, pi, actions, e_grid, lam_grid
