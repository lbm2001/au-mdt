"""Shared cost-minimizing backward induction for the EV charging MDP.

Generic over the mobility model: the caller supplies the (n_chi × n_chi)
one-step transition matrix P_t(χ'|χ), so the 2-state baseline and the
(k+1)-state Negative-Binomial model both plug into the same recursion. This
follows the paper's Algorithm (Backward Induction) directly — it **minimizes
expected cost** (charging cost + unserved-driving penalty) rather than
maximizing a reward.
"""
import numpy as np
from typing import Callable

from ev_mdt.models.common.model_utils import consumption as _consumption
from ev_mdt.params import N_e as _N_E, T_hours as _T_HOURS


def backward_induction(
    params,
    transition_matrix_fn: Callable[[int], np.ndarray],
    price_bin_probs_fn: Callable[[int], np.ndarray],
    n_chi: int,
    T: int = _T_HOURS * 60,
    N_e: int = _N_E,
    N_a: int | None = None,
    consumption_fn: Callable[[int], float] | None = None,
):
    """Solve the EV charging MDP by cost-minimizing backward induction.

    The discretised price bin λ̂_t is part of the state. Because λ̂_{t+1} is drawn
    i.i.d. from the time-dependent marginal (independent of the current price),
    the expected continuation cost factors as

        E[J_{t+1}(χ', e', λ̂')] = Σ_k  P_{t+1}(k) · J_{t+1}(χ', e', k)

    so the K-dimensional price expectation is contracted once per (t, χ').

    Parameters
    ----------
    params               : model params — must include K, lambda_max, beta, omega,
                           phi, eta_c, u_min, u_max, e_min, e_max.
    transition_matrix_fn : t -> (n_chi, n_chi) one-step matrix P_t(χ'|χ) (row=from).
    price_bin_probs_fn   : t -> (K,) bin probabilities for the price at t.
    n_chi                : number of mobility states (2 for baseline, k+1 for NegBin).
    T                    : horizon in minutes.
    N_e                  : battery grid points.
    N_a                  : non-zero charge rates; None → [0, u_min, u_max/2, u_max];
                           an int → [0] + linspace(u_min, u_max, N_a).
    consumption_fn       : chi -> kWh/min; default uses the shared consumption().

    Returns
    -------
    J        : ndarray (T+1, n_chi, N_e, K) — minimal expected cost-to-go
    pi       : ndarray (T,   n_chi, N_e, K) — index into `actions`
    actions  : ndarray (N_a+1,)
    e_grid   : ndarray (N_e,)
    lam_grid : ndarray (K,)                 — bin-centre prices (€/kWh)
    """
    K      = params.K
    e_grid = np.linspace(params.e_min, params.e_max, N_e)
    if N_a is None:
        actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])
    else:
        actions = np.concatenate([[0.0], np.linspace(params.u_min, params.u_max, N_a)])
    n_a = len(actions)

    lam_grid = np.array([(j + 0.5) * params.lambda_max / K for j in range(K)])

    if consumption_fn is None:
        consumption_fn = lambda chi: _consumption(chi, params)

    J  = np.zeros((T + 1, n_chi, N_e, K))               # J_T ≡ 0
    pi = np.zeros((T,     n_chi, N_e, K), dtype=int)

    all_p_next = np.array([price_bin_probs_fn(t + 1) for t in range(T)])  # (T, K)

    E = e_grid[:, np.newaxis]    # (N_e, 1)
    A = actions[np.newaxis, :]   # (1, n_a)

    for t in range(T - 1, -1, -1):
        P     = transition_matrix_fn(t)        # (n_chi, n_chi)
        J_bar = J[t + 1] @ all_p_next[t]       # price expectation → (n_chi, N_e)

        for chi in range(n_chi):
            driving = chi > 0

            # Available charge rates: while driving the car can only charge once
            # it has hit the battery floor (otherwise it is on the road, u = 0).
            if driving:
                u_a = np.where(E > params.e_min, 0.0, A)
            else:
                u_a = np.broadcast_to(A, (N_e, n_a)).copy()

            # Stage cost C_t(χ, e, λ̂, u): charging cost + unserved-driving penalty.
            cost = (u_a[:, :, np.newaxis]
                    * params.omega
                    * lam_grid[np.newaxis, np.newaxis, :])
            if driving:
                penalty = np.where(E > params.e_min, 0.0, params.omega * params.phi)
                cost = cost + penalty[:, :, np.newaxis]

            cons   = consumption_fn(chi)
            e_next = np.clip(
                E + params.eta_c * params.omega * u_a - cons,
                params.e_min, params.e_max,
            )
            e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            e_lo = np.floor(e_next_f).astype(int)
            e_hi = np.minimum(e_lo + 1, N_e - 1)
            w_hi = e_next_f - e_lo
            w_lo = 1.0 - w_hi

            # Expected continuation cost: Σ_χ' P_t(χ'|χ) · J̄_{t+1}(χ', e'(u)).
            EJ_next = np.zeros((N_e, n_a))
            for chi_next in range(n_chi):
                p = P[chi, chi_next]
                if p == 0.0:
                    continue
                EJ_next += p * (w_lo * J_bar[chi_next, e_lo]
                              + w_hi * J_bar[chi_next, e_hi])

            Q = cost + params.beta * EJ_next[:, :, np.newaxis]

            pi[t, chi] = np.argmin(Q, axis=1)
            J[t, chi]  = np.min(Q,   axis=1)

    return J, pi, actions, e_grid, lam_grid


def evaluate_policy(
    params,
    transition_matrix_fn: Callable[[int], np.ndarray],
    price_bin_probs_fn: Callable[[int], np.ndarray],
    n_chi: int,
    action_fn: Callable[[int, int], np.ndarray],
    T: int,
    N_e: int,
    beta: float | None = None,
    consumption_fn: Callable[[int], float] | None = None,
    progress_desc: str | None = None,
    cost_components: bool = False,
):
    """Exact policy evaluation: expected cost-to-go Jπ for a fixed deterministic policy.

    Same recursion as ``backward_induction`` but with no minimisation — the action
    is supplied by ``action_fn(t, chi) -> (N_e, K)`` desired charge rates (the
    driving/battery-floor gating is applied here, exactly as in the solver). With
    ``beta=1`` this yields the expected *undiscounted* total cost, matching the
    undiscounted cost summed in the Monte-Carlo rollouts.

    Pass ``progress_desc`` to show a per-timestep tqdm bar over the T backward
    steps (useful since each step calls a potentially slow scalar ``action_fn``).

    Returns the t=0 slice Jπ of shape (n_chi, N_e, K) — only J[0] is needed to
    compute the expected cost, so a rolling 2-slice buffer is used instead of
    storing the full (T+1, n_chi, N_e, K) trajectory.  With ``cost_components=True``
    returns ``(Jπ, Jc, Jp, Je, Jm)`` where Jc/Jp are the charging/penalty cost
    components, Je is expected kWh charged, and Jm is expected penalty minutes.
    """
    beta = params.beta if beta is None else beta
    K      = params.K
    e_grid = np.linspace(params.e_min, params.e_max, N_e)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / K for j in range(K)])
    if consumption_fn is None:
        consumption_fn = lambda chi: _consumption(chi, params)

    # Rolling 2-slice buffers: only the current and next timestep slices are kept,
    # reducing memory from O(T·n_chi·N_e·K) to O(n_chi·N_e·K) per accumulator.
    # Critical for large n_chi (NegBin k=10 → 11 states → ~6 GB per full array).
    J_next = np.zeros((n_chi, N_e, K))
    J_cur  = np.zeros((n_chi, N_e, K))
    if cost_components:
        Jc_next, Jc_cur = np.zeros((n_chi, N_e, K)), np.zeros((n_chi, N_e, K))
        Jp_next, Jp_cur = np.zeros((n_chi, N_e, K)), np.zeros((n_chi, N_e, K))
        Je_next, Je_cur = np.zeros((n_chi, N_e, K)), np.zeros((n_chi, N_e, K))
        Jm_next, Jm_cur = np.zeros((n_chi, N_e, K)), np.zeros((n_chi, N_e, K))

    all_p_next = np.array([price_bin_probs_fn(t + 1) for t in range(T)])
    E = e_grid[:, np.newaxis]    # (N_e, 1)

    steps = range(T - 1, -1, -1)
    if progress_desc is not None:
        from tqdm import tqdm
        steps = tqdm(steps, total=T, desc=progress_desc, unit="step",
                     position=3, leave=False)
    for t in steps:
        P     = transition_matrix_fn(t)
        J_bar = J_next @ all_p_next[t]       # (n_chi, N_e)
        if cost_components:
            Jc_bar = Jc_next @ all_p_next[t]
            Jp_bar = Jp_next @ all_p_next[t]
            Je_bar = Je_next @ all_p_next[t]
            Jm_bar = Jm_next @ all_p_next[t]

        for chi in range(n_chi):
            driving = chi > 0
            u = np.asarray(action_fn(t, chi), dtype=float)      # (N_e, K) desired rates
            if driving:
                u = np.where(E > params.e_min, 0.0, u)          # on the road → u = 0

            charge_cost = u * params.omega * lam_grid[np.newaxis, :]   # (N_e, K)
            charge_energy = u * params.eta_c * params.omega             # (N_e, K) kWh
            pen_cost = np.zeros_like(charge_cost)
            if driving:
                pen = np.where(E > params.e_min, 0.0, params.omega * params.phi)
                pen_cost = pen_cost + pen
            cost = charge_cost + pen_cost

            cons   = consumption_fn(chi)
            e_next = np.clip(E + params.eta_c * params.omega * u - cons,
                             params.e_min, params.e_max)
            e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
            e_lo = np.floor(e_next_f).astype(int)
            e_hi = np.minimum(e_lo + 1, N_e - 1)
            w_hi = e_next_f - e_lo
            w_lo = 1.0 - w_hi

            pen_minutes = np.zeros_like(charge_cost)
            if driving:
                pen_minutes = np.where(E > params.e_min, 0.0, np.ones_like(charge_cost))

            EJ = np.zeros((N_e, K))
            EJc = np.zeros((N_e, K)) if cost_components else None
            EJp = np.zeros((N_e, K)) if cost_components else None
            EJe = np.zeros((N_e, K)) if cost_components else None
            EJm = np.zeros((N_e, K)) if cost_components else None
            for chi_next in range(n_chi):
                p = P[chi, chi_next]
                if p == 0.0:
                    continue
                EJ += p * (w_lo * J_bar[chi_next][e_lo] + w_hi * J_bar[chi_next][e_hi])
                if cost_components:
                    EJc += p * (w_lo * Jc_bar[chi_next][e_lo] + w_hi * Jc_bar[chi_next][e_hi])
                    EJp += p * (w_lo * Jp_bar[chi_next][e_lo] + w_hi * Jp_bar[chi_next][e_hi])
                    EJe += p * (w_lo * Je_bar[chi_next][e_lo] + w_hi * Je_bar[chi_next][e_hi])
                    EJm += p * (w_lo * Jm_bar[chi_next][e_lo] + w_hi * Jm_bar[chi_next][e_hi])

            J_cur[chi] = cost + beta * EJ
            if cost_components:
                Jc_cur[chi] = charge_cost + beta * EJc
                Jp_cur[chi] = pen_cost + beta * EJp
                Je_cur[chi] = charge_energy + beta * EJe
                Jm_cur[chi] = pen_minutes + beta * EJm

        J_next, J_cur = J_cur, J_next
        if cost_components:
            Jc_next, Jc_cur = Jc_cur, Jc_next
            Jp_next, Jp_cur = Jp_cur, Jp_next
            Je_next, Je_cur = Je_cur, Je_next
            Jm_next, Jm_cur = Jm_cur, Jm_next

    # J_next now holds J[0] — the only slice needed for the expected cost.
    if cost_components:
        return J_next, Jc_next, Jp_next, Je_next, Jm_next
    return J_next


def scalar_policy_to_action_fn(
    policy_fn,
    e_grid: np.ndarray,
    lam_grid: np.ndarray,
    params,
    **kwargs,
) -> "Callable[[int, int], np.ndarray]":
    """Wrap a scalar policy ``f(t, chi, e, lam, params, **kwargs) -> float``
    into the ``(N_e, K)`` array form expected by :func:`evaluate_policy`."""
    def action_fn(t: int, chi: int) -> np.ndarray:
        return np.array(
            [[policy_fn(t, chi, float(e), float(lam), params, **kwargs)
              for lam in lam_grid]
             for e in e_grid],
            dtype=float,
        )
    return action_fn
