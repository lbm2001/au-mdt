import math

import numpy as np

from .params import NegBinParams

PARKED = 0


def is_driving(chi: int) -> bool:
    return chi > 0


def p_pd(t: int, params: NegBinParams) -> float:
    """Parked→Driving departure probability at minute t."""
    h = (t % 1440) / 60
    if 7.0 <= h < 9.0:
        return params.p_pd_morning
    elif 12.0 <= h < 14.0:
        return params.p_pd_lunch
    elif 16.0 <= h < 18.0:
        return params.p_pd_evening
    else:
        return params.p_pd_default


def _poisson_entry_probs(params: NegBinParams) -> np.ndarray:
    """PMF of k ~ Poisson(lambda_k) truncated to [1, k_max], length k_max."""
    lam   = params.lambda_k
    k_max = params.k
    pmf   = np.array([math.exp(-lam) * lam**r / math.factorial(r)
                      for r in range(1, k_max + 1)])
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def transition_matrix(t: int, params: NegBinParams) -> np.ndarray:
    """Return the (k+1)×(k+1) one-step transition matrix at minute t.

    State encoding — remaining phases:
      0       = Parked
      r ≥ 1   = Driving with r phases left
    """
    n = params.k + 1
    P = np.zeros((n, n))
    pPD = p_pd(t, params)
    q   = params.q

    P[0, 0] = 1.0 - pPD
    if params.lambda_k is None:
        P[0, params.k] = pPD
    else:
        entry = _poisson_entry_probs(params)
        P[0, 1:] = pPD * entry

    for r in range(1, params.k + 1):
        P[r, r]     = 1.0 - q
        P[r, r - 1] = q

    return P


def transition_probs(t: int, params: NegBinParams) -> tuple[float, float]:
    """Return (p_PD, p_DP_eff) at minute t.

    p_DP_eff is the effective Driving→Parked probability derived from the
    stationary distribution over driving phases.
    """
    p_PD = p_pd(t, params)
    if params.lambda_k is None:
        p_DP_eff = params.q / params.k
    else:
        entry = _poisson_entry_probs(params)
        sf = np.array([entry[r - 1:].sum() for r in range(1, params.k + 1)])
        p_DP_eff = float(params.q * sf[0] / sf.sum())
    return p_PD, p_DP_eff
