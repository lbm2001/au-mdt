import numpy as np
from math import erf as _erf, sqrt as _sqrt

from .params import NegBinParams

PARKED = 0
# Driving states are 1 .. k (D_1 .. D_k); is_driving(chi) = chi > 0


def is_driving(chi: int) -> bool:
    return chi > 0


def mean_price(t: int, params: NegBinParams) -> float:
    """Time-dependent mean electricity price λ̄_t (€/kWh); periodic over 1440 min."""
    h = (t % 1440) / 60
    if h < 6:
        return params.price_night
    elif h < 9:
        return params.price_morning
    elif h < 16:
        return params.price_midday
    elif h < 21:
        return params.price_evening
    else:
        return params.price_late


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
    import math
    lam = params.lambda_k
    k_max = params.k
    pmf = np.array([math.exp(-lam) * lam**r / math.factorial(r)
                    for r in range(1, k_max + 1)])
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def transition_matrix(t: int, params: NegBinParams) -> np.ndarray:
    """Return the (k+1)×(k+1) one-step transition matrix at minute t.

    State encoding — remaining phases:
      0       = Parked
      r ≥ 1   = Driving with r phases left

    Transitions:
      P    → r    w.p. p_PD × entry(r)   (entry: point mass at k or Poisson PMF)
      r>1  → r-1  w.p. q
      r=1  → P    w.p. q
      all states self-loop with residual probability.
    """
    n = params.k + 1
    P = np.zeros((n, n))
    pPD = p_pd(t, params)
    q = params.q

    P[0, 0] = 1.0 - pPD
    if params.lambda_k is None:
        P[0, params.k] = pPD                          # fixed k: point mass
    else:
        entry = _poisson_entry_probs(params)
        P[0, 1:] = pPD * entry                        # distribute over 1..k

    for r in range(1, params.k + 1):                  # remaining r phases
        P[r, r] = 1.0 - q
        P[r, r - 1] = q                               # r-1=0 means → Parked

    return P


def consumption(chi: int, params: NegBinParams) -> float:
    """Energy consumed per minute in state chi (kWh/min)."""
    return params.mu * params.v * params.omega if is_driving(chi) else 0.0


def price_bin(lam: float, params: NegBinParams) -> int:
    delta = params.lambda_max / params.K
    return int(min(max(0.0, lam) / delta, params.K - 1))


def bin_center_price(k: int, params: NegBinParams) -> float:
    return (k + 0.5) * params.lambda_max / params.K


def transition_probs(t: int, params: NegBinParams) -> tuple[float, float]:
    """Return (p_PD, p_DP_eff) at minute t.

    p_DP_eff is the effective Driving→Parked probability, derived from the
    stationary distribution over driving phases.  For fixed k: q/k.  For
    Poisson-sampled k: q / E[k_trip], where E[k_trip] = sum_r P(k≥r) over
    the truncated Poisson distribution.
    """
    p_PD = p_pd(t, params)
    if params.lambda_k is None:
        p_DP_eff = params.q / params.k
    else:
        entry = _poisson_entry_probs(params)   # PMF of k in [1..k_max]
        # Stationary P(remaining=r | driving) ∝ P(k_trip ≥ r) = sum_{j≥r} entry[j-1]
        sf = np.array([entry[r - 1:].sum() for r in range(1, params.k + 1)])
        p_DP_eff = float(params.q * sf[0] / sf.sum())
    return p_PD, p_DP_eff


def price_bin_probs(t: int, params: NegBinParams) -> np.ndarray:
    """Return K-vector of bin probabilities for the price at time t."""
    lam_bar = mean_price(t, params)
    sigma   = params.sigma_lambda
    delta   = params.lambda_max / params.K

    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + _erf((x - lam_bar) / (sigma * _sqrt(2.0))))

    edges = [j * delta for j in range(params.K + 1)]
    probs = np.empty(params.K)
    probs[0] = _cdf(edges[1])
    for j in range(1, params.K - 1):
        probs[j] = _cdf(edges[j + 1]) - _cdf(edges[j])
    probs[-1] = 1.0 - _cdf(edges[-2])
    return probs
