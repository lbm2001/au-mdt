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


def transition_matrix(t: int, params: NegBinParams) -> np.ndarray:
    """Return the (k+1)×(k+1) one-step transition matrix at minute t.

    States: 0=P, 1=D_1, …, k=D_k.
      P   → D_1      w.p. p_PD  (time-dependent)
      D_i → D_{i+1}  w.p. q     (i = 1, …, k-1)
      D_k → P        w.p. q
      all self-loops with residual probability.
    """
    n = params.k + 1
    P = np.zeros((n, n))
    pPD = p_pd(t, params)
    q = params.q

    P[0, 0] = 1.0 - pPD
    P[0, 1] = pPD

    for i in range(1, params.k):        # D_1 .. D_{k-1}
        P[i, i]     = 1.0 - q
        P[i, i + 1] = q

    P[params.k, params.k] = 1.0 - q    # D_k stays
    P[params.k, 0]         = q          # D_k → P

    return P


def consumption(chi: int, params: NegBinParams) -> float:
    """Energy consumed per minute in state chi (kWh/min)."""
    return params.mu * params.v * params.omega if is_driving(chi) else 0.0


def price_bin(lam: float, params: NegBinParams) -> int:
    delta = params.lambda_max / params.K
    return int(min(max(0.0, lam) / delta, params.K - 1))


def bin_center_price(k: int, params: NegBinParams) -> float:
    return (k + 0.5) * params.lambda_max / params.K


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
