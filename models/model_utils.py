"""Model utilities shared by all EV charging MDP models."""
import numpy as np
from math import erf as _erf, sqrt as _sqrt


def mean_price(t: int, params) -> float:
    """Time-dependent mean electricity price λ̄_t (€/kWh); periodic over 1440 min."""
    h = (t % 1440) / 60
    if h < 6:   return params.price_night
    if h < 9:   return params.price_morning
    if h < 16:  return params.price_midday
    if h < 21:  return params.price_evening
    return params.price_late


def price_bin(lam: float, params) -> int:
    """Map a price (€/kWh) to a bin index in [0, K-1]."""
    delta = params.lambda_max / params.K
    return int(min(max(0.0, lam) / delta, params.K - 1))


def bin_center_price(k: int, params) -> float:
    """Centre price (€/kWh) for bin k."""
    return (k + 0.5) * params.lambda_max / params.K


def price_bin_probs(t: int, params) -> np.ndarray:
    """Return K-vector of bin probabilities for the price at time t."""
    lam_bar = mean_price(t, params)
    sigma   = params.sigma_lambda
    delta   = params.lambda_max / params.K

    def _cdf(x: float) -> float:
        if sigma <= 0:
            return 1.0 if x > lam_bar else 0.0
        return 0.5 * (1.0 + _erf((x - lam_bar) / (sigma * _sqrt(2.0))))

    edges = [j * delta for j in range(params.K + 1)]
    probs = np.empty(params.K)
    probs[0] = _cdf(edges[1])
    for j in range(1, params.K - 1):
        probs[j] = _cdf(edges[j + 1]) - _cdf(edges[j])
    probs[-1] = 1.0 - _cdf(edges[-2])
    return probs


def consumption(chi: int, params) -> float:
    """Energy consumed per minute in state chi (kWh/min). chi > 0 means driving."""
    return params.mu * params.v * params.omega if chi > 0 else 0.0
