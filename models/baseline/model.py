import numpy as np
from math import erf as _erf, sqrt as _sqrt

from .params import BaselineParams

PARKED = 0
DRIVING = 1


def mean_price(t: int, params: BaselineParams) -> float:
    """Time-dependent mean electricity price λ̄_t (€/kWh); t is absolute minute, periodic over 1440."""
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


def transition_probs(t: int, params: BaselineParams) -> tuple[float, float]:
    """Returns (p_PD, p_DP) at minute t; periodic over 1440 minutes."""
    h = (t % 1440) / 60

    if 7.0 <= h < 9.0:
        p_pd = params.p_pd_morning
    elif 12.0 <= h < 14.0:
        p_pd = params.p_pd_lunch
    elif 16.0 <= h < 18.0:
        p_pd = params.p_pd_evening
    else:
        p_pd = params.p_pd_default

    if 7.5 <= h < 9.5:
        p_dp = params.p_dp_morning
    elif 12.25 <= h < 14.25:
        p_dp = params.p_dp_lunch
    elif 16.5 <= h < 18.5:
        p_dp = params.p_dp_evening
    else:
        p_dp = params.p_dp_default

    return p_pd, p_dp


def consumption(chi: int, params: BaselineParams) -> float:
    """Energy consumed per minute in state chi (kWh/min)."""
    return params.mu * params.v * params.omega if chi == DRIVING else 0.0


def price_bin(lam: float, params: BaselineParams) -> int:
    """Map a price (€/kWh) to a bin index in [0, K-1]. Negative prices map to bin 0."""
    delta = params.lambda_max / params.K
    return int(min(max(0.0, lam) / delta, params.K - 1))


def bin_center_price(k: int, params: BaselineParams) -> float:
    """Centre price (€/kWh) for bin k."""
    return (k + 0.5) * params.lambda_max / params.K


def price_bin_probs(t: int, params: BaselineParams) -> np.ndarray:
    """Return K-vector of bin probabilities for the price at time t.

    Negative prices are folded into bin 0 (truncation at 0).
    The last bin absorbs all probability above (K-1)·Δ.
    """
    lam_bar = mean_price(t, params)
    sigma   = params.sigma_lambda
    delta   = params.lambda_max / params.K  # bin width in €/kWh

    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + _erf((x - lam_bar) / (sigma * _sqrt(2.0))))

    edges = [k * delta for k in range(params.K + 1)]
    probs = np.empty(params.K)
    probs[0] = _cdf(edges[1])                  # includes all probability mass ≤ 0
    for k in range(1, params.K - 1):
        probs[k] = _cdf(edges[k + 1]) - _cdf(edges[k])
    probs[-1] = 1.0 - _cdf(edges[-2])          # absorbs tail above (K-1)·Δ
    return probs
