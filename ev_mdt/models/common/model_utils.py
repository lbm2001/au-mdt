"""Shared model utilities for all EV charging MDP models."""
from functools import lru_cache

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


def _hazard(h: float, p_morning: float, p_lunch: float, p_evening: float, p_default: float) -> float:
    """Parked→Driving departure hazard for hour-of-day h. Windows shared by both models."""
    if 7.0 <= h < 9.0:
        return p_morning
    if 12.0 <= h < 14.0:
        return p_lunch
    if 16.0 <= h < 18.0:
        return p_evening
    return p_default


def departure_prob(t: int, params) -> float:
    """Parked→Driving departure probability p_t^{P→D} at minute t (per minute, periodic).

    Identical to baseline.transition_probs/negbin.p_pd, but model-agnostic: reads only
    the p_pd_* fields on SharedParams so shared policies can use it without a model object.
    """
    return _hazard((t % 1440) / 60,
                   params.p_pd_morning, params.p_pd_lunch,
                   params.p_pd_evening, params.p_pd_default)


@lru_cache(maxsize=8)
def _tau_table(p_morning: float, p_lunch: float, p_evening: float, p_default: float) -> tuple:
    """Exact E[minutes until next departure] for every minute-of-day, given the hazards.

    Solves the periodic recurrence  τ(m) = 1 + (1 − p_{m+1})·τ(m+1)  over the 1440-minute
    ring by Gauss–Seidel sweeps (backward, so each sweep uses freshly-updated successors).
    Contraction per cycle is the survival product ∏(1−p) ≪ 1, so it converges in a few
    sweeps. Cached on the four hazard values, so it is computed once per parameter set.
    """
    haz = [_hazard(m / 60, p_morning, p_lunch, p_evening, p_default) for m in range(1440)]
    tau = [0.0] * 1440
    for _ in range(200):
        max_delta = 0.0
        for m in range(1439, -1, -1):
            nxt = (m + 1) % 1440
            new = 1.0 + (1.0 - haz[m]) * tau[nxt]
            max_delta = max(max_delta, abs(new - tau[m]))
            tau[m] = new
        if max_delta < 1e-9:
            break
    return tuple(tau)


def expected_trip_minutes(params) -> float:
    """Expected duration of the next trip in minutes (certainty-equivalent of Kempker's deadline).

    NegBin: E[T_trip] = k/q (or lambda_k/q for Poisson-sampled k).
    Baseline: E[T_trip] = 1/p_dp_default (geometric trips, time-averaged by the dominant rate).
    """
    if hasattr(params, "q"):
        k_mean = params.lambda_k if params.lambda_k is not None else float(params.k)
        return k_mean / params.q
    return 1.0 / params.p_dp_default


def expected_trips_per_day(params) -> float:
    """Expected number of Parked→Driving departures in a 1440-minute day.

    Uses time-averaged departure and return hazards:
        E[trips/day] = 1440 · p_pd_avg · p_dp_avg / (p_pd_avg + p_dp_avg)

    p_pd_avg is the minute-weighted average departure hazard over the day.
    p_dp_avg is approximated from the params' return-hazard fields (baseline)
    or as 1/E[T_trip] for NegBin (geometric-phase equivalent).
    """
    p_pd_avg = (
        120 * params.p_pd_morning
        + 120 * params.p_pd_lunch
        + 120 * params.p_pd_evening
        + 1080 * params.p_pd_default
    ) / 1440

    if hasattr(params, "q"):
        k_mean = params.lambda_k if params.lambda_k is not None else float(params.k)
        p_dp_avg = params.q / k_mean  # 1 / E[T_trip]
    else:
        p_dp_avg = (
            120 * params.p_dp_morning
            + 120 * params.p_dp_lunch
            + 120 * params.p_dp_evening
            + 1080 * params.p_dp_default
        ) / 1440

    return 1440 * p_pd_avg * p_dp_avg / (p_pd_avg + p_dp_avg)


def minutes_to_departure(t: int, params) -> float:
    """Charging window τ(t): exact E[minutes until the next Parked→Driving departure].

    Forward expectation over the periodic departure hazard departure_prob(·), i.e.
    τ(t) = Σ_{s≥1} s · P(first departure at t+s). Computed via the cached ring solve.
    """
    table = _tau_table(params.p_pd_morning, params.p_pd_lunch,
                       params.p_pd_evening, params.p_pd_default)
    return table[t % 1440]


def max_minutes_to_departure(params) -> float:
    """Maximum of τ(t) over all minutes of the day — the quietest off-peak value."""
    table = _tau_table(params.p_pd_morning, params.p_pd_lunch,
                       params.p_pd_evening, params.p_pd_default)
    return max(table)


@lru_cache(maxsize=8)
def _mean_tau(p_morning: float, p_lunch: float, p_evening: float, p_default: float) -> float:
    """Flat average of τ(t) over all 1440 minutes, reusing the cached ring solve."""
    table = _tau_table(p_morning, p_lunch, p_evening, p_default)
    return sum(table) / 1440.0


def mean_minutes_to_departure(params) -> float:
    """Average of τ(t) = E[minutes to next departure] over the full 1440-minute day."""
    return _mean_tau(params.p_pd_morning, params.p_pd_lunch,
                     params.p_pd_evening, params.p_pd_default)
