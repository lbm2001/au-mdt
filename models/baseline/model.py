from .params import BaselineParams

PARKED = 0
DRIVING = 1


def mean_price(t: int, params: BaselineParams) -> float:
    """Time-dependent mean electricity price λ̄_t (€/MWh); t is minute of day [0, 1440)."""
    h = t / 60
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
    """Returns (p_PD, p_DP) at minute t."""
    h = t / 60

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
