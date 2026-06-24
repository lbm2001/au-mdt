import numpy as np

from ev_mdt.params import BaselineParams

PARKED  = 0
DRIVING = 1


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


def transition_matrix(t: int, params: BaselineParams) -> np.ndarray:
    """The 2×2 one-step mobility transition matrix at minute t (rows = from-state).

        [[1 - p_PD, p_PD],
         [p_DP,     1 - p_DP]]

    Mirrors negbin.model.transition_matrix so both models plug into the shared
    backward-induction solver.
    """
    p_pd, p_dp = transition_probs(t, params)
    return np.array([[1.0 - p_pd, p_pd],
                     [p_dp,       1.0 - p_dp]])
