import math

from ev_mdt.models.common.rollout_utils import simulate_policy_rollout as _simulate
from ev_mdt.models.negbin.model import PARKED, p_pd


def _sample_poisson_k(uniform_draw: float, lambda_k: float, k_max: int) -> int:
    """Inverse-CDF sample from Poisson(lambda_k) truncated to [1, k_max]."""
    pmf_r = math.exp(-lambda_k)
    total = 1.0 - pmf_r
    cdf   = 0.0
    for r in range(1, k_max + 1):
        pmf_r *= lambda_k / r
        cdf   += pmf_r / total
        if uniform_draw <= cdf:
            return r
    return k_max


def _next_state(chi: int, scenario: dict, t: int, params) -> int:
    mob_draw   = float(scenario["mobility_draws"][t])
    phase_draw = float(scenario["phase_draws"][t])
    if chi == PARKED:
        if mob_draw < p_pd(t, params):
            if params.lambda_k is None:
                return params.k
            return _sample_poisson_k(phase_draw, params.lambda_k, params.k)
        return PARKED
    return chi - 1 if mob_draw < params.q else chi


def simulate_policy_rollout(policy_fn, scenario, e0, chi0, params, **policy_kwargs):
    return _simulate(policy_fn, scenario, e0, chi0, params, _next_state, **policy_kwargs)
