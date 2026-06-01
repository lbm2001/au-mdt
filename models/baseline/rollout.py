from models.rollout_utils import simulate_policy_rollout as _simulate
from .model import PARKED, DRIVING, transition_probs


def _next_state(chi: int, scenario: dict, t: int, params) -> int:
    p_PD, p_DP = transition_probs(t, params)
    draw = float(scenario["mobility_draws"][t])
    if chi == PARKED:
        return DRIVING if draw < p_PD else PARKED
    return PARKED if draw < p_DP else DRIVING


def simulate_policy_rollout(policy_fn, scenario, e0, chi0, params, **policy_kwargs):
    return _simulate(policy_fn, scenario, e0, chi0, params, _next_state, **policy_kwargs)
