import numpy as np

from ev_mdt.models.common.rollout_utils import simulate_policy_rollout as _simulate
from ev_mdt.models.negbin.model import PARKED, p_pd, _poisson_entry_probs


def _sample_poisson_k(uniform_draw: float, params) -> int:
    """Inverse-CDF sample of the trip's phase count k ∈ [1, k_max].

    Uses the *same* truncated-and-renormalised Poisson PMF the solver assumes
    (model._poisson_entry_probs), so the simulated trip-length distribution
    matches the one backward induction optimises against.
    """
    cdf = np.cumsum(_poisson_entry_probs(params))
    return min(int(np.searchsorted(cdf, uniform_draw, side="left")) + 1, params.k)


def _next_state(chi: int, scenario: dict, t: int, params) -> int:
    mob_draw   = float(scenario["mobility_draws"][t])
    phase_draw = float(scenario["phase_draws"][t])
    if chi == PARKED:
        if mob_draw < p_pd(t, params):
            if params.lambda_k is None:
                return params.k
            return _sample_poisson_k(phase_draw, params)
        return PARKED
    return chi - 1 if mob_draw < params.q else chi


def simulate_policy_rollout(policy_fn, scenario, e0, chi0, params, **policy_kwargs):
    return _simulate(policy_fn, scenario, e0, chi0, params, _next_state, **policy_kwargs)
