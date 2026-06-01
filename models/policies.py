"""Shared charging policies — work with any params that has the standard fields."""
import numpy as np

from models.model_utils import price_bin, price_bin_probs


def _transition_probs(t: int, params) -> tuple[float, float]:
    """Dispatch to the model-specific transition_probs via sys.modules (O(1) dict lookup)."""
    import sys
    mod_name = type(params).__module__.rsplit(".", 1)[0] + ".model"
    return sys.modules[mod_name].transition_probs(t, params)


# ── Shared policies ────────────────────────────────────────────────────────────

def actual_charge_rate(chi: int, e: float, desired_u: float, params) -> float:
    if chi > 0 and e > params.e_min:
        return 0.0
    return float(np.clip(desired_u, 0.0, params.u_max))


def backward_induction_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, pi: np.ndarray, actions: np.ndarray, e_grid: np.ndarray,
) -> float:
    e_idx   = int(np.argmin(np.abs(e_grid - e)))
    lam_idx = price_bin(lam, params)
    return float(actions[pi[t, chi, e_idx, lam_idx]])


def maximal_charging_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_max)


def price_oriented_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, low_threshold: float, high_threshold: float,
) -> float:
    if lam <= low_threshold:
        return float(params.u_max)
    if lam <= high_threshold:
        return float(params.u_max / 2)
    return 0.0


def night_charging_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_max) if t % 1440 < 360 else 0.0


def minimum_soc_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, soc_threshold: float,
) -> float:
    return float(params.u_max) if e < soc_threshold else 0.0


def always_minimum_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    return float(params.u_min)


def random_policy(
    t: int, chi: int, e: float, lam: float, params,
    *, rng: np.random.Generator,
) -> float:
    return float(rng.choice([0.0, params.u_min, params.u_max / 2, params.u_max]))


def dp_heuristic_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    """SoC-urgency heuristic: charge at u_max when F_t(lam) ≤ 1 − e/e_max."""
    if chi > 0 and e > params.e_min:
        return 0.0
    if e >= params.e_max:
        return 0.0
    thresh   = 1.0 - e / params.e_max
    probs    = price_bin_probs(t, params)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K for j in range(params.K)])
    F_p      = float(probs[lam_grid <= lam].sum())
    return float(params.u_max) if F_p <= thresh else 0.0


def expected_parking_policy(
    t: int, chi: int, e: float, lam: float, params,
) -> float:
    """Textbook three-band rule with rem = expected parked minutes per day at time t."""
    if chi > 0 and e > params.e_min:
        return 0.0
    x = params.e_max - e
    if x <= 0:
        return 0.0

    energy_per_step = params.u_max * params.omega * params.eta_c
    k = int(x // energy_per_step)

    p_PD, p_DP = _transition_probs(t, params)
    denom = p_PD + p_DP
    pi_P  = p_DP / denom if denom > 0 else 0.5
    rem   = max(int(pi_P * 1440), k + 1)

    probs    = price_bin_probs(t, params)
    lam_grid = np.array([(j + 0.5) * params.lambda_max / params.K for j in range(params.K)])
    F_p      = float(probs[lam_grid <= lam].sum())

    thresh_k  = k / rem
    thresh_k1 = (k + 1) / rem

    if F_p <= thresh_k:
        u = params.u_max
    elif F_p <= thresh_k1:
        u = (x - k * energy_per_step) / (params.omega * params.eta_c)
    else:
        u = 0.0
    return float(np.clip(u, 0.0, params.u_max))
