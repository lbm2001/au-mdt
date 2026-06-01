import numpy as np

from .model import is_driving, price_bin
from .params import NegBinParams
from .rollout import actual_charge_rate


def backward_induction_policy(
    t: int,
    chi: int,
    e: float,
    lam: float,
    params: NegBinParams,
    *,
    pi: np.ndarray,
    actions: np.ndarray,
    e_grid: np.ndarray,
) -> float:
    e_idx   = int(np.argmin(np.abs(e_grid - e)))
    lam_idx = price_bin(lam, params)
    a_idx   = pi[t, chi, e_idx, lam_idx]
    return float(actions[a_idx])


def maximal_charging_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
) -> float:
    return float(params.u_max)


def price_oriented_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
    *, low_threshold: float, high_threshold: float,
) -> float:
    if lam <= low_threshold:
        return float(params.u_max)
    if lam <= high_threshold:
        return float(params.u_max / 2)
    return 0.0


def night_charging_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
) -> float:
    return float(params.u_max) if t % 1440 < 360 else 0.0


def minimum_soc_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
    *, soc_threshold: float,
) -> float:
    return float(params.u_max) if e < soc_threshold else 0.0


def always_minimum_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
) -> float:
    return float(params.u_min)


def random_policy(
    t: int, chi: int, e: float, lam: float, params: NegBinParams,
    *, rng: np.random.Generator,
) -> float:
    return float(rng.choice([0.0, params.u_min, params.u_max / 2, params.u_max]))
