"""Single source of truth for all model parameters and solver defaults.

Import hierarchy:
    SharedParams       — battery, vehicle, price, mobility, discretisation
    BaselineParams     — extends SharedParams with 2-state Markov (Geom trips)
    NegBinParams       — extends SharedParams with k-phase chain (NegBin trips)
    SolverConfig       — default solver hyper-parameters (N_e, T_hours)
    BASELINE_MODEL / NEGBIN_MODEL — canonical string keys used by the CLI and
                                    sensitivity analysis to select a model.
"""
from dataclasses import dataclass

BASELINE_MODEL = "Baseline"
NEGBIN_FIXED_MODEL = "Negative Binomial trips (fixed k)"
NEGBIN_SAMPLED_MODEL = "Negative Binomial trips (sampled k)"
MODEL_LABELS = [BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL]


@dataclass
class SharedParams:
    # ── Battery ───────────────────────────────────────────────────────────────
    u_max: float = 11.0
    u_min: float = 1.4
    e_max: float = 40.0
    e_min: float = 0.0

    # ── Charging / cost ───────────────────────────────────────────────────────
    eta_c: float = 0.95
    phi: float = 1000.0
    beta: float = 1.0

    # ── Vehicle dynamics ──────────────────────────────────────────────────────
    v: float = 50.0
    mu: float = 0.2

    # ── Electricity price (€/kWh) ─────────────────────────────────────────────
    # Wholesale DK1 day-ahead means by time-of-day window, fitted to ENTSO-E data
    # excluding the 2021–2023 energy crisis (a "typical" normal-market day).
    # Note the duck-curve shape: midday is cheapest (wind+solar), peaks morning/evening.
    price_night: float = 0.053    # 00–06 h
    price_morning: float = 0.068  # 06–09 h
    price_midday: float = 0.047   # 09–16 h
    price_evening: float = 0.073  # 16–21 h
    price_late: float = 0.066     # 21–24 h
    sigma_lambda: float = 0.045

    # ── Parked → Driving departure probabilities (per minute) ─────────────────
    p_pd_morning: float = 0.04
    p_pd_lunch: float = 0.015
    p_pd_evening: float = 0.035
    p_pd_default: float = 0.0025

    # ── Price discretisation ──────────────────────────────────────────────────
    K: int = 100
    lambda_max: float = 0.25  # max_t price + 4*sigma_lambda; covers ~99.6% of normal-market DK1 prices

    # ── Fixed conversion factor ───────────────────────────────────────────────
    omega: float = 1 / 60


@dataclass
class BaselineParams(SharedParams):
    # ── Driving → Parked return probabilities (per minute) ────────────────────
    p_dp_morning: float = 0.05   # 07:30 – 09:30
    p_dp_lunch: float = 0.1     # 12:15 – 14:15
    p_dp_evening: float = 0.05   # 16:30 – 18:30
    p_dp_default: float = 0.1   # all other times


@dataclass
class NegBinParams(SharedParams):
    # ── NegBin trip-duration parameters ──────────────────────────────────────
    # T_trip ~ NegBin(k, q):  E[T] = k/q,  Var[T] = k(1-q)/q²
    k: int = 5       # fixed phases (lambda_k=None) or k_max (lambda_k set)
    q: float = 0.20  # per-phase transition probability (timescale)
                     # default: E[T] = 25 min, Var[T] = 100 min²

    # ── Poisson-sampled phases ────────────────────────────────────────────────
    # When set, k ~ Poisson(lambda_k) is sampled at the start of each trip
    # (truncated to [1, k]).  When None, every trip has exactly k phases.
    lambda_k: float | None = None


@dataclass
class SolverConfig:
    """Default solver hyper-parameters. Override per-call via keyword args."""
    N_e: int = 500       # battery grid points
    T_hours: int = 24    # planning horizon; T = T_hours * 60 minutes


# Module-level constants so existing code can do:
#   from ev_mdt.params import N_e, T_hours
_defaults = SolverConfig()
N_e: int = _defaults.N_e
T_hours: int = _defaults.T_hours
