from dataclasses import dataclass

from models.params import SharedParams


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
