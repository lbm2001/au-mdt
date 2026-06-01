from dataclasses import dataclass

from models.params import SharedParams


@dataclass
class BaselineParams(SharedParams):
    # ── Driving → Parked return probabilities (per minute) ────────────────────
    p_dp_morning: float = 0.15   # 07:30 – 09:30
    p_dp_lunch: float = 0.20     # 12:15 – 14:15
    p_dp_evening: float = 0.15   # 16:30 – 18:30
    p_dp_default: float = 0.25   # all other times
