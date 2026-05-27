from dataclasses import dataclass, field


@dataclass
class BaselineParams:
    # ── Battery ───────────────────────────────────────────────────────────────
    u_max: float = 11.0    # maximum charge rate (kW)
    u_min: float = 1.4     # minimum charge rate (kW)
    e_max: float = 40.0    # maximum battery level (kWh)
    e_min: float = 0.0     # minimum battery level (kWh)

    # ── Charging / cost ───────────────────────────────────────────────────────
    eta_c: float = 0.95    # charging efficiency
    phi: float = 1000.0    # unserved-driving penalty (€/h)
    beta: float = 0.999    # discount factor

    # ── Vehicle dynamics ──────────────────────────────────────────────────────
    v: float = 50.0        # average driving speed (km/h)
    mu: float = 0.2        # energy consumption (kWh/km)

    # ── Electricity price (€/kWh) ─────────────────────────────────────────────
    price_night: float = 0.30     # 00:00 – 06:00
    price_morning: float = 0.48   # 06:00 – 09:00
    price_midday: float = 0.39    # 09:00 – 16:00
    price_evening: float = 0.55   # 16:00 – 21:00
    price_late: float = 0.34      # 21:00 – 24:00
    sigma_lambda: float = 0.05    # price standard deviation (€/kWh)

    # ── Mobility transition probabilities (per minute) ────────────────────────
    # Parked → Driving
    p_pd_morning: float = 0.08   # 07:00 – 09:00
    p_pd_lunch: float = 0.03     # 12:00 – 14:00
    p_pd_evening: float = 0.07   # 16:00 – 18:00
    p_pd_default: float = 0.005  # all other times

    # Driving → Parked
    p_dp_morning: float = 0.15   # 07:30 – 09:30
    p_dp_lunch: float = 0.20     # 12:15 – 14:15
    p_dp_evening: float = 0.15   # 16:30 – 18:30
    p_dp_default: float = 0.25   # all other times

    # ── Price discretisation ──────────────────────────────────────────────────
    K: int = 20              # number of price bins
    lambda_max: float = 0.75   # upper price bound for binning (€/kWh);
                               # covers mean + 4σ of the most expensive period

    # ── Fixed conversion factor (not user-configurable) ───────────────────────
    omega: float = 1 / 60  # minutes → hours (h/min)
