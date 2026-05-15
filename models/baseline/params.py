from dataclasses import dataclass


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

    # ── Electricity price (€/MWh) ─────────────────────────────────────────────
    price_night: float = 70.0     # 00:00 – 06:00
    price_morning: float = 150.0  # 06:00 – 09:00
    price_midday: float = 110.0   # 09:00 – 16:00
    price_evening: float = 170.0  # 16:00 – 21:00
    price_late: float = 100.0     # 21:00 – 24:00
    sigma_lambda: float = 20.0    # price standard deviation (€/MWh)

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

    # ── Fixed conversion factor (not user-configurable) ───────────────────────
    omega: float = 1 / 60  # minutes → hours (h/min)
