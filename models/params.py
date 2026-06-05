from dataclasses import dataclass


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
    beta: float = 0.999

    # ── Vehicle dynamics ──────────────────────────────────────────────────────
    v: float = 50.0
    mu: float = 0.2

    # ── Electricity price (€/kWh) ─────────────────────────────────────────────
    price_night: float = 0.30
    price_morning: float = 0.48
    price_midday: float = 0.39
    price_evening: float = 0.55
    price_late: float = 0.34
    sigma_lambda: float = 0.05

    # ── Parked → Driving departure probabilities (per minute) ─────────────────
    p_pd_morning: float = 0.04
    p_pd_lunch: float = 0.015
    p_pd_evening: float = 0.035
    p_pd_default: float = 0.0025

    # ── Price discretisation ──────────────────────────────────────────────────
    K: int = 20
    lambda_max: float = 0.75

    # ── Fixed conversion factor ───────────────────────────────────────────────
    omega: float = 1 / 60
