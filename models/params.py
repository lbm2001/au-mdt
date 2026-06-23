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
    lambda_max: float = 0.25  # max_t price + 4*sigma_lambda (covers ~99.6% of normal-market DK1 prices)

    # ── Fixed conversion factor ───────────────────────────────────────────────
    omega: float = 1 / 60
