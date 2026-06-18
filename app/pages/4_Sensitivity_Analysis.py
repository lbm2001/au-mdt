"""
=============================================================================
Sensitivity Analysis — EV Charging MDP
=============================================================================
Baseline configuration (SharedParams / BaselineParams defaults)
---------------------------------------------------------------
  Battery  : e_max=40 kWh, e_min=0 kWh, η_c=0.95, u_max=11 kW, u_min=1.4 kW
  Vehicle  : v=50 km/h, μ=0.20 kWh/km
  Cost     : φ=1000 €/h (unserved-driving penalty), β=0.999
  Solver   : K=20 price bins, λ_max=0.75 €/kWh

Mobility model (sidebar selector, applies to every sweep):
  • Baseline           — trip duration ~ Geom(p_DP); 2-state mobility chain
  • NegBin (fixed k)   — trip ~ NegBin(k, q) via a k-phase chain; mean = k/q min
  • NegBin (sampled k) — k ~ Poisson(λ_k) drawn at each trip start

Four independent sweep dimensions (one at a time, others held at baseline):
  1. Pricing model  — Gaussian parametric · Gaussian bins · GMM · MDN
  2. Penalty φ      — {0,100,500,1000,2000,5000,10 000} €/h
  3. Horizon T      — {24 h, 48 h, 168 h}
  4. Season × Day type — all 8 combinations of {winter,spring,summer,autumn} × {weekday,weekend}

Policies compared
-----------------
  • Optimal (Backward induction) — solves the MDP exactly for each config
  • Night charging               — charges only during 00:00–06:00
  • DP heuristic                 — SoC-urgency rule using price CDF
  • Maximal charging             — always charges at u_max

Note on pricing sweep: the DP-heuristic benchmark uses `price_bin_probs(t, params)`
(Gaussian parametric) internally regardless of the sweep value.  This is a known
limitation — the other policies are fully agnostic to the pricing model.

Re-running a single sweep: click its Run button in the corresponding tab.
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models.baseline import (
    BaselineParams, transition_probs as _baseline_transition_probs,
    consumption as _baseline_consumption,
    backward_induction_policy, maximal_charging_policy, price_oriented_policy,
    night_charging_policy, minimum_soc_policy, always_minimum_policy, dp_heuristic_policy,
    simulate_policy_rollout as _baseline_simulate_rollout, rollout_metrics,
)
from models.negative_binomial_trips import (
    NegBinParams, simulate_policy_rollout as _negbin_simulate_rollout,
)
from models.negative_binomial_trips.backward_induction import (
    backward_induction as _negbin_run_bi,
)
from models.model_utils import mean_price, price_bin_probs as _gaussian_pbp
from utils.backward_induction import backward_induction as _baseline_run_bi
from pricing_models.pricing import GaussianBinnedSampler, make_price_bin_probs_fn
from pricing_models.entsoe_loader import load_prices


# ── Constants ─────────────────────────────────────────────────────────────────

from utils.viz import POLICY_COLORS    # shared canonical colour map

# Policies compared in this page (the canonical set), in legend order.
POLICY_ORDER = ["Backward induction", "Night charging", "DP heuristic", "Maximal charging",
                "Always minimum", "Price-oriented", "Minimum SoC"]

# Distinct line colours for per-swept-value overlays (cycled if more values than colours).
SWEEP_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
                 "#BBBBBB", "#882255", "#44AA99", "#999933", "#DDCC77"]

PHI_VALUES       = [0, 0.05, 1, 2, 5, 50, 100, 1000, 2000]
HORIZON_HOURS    = [24, 48, 168]

# Departure profiles: each overrides the four p_PD_* departure probabilities only
# (trip-duration p_DP_* / NegBin k,q stay at model defaults), so the sweep isolates
# the effect of *when/how often the car departs*.
DEPARTURE_PROFILES = {
    "Single morning trip": dict(p_pd_morning=0.060, p_pd_lunch=0.000, p_pd_evening=0.000, p_pd_default=0.0005),
    "Stay-at-home":        dict(p_pd_morning=0.002, p_pd_lunch=0.001, p_pd_evening=0.002, p_pd_default=0.0005),
    "All-day errands":     dict(p_pd_morning=0.015, p_pd_lunch=0.015, p_pd_evening=0.015, p_pd_default=0.0150),
}
# Years dropped when the energy-crisis sweep excludes them (the 2021–23 price spike).
CRISIS_YEARS = (2021, 2022, 2023)

# ── Mobility models ─────────────────────────────────────────────────────────────
# Each sweep can run against any of these.  The Baseline uses a 2-state mobility
# chain; the NegBin variants use a (k+1)-state phase chain (parked + k driving
# phases), so they have their own solver and rollout dynamics.
MODEL_LABELS    = ["Baseline", "NegBin trips (fixed k)", "NegBin trips (sampled k)"]
BASELINE_MODEL  = MODEL_LABELS[0]   # mobility model the non-mobility sweeps hold fixed
NEGBIN_LAMBDA_K = 5.0   # mean phases for the Poisson-sampled-k NegBin model


def _poisson_kmax(lambda_k: float, quantile: float = 0.999) -> int:
    """Smallest k_max with P(X ≤ k_max) ≥ quantile for X ~ Poisson(lambda_k)."""
    import math
    pmf = math.exp(-lambda_k)
    cdf = pmf
    k   = 0
    while cdf < quantile:
        k   += 1
        pmf *= lambda_k / k
        cdf += pmf
    return max(k, 1)


# ── Pure computation helpers ───────────────────────────────────────────────────

def _build_params(model_label: str, **overrides):
    """Build the params object for the selected mobility model, applying overrides.

    Baseline → BaselineParams.  NegBin variants → NegBinParams; the sampled-k
    variant additionally sets lambda_k (Poisson mean) and truncates the phase
    chain at the 99.9th-percentile k.
    """
    if model_label == "Baseline":
        return BaselineParams(**overrides)
    if model_label == "NegBin trips (sampled k)":
        return NegBinParams(**overrides, lambda_k=NEGBIN_LAMBDA_K,
                            k=_poisson_kmax(NEGBIN_LAMBDA_K))
    return NegBinParams(**overrides)  # fixed-k NegBin (lambda_k=None by default)


def _rollout_fn(model_label: str):
    """Return the model-specific simulate_policy_rollout."""
    return _baseline_simulate_rollout if model_label == "Baseline" else _negbin_simulate_rollout


def _solve(model_label: str, params, pbp_fn, T: int, N_e: int):
    """Run the model-appropriate backward induction; returns (pi, actions, e_grid, lam_grid).

    The NegBin solver carries the phase chain internally, so it takes only
    price_bin_probs_fn; the Baseline solver is parameterised by transition and
    consumption functions.
    """
    if model_label == "Baseline":
        _, pi, actions, e_grid, lam_grid = _baseline_run_bi(
            params,
            transition_probs_fn=lambda t: _baseline_transition_probs(t, params),
            consumption_fn=lambda chi: _baseline_consumption(chi, params),
            price_bin_probs_fn=pbp_fn,
            T=T, N_e=N_e,
        )
    else:
        _, pi, actions, e_grid, lam_grid = _negbin_run_bi(
            params, price_bin_probs_fn=pbp_fn, T=T, N_e=N_e,
        )
    return pi, actions, e_grid, lam_grid


def _make_scenario(params, seed: int, horizon: int,
                   sampler=None, season: str = "winter", is_weekend: bool = False) -> dict:
    """
    Generate one rollout scenario.  Uses separate sub-seeds for mobility and
    prices so that mobility draws are identical across pricing-model comparisons
    (common random numbers for variance reduction).
    """
    rng_mob = np.random.default_rng([seed, 0])
    rng_lam = np.random.default_rng([seed, 1])
    rng_e0  = np.random.default_rng([seed, 2])
    mobility_draws = rng_mob.random(horizon)
    phase_draws    = rng_mob.random(horizon)
    e0 = float(rng_e0.uniform(params.e_min, params.e_max))   # random initial SoC per scenario
    if sampler is None:
        lam_path = np.array([
            max(0.0, float(rng_lam.normal(mean_price(t, params), params.sigma_lambda)))
            for t in range(horizon)
        ])
    else:
        dow = 5 if is_weekend else 0
        lam_path = np.array([
            max(0.0, sampler.sample(dow, (t // 60) % 24, season, rng=rng_lam))
            for t in range(horizon)
        ])
    return {"lam_path": lam_path, "mobility_draws": mobility_draws,
            "phase_draws": phase_draws, "e0": e0}


def _run_rollouts(pi, actions, e_grid, params, scenarios: list, rollout_fn, pbp_fn) -> dict:
    """
    Run all four policies on each scenario using the model-specific rollout_fn.
    The DP heuristic is fed the active world's price distribution (pbp_fn) so it
    judges prices against the same distribution the world is drawn from, not the
    hard-coded Gaussian-parametric one.
    Returns {policy_name: [metrics_dict, ...]} for N_rollouts scenarios.
    """
    chi0 = 0  # start parked

    # (name, fn, extra policy kwargs) — thresholds default to baseline price/SoC values
    benchmarks = [
        ("Night charging",   night_charging_policy,   {}),
        ("DP heuristic",     dp_heuristic_policy,      {"price_bin_probs_fn": pbp_fn}),
        ("Maximal charging", maximal_charging_policy,  {}),
        ("Always minimum",   always_minimum_policy,    {}),
        ("Price-oriented",   price_oriented_policy,
         {"low_threshold": params.price_night, "high_threshold": params.price_evening}),
        ("Minimum SoC",      minimum_soc_policy,       {"soc_threshold": params.e_max * 0.25}),
    ]

    results: dict[str, list] = {p: [] for p in POLICY_ORDER}
    for sc in scenarios:
        e0 = float(sc["e0"])   # random initial SoC, shared by all policies in this scenario
        # Backward induction (optimal)
        ro = rollout_fn(
            backward_induction_policy, sc, e0, chi0, params,
            pi=pi, actions=actions, e_grid=e_grid,
        )
        results["Backward induction"].append(rollout_metrics(ro, params))

        # Benchmark policies
        for name, fn, kw in benchmarks:
            ro = rollout_fn(fn, sc, e0, chi0, params, **kw)
            results[name].append(rollout_metrics(ro, params))

    return results


def _costs(rollout_results: dict[str, list], policy: str) -> np.ndarray:
    return np.array([m["Total cost (€)"] for m in rollout_results[policy]])


def _mean_u(rollout_results: dict[str, list], policy: str) -> float:
    return float(np.mean([m["Mean charge rate while parked (kW)"] for m in rollout_results[policy]]))


def _opt_rates_averaged(pi, actions, params: BaselineParams, pbp_fn, T: int):
    """u*(t, e) averaged over price bins for parked state (chi=0)."""
    desired = actions[pi[:, 0, :, :]]                              # (T, N_e, K)
    weights = np.array([pbp_fn(t) for t in range(T)])             # (T, K)
    avg = (desired * weights[:, np.newaxis, :]).sum(axis=2)        # (T, N_e)
    return np.clip(avg, 0.0, params.u_max)


def _bin_heatmap(rates, e_grid, T: int, time_bin_min: int, battery_bin_kwh: float,
                 e_min: float, e_max: float):
    """Aggregate (T, N_e) charge rates into time × battery bins.

    Returns (z, t_centers, b_centers) with z shaped (n_battery_bins, n_time_bins) for a
    heatmap (y = battery, x = time).
    """
    n_t    = max(1, T // time_bin_min)
    usable = n_t * time_bin_min
    rt     = rates[:usable].reshape(n_t, time_bin_min, rates.shape[1]).mean(axis=1)  # (n_t, N_e)

    edges = np.arange(e_min, e_max + battery_bin_kwh, battery_bin_kwh)
    n_b   = len(edges) - 1
    z     = np.full((n_t, n_b), np.nan)
    for i in range(n_b):
        lo, hi = edges[i], edges[i + 1]
        mask = (e_grid >= lo) & (e_grid < hi) if i < n_b - 1 else (e_grid >= lo) & (e_grid <= hi)
        if mask.any():
            z[:, i] = rt[:, mask].mean(axis=1)
    t_centers = (np.arange(n_t) + 0.5) * time_bin_min / 60
    b_centers = (edges[:-1] + edges[1:]) / 2
    return z.T, t_centers, b_centers


def _run_sweep_step(model_label: str, label: str, params, pbp_fn,
                    T: int, N_e: int, N_rollouts: int, seed: int,
                    sampler=None, season: str = "winter", is_weekend: bool = False) -> dict:
    """Solve + run rollouts for one sweep configuration."""
    pi, actions, e_grid, lam_grid = _solve(model_label, params, pbp_fn, T, N_e)
    scenarios = [
        _make_scenario(params, seed + i, T, sampler=sampler,
                       season=season, is_weekend=is_weekend)
        for i in range(N_rollouts)
    ]
    rollouts = _run_rollouts(pi, actions, e_grid, params, scenarios, _rollout_fn(model_label), pbp_fn)
    # Keep one full optimal-policy trajectory (first scenario) for the rollout plot.
    sample_rollout = _rollout_fn(model_label)(
        backward_induction_policy, scenarios[0], float(scenarios[0]["e0"]), 0, params,
        pi=pi, actions=actions, e_grid=e_grid,
    )
    return {
        "model":   model_label,
        "label":   label,
        "params":  params,
        "pbp_fn":  pbp_fn,
        "pi":      pi,
        "actions": actions,
        "e_grid":  e_grid,
        "lam_grid": lam_grid,
        "T":       T,
        "rollouts": rollouts,
        "sample_rollout": sample_rollout,
    }


# ── Sweep orchestrators ────────────────────────────────────────────────────────

# The pricing sweeps fix the model to the real-data Gaussian-bins sampler and vary the
# context (season / day-type / crisis inclusion) instead of comparing price models.

def _gbins_step(label: str, sampler, season: str, is_weekend: bool,
                N_rollouts: int, N_e: int, seed: int) -> dict:
    """One Gaussian-bins sweep step at a fixed (season, day-type) context."""
    params = _build_params(BASELINE_MODEL)
    pbp_fn = make_price_bin_probs_fn(sampler, params, season, is_weekend)
    return _run_sweep_step(
        BASELINE_MODEL, label, params, pbp_fn, T=24 * 60, N_e=N_e,
        N_rollouts=N_rollouts, seed=seed,
        sampler=sampler, season=season, is_weekend=is_weekend,
    )


def sweep_pricing_season(sampler, N_rollouts: int, N_e: int, seed: int,
                         progress_cb=None) -> list[dict]:
    """Gaussian bins (crisis-excluded): vary season, held at weekday."""
    seasons = ["winter", "spring", "summer", "autumn"]
    results = []
    for i, s in enumerate(seasons):
        if progress_cb:
            progress_cb(i / len(seasons), f"Solving {s}…")
        results.append(_gbins_step(s.capitalize(), sampler, s, False, N_rollouts, N_e, seed))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_pricing_daytype(sampler, N_rollouts: int, N_e: int, seed: int,
                          progress_cb=None) -> list[dict]:
    """Gaussian bins (crisis-excluded): vary weekday/weekend, held at spring."""
    combos = [("Weekday", False), ("Weekend", True)]
    results = []
    for i, (label, we) in enumerate(combos):
        if progress_cb:
            progress_cb(i / len(combos), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", we, N_rollouts, N_e, seed))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_pricing_crisis(sampler_excl, sampler_incl, N_rollouts: int, N_e: int, seed: int,
                         progress_cb=None) -> list[dict]:
    """Gaussian bins: vary crisis inclusion, held at spring + weekday."""
    items = [("Excl. crisis", sampler_excl), ("Incl. crisis", sampler_incl)]
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb:
            progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_rollouts, N_e, seed))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_penalty(
    model_label: str, N_rollouts: int, N_e: int, seed: int, progress_cb=None,
) -> list[dict]:
    """
    Sweep φ ∈ PHI_VALUES.  Uses Gaussian parametric pricing.
    Returns a list of sweep-step result dicts, one per φ value.
    """
    results = []
    for i, phi in enumerate(PHI_VALUES):
        if progress_cb:
            progress_cb(i / len(PHI_VALUES), f"Solving φ = {phi} €/h…")
        params = _build_params(model_label, phi=float(phi))
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        result = _run_sweep_step(
            model_label, f"φ={phi}", params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed,
        )
        results.append(result)
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_horizon(
    model_label: str, N_rollouts: int, N_e: int, seed: int, progress_cb=None,
) -> list[dict]:
    """
    Compare T ∈ {24h, 48h, 168h}.  Uses Gaussian parametric pricing.
    Returns a list of sweep-step result dicts, one per horizon.
    """
    results = []
    for i, T_h in enumerate(HORIZON_HOURS):
        if progress_cb:
            progress_cb(i / len(HORIZON_HOURS), f"Solving T = {T_h} h…")
        params = _build_params(model_label)
        T      = T_h * 60
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        result = _run_sweep_step(
            model_label, f"{T_h} h", params, pbp_fn, T=T, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed,
        )
        results.append(result)
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_departure_profiles(
    model_label: str, N_rollouts: int, N_e: int, seed: int, progress_cb=None,
) -> list[dict]:
    """
    Compare departure profiles (p_PD_* overrides) over a 24 h horizon.
    Uses Gaussian parametric pricing.  All other params at baseline.
    Returns a list of sweep-step result dicts, one per profile.
    """
    results = []
    profiles = list(DEPARTURE_PROFILES.items())
    for i, (label, overrides) in enumerate(profiles):
        if progress_cb:
            progress_cb(i / len(profiles), f"Solving {label}…")
        params = _build_params(model_label, **overrides)
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        result = _run_sweep_step(
            model_label, label, params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed,
        )
        results.append(result)
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_mobility_models(
    N_rollouts: int, N_e: int, seed: int, progress_cb=None,
) -> list[dict]:
    """
    Compare the mobility models (Baseline · NegBin fixed-k · NegBin sampled-k) over a
    24 h horizon.  Uses Gaussian parametric pricing.  All other params at baseline.
    Returns a list of sweep-step result dicts, one per model.
    """
    def _label(model, params):
        if model == "Baseline":
            return "Baseline"
        if getattr(params, "lambda_k", None) is not None:   # Poisson-sampled k
            return f"NegBin (Poisson k̄={params.lambda_k:g}, k_max={params.k})"
        return f"NegBin (fixed k={params.k})"

    results = []
    for i, model in enumerate(MODEL_LABELS):
        if progress_cb:
            progress_cb(i / len(MODEL_LABELS), f"Solving {model}…")
        params = _build_params(model)
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        result = _run_sweep_step(
            model, _label(model, params), params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed,
        )
        results.append(result)
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


# ── Figure factories ───────────────────────────────────────────────────────────

def _grid_dims(n: int) -> tuple[int, int]:
    cols = 2 if n == 4 else min(n, 3)   # 4 panels → 2×2 (e.g. the season sweep)
    rows = int(np.ceil(n / cols))
    return rows, cols


def fig_heatmap_grid(results: list[dict], ncols: int = 1, time_bin_min: int = 1,
                     battery_bin_kwh: float = 0.5) -> go.Figure:
    """Optimal-policy heatmaps (price-averaged). ncols=1 → one per row; ncols>1 → grid."""
    n    = len(results)
    rows = int(np.ceil(n / ncols))
    fig = make_subplots(
        rows=rows, cols=ncols, subplot_titles=[r["label"] for r in results],
        horizontal_spacing=0.08 if ncols > 1 else 0.0,
        vertical_spacing=min(0.08, 0.8 / max(rows, 1)),
    )
    for idx, r in enumerate(results):
        row = idx // ncols + 1
        col = idx %  ncols + 1
        T   = r["T"]
        rates = _opt_rates_averaged(r["pi"], r["actions"], r["params"], r["pbp_fn"], T)
        z, t_centers, b_centers = _bin_heatmap(
            rates, r["e_grid"], T, time_bin_min, battery_bin_kwh,
            r["params"].e_min, r["params"].e_max,
        )
        fig.add_trace(go.Heatmap(
            x=t_centers, y=b_centers, z=z,
            zmin=0, zmax=r["params"].u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.01) if idx == 0 else None,
            hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u*: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Hour (h)" if row == rows else "", range=[0, T // 60],
                         row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "", row=row, col=col)
    fig.update_layout(height=280 * rows + 60, margin=dict(l=40, r=60, t=40, b=40))
    return fig


def fig_policy_price_grid(results: list[dict], hour: int) -> go.Figure:
    """Battery × price decision map per swept value: u*(e, λ̂) at a fixed hour, parked state.

    This is the view the (t, e) heatmap can't show — it keeps the price axis instead of
    averaging it out, so the policy's price response (charge-vs-defer threshold) is visible.
    """
    n = len(results)
    rows, cols = _grid_dims(n)
    titles = [r["label"] for r in results]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )
    for idx, r in enumerate(results):
        row = idx // cols + 1
        col = idx % cols  + 1
        t   = min(hour * 60, r["T"] - 1)
        z   = r["actions"][r["pi"][t, 0, :, :]]   # (N_e, K) — parked state
        fig.add_trace(go.Heatmap(
            x=r["lam_grid"],
            y=r["e_grid"],
            z=z,
            zmin=0, zmax=r["params"].u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.01) if idx == 0 else None,
            hovertemplate="Price: %{x:.3f} €/kWh<br>Battery: %{y:.1f} kWh<br>u*: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == rows else "", row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "", row=row, col=col)
    fig.update_layout(height=350 * rows + 60, margin=dict(l=40, r=60, t=40, b=40))
    return fig


def _charge_threshold(pi, actions, lam_grid, T: int) -> np.ndarray:
    """(T, N_e) highest price (€/kWh) at which the parked policy still charges, NaN if never."""
    u        = actions[pi[:, 0, :, :]]                  # (T, N_e, K)
    charging = u > 0
    last_k   = (charging * (np.arange(len(lam_grid)) + 1)).max(axis=2) - 1   # (T, N_e)
    return np.where(charging.any(axis=2),
                    lam_grid[np.clip(last_k, 0, len(lam_grid) - 1)], np.nan)


def fig_policy_threshold_grid(results: list[dict]) -> go.Figure:
    """Charging price-threshold per swept value: charge while current price ≤ colour."""
    import warnings
    n = len(results)
    rows, cols = _grid_dims(n)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[r["label"] for r in results],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for idx, r in enumerate(results):
        row = idx // cols + 1
        col = idx % cols  + 1
        T   = r["T"]
        thr = _charge_threshold(r["pi"], r["actions"], r["lam_grid"], T)   # (T, N_e)
        tb  = max(1, T // 48)
        n_t = T // tb
        with warnings.catch_warnings():           # all-NaN time bins are expected
            warnings.simplefilter("ignore")
            thr_b = np.nanmean(thr[:n_t * tb].reshape(n_t, tb, -1), axis=1)   # (n_t, N_e)
        t_centers = (np.arange(n_t) + 0.5) * tb / 60
        fig.add_trace(go.Heatmap(
            x=t_centers, y=r["e_grid"], z=thr_b.T,
            colorscale="Viridis",
            showscale=(idx == 0),
            colorbar=dict(title="€/kWh", x=1.01) if idx == 0 else None,
            hovertemplate="Hour: %{x:.1f}<br>Battery: %{y:.1f} kWh<br>charge if price ≤ %{z:.3f} €/kWh<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Hour (h)" if row == rows else "", range=[0, T // 60],
                         row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "", row=row, col=col)
    fig.update_layout(height=350 * rows + 60, margin=dict(l=40, r=60, t=40, b=40))
    return fig


def _charge_battery_ceiling(pi, actions, e_grid, t: int) -> np.ndarray:
    """(K,) highest battery level at which the parked policy still charges at each price bin.

    The charge/no-charge border in the (price, battery) plane: at price bin k the policy
    charges iff battery ≤ ceiling[k].  NaN where it never charges at that price.
    """
    charging = actions[pi[t, 0, :, :]] > 0                        # (N_e, K)
    e_rank   = (np.arange(len(e_grid)) + 1)[:, np.newaxis]        # (N_e, 1)
    top_e    = (charging * e_rank).max(axis=0) - 1                # (K,) highest charging e-index
    return np.where(charging.any(axis=0),
                    e_grid[np.clip(top_e, 0, len(e_grid) - 1)], np.nan)


def fig_charge_boundary_grid(results: list[dict]) -> go.Figure:
    """Charge/no-charge border in the (price, battery) plane, one curve per hour of the day.

    Distils each price-decision map to just its red/blue boundary, overlaid for all 24 h so
    the border's daily movement is visible (charge below each curve, defer above).
    """
    from plotly.colors import sample_colorscale
    n = len(results)
    rows, cols = _grid_dims(n)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[r["label"] for r in results],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for idx, r in enumerate(results):
        row = idx // cols + 1
        col = idx % cols  + 1
        n_h = min(24, r["T"] // 60)
        for h in range(n_h):
            ceil  = _charge_battery_ceiling(r["pi"], r["actions"], r["e_grid"], h * 60)
            color = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
            fig.add_trace(go.Scatter(
                x=r["lam_grid"], y=ceil, mode="lines", line=dict(color=color, width=1.3),
                showlegend=False,
                hovertemplate=f"Hour {h:02d}:00<br>Price %{{x:.3f}} €/kWh<br>charge if battery ≤ %{{y:.1f}} kWh<extra></extra>",
            ), row=row, col=col)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == rows else "", row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         range=[0, r["params"].e_max], row=row, col=col)
    # Hour colour bar (dummy trace; lines themselves can't carry a colour scale)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.01)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.update_layout(height=350 * rows + 60, margin=dict(l=40, r=60, t=40, b=40))
    return fig


def fig_breakeven_penalty(phi_results: list[dict], soc_fracs=(0.25, 0.5, 0.75)) -> go.Figure:
    """Break-even penalty along a sampled price path (penalty sweep only).

    Reuses the per-φ optimal policies already solved by the sweep.  At each minute of the
    sampled price path, and for a few fixed battery levels, reports the smallest swept φ at
    which the parked optimal policy would charge — i.e. how much unserved-driving penalty it
    takes to justify charging *now*, given the current price and SoC.
    """
    rs = sorted(phi_results, key=lambda r: r["params"].phi)
    if not rs or rs[0].get("sample_rollout") is None:
        return go.Figure()
    phis     = np.array([r["params"].phi for r in rs])
    p0       = rs[0]["params"]
    e_grid   = rs[0]["e_grid"]
    T        = rs[0]["T"]
    lam_traj = np.asarray(rs[0]["sample_rollout"]["lam_traj"])
    k_t      = np.clip((lam_traj / (p0.lambda_max / p0.K)).astype(int), 0, p0.K - 1)   # (T,)
    t_idx    = np.arange(T)
    hours    = t_idx / 60

    fig = go.Figure()
    for i, frac in enumerate(soc_fracs):
        e_idx   = int(np.argmin(np.abs(e_grid - frac * p0.e_max)))
        charges = np.array([r["actions"][r["pi"][t_idx, 0, e_idx, k_t]] > 0 for r in rs])  # (n_φ, T)
        any_c   = charges.any(axis=0)
        phi_star = np.where(any_c, phis[np.argmax(charges, axis=0)], np.nan)
        phi_star = np.where(phi_star > 0, phi_star, np.nan)   # guard log axis (φ=0 never charges)
        fig.add_trace(go.Scatter(
            x=hours, y=phi_star, mode="lines", name=f"SoC {int(frac * 100)}%",
            line=dict(color=SWEEP_PALETTE[i], width=2, shape="hv"),
            hovertemplate=f"Hour %{{x:.1f}}<br>need φ ≥ %{{y}} €/h<extra>SoC {int(frac*100)}%</extra>",
        ))
    fig.add_trace(go.Scatter(x=hours, y=lam_traj, mode="lines", name="price λ_t",
                             line=dict(color="#bbbbbb", width=1), yaxis="y2", hoverinfo="skip"))
    fig.update_layout(
        xaxis_title="Hour (h)", height=440, margin=dict(l=55, r=55, t=30, b=40),
        yaxis=dict(title="φ required to charge (€/h)", type="log"),
        yaxis2=dict(title="price (€/kWh)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_rollout_trajectories(results: list[dict]) -> go.Figure:
    """One sample optimal-policy trajectory per swept value: price path (top) + battery SoC (bottom)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.4, 0.6],
        subplot_titles=("Sample price path (λ_t)", "Battery SoC under the optimal policy"),
    )
    for i, r in enumerate(results):
        ro = r.get("sample_rollout")
        if ro is None:
            continue
        color = SWEEP_PALETTE[i % len(SWEEP_PALETTE)]
        hours = np.arange(len(ro["e_traj"])) / 60
        fig.add_trace(go.Scatter(
            x=hours, y=ro["lam_traj"], mode="lines", legendgroup=r["label"], showlegend=False,
            line=dict(color=color, width=1, shape="hv"), name=r["label"],
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hours, y=ro["e_traj"], mode="lines", legendgroup=r["label"],
            line=dict(color=color, width=2), name=r["label"],
        ), row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.05))
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Battery (kWh)", row=2, col=1)
    fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
    return fig


def fig_cost_distribution(results: list[dict], log_y: bool = True,
                          x_label: str = "Swept value", error: str = "sem") -> go.Figure:
    """Mean total cost (incl. penalty) over sampled rollouts, grouped bars per swept value.

    error: "sem" → ±1 standard error of the mean (uncertainty of the plotted mean; default);
           "std" → ±1 standard deviation (spread of individual trips).
    The lower error bar is clamped at 0 (cost cannot be negative).  A log y-axis (default)
    keeps low-cost policies readable when a high-variance benchmark dominates the scale.
    """
    labels = [r["label"] for r in results]
    # Order policies ascending by overall mean cost (across swept values) → cheapest bar first.
    overall = {p: float(np.mean([np.mean(_costs(r["rollouts"], p)) for r in results]))
               for p in POLICY_ORDER}
    ordered_policies = sorted(POLICY_ORDER, key=lambda p: overall[p])
    fig = go.Figure()
    for policy in ordered_policies:
        means, errs = [], []
        for r in results:
            costs = _costs(r["rollouts"], policy)
            n = len(costs)
            sd = float(np.std(costs, ddof=1)) if n > 1 else 0.0
            means.append(float(np.mean(costs)))
            errs.append(sd / np.sqrt(n) if (error == "sem" and n > 0) else sd)
        minus = [min(e, m) for m, e in zip(means, errs)]   # cost ≥ 0 → don't dip below 0
        fig.add_trace(go.Bar(
            x=labels, y=means, name=policy, marker_color=POLICY_COLORS[policy],
            error_y=dict(type="data", symmetric=False, array=errs, arrayminus=minus,
                         visible=True, thickness=1.2, width=4),
            hovertemplate="%{x}<br>mean %{y:.3f} € (± %{error_y.array:.3f})<extra>" + policy + "</extra>",
        ))
    yaxis = dict(title="Mean total cost incl. penalty (€)" + ("  [log]" if log_y else ""),
                 type="log" if log_y else "linear")
    if log_y:
        yaxis["dtick"] = 1   # one labelled tick per decade (10ⁿ) — drop the 2·/5· minor labels
    fig.update_layout(
        barmode="group",
        xaxis_title=x_label,
        yaxis=yaxis,
        height=440,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """One row per (swept_value, policy) with key metrics."""
    rows = []
    for r in results:
        for policy in POLICY_ORDER:
            costs = _costs(r["rollouts"], policy)
            rows.append({
                "Swept value":         r["label"],
                "Policy":              policy,
                "Mean cost (€)":       round(float(np.mean(costs)), 4),
                "Std cost (€)":        round(float(np.std(costs, ddof=1)), 4),
                "Mean u parked (kW)":  round(_mean_u(r["rollouts"], policy), 3),
            })
    return pd.DataFrame(rows)


# ── Streamlit helpers ──────────────────────────────────────────────────────────

def _get_gbins(exclude_crisis: bool) -> GaussianBinnedSampler:
    """Fitted Gaussian-bins sampler, cached per crisis setting (loads ENTSO-E data once)."""
    key = "sa_gbins_excl" if exclude_crisis else "sa_gbins_incl"
    if key not in st.session_state:
        if "sa_price_df" not in st.session_state:
            with st.spinner("Loading ENTSO-E price data…"):
                st.session_state["sa_price_df"] = load_prices()
        df = st.session_state["sa_price_df"]
        if exclude_crisis:
            df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
        with st.spinner(f"Fitting Gaussian-bins price model ({'excl.' if exclude_crisis else 'incl.'} crisis)…"):
            st.session_state[key] = GaussianBinnedSampler().fit(df)
    return st.session_state[key]


def _paper_config(filename: str) -> dict:
    """Plotly config so the modebar download button exports a clean, paper-ready vector SVG."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 4},
    }


def _chart(fig, filename: str):
    """Render a chart full-width with a paper-ready SVG download configured."""
    st.plotly_chart(fig, use_container_width=True, config=_paper_config(filename))


# (session-state key, folder name) for every sweep that can hold results.
_SWEEP_RESULT_KEYS = [
    ("sa_pricing_season_results",  "pricing_season"),
    ("sa_pricing_daytype_results", "pricing_daytype"),
    ("sa_pricing_crisis_results",  "pricing_crisis"),
    ("sa_phi_results",             "penalty"),
    ("sa_horizon_results",         "horizon"),
    ("sa_departure_results",       "departure_profile"),
    ("sa_mobility_results",        "mobility_model"),
]


def _build_figures_zip() -> tuple[bytes, int]:
    """Render every computed sweep's figures to high-res PNG, bundled into a ZIP.

    Returns (zip_bytes, n_figures).  One folder per sweep that has results.
    """
    import io, zipfile
    buf, n = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, folder in _SWEEP_RESULT_KEYS:
            results = st.session_state.get(key)
            if not results:
                continue
            figs = {
                "policy_heatmaps": fig_heatmap_grid(
                    results, ncols={"sa_phi_results": 3, "sa_pricing_season_results": 2}.get(key, 1)),
                "policy_vs_price": fig_policy_price_grid(results, 18),
                "charge_border":   fig_charge_boundary_grid(results),
                "price_threshold": fig_policy_threshold_grid(results),
                "cost":            fig_cost_distribution(results),
            }
            if any(r.get("sample_rollout") for r in results):
                figs["sample_rollout"] = fig_rollout_trajectories(results)
            if key == "sa_phi_results":
                figs["breakeven_penalty"] = fig_breakeven_penalty(results)
            for name, fig in figs.items():
                h = int(fig.layout.height or 500)
                zf.writestr(f"{folder}/{name}.png",
                            fig.to_image(format="png", width=1400, height=h, scale=2))
                n += 1
    return buf.getvalue(), n


# ── Save / load snapshots ───────────────────────────────────────────────────────
# Snapshots persist a fully-computed analysis to disk so plot tweaks (labels, axes,
# colours) don't require re-solving.  The only non-picklable field in a result dict is
# `pbp_fn` (a closure); it's used at render time only as pbp_fn(t) for t in range(T),
# so we store a precomputed (T, K) weight table and rebuild the lookup on load.

SAVED_DIR = Path(__file__).parent.parent.parent / "saved_analyses"


def _serialize_results(results: list[dict]) -> list[dict]:
    """Strip the unpicklable pbp_fn closure, replacing it with a (T, K) weight table."""
    out = []
    for r in results:
        d = {k: v for k, v in r.items() if k != "pbp_fn"}
        d["pbp_weights"] = np.stack([r["pbp_fn"](t) for t in range(r["T"])])
        out.append(d)
    return out


def _deserialize_results(raw: list[dict]) -> list[dict]:
    """Rebuild pbp_fn as a lookup over the stored (T, K) weight table."""
    out = []
    for d in raw:
        r = {k: v for k, v in d.items() if k != "pbp_weights"}
        w = d["pbp_weights"]
        r["pbp_fn"] = lambda t, w=w: w[t]
        out.append(r)
    return out


def _save_snapshot(name: str) -> Path:
    """Pickle every computed sweep (gzip-compressed) under a timestamped file name."""
    import gzip, pickle, datetime
    bundle = {
        "meta": {
            "name":       name,
            "saved_at":   datetime.datetime.now().isoformat(timespec="seconds"),
            "N_rollouts": st.session_state.get("sa_N_rollouts"),
            "N_e":        st.session_state.get("sa_N_e"),
            "seed":       st.session_state.get("sa_seed"),
            "sweeps":     [],
        },
        "sweeps": {},
    }
    for key, _folder in _SWEEP_RESULT_KEYS:
        results = st.session_state.get(key)
        if results:
            bundle["sweeps"][key] = _serialize_results(results)
            bundle["meta"]["sweeps"].append(key)

    SAVED_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "snapshot"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAVED_DIR / f"{safe}_{stamp}.pkl.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _list_snapshots() -> list[Path]:
    """Saved snapshot files, newest first."""
    if not SAVED_DIR.exists():
        return []
    return sorted(SAVED_DIR.glob("*.pkl.gz"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_snapshot(path: Path) -> dict:
    """Read a snapshot and restore every sweep into session_state. Returns its meta dict."""
    import gzip, pickle
    with gzip.open(path, "rb") as f:
        bundle = pickle.load(f)
    for key, raw in bundle["sweeps"].items():
        st.session_state[key] = _deserialize_results(raw)
    return bundle["meta"]


SWEEP_AXIS_LABEL = {
    "pricing_season":    "Season",
    "pricing_daytype":   "Day type",
    "pricing_crisis":    "Energy-crisis data",
    "penalty":           "Penalty φ (€/h)",
    "horizon":           "Horizon T (h)",
    "departure_profile": "Departure profile",
    "mobility_model":    "Mobility model",
}


def _show_results(results: list[dict], sweep_label: str):
    """Render all output plots and tables for a completed sweep."""
    if results:
        models = {r.get("model", "?") for r in results}
        st.caption("Mobility model: **varies by panel**" if len(models) > 1
                   else f"Mobility model: **{next(iter(models))}**")
    st.subheader("Policy heatmaps")
    _heatmap_ncols = {"penalty": 3, "pricing_season": 2}.get(sweep_label, 1)
    _chart(fig_heatmap_grid(results, ncols=_heatmap_ncols), f"{sweep_label}_policy_heatmaps")

    st.subheader("Optimal policy vs price")
    st.caption(
        "u*(battery × price) at the selected hour — keeps the price axis instead of averaging "
        "it out, so the charge-vs-defer price response is visible (it's flat in the heatmap above)."
    )
    ppmap_hour = st.slider("Hour of day", 0, 23, 12, key=f"sa_ppmap_hour_{sweep_label}")
    _chart(fig_policy_price_grid(results, ppmap_hour), f"{sweep_label}_policy_vs_price")

    st.subheader("Charge / no-charge border (all hours)")
    st.caption(
        "Just the charge-vs-defer boundary of the map above, drawn in the price × battery plane, "
        "with one curve per hour of the day (colour = hour). Charge below each curve, defer above — "
        "so you can see the border move across the day without picking a single hour."
    )
    _chart(fig_charge_boundary_grid(results), f"{sweep_label}_charge_border")

    st.subheader("Charging price threshold")
    st.caption(
        "Colour = the highest price at which the parked policy still charges (charge while "
        "price ≤ colour). Collapses the price response to one surface; blank = never charges."
    )
    _chart(fig_policy_threshold_grid(results), f"{sweep_label}_price_threshold")

    if any(r.get("sample_rollout") for r in results):
        st.subheader("Sample rollout (optimal policy)")
        st.caption(
            "One simulated 24 h+ trajectory per swept value — the price path it faced (top) and "
            "the resulting battery SoC under the optimal policy (bottom). Shows what the policy does."
        )
        _chart(fig_rollout_trajectories(results), f"{sweep_label}_sample_rollout")

    st.subheader("Mean cost")
    st.caption("Mean total cost per sampled trip — **including the unserved-driving penalty** — "
               "grouped by swept value, one bar per policy. Error bars: **SEM** = uncertainty of "
               "the mean (std/√N); **Std** = spread of individual trips. Lower bar clamped at 0.")
    cc1, cc2 = st.columns(2)
    with cc1:
        cost_axis = st.radio("Cost axis", ["Log", "Linear"], horizontal=True,
                             key=f"sa_cost_axis_{sweep_label}")
    with cc2:
        err_mode = st.radio("Error bars", ["SEM", "Std"], horizontal=True,
                            key=f"sa_cost_err_{sweep_label}")
    x_label = SWEEP_AXIS_LABEL.get(sweep_label, "Swept value")
    _chart(fig_cost_distribution(results, log_y=(cost_axis == "Log"), x_label=x_label,
                                 error=err_mode.lower()), f"{sweep_label}_cost")

    st.subheader("Summary table")
    df = build_summary_df(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        f"sensitivity_{sweep_label.replace(' ', '_')}.csv",
        "text/csv",
    )


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sensitivity Analysis — EV Charging MDP", layout="wide")
st.title("Sensitivity Analysis")
with st.expander("About this page", expanded=False):
    st.markdown("""
**Baseline configuration** (SharedParams / BaselineParams defaults):
battery e_max = 40 kWh · η_c = 0.95 · u_max = 11 kW · φ = 1000 €/h · K = 20 bins · λ_max = 0.30 €/kWh

**Prices** are wholesale DK1 day-ahead levels (€/kWh). The Gaussian-parametric means are fitted to
ENTSO-E data **excluding** the 2021–23 crisis; the data-driven models (bins/GMM/MDN) train on **all**
years. Negative wholesale prices (~2.6% of hours) are floored to 0.

**Policies compared:** Optimal (Backward induction) · Night charging · DP heuristic · Maximal charging · Always minimum

**Mobility model** (sidebar — applies to every sweep):
- **Baseline** — trip ~ Geom(p_DP); 2-state chain; default E[T] ≈ 11 min.
- **NegBin (fixed k)** — trip ~ NegBin(k, q); k-phase chain; default E[T] = k/q = 25 min.
- **NegBin (sampled k)** — k ~ Poisson(λ_k) drawn at each trip start; default E[T] ≈ 25 min.

> Default trip durations differ across models, so switching the model is **not** a controlled
> comparison — read each model's sweeps on their own.

**Four independent sweep dimensions** (others held at baseline):
1. **Pricing model** — Gaussian parametric · Gaussian bins · GMM · MDN
2. **Penalty φ** — {0, 100, 500, 1000, 2000, 5000, 10 000} €/h
3. **Horizon T** — {24 h, 48 h, 168 h}
4. **Season × Day type** — all 8 combinations of {winter, spring, summer, autumn} × {weekday, weekend}

> **Reading the Pricing tab:** each pricing model is solved *and* evaluated in its **own** price
> world. Compare policies *within* a column (which policy wins, optimality gap, feasibility) — not
> absolute costs *across* columns. The DP heuristic uses each world's own price distribution.
> NegBin models have more mobility states → slower solves; lower **N_e** if needed.
> Re-run a single sweep with its **Run** button.
    """)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sweep settings")

    N_rollouts = st.slider("Rollouts per config", 10, 500, 500, 10, key="sa_N_rollouts")
    N_e        = st.select_slider("Battery grid points N_e",
                                  [50, 100, 200, 500], value=500, key="sa_N_e")
    seed       = st.number_input("Random seed", 0, 9999, 42, key="sa_seed")

    st.divider()
    if st.button("▶ Run all sweeps", type="primary", use_container_width=True, key="sa_run_all"):
        st.session_state["sa_run_all_triggered"] = True
        st.rerun()
    st.caption("Computes every sweep once; results persist across tabs.")

    if st.button("📦 Build figure ZIP", use_container_width=True, key="sa_build_zip"):
        with st.spinner("Rendering all figures to PNG…"):
            data, n_figs = _build_figures_zip()
        if n_figs == 0:
            st.warning("No results yet — run a sweep first.")
        else:
            st.session_state["sa_fig_zip"] = data
            st.session_state["sa_fig_zip_n"] = n_figs
    if "sa_fig_zip" in st.session_state:
        st.download_button(
            f"⬇ Download {st.session_state['sa_fig_zip_n']} figures (ZIP)",
            st.session_state["sa_fig_zip"], "sensitivity_figures.zip",
            "application/zip", use_container_width=True, key="sa_dl_zip",
        )

    st.divider()
    st.header("Saved analyses")
    st.caption("Snapshot computed results to disk, then reload them later — so tweaking "
               "plot labels/axes doesn't require re-solving.")

    # Save: bundle every computed sweep under a user-supplied name.
    _computed = [k for k, _ in _SWEEP_RESULT_KEYS if st.session_state.get(k)]
    snap_name = st.text_input("Snapshot name", value="analysis", key="sa_snap_name")
    if st.button("💾 Save current results", use_container_width=True, key="sa_save_snap",
                 disabled=not _computed):
        path = _save_snapshot(snap_name)
        st.success(f"Saved {len(_computed)} sweep(s) → {path.name}")
    if not _computed:
        st.caption("No results computed yet — run a sweep first.")

    # Load: pick an existing snapshot and restore it into session_state.
    snaps = _list_snapshots()
    if snaps:
        labels = {p.name: p for p in snaps}
        chosen = st.selectbox("Load snapshot", list(labels), key="sa_load_pick")
        lc1, lc2 = st.columns(2)
        with lc1:
            if st.button("📂 Load", use_container_width=True, key="sa_load_snap"):
                meta = _load_snapshot(labels[chosen])
                st.session_state["sa_loaded_meta"] = meta
                st.rerun()
        with lc2:
            if st.button("🗑 Delete", use_container_width=True, key="sa_del_snap"):
                labels[chosen].unlink()
                st.rerun()
        if "sa_loaded_meta" in st.session_state:
            m = st.session_state["sa_loaded_meta"]
            st.caption(f"Loaded **{m.get('name')}** — {len(m.get('sweeps', []))} sweep(s), "
                       f"N_rollouts={m.get('N_rollouts')}, N_e={m.get('N_e')}, "
                       f"seed={m.get('seed')} · saved {m.get('saved_at')}")

# ── Run-all orchestration ─────────────────────────────────────────────────────

if st.session_state.pop("sa_run_all_triggered", False):
    bar = st.progress(0.0, text="Starting…")
    _s_excl = _get_gbins(exclude_crisis=True)
    _s_incl = _get_gbins(exclude_crisis=False)
    _steps = [
        ("Pricing · season",    "sa_pricing_season_results",
         lambda cb: sweep_pricing_season(_s_excl, N_rollouts, N_e, seed, cb)),
        ("Pricing · day-type",  "sa_pricing_daytype_results",
         lambda cb: sweep_pricing_daytype(_s_excl, N_rollouts, N_e, seed, cb)),
        ("Pricing · crisis",    "sa_pricing_crisis_results",
         lambda cb: sweep_pricing_crisis(_s_excl, _s_incl, N_rollouts, N_e, seed, cb)),
        ("Penalty",             "sa_phi_results",
         lambda cb: sweep_penalty(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Horizon",             "sa_horizon_results",
         lambda cb: sweep_horizon(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Departure",           "sa_departure_results",
         lambda cb: sweep_departure_profiles(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Mobility",            "sa_mobility_results",
         lambda cb: sweep_mobility_models(N_rollouts, N_e, seed, cb)),
    ]
    n = len(_steps)
    for i, (name, key, fn) in enumerate(_steps):
        st.session_state[key] = fn(
            lambda f, m, i=i, name=name: bar.progress((i + f) / n, text=f"{name}: {m}"))
    bar.progress(1.0, text="Rendering figure ZIP…")
    data, n_figs = _build_figures_zip()
    st.session_state["sa_fig_zip"]   = data
    st.session_state["sa_fig_zip_n"] = n_figs
    bar.empty()
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_price, tab_phi, tab_T, tab_departure, tab_mobility = st.tabs(
    ["Pricing Model", "Penalty φ", "Horizon T", "Departure Profile", "Mobility Model"]
)

# ─── Tab 1: Pricing model (Gaussian bins, real data) ──────────────────────────
with tab_price:
    st.markdown(
        "Real-data pricing via the **Gaussian-bins** model. Three sub-sweeps vary the price "
        "context one factor at a time; the others are held at baseline (spring · weekday · "
        "crisis-excluded). All use the Baseline mobility model."
    )
    sub_season, sub_daytype, sub_crisis = st.tabs(["Season", "Weekday/Weekend", "Energy crisis"])

    with sub_season:
        st.caption("Vary season — held at weekday, crisis-excluded.")
        if st.button("Run season sweep", key="sa_run_pseason"):
            st.session_state.pop("sa_pricing_season_results", None)
            sampler = _get_gbins(exclude_crisis=True)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running season sweep…"):
                st.session_state["sa_pricing_season_results"] = sweep_pricing_season(
                    sampler, N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_season_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_season_results"], "pricing_season")
        else:
            st.info("Click **Run season sweep** to compute results.")

    with sub_daytype:
        st.caption("Vary weekday vs weekend — held at spring, crisis-excluded.")
        if st.button("Run weekday/weekend sweep", key="sa_run_pdaytype"):
            st.session_state.pop("sa_pricing_daytype_results", None)
            sampler = _get_gbins(exclude_crisis=True)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running weekday/weekend sweep…"):
                st.session_state["sa_pricing_daytype_results"] = sweep_pricing_daytype(
                    sampler, N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_daytype_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_daytype_results"], "pricing_daytype")
        else:
            st.info("Click **Run weekday/weekend sweep** to compute results.")

    with sub_crisis:
        st.caption("Vary whether the 2021–23 crisis years are included in the fitted price "
                   "data — held at spring, weekday.")
        if st.button("Run energy-crisis sweep", key="sa_run_pcrisis"):
            st.session_state.pop("sa_pricing_crisis_results", None)
            s_excl = _get_gbins(exclude_crisis=True)
            s_incl = _get_gbins(exclude_crisis=False)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running energy-crisis sweep…"):
                st.session_state["sa_pricing_crisis_results"] = sweep_pricing_crisis(
                    s_excl, s_incl, N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_crisis_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_crisis_results"], "pricing_crisis")
        else:
            st.info("Click **Run energy-crisis sweep** to compute results.")

# ─── Tab 2: Penalty φ ─────────────────────────────────────────────────────────
with tab_phi:
    st.markdown(
        f"Sweeps the unserved-driving penalty φ ∈ {PHI_VALUES} €/h over a 24 h horizon.  "
        "Uses Gaussian parametric pricing.  All other params at baseline."
    )
    if st.button("Run penalty sweep", key="sa_run_phi"):
        st.session_state.pop("sa_phi_results", None)

        bar  = st.progress(0.0, text="Starting…")
        with st.spinner("Running penalty sweep…"):
            results = sweep_penalty(
                model_label=BASELINE_MODEL, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_phi_results"] = results
        bar.empty()
        st.rerun()

    if "sa_phi_results" in st.session_state:
        _show_results(st.session_state["sa_phi_results"], "penalty")
        st.subheader("Break-even penalty along a trajectory")
        st.caption(
            "Along one sampled price path, the smallest φ at which the optimal policy would "
            "charge *now* — evaluated at fixed SoC levels. Low = charging pays off even with a "
            "tiny penalty (low SoC / pre-departure); high = needs a large penalty to bother."
        )
        _chart(fig_breakeven_penalty(st.session_state["sa_phi_results"]),
               "penalty_breakeven")
    else:
        st.info("Click **Run penalty sweep** to compute results.")

# ─── Tab 3: Horizon T ─────────────────────────────────────────────────────────
with tab_T:
    st.markdown(
        f"Compares horizon lengths T ∈ {HORIZON_HOURS} h.  "
        "Uses Gaussian parametric pricing.  All other params at baseline.  "
        "Note: the 168 h solve is the slow one — and NegBin models add mobility "
        "states on top, so lower **N_e** (sidebar) if it drags."
    )
    if st.button("Run horizon sweep", key="sa_run_T"):
        st.session_state.pop("sa_horizon_results", None)

        bar  = st.progress(0.0, text="Starting…")
        with st.spinner("Running horizon sweep…"):
            results = sweep_horizon(
                model_label=BASELINE_MODEL, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_horizon_results"] = results
        bar.empty()
        st.rerun()

    if "sa_horizon_results" in st.session_state:
        _show_results(st.session_state["sa_horizon_results"], "horizon")
    else:
        st.info("Click **Run horizon sweep** to compute results.")

# ─── Tab 4: Departure profile ─────────────────────────────────────────────────
with tab_departure:
    st.markdown(
        f"Compares departure profiles {list(DEPARTURE_PROFILES)} over a 24 h horizon.  "
        "Each profile overrides only the **p_PD_*** departure probabilities — trip duration, "
        "pricing (Gaussian parametric) and all other params are held at baseline — so the "
        "differences isolate the effect of *when/how often the car departs*."
    )
    if st.button("Run departure-profile sweep", key="sa_run_departure"):
        st.session_state.pop("sa_departure_results", None)

        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running departure-profile sweep…"):
            results = sweep_departure_profiles(
                model_label=BASELINE_MODEL, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_departure_results"] = results
        bar.empty()
        st.rerun()

    if "sa_departure_results" in st.session_state:
        _show_results(st.session_state["sa_departure_results"], "departure_profile")
    else:
        st.info("Click **Run departure-profile sweep** to compute results.")

# ─── Tab 6: Mobility model ────────────────────────────────────────────────────
with tab_mobility:
    st.markdown(
        f"Compares the mobility models {MODEL_LABELS} over a 24 h horizon.  "
        "Uses Gaussian parametric pricing; all other params at baseline — so the differences "
        "isolate the effect of the *trip-duration / departure dynamics* of each model."
    )
    if st.button("Run mobility-model sweep", key="sa_run_mobility"):
        st.session_state.pop("sa_mobility_results", None)

        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running mobility-model sweep…"):
            results = sweep_mobility_models(
                N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_mobility_results"] = results
        bar.empty()
        st.rerun()

    if "sa_mobility_results" in st.session_state:
        _show_results(st.session_state["sa_mobility_results"], "mobility_model")
    else:
        st.info("Click **Run mobility-model sweep** to compute results.")
