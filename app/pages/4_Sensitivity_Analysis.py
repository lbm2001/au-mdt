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
  • Optimal (Backward Induction) — solves the MDP exactly for each config
  • Night Charging               — charges only during 00:00–06:00
  • DP-Heuristic                 — SoC-urgency rule using price CDF
  • Always-Maximum             — always charges at u_max

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
import streamlit.components.v1 as components

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
from pricing_models.pricing import (GaussianBinnedSampler, GMMSampler, MDNSampler,
                                     make_price_bin_probs_fn)
from pricing_models.entsoe_loader import load_prices


# ── Constants ─────────────────────────────────────────────────────────────────

from utils.viz import POLICY_COLORS, POLICY_ORDER    # shared canonical colours + order
from utils.trip_duration import compute_trip_durations, trip_duration_figure

# Distinct line colours for per-swept-value overlays (cycled if more values than colours).
SWEEP_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
                 "#BBBBBB", "#882255", "#44AA99", "#999933", "#DDCC77"]

PHI_VALUES       = [0, 0.05, 1, 50, 500, 5000]
BETA_VALUES      = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]   # discount factor sweep
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
    The DP-Heuristic is fed the active world's price distribution (pbp_fn) so it
    judges prices against the same distribution the world is drawn from, not the
    hard-coded Gaussian-parametric one.
    Returns {policy_name: [metrics_dict, ...]} for N_rollouts scenarios.
    """
    chi0 = 0  # start parked

    # (name, fn, extra policy kwargs) — thresholds default to baseline price/SoC values
    benchmarks = [
        ("Night Charging",   night_charging_policy,   {}),
        ("DP-Heuristic",     dp_heuristic_policy,      {"price_bin_probs_fn": pbp_fn}),
        ("Always-Maximum", maximal_charging_policy,  {}),
        ("Always-Minimum",   always_minimum_policy,    {}),
        ("Price-Oriented",   price_oriented_policy,
         {"low_threshold": params.price_night, "high_threshold": params.price_evening}),
        ("Minimum-Charge",      minimum_soc_policy,       {"soc_threshold": params.e_max * 0.25}),
    ]

    results: dict[str, list] = {p: [] for p in POLICY_ORDER}
    for sc in scenarios:
        e0 = float(sc["e0"])   # random initial SoC, shared by all policies in this scenario
        # Backward Induction (optimal)
        ro = rollout_fn(
            backward_induction_policy, sc, e0, chi0, params,
            pi=pi, actions=actions, e_grid=e_grid,
        )
        results["Backward Induction"].append(rollout_metrics(ro, params))

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
    items = [("Excluding crisis", sampler_excl), ("Including crisis", sampler_incl)]
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb:
            progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_rollouts, N_e, seed))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


def sweep_pricing_model(samplers: dict, N_rollouts: int, N_e: int, seed: int,
                        progress_cb=None) -> list[dict]:
    """Vary the price model (Gaussian bins / GMM / MDN), held at spring · weekday · crisis-excluded.

    `samplers` maps a model label to a fitted sampler; each drives its own sampled price world.
    """
    items = list(samplers.items())
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


def sweep_beta(
    model_label: str, N_rollouts: int, N_e: int, seed: int, progress_cb=None,
) -> list[dict]:
    """
    Sweep the discount factor β ∈ BETA_VALUES over a 24 h horizon.  Gaussian parametric pricing.
    Returns a list of sweep-step result dicts, one per β value.
    """
    results = []
    for i, beta in enumerate(BETA_VALUES):
        if progress_cb:
            progress_cb(i / len(BETA_VALUES), f"Solving β = {beta:g}…")
        params = _build_params(model_label, beta=float(beta))
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        result = _run_sweep_step(
            model_label, f"β={beta:g}", params, pbp_fn, T=24 * 60, N_e=N_e,
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
    Compare NegBin mobility models over a 24 h horizon: {fixed-k, Poisson-k} × {k=5, k=10}
    (4 configs).  Gaussian parametric pricing.  All other params at baseline.
    Returns a list of sweep-step result dicts, one per config.
    """
    FIXED, SAMPLED = "NegBin trips (fixed k)", "NegBin trips (sampled k)"
    configs = [
        (FIXED,   "NegBin fixed k=5",    NegBinParams(k=5)),
        (FIXED,   "NegBin fixed k=10",   NegBinParams(k=10)),
        (SAMPLED, "NegBin Poisson k=5",  NegBinParams(lambda_k=5.0,  k=_poisson_kmax(5.0))),
        (SAMPLED, "NegBin Poisson k=10", NegBinParams(lambda_k=10.0, k=_poisson_kmax(10.0))),
    ]
    results = []
    for i, (model, label, params) in enumerate(configs):
        if progress_cb:
            progress_cb(i / len(configs), f"Solving {label}…")
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model, label, params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed,
        ))
    if progress_cb:
        progress_cb(1.0, "Done.")
    return results


# ── Figure factories ───────────────────────────────────────────────────────────

def _grid_dims(n: int) -> tuple[int, int]:
    cols = 2 if n == 4 else min(n, 3)   # 4 panels → 2×2 (e.g. the season sweep)
    rows = int(np.ceil(n / cols))
    return rows, cols


def fig_heatmap_grid(results: list[dict], ncols: int = 1, time_bin_min: int = 1,
                     battery_bin_kwh: float = 0.5, show_titles: bool = True) -> go.Figure:
    """Optimal-policy heatmaps (price-averaged). ncols=1 → one per row; ncols>1 → grid.

    show_titles=False drops the per-panel labels (used for the single Baseline export).
    """
    n    = len(results)
    rows = int(np.ceil(n / ncols))
    fig = make_subplots(
        rows=rows, cols=ncols,
        subplot_titles=[r["label"] for r in results] if show_titles else None,
        horizontal_spacing=0.08 if ncols > 1 else 0.0,
        vertical_spacing=0.14 if rows > 1 else 0.0,   # room for each row's φ=… title
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
                         showticklabels=(row == rows), row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "", row=row, col=col)
    fig.update_layout(height=280 * rows + 70, margin=dict(l=40, r=60, t=55, b=40))
    for ann in fig.layout.annotations:   # lift the per-panel titles off the plots a bit
        ann.yshift = 10
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


def fig_cost_distribution(results: list[dict], log_y: bool = True,
                          x_label: str = "Swept value", error: str = "sem") -> go.Figure:
    """Mean total cost (incl. penalty) over sampled rollouts, grouped bars per swept value.

    error: "sem" → ±1 standard error of the mean (uncertainty of the plotted mean; default);
           "std" → ±1 standard deviation (spread of individual trips).
    The lower error bar is clamped at 0 (cost cannot be negative).  A log y-axis (default)
    keeps low-cost policies readable when a high-variance benchmark dominates the scale.
    """
    labels = [r["label"] for r in results]
    fig = go.Figure()
    for policy in POLICY_ORDER:   # fixed canonical order → same order/colour in every figure
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


_PRICE_MODEL_CLASSES = {
    "Gaussian bins": GaussianBinnedSampler,
    "GMM":           GMMSampler,
    "MDN":           MDNSampler,
}


def _get_price_model(model_name: str):
    """Fitted price sampler of the given type (crisis-excluded data, baseline context), cached."""
    if model_name == "Gaussian bins":
        return _get_gbins(exclude_crisis=True)   # reuse the already-cached bins fit
    key = f"sa_pmodel_{model_name}"
    if key not in st.session_state:
        if "sa_price_df" not in st.session_state:
            with st.spinner("Loading ENTSO-E price data…"):
                st.session_state["sa_price_df"] = load_prices()
        df = st.session_state["sa_price_df"]
        df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]   # baseline: crisis-excluded
        with st.spinner(f"Fitting {model_name} price model…"):
            st.session_state[key] = _PRICE_MODEL_CLASSES[model_name]().fit(df)
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
    ("sa_pricing_model_results",   "pricing_model"),
    ("sa_pricing_season_results",  "pricing_season"),
    ("sa_pricing_daytype_results", "pricing_daytype"),
    ("sa_pricing_crisis_results",  "pricing_crisis"),
    ("sa_phi_results",             "penalty"),
    ("sa_beta_results",            "beta"),
    ("sa_horizon_results",         "horizon"),
    ("sa_departure_results",       "departure_profile"),
    ("sa_mobility_results",        "mobility_model"),
]


# ── Per-model "baseline" figures (cost bar · optimal-policy heatmap · mean trajectory) ──
# Filename prefix in the (flat) baseline_models/ folder for each mobility model.
_MODEL_PREFIX = {
    "Baseline":                 "baseline",
    "NegBin trips (fixed k)":   "negbin",
    "NegBin trips (sampled k)": "negbin_poisson",
}

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"


def _rgba(hex_color: str, alpha: float) -> str:
    """rgba() string for a hex (#RRGGBB) or named colour, at the given opacity."""
    named = {"orange": "255,165,0", "lightgray": "211,211,211"}
    if hex_color.startswith("#"):
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
    return f"rgba({named.get(hex_color, '128,128,128')},{alpha})"


def _run_rollouts_full(pi, actions, e_grid, params, scenarios, rollout_fn, pbp_fn) -> dict:
    """Like _run_rollouts but keeps each raw rollout (u_traj/chi_traj/cost_traj), not summary metrics."""
    chi0 = 0
    benchmarks = [
        ("Night Charging",   night_charging_policy,   {}),
        ("DP-Heuristic",     dp_heuristic_policy,      {"price_bin_probs_fn": pbp_fn}),
        ("Always-Maximum", maximal_charging_policy,  {}),
        ("Always-Minimum",   always_minimum_policy,    {}),
        ("Price-Oriented",   price_oriented_policy,
         {"low_threshold": params.price_night, "high_threshold": params.price_evening}),
        ("Minimum-Charge",      minimum_soc_policy,       {"soc_threshold": params.e_max * 0.25}),
    ]
    out: dict[str, list] = {p: [] for p in POLICY_ORDER}
    for sc in scenarios:
        e0 = float(sc["e0"])
        out["Backward Induction"].append(rollout_fn(
            backward_induction_policy, sc, e0, chi0, params,
            pi=pi, actions=actions, e_grid=e_grid))
        for name, fn, kw in benchmarks:
            out[name].append(rollout_fn(fn, sc, e0, chi0, params, **kw))
    return out


def _baseline_cost_fig(full: dict) -> go.Figure:
    """Per-policy mean total cost (incl. penalty), log axis, ordered cheapest→dearest, ±SEM."""
    names, means, errs = list(full), [], []
    for name in names:
        costs = np.array([r["cost_traj"].sum() for r in full[name]])
        m  = len(costs)
        sd = float(costs.std(ddof=1)) if m > 1 else 0.0
        means.append(float(costs.mean()))
        errs.append(sd / np.sqrt(m) if m > 0 else 0.0)
    minus = [min(e, mu) for mu, e in zip(means, errs)]
    fig = go.Figure(go.Bar(
        x=names, y=means, marker_color=[POLICY_COLORS[n] for n in names],
        error_y=dict(type="data", symmetric=False, array=errs, arrayminus=minus,
                     visible=True, thickness=1.2, width=4)))
    fig.update_layout(yaxis=dict(title="Mean total cost incl. penalty (€)  [log]", type="log", dtick=1),
                      xaxis_title="Policy", height=460, margin=dict(l=40, r=20, t=20, b=110),
                      showlegend=False)
    fig.update_xaxes(categoryorder="array", categoryarray=POLICY_ORDER)   # fixed canonical order
    return fig


def _baseline_traj_fig(full: dict, scenarios: list, T: int, params) -> go.Figure:
    """Scenario-averaged trajectories: price, mobility, per-policy charge rate (±SEM bands)."""
    hours, T_hours = np.arange(T) / 60, T // 60
    n = max(len(scenarios), 1)
    sem = lambda a: a.std(axis=0) / np.sqrt(n)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Mean price λ̄<sub>t</sub>",
                                        "Mean mobility — 0 parked, 1 driving",
                                        "Mean charge rate u per policy"))

    def band(mean, half, color, name, row, legend=False):
        fill = _rgba(color, 0.12)
        fig.add_trace(go.Scatter(x=hours, y=mean + half, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=name), row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean - half, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=fill, showlegend=False,
                                 hoverinfo="skip", legendgroup=name), row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean, mode="lines", line=dict(color=color, width=1.6),
                                 name=name, legendgroup=name, showlegend=legend), row=row, col=1)

    P = np.array([sc["lam_path"] for sc in scenarios])
    band(P.mean(0), sem(P), "lightgray", "λ̄<sub>t</sub>", row=1)
    Mob = np.array([(r["chi_traj"] > 0).astype(float) for r in full["Backward Induction"]])
    band(Mob.mean(0), sem(Mob), "orange", "driving", row=2)
    for name, rolls in full.items():
        U = np.array([r["u_traj"] for r in rolls])
        band(U.mean(0), sem(U), POLICY_COLORS[name], name, row=3, legend=True)

    fig.update_layout(height=900, margin=dict(l=40, r=30, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(range=[0, T_hours], dtick=max(T_hours // 8, 1))
    fig.update_xaxes(title_text="Hour (h)", row=3, col=1)
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Fraction driving", tickvals=[0, 0.5, 1], row=2, col=1)
    fig.update_yaxes(title_text="u (kW)", range=[-0.2, params.u_max + 0.5], row=3, col=1)
    return fig


def _baseline_model_figs(result: dict, N_rollouts: int, seed: int) -> dict:
    """The three per-model figures: cost bar, optimal-policy heatmap, mean trajectory."""
    model, params, T, pbp_fn = result["model"], result["params"], result["T"], result["pbp_fn"]
    scenarios = [_make_scenario(params, seed + i, T) for i in range(N_rollouts)]  # Gaussian parametric
    full = _run_rollouts_full(result["pi"], result["actions"], result["e_grid"],
                              params, scenarios, _rollout_fn(model), pbp_fn)
    return {
        "baseline_cost":           _baseline_cost_fig(full),
        "baseline_optimal_policy": fig_heatmap_grid([result], show_titles=False),
        "baseline_trajectories":   _baseline_traj_fig(full, scenarios, T, params),
    }


def _export_baseline_result(model: str, N_e: int) -> dict:
    """Solve one canonical baseline-export model for single-figure export."""
    T = 24 * 60
    params = _build_params(model)
    pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
    pi, actions, e_grid, lam_grid = _solve(model, params, pbp_fn, T, N_e)
    return {"model": model, "label": model, "params": params, "pbp_fn": pbp_fn,
            "pi": pi, "actions": actions, "e_grid": e_grid, "lam_grid": lam_grid, "T": T}


def _figure_png_bytes(fig: go.Figure) -> bytes:
    """Render one export figure to high-res PNG bytes."""
    h = int(fig.layout.height or 500)
    return fig.to_image(format="png", width=1400, height=h, scale=2)


def _sweep_export_ids(key: str) -> list[str]:
    """Export figure IDs for one computed sweep."""
    return [f"sweep:{key}:{name}" for name in ("policy_heatmaps", "charge_border", "cost")]


def _auto_download_png(filename: str, data: bytes, token: str) -> None:
    """Ask the browser to download one PNG immediately, with no extra user click.

    Browsers may block repeated automatic downloads until the user allows them for the app.
    The sidebar also keeps fallback download buttons for completed run-all exports.
    """
    import base64
    import html
    import re

    element_id = "sa_dl_" + re.sub(r"[^A-Za-z0-9_-]", "_", token)
    safe_filename = html.escape(filename, quote=True)
    b64 = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
        <a id="{element_id}" download="{safe_filename}" href="data:image/png;base64,{b64}"></a>
        <script>
        const link = document.getElementById("{element_id}");
        if (link) link.click();
        </script>
        """,
        height=0,
        width=0,
    )


def _available_export_figures() -> list[dict]:
    """Figures that can currently be rendered one at a time for download."""
    items = [
        {"id": "baseline:Baseline:cost", "path": "baseline_models/baseline_cost.png",
         "label": "baseline_models / baseline_cost"},
        {"id": "baseline:Baseline:optimal_policy", "path": "baseline_models/baseline_optimal_policy.png",
         "label": "baseline_models / baseline_optimal_policy"},
        {"id": "baseline:Baseline:trajectories", "path": "baseline_models/baseline_trajectories.png",
         "label": "baseline_models / baseline_trajectories"},
        {"id": "baseline:NegBin trips (fixed k):trajectories", "path": "baseline_models/negbin_trajectories.png",
         "label": "baseline_models / negbin_trajectories"},
        {"id": "baseline:NegBin trips (sampled k):trajectories", "path": "baseline_models/negbin_poisson_trajectories.png",
         "label": "baseline_models / negbin_poisson_trajectories"},
        {"id": "trip_duration", "path": "baseline_models/trip_duration_by_model.png",
         "label": "baseline_models / trip_duration_by_model"},
    ]
    for key, folder in _SWEEP_RESULT_KEYS:
        if not st.session_state.get(key):
            continue
        for name in ("policy_heatmaps", "charge_border", "cost"):
            items.append({
                "id": f"sweep:{key}:{name}",
                "path": f"sensitivity_figures/{folder}/{name}.png",
                "label": f"sensitivity_figures / {folder} / {name}",
            })
    return items


def _render_export_figure(export_id: str) -> tuple[str, bytes]:
    """Render the selected export figure and return (relative export path, png bytes)."""
    if export_id == "trip_duration":
        path = "baseline_models/trip_duration_by_model.png"
        return path, _figure_png_bytes(trip_duration_figure(compute_trip_durations()))

    kind, *parts = export_id.split(":")
    if kind == "baseline":
        model, figure_name = parts
        result = _export_baseline_result(model, int(st.session_state.get("sa_N_e", 500)))
        if figure_name == "optimal_policy":
            fig = fig_heatmap_grid([result], show_titles=False)
        else:
            figs = _baseline_model_figs(
                result,
                int(st.session_state.get("sa_N_rollouts", 200)),
                int(st.session_state.get("sa_seed", 42)),
            )
            fig = figs[f"baseline_{figure_name}"]
        prefix = _MODEL_PREFIX[model]
        path = (f"baseline_models/{prefix}_{figure_name}.png"
                if model != "Baseline" else f"baseline_models/baseline_{figure_name}.png")
        return path, _figure_png_bytes(fig)

    if kind == "sweep":
        key, figure_name = parts
        results = st.session_state.get(key)
        if not results:
            raise ValueError("Selected sweep has no results.")
        folder = dict(_SWEEP_RESULT_KEYS)[key]
        if figure_name == "policy_heatmaps":
            ncols = {"sa_phi_results": 3, "sa_beta_results": 3, "sa_pricing_season_results": 2,
                     "sa_mobility_results": 2}.get(key, 1)
            fig = fig_heatmap_grid(results, ncols=ncols)
        elif figure_name == "charge_border":
            fig = fig_charge_boundary_grid(results)
        elif figure_name == "cost":
            fig = fig_cost_distribution(results)
        else:
            raise ValueError(f"Unknown export figure: {figure_name}")
        path = f"sensitivity_figures/{folder}/{figure_name}.png"
        return path, _figure_png_bytes(fig)

    raise ValueError(f"Unknown export id: {export_id}")


SWEEP_AXIS_LABEL = {
    "pricing_model":     "Pricing model",
    "pricing_season":    "Season",
    "pricing_daytype":   "Day type",
    "pricing_crisis":    "Energy-crisis data",
    "penalty":           "Penalty φ (€/h)",
    "beta":              "Discount factor β",
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
    _heatmap_ncols = {"penalty": 3, "beta": 3, "pricing_season": 2, "mobility_model": 2}.get(sweep_label, 1)
    _chart(fig_heatmap_grid(results, ncols=_heatmap_ncols), f"{sweep_label}_policy_heatmaps")

    st.subheader("Charge / no-charge border (all hours)")
    st.caption(
        "Just the charge-vs-defer boundary of the map above, drawn in the price × battery plane, "
        "with one curve per hour of the day (colour = hour). Charge below each curve, defer above — "
        "so you can see the border move across the day without picking a single hour."
    )
    _chart(fig_charge_boundary_grid(results), f"{sweep_label}_charge_border")

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

**Policies compared:** Optimal (Backward Induction) · Night Charging · DP-Heuristic · Always-Maximum · Always-Minimum

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
> absolute costs *across* columns. The DP-Heuristic uses each world's own price distribution.
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
    run_all_auto_export = st.checkbox(
        "Also auto-download export figures during run-all",
        value=True,
        key="sa_run_all_auto_export",
    )
    st.caption("Computes every sweep once; results persist across tabs. Each sweep's export PNGs "
               "are saved under `figures/` as soon as that sweep completes. Auto-download also "
               "asks the browser to download each PNG.")

    if st.session_state.get("sa_run_all_exports"):
        with st.expander("Completed run-all exports", expanded=False):
            for i, item in enumerate(st.session_state["sa_run_all_exports"]):
                st.download_button(
                    f"⬇ {item.get('path', item['label'])}",
                    item["data"],
                    item["filename"],
                    "image/png",
                    use_container_width=True,
                    key=f"sa_run_all_export_dl_{i}_{item['filename']}",
                )
    if st.session_state.get("sa_run_all_export_errors"):
        with st.expander("Run-all export errors", expanded=False):
            for err in st.session_state["sa_run_all_export_errors"]:
                st.caption(err)

# ── Run-all orchestration ─────────────────────────────────────────────────────

if st.session_state.pop("sa_run_all_triggered", False):
    bar = st.progress(0.0, text="Starting…")
    st.session_state["sa_run_all_exports"] = []
    st.session_state["sa_run_all_export_errors"] = []
    _s_excl = _get_gbins(exclude_crisis=True)
    _s_incl = _get_gbins(exclude_crisis=False)
    _steps = [
        ("Pricing · model",     "sa_pricing_model_results",
         lambda cb: sweep_pricing_model(
             {m: _get_price_model(m) for m in ("Gaussian bins", "GMM", "MDN")},
             N_rollouts, N_e, seed, cb)),
        ("Pricing · season",    "sa_pricing_season_results",
         lambda cb: sweep_pricing_season(_s_excl, N_rollouts, N_e, seed, cb)),
        ("Pricing · day-type",  "sa_pricing_daytype_results",
         lambda cb: sweep_pricing_daytype(_s_excl, N_rollouts, N_e, seed, cb)),
        ("Pricing · crisis",    "sa_pricing_crisis_results",
         lambda cb: sweep_pricing_crisis(_s_excl, _s_incl, N_rollouts, N_e, seed, cb)),
        ("Penalty",             "sa_phi_results",
         lambda cb: sweep_penalty(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Discount β",          "sa_beta_results",
         lambda cb: sweep_beta(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Horizon",             "sa_horizon_results",
         lambda cb: sweep_horizon(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Departure",           "sa_departure_results",
         lambda cb: sweep_departure_profiles(BASELINE_MODEL, N_rollouts, N_e, seed, cb)),
        ("Mobility",            "sa_mobility_results",
         lambda cb: sweep_mobility_models(N_rollouts, N_e, seed, cb)),
    ]
    n = len(_steps)

    def _emit_export(export_id: str, seq: int) -> int:
        labels_by_id = {item["id"]: item["label"] for item in _available_export_figures()}
        label = labels_by_id.get(export_id, export_id)
        bar.progress(min((i + 0.9) / n, 1.0), text=f"Rendering export: {label}")
        try:
            rel_path, data = _render_export_figure(export_id)
            out_path = FIGURES_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
        except Exception as exc:
            st.session_state["sa_run_all_export_errors"].append(f"{label}: {exc}")
            return seq
        st.session_state["sa_run_all_exports"].append({
            "label": label,
            "filename": rel_path.replace("/", "__"),
            "path": str(out_path.relative_to(Path(__file__).parent.parent.parent)),
            "data": data,
        })
        if run_all_auto_export:
            _auto_download_png(rel_path.replace("/", "__"), data, f"run_all_{seq}_{rel_path}")
        return seq + 1

    export_seq = 0
    for i, (name, key, fn) in enumerate(_steps):
        st.session_state[key] = fn(
            lambda f, m, i=i, name=name: bar.progress((i + f) / n, text=f"{name}: {m}"))
        for export_id in _sweep_export_ids(key):
            export_seq = _emit_export(export_id, export_seq)

    for export_id in [
        "baseline:Baseline:cost",
        "baseline:Baseline:optimal_policy",
        "baseline:Baseline:trajectories",
        "baseline:NegBin trips (fixed k):trajectories",
        "baseline:NegBin trips (sampled k):trajectories",
        "trip_duration",
    ]:
        export_seq = _emit_export(export_id, export_seq)

    bar.progress(1.0, text="Done.")
    bar.empty()
    st.success(
        f"Run-all complete. Saved {len(st.session_state['sa_run_all_exports'])} export PNG(s) "
        f"under `{FIGURES_DIR.relative_to(Path(__file__).parent.parent.parent)}/`. "
        "If the browser blocked automatic downloads, use the completed-export buttons in the sidebar."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_price, tab_phi, tab_beta, tab_T, tab_departure, tab_mobility = st.tabs(
    ["Pricing Model", "Penalty φ", "Discount β", "Horizon T", "Departure Profile", "Mobility Model"]
)

# ─── Tab 1: Pricing model (Gaussian bins, real data) ──────────────────────────
with tab_price:
    st.markdown(
        "Real-data pricing on ENTSO-E DK1 data. Four sub-sweeps vary one factor at a time; the "
        "others are held at baseline (Gaussian bins · spring · weekday · crisis-excluded). "
        "All use the Baseline mobility model."
    )
    sub_model, sub_season, sub_daytype, sub_crisis = st.tabs(
        ["Pricing model", "Season", "Weekday/Weekend", "Energy crisis"])

    with sub_model:
        st.caption("Vary the price model (Gaussian bins · GMM · MDN) — held at spring · weekday · "
                   "crisis-excluded. Each model is fitted on the same ENTSO-E data and drives its "
                   "own sampled price world. (MDN fitting trains a neural net — first run is slower.)")
        if st.button("Run pricing-model sweep", key="sa_run_pmodel"):
            st.session_state.pop("sa_pricing_model_results", None)
            samplers = {m: _get_price_model(m) for m in ("Gaussian bins", "GMM", "MDN")}
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running pricing-model sweep…"):
                st.session_state["sa_pricing_model_results"] = sweep_pricing_model(
                    samplers, N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_model_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_model_results"], "pricing_model")
        else:
            st.info("Click **Run pricing-model sweep** to compute results.")

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
    else:
        st.info("Click **Run penalty sweep** to compute results.")

# ─── Tab 3: Discount factor β ─────────────────────────────────────────────────
with tab_beta:
    st.markdown(
        f"Sweeps the discount factor β ∈ {BETA_VALUES} over a 24 h horizon.  "
        "Uses Gaussian parametric pricing.  All other params at baseline.  "
        "(Horizon T is its own sweep in the **Horizon T** tab.)"
    )
    if st.button("Run discount-β sweep", key="sa_run_beta"):
        st.session_state.pop("sa_beta_results", None)

        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running discount-β sweep…"):
            results = sweep_beta(
                model_label=BASELINE_MODEL, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_beta_results"] = results
        bar.empty()
        st.rerun()

    if "sa_beta_results" in st.session_state:
        _show_results(st.session_state["sa_beta_results"], "beta")
    else:
        st.info("Click **Run discount-β sweep** to compute results.")

# ─── Tab 4: Horizon T ─────────────────────────────────────────────────────────
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
        "Compares NegBin mobility models over a 24 h horizon: **{fixed-k, Poisson-k} × {k=5, k=10}** "
        "(4 configs).  Uses Gaussian parametric pricing; all other params at baseline — so the "
        "differences isolate the effect of the *trip-duration dynamics* (larger k → longer trips).  "
        "The Baseline (binomial) model is shown in the figure-export's `baseline_models/` instead."
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
