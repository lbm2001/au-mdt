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
    backward_induction_policy, maximal_charging_policy,
    night_charging_policy, always_minimum_policy, dp_heuristic_policy,
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
from pricing_models.pricing import (
    GaussianBinnedSampler, GMMSampler, MDNSampler,
    make_price_bin_probs_fn,
)
from pricing_models.entsoe_loader import load_prices


# ── Constants ─────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    "Optimal (BI)":     "#4477AA",
    "Night charging":   "#AA3377",
    "DP heuristic":     "#009988",
    "Maximal charging": "#228833",
    "Always minimum":   "#CCBB44",
}
POLICY_ORDER = list(POLICY_COLORS)

PHI_VALUES       = [0, 100, 500, 1000, 2000, 5000, 10_000]
HORIZON_HOURS    = [24, 48, 168]
PRICING_LABELS   = ["Gaussian (parametric)", "Gaussian bins", "GMM", "MDN"]
SAMPLER_CLASSES  = {
    "Gaussian bins": GaussianBinnedSampler,
    "GMM":           GMMSampler,
    "MDN":           MDNSampler,
}

# ── Mobility models ─────────────────────────────────────────────────────────────
# Each sweep can run against any of these.  The Baseline uses a 2-state mobility
# chain; the NegBin variants use a (k+1)-state phase chain (parked + k driving
# phases), so they have their own solver and rollout dynamics.
MODEL_LABELS    = ["Baseline", "NegBin trips (fixed k)", "NegBin trips (sampled k)"]
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


def _make_pbp_fn(label: str, params, season: str, is_weekend: bool,
                 sampler_cache: dict):
    """Return a price_bin_probs_fn for the given pricing label."""
    if label == "Gaussian (parametric)":
        return lambda t: _gaussian_pbp(t, params)
    sampler = sampler_cache[label]
    return make_price_bin_probs_fn(sampler, params, season, is_weekend)


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
    mobility_draws = rng_mob.random(horizon)
    phase_draws    = rng_mob.random(horizon)
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
    return {"lam_path": lam_path, "mobility_draws": mobility_draws, "phase_draws": phase_draws}


def _run_rollouts(pi, actions, e_grid, params, scenarios: list, rollout_fn) -> dict:
    """
    Run all four policies on each scenario using the model-specific rollout_fn.
    Returns {policy_name: [metrics_dict, ...]} for N_rollouts scenarios.
    """
    e0   = float(params.e_max / 2)
    chi0 = 0  # start parked

    results: dict[str, list] = {p: [] for p in POLICY_ORDER}
    for sc in scenarios:
        # Optimal BI
        ro = rollout_fn(
            backward_induction_policy, sc, e0, chi0, params,
            pi=pi, actions=actions, e_grid=e_grid,
        )
        results["Optimal (BI)"].append(rollout_metrics(ro, params))

        # Benchmark policies
        for name, fn in [
            ("Night charging",   night_charging_policy),
            ("DP heuristic",     dp_heuristic_policy),
            ("Maximal charging", maximal_charging_policy),
            ("Always minimum",   always_minimum_policy),
        ]:
            ro = rollout_fn(fn, sc, e0, chi0, params)
            results[name].append(rollout_metrics(ro, params))

    return results


def _costs(rollout_results: dict[str, list], policy: str) -> np.ndarray:
    return np.array([m["Total cost (€)"] for m in rollout_results[policy]])


def _feasibility(rollout_results: dict[str, list], policy: str) -> float:
    return float(np.mean([m["Penalty minutes"] == 0 for m in rollout_results[policy]]))


def _mean_u(rollout_results: dict[str, list], policy: str) -> float:
    return float(np.mean([m["Mean charge rate while parked (kW)"] for m in rollout_results[policy]]))


def _opt_gap(rollout_results: dict[str, list], benchmark: str) -> tuple[float, float]:
    """Returns (mean_gap, half_CI95) where gap = benchmark_cost - optimal_cost."""
    opt  = _costs(rollout_results, "Optimal (BI)")
    bench = _costs(rollout_results, benchmark)
    gaps  = bench - opt
    n = len(gaps)
    mean = float(np.mean(gaps))
    ci   = 1.96 * float(np.std(gaps, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return mean, ci


def _opt_rates_averaged(pi, actions, params: BaselineParams, pbp_fn, T: int):
    """u*(t, e) averaged over price bins for parked state (chi=0)."""
    desired = actions[pi[:, 0, :, :]]                              # (T, N_e, K)
    weights = np.array([pbp_fn(t) for t in range(T)])             # (T, K)
    avg = (desired * weights[:, np.newaxis, :]).sum(axis=2)        # (T, N_e)
    return np.clip(avg, 0.0, params.u_max)


def _heatmap_z(rates, T: int, N_e: int, time_bin_min: int = 60):
    """Aggregate charge rates into (n_time_bins, N_e) then return transposed for heatmap."""
    n_bins  = T // time_bin_min
    usable  = n_bins * time_bin_min
    binned  = rates[:usable].reshape(n_bins, time_bin_min, N_e).mean(axis=1)
    return binned.T  # (N_e, n_time_bins)


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
    rollouts = _run_rollouts(pi, actions, e_grid, params, scenarios, _rollout_fn(model_label))
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
    }


# ── Sweep orchestrators ────────────────────────────────────────────────────────

def sweep_pricing_models(
    model_label: str, N_rollouts: int, N_e: int, seed: int,
    season: str, is_weekend: bool,
    sampler_cache: dict, progress_cb=None,
) -> list[dict]:
    """
    Compare all pricing models for the selected mobility model.
    Returns a list of sweep-step result dicts, one per pricing label.
    """
    base = _build_params(model_label)
    results = []
    for i, label in enumerate(PRICING_LABELS):
        if progress_cb:
            progress_cb(i / len(PRICING_LABELS), f"Solving {label}…")
        sampler = sampler_cache.get(label)  # None for Gaussian parametric
        pbp_fn  = _make_pbp_fn(label, base, season, is_weekend, sampler_cache)
        result  = _run_sweep_step(
            model_label, label, base, pbp_fn, T=24 * 60, N_e=N_e, N_rollouts=N_rollouts,
            seed=seed, sampler=sampler, season=season, is_weekend=is_weekend,
        )
        results.append(result)
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


def sweep_season_weekend(
    model_label: str, N_rollouts: int, N_e: int, seed: int,
    pricing_label: str, sampler_cache: dict, progress_cb=None,
) -> list[dict]:
    """
    Compare all 8 (season, day-type) combinations using a data-driven pricing model.
    Returns a list of sweep-step result dicts, one per combination.
    """
    base   = _build_params(model_label)
    combos = [(s, w) for s in ["winter", "spring", "summer", "autumn"] for w in [False, True]]
    results = []
    for i, (season, is_weekend) in enumerate(combos):
        label = f"{season.capitalize()} · {'Weekend' if is_weekend else 'Weekday'}"
        if progress_cb:
            progress_cb(i / len(combos), f"Solving {label}…")
        sampler = sampler_cache.get(pricing_label)
        pbp_fn  = _make_pbp_fn(pricing_label, base, season, is_weekend, sampler_cache)
        result  = _run_sweep_step(
            model_label, label, base, pbp_fn, T=24 * 60, N_e=N_e, N_rollouts=N_rollouts,
            seed=seed, sampler=sampler, season=season, is_weekend=is_weekend,
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


# ── Figure factories ───────────────────────────────────────────────────────────

def _grid_dims(n: int) -> tuple[int, int]:
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    return rows, cols


def fig_heatmap_grid(results: list[dict]) -> go.Figure:
    """Policy heatmap grid: one subplot per swept value, optimal policy, price-averaged."""
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
        T    = r["T"]
        N_e  = len(r["e_grid"])
        rates = _opt_rates_averaged(r["pi"], r["actions"], r["params"], r["pbp_fn"], T)
        time_bin = max(1, T // 48)  # ~48 time slices regardless of horizon
        z    = _heatmap_z(rates, T, N_e, time_bin_min=time_bin)
        T_h  = T // 60
        t_centers = (np.arange(z.shape[1]) + 0.5) * time_bin / 60
        fig.add_trace(go.Heatmap(
            x=t_centers,
            y=r["e_grid"],
            z=z,
            zmin=0, zmax=r["params"].u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="kW", x=1.01) if idx == 0 else None,
            hovertemplate="Hour: %{x:.1f} h<br>Battery: %{y:.1f} kWh<br>u*: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Hour (h)" if row == rows else "", range=[0, T_h],
                         row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         row=row, col=col)
    fig.update_layout(height=350 * rows + 60, margin=dict(l=40, r=60, t=60, b=40),
                      title_text="Optimal policy u*(t, e) — parked state, averaged over price bins")
    return fig


def fig_cost_distribution(results: list[dict]) -> go.Figure:
    """Box plots of total cost per swept value; all policies overlaid."""
    fig = go.Figure()
    for r in results:
        for policy in POLICY_ORDER:
            costs = _costs(r["rollouts"], policy)
            fig.add_trace(go.Box(
                y=costs,
                name=policy,
                legendgroup=policy,
                showlegend=(r is results[0]),
                marker_color=POLICY_COLORS[policy],
                x=[r["label"]] * len(costs),
                boxmean=True,
            ))
    fig.update_layout(
        boxmode="group",
        xaxis_title="Swept value",
        yaxis_title="Total charging cost (€)",
        height=440,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_opt_gap(results: list[dict]) -> go.Figure:
    """Optimality gap (benchmark − optimal) mean ± CI95 vs swept value."""
    fig = go.Figure()
    for policy in POLICY_ORDER:
        if policy == "Optimal (BI)":
            continue
        x, y_mean, y_lo, y_hi = [], [], [], []
        for r in results:
            mean, ci = _opt_gap(r["rollouts"], policy)
            x.append(r["label"])
            y_mean.append(mean)
            y_lo.append(mean - ci)
            y_hi.append(mean + ci)
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=y_hi + y_lo[::-1],
            fill="toself", fillcolor=POLICY_COLORS[policy].replace(")", ",0.15)").replace("rgb", "rgba"),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines+markers",
            name=policy, line=dict(color=POLICY_COLORS[policy], width=2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Gap: %{y:.4f} €<extra>" + policy + "</extra>",
        ))
    fig.update_layout(
        xaxis_title="Swept value",
        yaxis_title="Optimality gap (€)",
        height=380, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_feasibility(results: list[dict]) -> go.Figure:
    """Feasibility rate (fraction penalty-free rollouts) vs swept value."""
    fig = go.Figure()
    for policy in POLICY_ORDER:
        x = [r["label"] for r in results]
        y = [_feasibility(r["rollouts"], policy) for r in results]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=policy, line=dict(color=POLICY_COLORS[policy], width=2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Feasibility: %{y:.2%}<extra>" + policy + "</extra>",
        ))
    fig.update_layout(
        xaxis_title="Swept value",
        yaxis_title="Feasibility rate",
        yaxis=dict(tickformat=".0%", range=[-0.05, 1.05]),
        height=380, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_mean_u(results: list[dict]) -> go.Figure:
    """Mean charge rate while parked vs swept value, per policy."""
    fig = go.Figure()
    for policy in POLICY_ORDER:
        x = [r["label"] for r in results]
        y = [_mean_u(r["rollouts"], policy) for r in results]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=policy, line=dict(color=POLICY_COLORS[policy], width=2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Mean u: %{y:.2f} kW<extra>" + policy + "</extra>",
        ))
    fig.update_layout(
        xaxis_title="Swept value",
        yaxis_title="Mean charge rate while parked (kW)",
        height=380, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """One row per (swept_value, policy) with key metrics."""
    rows = []
    for r in results:
        for policy in POLICY_ORDER:
            costs = _costs(r["rollouts"], policy)
            if policy != "Optimal (BI)":
                mean_gap, ci = _opt_gap(r["rollouts"], policy)
            else:
                mean_gap, ci = 0.0, 0.0
            rows.append({
                "Swept value":         r["label"],
                "Policy":              policy,
                "Mean cost (€)":       round(float(np.mean(costs)), 4),
                "Std cost (€)":        round(float(np.std(costs, ddof=1)), 4),
                "Opt gap mean (€)":    round(mean_gap, 4),
                "Opt gap CI95 (€)":    round(ci, 4),
                "Feasibility rate":    round(_feasibility(r["rollouts"], policy), 4),
                "Mean u parked (kW)":  round(_mean_u(r["rollouts"], policy), 3),
            })
    return pd.DataFrame(rows)


# ── PDF report ────────────────────────────────────────────────────────────────

def _fig_to_png(fig: go.Figure, width: int = 1200, height: int = None) -> bytes:
    h = height or max(400, fig.layout.height or 400)
    return fig.to_image(format="png", width=width, height=h, scale=2)


def _pdf_str(s: str) -> str:
    """Replace characters outside ISO-8859-1 (unsupported by Helvetica) with ASCII equivalents."""
    return (
        str(s)
        .replace("€", "EUR")
        .replace("—", " - ")
        .replace("–", " - ")
        .replace("×", "x")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _append_sweep_pages(pdf, W, XPos, YPos, results: list[dict], sweep_label: str):
    """Write one sweep's cover + figures + table into an existing FPDF instance."""
    import io
    from PIL import Image as _PILImage

    def _add_section(title: str, img_bytes: bytes):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, _pdf_str(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        img = _PILImage.open(io.BytesIO(img_bytes))
        iw, ih = img.size
        h_mm = W * ih / iw
        pdf.image(io.BytesIO(img_bytes), x=15, y=pdf.get_y(), w=W, h=h_mm)

    # Section cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _pdf_str(f"Sensitivity Analysis - {sweep_label}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 11)
    _model = results[0].get("model", "?") if results else "?"
    pdf.cell(0, 8, _pdf_str(f"EV Charging MDP  |  Model: {_model}  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    for r in results:
        pdf.cell(0, 6, _pdf_str(f"  * {r['label']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Figures
    for title, fig in [
        ("Policy heatmaps",           fig_heatmap_grid(results)),
        ("Cost distribution",         fig_cost_distribution(results)),
        ("Mean charge rate (parked)", fig_mean_u(results)),
        ("Optimality gap",            fig_opt_gap(results)),
        ("Feasibility rate",          fig_feasibility(results)),
    ]:
        _add_section(title, _fig_to_png(fig))

    # Summary table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Summary table", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    df = build_summary_df(results)
    col_widths = {c: max(len(c), 12) * 2.5 for c in df.columns}
    row_h = 6
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    for col in df.columns:
        pdf.cell(col_widths[col], row_h, _pdf_str(col), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for col in df.columns:
            pdf.cell(col_widths[col], row_h, _pdf_str(row[col]), border=1)
        pdf.ln()


def build_report_pdf(results: list[dict], sweep_label: str) -> bytes:
    """Single-sweep PDF report."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    W = pdf.w - 30
    _append_sweep_pages(pdf, W, XPos, YPos, results, sweep_label)
    return bytes(pdf.output())


def build_combined_pdf(sweeps: list[tuple[list[dict], str]]) -> bytes:
    """Concatenate multiple sweeps into one PDF. sweeps = [(results, label), ...]"""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    W = pdf.w - 30
    for results, label in sweeps:
        _append_sweep_pages(pdf, W, XPos, YPos, results, label)
    return bytes(pdf.output())


# ── Streamlit helpers ──────────────────────────────────────────────────────────

def _ensure_samplers(sampler_cache: dict, labels: list[str], status_fn=None):
    """Fit and cache any missing samplers (loads ENTSO-E data once)."""
    need = [l for l in labels if l in SAMPLER_CLASSES and l not in sampler_cache]
    if not need:
        return
    with st.spinner("Loading ENTSO-E price data…"):
        df = load_prices()
    for label in need:
        msg = f"Fitting {label} sampler…"
        if status_fn:
            status_fn(msg)
        else:
            st.toast(msg)
        sampler_cache[label] = SAMPLER_CLASSES[label]().fit(df)


def _show_results(results: list[dict], sweep_label: str):
    """Render all output plots and tables for a completed sweep."""
    if results:
        st.caption(f"Mobility model: **{results[0].get('model', '?')}**")
    st.subheader("Policy heatmaps")
    st.plotly_chart(fig_heatmap_grid(results), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost distribution")
        st.plotly_chart(fig_cost_distribution(results), use_container_width=True)
    with col2:
        st.subheader("Mean charge rate (parked)")
        st.plotly_chart(fig_mean_u(results), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Optimality gap")
        st.plotly_chart(fig_opt_gap(results), use_container_width=True)
    with col4:
        st.subheader("Feasibility rate")
        st.plotly_chart(fig_feasibility(results), use_container_width=True)

    st.subheader("Summary table")
    df = build_summary_df(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    dl_col1, dl_col2 = st.columns([1, 1])
    fname = f"sensitivity_{sweep_label.replace(' ', '_')}"
    with dl_col1:
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            csv,
            f"{fname}.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl_col2:
        with st.spinner("Building PDF…"):
            pdf_bytes = build_report_pdf(results, sweep_label)
        st.download_button(
            "Download PDF report",
            pdf_bytes,
            f"{fname}.pdf",
            "application/pdf",
            use_container_width=True,
        )


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sensitivity Analysis — EV Charging MDP", layout="wide")
st.title("Sensitivity Analysis")
with st.expander("About this page", expanded=False):
    st.markdown("""
**Baseline configuration** (SharedParams / BaselineParams defaults):
battery e_max = 40 kWh · η_c = 0.95 · u_max = 11 kW · φ = 1000 €/h · K = 20 bins · λ_max = 0.75 €/kWh

**Policies compared:** Optimal (Backward induction) · Night charging · DP heuristic · Maximal charging

**Mobility model** (sidebar — applies to every sweep):
- **Baseline** — trip duration ~ Geom(p_DP); 2-state mobility chain.
- **NegBin (fixed k)** — trip ~ NegBin(k, q) via a k-phase chain; mean = k/q min.
- **NegBin (sampled k)** — k ~ Poisson(λ_k) drawn at each trip start.

**Four independent sweep dimensions** (others held at baseline):
1. **Pricing model** — Gaussian parametric · Gaussian bins · GMM · MDN
2. **Penalty φ** — {0, 100, 500, 1000, 2000, 5000, 10 000} €/h
3. **Horizon T** — {24 h, 48 h, 168 h}
4. **Season × Day type** — all 8 combinations of {winter, spring, summer, autumn} × {weekday, weekend}

> **Note:** The DP-heuristic benchmark uses the Gaussian parametric price model internally
> regardless of the pricing-model sweep. All other policies are agnostic to the pricing model.
> The NegBin models have more mobility states, so their solves are slower — lower **N_e** if needed.
> Re-run a single sweep by clicking its **Run** button in the corresponding tab.
    """)

# ── Sidebar controls ──────────────────────────────────────────────────────────

def _clear_sweep_results():
    """Drop cached sweep results so they are recomputed after a model switch."""
    for k in ["sa_pricing_results", "sa_phi_results",
              "sa_horizon_results", "sa_season_results", "sa_combined_pdf"]:
        st.session_state.pop(k, None)


with st.sidebar:
    st.header("Sweep settings")

    model_label = st.selectbox(
        "Mobility model",
        MODEL_LABELS,
        key="sa_model",
        on_change=_clear_sweep_results,
        help=(
            "**Baseline** — trip duration ~ Geom(p_DP), 2-state mobility chain.  "
            "**NegBin (fixed k)** — trip ~ NegBin(k, q) via a k-phase chain, mean = k/q min.  "
            f"**NegBin (sampled k)** — k ~ Poisson(λ_k={NEGBIN_LAMBDA_K:g}) drawn at each trip start.  "
            "NegBin solves are slower (more mobility states) — lower N_e if needed."
        ),
    )

    inherit = st.toggle(
        "Inherit params from Settings page",
        value=False, key="sa_inherit",
        help="When on, uses the params object from the last backward-induction run in Settings.",
    )
    N_rollouts = st.slider("Rollouts per config", 10, 500, 50, 10, key="sa_N_rollouts")
    N_e        = st.select_slider("Battery grid points N_e",
                                  [50, 100, 200, 500], value=500, key="sa_N_e")
    seed       = st.number_input("Random seed", 0, 9999, 42, key="sa_seed")

    st.divider()
    st.markdown("**Pricing sweep context**")
    season     = st.radio("Season", ["winter", "spring", "summer", "autumn"],
                          horizontal=True, key="sa_season")
    is_weekend = st.toggle("Weekend", key="sa_is_weekend")

    st.divider()
    st.markdown("**Run all sweeps**")
    run_all_pricing_label = st.radio(
        "Pricing model for season sweep",
        ["Gaussian bins", "GMM", "MDN"],
        horizontal=True, key="sa_run_all_pricing_label",
    )
    if st.button("Run all & build combined PDF", key="sa_run_all", use_container_width=True):
        for k in ["sa_pricing_results", "sa_phi_results",
                  "sa_horizon_results", "sa_season_results"]:
            st.session_state.pop(k, None)
        st.session_state["sa_run_all_triggered"] = True
        st.rerun()

    if "sa_combined_pdf" in st.session_state:
        st.download_button(
            "Download combined PDF",
            st.session_state["sa_combined_pdf"],
            "sensitivity_all_sweeps.pdf",
            "application/pdf",
            use_container_width=True,
            key="sa_dl_combined",
        )

# Sampler cache persists for the whole session
if "sa_sampler_cache" not in st.session_state:
    st.session_state["sa_sampler_cache"] = {}
sampler_cache = st.session_state["sa_sampler_cache"]

# ── Run-all orchestration ─────────────────────────────────────────────────────

if st.session_state.pop("sa_run_all_triggered", False):
    _ensure_samplers(sampler_cache, PRICING_LABELS + [run_all_pricing_label])
    bar = st.progress(0.0, text="Starting…")

    bar.progress(0.05, "Pricing model sweep…")
    st.session_state["sa_pricing_results"] = sweep_pricing_models(
        model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
        season=season, is_weekend=is_weekend,
        sampler_cache=sampler_cache,
        progress_cb=lambda f, m: bar.progress(0.05 + f * 0.2, m),
    )

    bar.progress(0.25, "Penalty sweep…")
    st.session_state["sa_phi_results"] = sweep_penalty(
        model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
        progress_cb=lambda f, m: bar.progress(0.25 + f * 0.2, m),
    )

    bar.progress(0.45, "Horizon sweep…")
    st.session_state["sa_horizon_results"] = sweep_horizon(
        model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
        progress_cb=lambda f, m: bar.progress(0.45 + f * 0.2, m),
    )

    bar.progress(0.65, "Season x day-type sweep…")
    st.session_state["sa_season_results"] = sweep_season_weekend(
        model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
        pricing_label=run_all_pricing_label,
        sampler_cache=sampler_cache,
        progress_cb=lambda f, m: bar.progress(0.65 + f * 0.2, m),
    )

    bar.progress(0.85, "Building combined PDF…")
    st.session_state["sa_combined_pdf"] = build_combined_pdf([
        (st.session_state["sa_pricing_results"], "Pricing Model"),
        (st.session_state["sa_phi_results"],     "Penalty phi"),
        (st.session_state["sa_horizon_results"], "Horizon T"),
        (st.session_state["sa_season_results"],  "Season x Day Type"),
    ])
    bar.progress(1.0, "Done.")
    bar.empty()
    st.rerun()

# ── Four tabs ─────────────────────────────────────────────────────────────────

tab_price, tab_phi, tab_T, tab_season = st.tabs(
    ["Pricing Model", "Penalty φ", "Horizon T", "Season × Day Type"]
)

# ─── Tab 1: Pricing model ─────────────────────────────────────────────────────
with tab_price:
    st.markdown(
        "Sweeps all implemented pricing models over a 24 h horizon.  "
        "Season and weekend/weekday are set in the sidebar.  "
        "Benchmark policies use the Gaussian parametric model internally (see module docstring)."
    )
    if st.button("Run pricing model sweep", key="sa_run_price"):
        st.session_state.pop("sa_pricing_results", None)

        _ensure_samplers(sampler_cache, PRICING_LABELS)

        bar  = st.progress(0.0, text="Starting…")
        with st.spinner("Running pricing model sweep…"):
            results = sweep_pricing_models(
                model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                season=season, is_weekend=is_weekend,
                sampler_cache=sampler_cache,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_pricing_results"] = results
        bar.empty()
        st.rerun()

    if "sa_pricing_results" in st.session_state:
        _show_results(st.session_state["sa_pricing_results"], "pricing_model")
    else:
        st.info("Click **Run pricing model sweep** to compute results.")

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
                model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_phi_results"] = results
        bar.empty()
        st.rerun()

    if "sa_phi_results" in st.session_state:
        _show_results(st.session_state["sa_phi_results"], "penalty")
    else:
        st.info("Click **Run penalty sweep** to compute results.")

# ─── Tab 3: Horizon T ─────────────────────────────────────────────────────────
with tab_T:
    st.markdown(
        f"Compares horizon lengths T ∈ {HORIZON_HOURS} h.  "
        "Uses Gaussian parametric pricing.  All other params at baseline.  "
        "Note: the 168 h solve is slower (~30 s at N_e=100)."
    )
    if st.button("Run horizon sweep", key="sa_run_T"):
        st.session_state.pop("sa_horizon_results", None)

        bar  = st.progress(0.0, text="Starting…")
        with st.spinner("Running horizon sweep…"):
            results = sweep_horizon(
                model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_horizon_results"] = results
        bar.empty()
        st.rerun()

    if "sa_horizon_results" in st.session_state:
        _show_results(st.session_state["sa_horizon_results"], "horizon")
    else:
        st.info("Click **Run horizon sweep** to compute results.")

# ─── Tab 4: Season × Day type ─────────────────────────────────────────────────
with tab_season:
    st.markdown(
        "Sweeps all 8 combinations of season × day type over a 24 h horizon.  "
        "Requires a data-driven pricing model (Gaussian bins, GMM, or MDN).  "
        "All other params at baseline."
    )
    pricing_label_season = st.radio(
        "Pricing model",
        ["Gaussian bins", "GMM", "MDN"],
        horizontal=True,
        key="sa_season_pricing_label",
    )
    if st.button("Run season × day-type sweep", key="sa_run_season"):
        st.session_state.pop("sa_season_results", None)

        _ensure_samplers(sampler_cache, [pricing_label_season])

        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running season × day-type sweep…"):
            results = sweep_season_weekend(
                model_label=model_label, N_rollouts=N_rollouts, N_e=N_e, seed=seed,
                pricing_label=pricing_label_season,
                sampler_cache=sampler_cache,
                progress_cb=lambda frac, msg: bar.progress(frac, text=msg),
            )
        st.session_state["sa_season_results"] = results
        bar.empty()
        st.rerun()

    if "sa_season_results" in st.session_state:
        _show_results(st.session_state["sa_season_results"], "season_daytype")
    else:
        st.info("Click **Run season × day-type sweep** to compute results.")
