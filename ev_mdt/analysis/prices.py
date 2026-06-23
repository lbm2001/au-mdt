"""Price-model simulation and comparison.

Public API
----------
fit_samplers(df, progress_cb)          — fit all three data-driven samplers
simulate_price_paths(...)              — sample hourly prices for N days per model
price_figures(results)                 — (mean-profile fig, std fig) as Plotly Figures
"""

from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ev_mdt.params import SharedParams
from ev_mdt.models.common.model_utils import mean_price
from ev_mdt.pricing.samplers import (
    GaussianBinnedSampler, GMMSampler, MDNSampler, SEASONS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

PRICE_MODEL_COLORS = {
    "Gaussian (parametric)": "#4477AA",
    "Gaussian Bins":          "#EE7733",
    "GMM":                    "#228833",
    "MDN":                    "#EE6677",
}

_SAMPLER_CLASSES: dict[str, type] = {
    "Gaussian Bins": GaussianBinnedSampler,
    "GMM":           GMMSampler,
    "MDN":           MDNSampler,
}


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_samplers(
    df: pd.DataFrame,
    progress_cb: Callable[[str, float, str], None] | None = None,
) -> dict:
    """Fit all three data-driven samplers on a preprocessed ENTSO-E DataFrame.

    progress_cb(model_name, fraction 0-1, message) — optional progress hook.
    Returns {name: fitted_sampler}.
    """
    samplers: dict = {}
    n = len(_SAMPLER_CLASSES)
    for i, (name, cls) in enumerate(_SAMPLER_CLASSES.items()):
        def _prog(frac: float, msg: str, _i: int = i, _n: str = name) -> None:
            if progress_cb is not None:
                progress_cb(_n, (_i + frac) / n, msg)

        sampler = cls()
        samplers[name] = sampler.fit(df, _progress=_prog)

    return samplers


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_price_paths(
    samplers: dict,
    n_days: int = 1000,
    season: str | None = None,
    daytype: str = "all",
    seed: int = 42,
    params: SharedParams | None = None,
) -> dict[str, np.ndarray]:
    """Simulate hourly price paths for every pricing model.

    Parameters
    ----------
    samplers    : fitted sampler dict from fit_samplers()
    n_days      : number of simulated days
    season      : one of SEASONS, or None / "all" for random per day
    daytype     : "all" | "weekday" | "weekend"
    seed        : RNG seed
    params      : SharedParams (defaults used if None)

    Returns
    -------
    dict mapping model name → ndarray of shape (n_days, 24), prices in €/kWh.
    Includes "Gaussian (parametric)" derived from params.
    """
    rng    = np.random.default_rng(seed)
    params = params or SharedParams()

    # ── Draw per-day contexts ─────────────────────────────────────────────────
    season_lower = season.lower() if season and season.lower() != "all" else None

    if season_lower is not None and season_lower not in SEASONS:
        raise ValueError(f"season must be one of {SEASONS} or None/'all', got {season!r}")

    if season_lower is not None:
        day_seasons: list[str] = [season_lower] * n_days
    else:
        day_seasons = [SEASONS[i] for i in rng.integers(0, 4, n_days)]

    daytype = daytype.lower()
    if daytype == "weekday":
        day_dows: list[int] = rng.integers(0, 5, n_days).tolist()
    elif daytype == "weekend":
        day_dows = (rng.integers(0, 2, n_days) + 5).tolist()
    else:
        day_dows = rng.integers(0, 7, n_days).tolist()

    day_is_weekend = [dow >= 5 for dow in day_dows]

    results: dict[str, np.ndarray] = {}

    # ── Gaussian parametric (vectorised) ─────────────────────────────────────
    hour_means = np.array([mean_price(h * 60, params) for h in range(24)])
    results["Gaussian (parametric)"] = np.clip(
        rng.normal(hour_means[np.newaxis, :], params.sigma_lambda, size=(n_days, 24)),
        0.0, None,
    )

    # ── Data-driven models ────────────────────────────────────────────────────
    # One sample per hour — consistent with the hourly ENTSO-E data resolution.
    for name, sampler in samplers.items():
        prices = np.zeros((n_days, 24))
        for is_wknd, s in product([False, True], SEASONS):
            idx = [d for d in range(n_days)
                   if day_is_weekend[d] == is_wknd and day_seasons[d] == s]
            if not idx:
                continue
            dow = 5 if is_wknd else 0
            for h in range(24):
                draws = [max(0.0, sampler.sample(dow, h, s, rng)) for _ in idx]
                for j, d in enumerate(idx):
                    prices[d, h] = draws[j]
        results[name] = prices

    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def price_figures(
    results: dict[str, np.ndarray],
) -> tuple[go.Figure, go.Figure]:
    """Return (mean-profile figure, std figure) for the simulation results.

    Each figure has one trace per pricing model.
    """
    hours = np.arange(24)
    n     = next(iter(results.values())).shape[0]

    fig_mean = go.Figure()
    fig_std  = go.Figure()

    for name, prices in results.items():
        col = PRICE_MODEL_COLORS.get(name, "#888888")
        mu  = prices.mean(axis=0)
        sem = prices.std(axis=0) / np.sqrt(n)
        std = prices.std(axis=0)

        fig_mean.add_trace(go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([mu + sem, (mu - sem)[::-1]]),
            fill="toself", fillcolor=_rgba(col, 0.12),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=name,
        ))
        fig_mean.add_trace(go.Scatter(
            x=hours, y=mu, mode="lines",
            line=dict(color=col, width=1.6),
            name=name, legendgroup=name,
            hovertemplate=f"<b>{name}</b><br>Hour %{{x:02d}}:00<br>%{{y:.4f}} €/kWh<extra></extra>",
        ))

        fig_std.add_trace(go.Scatter(
            x=hours, y=std, mode="lines",
            line=dict(color=col, width=1.6),
            name=name, legendgroup=name,
            hovertemplate=f"<b>{name}</b><br>Hour %{{x:02d}}:00<br>σ = %{{y:.4f}} €/kWh<extra></extra>",
        ))

    _legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    _xaxis  = dict(title="Hour of day", dtick=2, range=[0, 23])
    _margin = dict(l=50, r=20, t=20, b=40)

    fig_mean.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480, hovermode="x unified", margin=_margin, legend=_legend,
        xaxis=_xaxis, yaxis=dict(title="Mean price (€/kWh)"),
    )
    fig_std.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380, hovermode="x unified", margin=_margin, legend=_legend,
        xaxis=_xaxis, yaxis=dict(title="Std (€/kWh)"),
    )
    return fig_mean, fig_std
