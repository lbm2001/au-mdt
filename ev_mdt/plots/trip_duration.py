"""Trip-duration comparison across the three mobility models.

Used by the Policy Rollout app page and the CLI so the chart and its underlying
sampling stay identical in both places.
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ev_mdt.params import BaselineParams, NegBinParams
from ev_mdt.models.baseline.rollout import _next_state as _ns_base
from ev_mdt.models.negbin.rollout import _next_state as _ns_nb

MOBILITY_COLORS = {
    "Baseline":                           "#4477AA",
    "NegBin (fixed k=5)":                 "#EE6677",
    "NegBin (Poisson k=5, k_max=13)":    "#228833",
}


def _kmax(lam: float, q: float = 0.999) -> int:
    """Smallest k_max such that a Poisson(lam) puts ≥ q of its mass in [1, k_max]."""
    pmf, cdf, k = math.exp(-lam), math.exp(-lam), 0
    while cdf < q:
        k += 1
        pmf *= lam / k
        cdf += pmf
    return max(k, 1)


def compute_trip_durations(n_scen: int = 10000, horizon: int = 1440, seed: int = 0) -> dict:
    """Driving-spell lengths (minutes) per mobility model, from simulated mobility only."""
    specs = {
        "Baseline":                       (BaselineParams(),                          _ns_base),
        "NegBin (fixed k=5)":             (NegBinParams(),                            _ns_nb),
        "NegBin (Poisson k=5, k_max=13)": (NegBinParams(lambda_k=5.0, k=_kmax(5.0)), _ns_nb),
    }
    out = {}
    for name, (p, next_state) in specs.items():
        durs = []
        for i in range(n_scen):
            rng = np.random.default_rng(seed + i)
            sc  = {"mobility_draws": rng.random(horizon), "phase_draws": rng.random(horizon)}
            chi = np.empty(horizon, dtype=int)
            c = 0
            for t in range(horizon):
                chi[t] = c
                c = next_state(c, sc, t, p)
            d  = (chi > 0).astype(int)
            ed = np.diff(np.concatenate([[0], d, [0]]))
            durs.extend((np.where(ed == -1)[0] - np.where(ed == 1)[0]).tolist())
        out[name] = np.asarray(durs, dtype=float)
    return out


def trip_duration_figure(durs: dict) -> go.Figure:
    """Two panels: trip-duration density (left) and survival P(duration > t) (right, log-y)."""
    cap   = int(np.ceil(max((np.percentile(d, 99) for d in durs.values() if len(d)), default=30)))
    edges = np.arange(0, cap + 2, 2)
    ctr   = (edges[:-1] + edges[1:]) / 2
    tgrid = np.arange(0, cap + 1)
    fig = make_subplots(rows=1, cols=2)
    for name, d in durs.items():
        col = MOBILITY_COLORS.get(name, "#888888")
        dens, _ = np.histogram(d, bins=edges, density=True)
        fig.add_trace(go.Scatter(x=ctr, y=dens, mode="lines",
                                 line=dict(color=col, width=2, shape="spline"),
                                 name=name, legendgroup=name),
                      row=1, col=1)
        surv = np.array([float((d > t).mean()) for t in tgrid])
        fig.add_trace(go.Scatter(x=tgrid, y=surv, mode="lines", line=dict(color=col, width=2),
                                 name=name, legendgroup=name, showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Trip duration (min)", row=1, col=1)
    fig.update_xaxes(title_text="Trip duration (min)", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="P(duration > t)", type="log", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06))
    return fig
