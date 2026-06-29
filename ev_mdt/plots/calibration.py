"""Figures for the Departure-Urgency calibration sweeps (e_base and γ).

Both consume the exact-sweep row dicts produced by
``ev_mdt.analysis.sensitivity.sweep_target_ceiling_exact`` / ``sweep_gamma_exact``.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_target_sweep(rows: list[dict]) -> go.Figure:
    """Total / penalty / charging expected cost vs the DU target ceiling (e_base).

    The cost-minimising ceiling is marked with a star.
    """
    df   = pd.DataFrame(rows)
    best = df.loc[df["mean_cost"].idxmin()]
    x    = df["target_kwh"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df["mean_cost"], mode="lines+markers",
        line=dict(color="#4477AA", width=2), marker=dict(size=7),
        name="Total cost", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["mean_penalty_cost"], mode="lines+markers",
        line=dict(color="#CC3311", width=2, dash="dash"), marker=dict(size=6),
        name="Penalty cost", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=[best["target_kwh"]], y=[best["mean_cost"]],
        mode="markers", marker=dict(color="#EE6677", size=12, symbol="star"),
        name=f"Best: {best['target_kwh']:.1f} kWh ({best['target_frac']:.1%})", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["mean_charge_cost"], mode="lines+markers",
        line=dict(color="#228833", width=2, dash="dot"), marker=dict(size=6),
        name="Charging cost (right)", yaxis="y2",
    ))
    fig.update_layout(
        xaxis_title="Target ceiling (kWh)",
        yaxis=dict(title="Total / penalty cost (€)"),
        yaxis2=dict(title="Charging cost (€)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white", height=500,
    )
    return fig


def fig_gamma_sweep(results: dict[str, list[dict]]) -> go.Figure:
    """Expected cost vs the ceiling scaling exponent γ, one subplot per mobility model."""
    cost_label  = "Expected cost (€)"
    model_names = list(results.keys())
    n_models    = len(model_names)
    fig = make_subplots(
        rows=n_models, cols=1, shared_xaxes=True,
        subplot_titles=model_names, vertical_spacing=0.10,
    )
    colors = {"total": "#4477AA", "charging": "#228833", "penalty": "#CC3311"}
    for row_i, model_name in enumerate(model_names, start=1):
        df   = pd.DataFrame(results[model_name])
        x    = df["gamma"]
        show = row_i == 1
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_cost"], mode="lines+markers",
            line=dict(color=colors["total"], width=2), marker=dict(size=6),
            name="Total cost", legendgroup="total", showlegend=show,
        ), row=row_i, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_charge_cost"], mode="lines+markers",
            line=dict(color=colors["charging"], width=2, dash="dot"), marker=dict(size=5),
            name="Charging cost", legendgroup="charging", showlegend=show,
        ), row=row_i, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_penalty_cost"], mode="lines+markers",
            line=dict(color=colors["penalty"], width=2, dash="dash"), marker=dict(size=5),
            name="Penalty cost", legendgroup="penalty", showlegend=show,
        ), row=row_i, col=1)
        best = df.loc[df["mean_cost"].idxmin()]
        fig.add_trace(go.Scatter(
            x=[best["gamma"]], y=[best["mean_cost"]],
            mode="markers", marker=dict(color="#EE6677", size=11, symbol="star"),
            name="Best γ", legendgroup=f"best_{row_i}", showlegend=show,
        ), row=row_i, col=1)
        fig.update_yaxes(title_text=cost_label, row=row_i, col=1)
    fig.update_xaxes(title_text="γ", row=n_models, col=1)
    fig.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=320 * n_models,
        margin=dict(l=70, r=80, t=60, b=40),
    )
    return fig
