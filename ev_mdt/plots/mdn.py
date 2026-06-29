"""Training-curve figures for the MDN price sampler.

Consumes the per-epoch history list recorded by ``MDNSampler.fit(_history=[…])``:
each row is ``{"step", "loss", "loss_original_space", "pi_0", "pi_1", …}``.
Used by the ``fit-mdn`` CLI command and the Price/Settings app pages so the
training plots stay identical in both places.
"""
import plotly.graph_objects as go

from ev_mdt.plots.viz import SWEEP_PALETTE


def fig_mdn_nll(history: list[dict]) -> go.Figure:
    """Negative log-likelihood (original price space) vs training epoch."""
    steps = [r["step"] for r in history]
    nll   = [r["loss_original_space"] for r in history]
    fig = go.Figure(go.Scatter(
        x=steps, y=nll, mode="lines",
        line=dict(color=SWEEP_PALETTE[0], width=1.8),
        hovertemplate="Epoch %{x}<br>NLL %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Epoch",
        yaxis_title="Neg. log-likelihood (original space)",
        height=420,
        margin=dict(l=70, r=30, t=40, b=50),
    )
    return fig


def fig_mdn_components(history: list[dict], n_components: int) -> go.Figure:
    """Mean mixture weight π_k of each component vs training epoch."""
    steps = [r["step"] for r in history]
    fig = go.Figure()
    for k in range(n_components):
        key = f"pi_{k}"
        weights = [r[key] for r in history if key in r]
        fig.add_trace(go.Scatter(
            x=steps, y=weights, mode="lines",
            name=f"Component {k}",
            line=dict(color=SWEEP_PALETTE[k % len(SWEEP_PALETTE)], width=1.8),
            hovertemplate=f"Component {k}<br>Epoch %{{x}}<br>π=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Epoch",
        yaxis_title="Mean mixture weight π_k",
        height=420,
        margin=dict(l=70, r=30, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
