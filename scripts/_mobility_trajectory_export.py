"""One-off: export mean price + mobility trajectories for all three mobility models."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL,
    NEGBIN_LAMBDA_K, build_params, make_scenario,
)
from ev_mdt.models.common.policies import always_minimum_policy
from ev_mdt.models.baseline.rollout import simulate_policy_rollout as baseline_rollout
from ev_mdt.models.negbin.rollout import simulate_policy_rollout as negbin_rollout
from ev_mdt.params import NegBinParams
from ev_mdt.plots.sensitivity import figure_to_png
from ev_mdt.plots.viz import MODEL_COLORS

ROOT    = Path(__file__).resolve().parent
N       = 1000
SEED    = 42
T       = 24 * 60
T_HOURS = T // 60

MODELS = [
    ("Baseline",                     BASELINE_MODEL,       build_params(BASELINE_MODEL),       baseline_rollout),
    ("Negative Binomial (fixed k)",  NEGBIN_FIXED_MODEL,   NegBinParams(k=5),                 negbin_rollout),
    ("Negative Binomial (Poisson k)",NEGBIN_SAMPLED_MODEL, NegBinParams(
        lambda_k=NEGBIN_LAMBDA_K, k=NegBinParams.k_max_for_lambda(NEGBIN_LAMBDA_K)
    ), negbin_rollout),
]

# Generate scenarios from baseline params (shared price paths).
base_params = build_params(BASELINE_MODEL)
print(f"Generating {N} scenarios…", flush=True)
scenarios = [make_scenario(base_params, SEED + i, T) for i in range(N)]

h_axis = np.arange(T_HOURS)
m_axis = np.arange(T) / 60

# Mean hourly price (same across models — shared scenarios).
P = np.array([sc["lam_path"] for sc in scenarios])           # (N, T)
P_hourly = P.reshape(N, T_HOURS, T // T_HOURS).mean(axis=2)  # (N, T_HOURS)
mean_price = P_hourly.mean(axis=0)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Mean sampled price", "Mean sampled mobility (0 parked, 1 driving)"),
)

# Price (single grey line, same for all models).
fig.add_trace(go.Scatter(
    x=h_axis, y=mean_price, mode="lines",
    line=dict(color="lightgray", width=1.8),
    name="Price", showlegend=False,
), row=1, col=1)

# Mobility per model.
for label, model_label, params, rollout_fn in MODELS:
    print(f"Rolling out {label}…", flush=True)
    chi_trajs = [
        rollout_fn(always_minimum_policy, sc, sc["e0"], 0, params)["chi_traj"]
        for sc in scenarios
    ]
    Mob = np.array([(chi > 0).astype(float) for chi in chi_trajs])  # (N, T)
    mean_mob = Mob.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=m_axis, y=mean_mob, mode="lines",
        line=dict(color=MODEL_COLORS[label], width=1.8),
        name=label, showlegend=True,
    ), row=2, col=1)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=560,
    hovermode="x unified",
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(range=[0, T_HOURS], dtick=max(1, T_HOURS // 8))
fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
fig.update_yaxes(title_text="€/kWh", row=1, col=1)
fig.update_yaxes(title_text="Fraction driving", tickvals=[0, 0.5, 1], row=2, col=1)

out = ROOT / "mobility_trajectories.png"
out.write_bytes(figure_to_png(fig))
print(f"Saved: {out}")
