import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Policy Rollout — EV Charging MDP", layout="wide")
st.title("Policy Rollout")

# ── Guard ─────────────────────────────────────────────────────────────────────

if "pi" not in st.session_state:
    st.warning("No solution found. Please go to **Settings** and click **Run Backward Induction** first.")
    st.stop()

V        = st.session_state["V"]
pi       = st.session_state["pi"]
actions  = st.session_state["actions"]
e_grid   = st.session_state["e_grid"]
lam_grid = st.session_state["lam_grid"]
params   = st.session_state["params"]
T        = st.session_state["T"]
T_hours  = T // 60

is_negbin = st.session_state.get("solved_model", "").startswith("NegBin")

# ── Model-specific imports ────────────────────────────────────────────────────

if is_negbin:
    from models.negative_binomial_trips import (
        mean_price,
        backward_induction_policy, maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy,
        dp_heuristic_policy,
    )
    from models.negative_binomial_trips import (
        generate_rollout_scenario, rollout_metrics, simulate_policy_rollout,
    )
else:
    from models.baseline import (
        mean_price,
        backward_induction_policy, maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy,
        dp_heuristic_policy,
        generate_rollout_scenario, rollout_metrics, simulate_policy_rollout,
    )

from utils.viz import POLICY_COLORS, POLICY_ORDER    # shared canonical colours + order

# Trip-duration comparison across the three mobility models (shared with the
# Sensitivity figure-ZIP builder so both render the identical chart).
from utils.trip_duration import compute_trip_durations, trip_duration_figure

# Named colours used for non-policy bands (price/mobility) → "r,g,b".
_CSS_RGB = {"orange": "255,165,0", "lightgray": "211,211,211"}


def _rgba(color: str, alpha: float) -> str:
    """rgba() string for a hex (#RRGGBB) or named colour, at the given opacity."""
    if color.startswith("#"):
        h = color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
    return f"rgba({_CSS_RGB.get(color, '128,128,128')},{alpha})"

# Read policy thresholds set in Policy Explorer (with sensible defaults)
low_threshold  = float(st.session_state.get("benchmark_low_threshold",  params.price_night))
high_threshold = float(st.session_state.get("benchmark_high_threshold", params.price_evening))
soc_threshold  = float(st.session_state.get("soc_threshold",            params.e_max * 0.25))

# ── Single rollout ────────────────────────────────────────────────────────────

st.subheader("Single-Scenario Rollout")

sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
with sim_col1:
    e0 = st.slider("Initial battery (kWh)", float(params.e_min), float(params.e_max),
                   float(params.e_max / 2),
                   float((params.e_max - params.e_min) / (len(e_grid) - 1)))
with sim_col2:
    chi0 = st.radio("Initial state", ["Parked", "Driving"], horizontal=True)
with sim_col3:
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0, step=1)
with sim_col4:
    st.write("")
    st.write("")
    rerun_sim = st.button("Re-run")

if rerun_sim:
    st.session_state["sim_seed"] = int(seed) + 1
    seed = st.session_state["sim_seed"]
else:
    seed = st.session_state.get("sim_seed", int(seed))

chi0_int = 0 if chi0 == "Parked" else 1  # D_1 (state 1) is the natural driving start for NegBin
scenario = generate_rollout_scenario(params, int(seed), horizon=T)

single_rollouts = {
    "Backward Induction": simulate_policy_rollout(
        backward_induction_policy, scenario, float(e0), chi0_int, params,
        pi=pi, actions=actions, e_grid=e_grid),
    "DP-Heuristic": simulate_policy_rollout(
        dp_heuristic_policy, scenario, float(e0), chi0_int, params),
    "Always-Maximum": simulate_policy_rollout(
        maximal_charging_policy, scenario, float(e0), chi0_int, params),
    "Price-Oriented": simulate_policy_rollout(
        price_oriented_policy, scenario, float(e0), chi0_int, params,
        low_threshold=low_threshold, high_threshold=high_threshold),
    "Night Charging": simulate_policy_rollout(
        night_charging_policy, scenario, float(e0), chi0_int, params),
    "Minimum-Charge": simulate_policy_rollout(
        minimum_soc_policy, scenario, float(e0), chi0_int, params,
        soc_threshold=soc_threshold),
    "Always-Minimum": simulate_policy_rollout(
        always_minimum_policy, scenario, float(e0), chi0_int, params),
}

hours        = np.arange(T) / 60
lam_traj     = scenario["lam_path"]
chi_traj_ref = single_rollouts["Backward Induction"]["chi_traj"]

# Binarise for display: parked=0, any driving phase=1
driving_traj = (chi_traj_ref > 0).astype(int)


st.divider()

# ── N-scenario simulation ─────────────────────────────────────────────────────

st.subheader("N-Scenario Policy Comparison")

nd_col1, nd_col2, nd_col3 = st.columns(3)
with nd_col1:
    n_days = st.slider("Number of scenarios N", 10, 500, 500, 10, key="n_days")
with nd_col2:
    nd_chi0 = st.radio("Initial state ", ["Parked", "Driving"], horizontal=True, key="nd_chi0")
with nd_col3:
    nd_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="nd_seed")
st.caption("Initial battery is randomised uniformly over [e_min, e_max] per scenario (seeded; "
           "same start for all policies within a scenario).")

nd_chi0_int = 0 if nd_chi0 == "Parked" else 1

# Cache the rollouts so re-renders (e.g. toggling a plot option) don't recompute them.
_nd_key = (n_days, nd_chi0_int, int(nd_seed),
           low_threshold, high_threshold, soc_threshold, T, id(pi))
if st.session_state.get("_nd_key") != _nd_key:
    with st.spinner(f"Rolling out all policies over {n_days} scenarios…"):
        rng_nd = np.random.default_rng(int(nd_seed))
        nd_scenarios = [
            generate_rollout_scenario(params, int(rng_nd.integers(0, 1_000_000)), horizon=T)
            for _ in range(n_days)
        ]
        nd_rollouts = {name: [] for name in POLICY_COLORS}
        for sc in nd_scenarios:
            e0_i = float(rng_nd.uniform(params.e_min, params.e_max))   # random SoC per scenario
            nd_rollouts["Backward Induction"].append(simulate_policy_rollout(
                backward_induction_policy, sc, e0_i, nd_chi0_int, params,
                pi=pi, actions=actions, e_grid=e_grid))
            nd_rollouts["DP-Heuristic"].append(simulate_policy_rollout(
                dp_heuristic_policy, sc, e0_i, nd_chi0_int, params))
            nd_rollouts["Always-Maximum"].append(simulate_policy_rollout(
                maximal_charging_policy, sc, e0_i, nd_chi0_int, params))
            nd_rollouts["Price-Oriented"].append(simulate_policy_rollout(
                price_oriented_policy, sc, e0_i, nd_chi0_int, params,
                low_threshold=low_threshold, high_threshold=high_threshold))
            nd_rollouts["Night Charging"].append(simulate_policy_rollout(
                night_charging_policy, sc, e0_i, nd_chi0_int, params))
            nd_rollouts["Minimum-Charge"].append(simulate_policy_rollout(
                minimum_soc_policy, sc, e0_i, nd_chi0_int, params,
                soc_threshold=soc_threshold))
            nd_rollouts["Always-Minimum"].append(simulate_policy_rollout(
                always_minimum_policy, sc, e0_i, nd_chi0_int, params))
    st.session_state["_nd_rollouts"]  = nd_rollouts
    st.session_state["_nd_scenarios"] = nd_scenarios
    st.session_state["_nd_key"]       = _nd_key

nd_rollouts  = st.session_state["_nd_rollouts"]
nd_scenarios = st.session_state["_nd_scenarios"]


def _nd_stats(rollout_list: list) -> dict:
    costs    = np.array([r["cost_traj"].sum() for r in rollout_list])
    pen_mins = np.array([int(((r["chi_traj"] > 0) & (r["e_traj"] <= params.e_min)).sum())
                         for r in rollout_list])
    energy   = np.array([(r["u_traj"] * params.omega).sum() for r in rollout_list])
    final_e  = np.array([r["final_e"] for r in rollout_list])
    return {
        "Mean cost (€)":             costs.mean(),
        "Std cost (€)":              costs.std(),
        "Median cost (€)":           float(np.median(costs)),
        "Mean penalty min":          pen_mins.mean(),
        "% scenarios with penalty":  float((pen_mins > 0).mean() * 100),
        "Mean energy charged (kWh)": energy.mean(),
        "Mean final battery (kWh)":  final_e.mean(),
    }


stats_df = pd.DataFrame({name: _nd_stats(rolls) for name, rolls in nd_rollouts.items()}).T
st.dataframe(
    stats_df.style.format({
        "Mean cost (€)":             "{:.3f}",
        "Std cost (€)":              "{:.3f}",
        "Median cost (€)":           "{:.3f}",
        "Mean penalty min":          "{:.1f}",
        "% scenarios with penalty":  "{:.1f}%",
        "Mean energy charged (kWh)": "{:.2f}",
        "Mean final battery (kWh)":  "{:.2f}",
    }),
    use_container_width=True,
)


def cost_bar_figure(error: str, log_y: bool) -> go.Figure:
    """Mean total cost (incl. penalty) ± error per policy — matches the sensitivity page."""
    names, means, errs = list(nd_rollouts), [], []
    for name in names:
        costs = np.array([r["cost_traj"].sum() for r in nd_rollouts[name]])
        m  = len(costs)
        sd = float(costs.std(ddof=1)) if m > 1 else 0.0
        means.append(float(costs.mean()))
        errs.append(sd / np.sqrt(m) if (error == "sem" and m > 0) else sd)
    minus = [min(e, mu) for mu, e in zip(means, errs)]   # cost ≥ 0 → don't dip below 0
    fig = go.Figure(go.Bar(
        x=names, y=means, marker_color=[POLICY_COLORS[n] for n in names],
        error_y=dict(type="data", symmetric=False, array=errs, arrayminus=minus,
                     visible=True, thickness=1.2, width=4),
    ))
    yaxis = dict(title="Mean total cost incl. penalty (€)" + ("  [log]" if log_y else ""),
                 type="log" if log_y else "linear")
    if log_y:
        yaxis["dtick"] = 1
    fig.update_layout(yaxis=yaxis, xaxis_title="Policy", height=460,
                      margin=dict(l=40, r=20, t=20, b=110), showlegend=False)
    fig.update_xaxes(categoryorder="array", categoryarray=POLICY_ORDER)   # fixed canonical order
    return fig


st.subheader("Mean cost")
st.caption("Mean total cost per scenario — **including the unserved-driving penalty** — "
           "one bar per policy. SEM = uncertainty of the mean (std/√N); Std = spread across "
           "scenarios. Lower bar clamped at 0.")
cc1, cc2 = st.columns(2)
with cc1:
    nd_cost_axis = st.radio("Cost axis", ["Log", "Linear"], horizontal=True, key="nd_cost_axis")
with cc2:
    nd_err = st.radio("Error bars", ["SEM", "Std"], horizontal=True, key="nd_cost_err")
st.plotly_chart(cost_bar_figure(nd_err.lower(), nd_cost_axis == "Log"), use_container_width=True)


def mean_trajectory_figure() -> go.Figure:
    """Scenario-averaged trajectories vs time: price and mobility (shared across policies)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Mean price λ̄<sub>t</sub>",
                        "Mean mobility — 0 parked, 1 driving"),
    )

    def band(mean, half, color, name, row):
        fill = _rgba(color, 0.12)
        fig.add_trace(go.Scatter(x=hours, y=mean + half, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=name),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean - half, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=fill, showlegend=False,
                                 hoverinfo="skip", legendgroup=name), row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean, mode="lines", line=dict(color=color, width=1.6),
                                 name=name, legendgroup=name, showlegend=False), row=row, col=1)

    P = np.array([sc["lam_path"] for sc in nd_scenarios])                       # (N, T)
    n_scen = max(P.shape[0], 1)
    sem = lambda a: a.std(axis=0) / np.sqrt(n_scen)    # ±1 standard error of the mean
    band(P.mean(0), sem(P), "lightgray", "λ̄<sub>t</sub>", row=1)
    # mobility is policy-independent → take it from any policy's rollouts
    Mob = np.array([(r["chi_traj"] > 0).astype(float)
                    for r in nd_rollouts["Backward Induction"]])               # (N, T)
    band(Mob.mean(0), sem(Mob), "orange", "driving", row=2)

    fig.update_layout(height=560, hovermode="x unified",
                      margin=dict(l=50, r=30, t=50, b=40), showlegend=False)
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Fraction driving", tickvals=[0, 0.5, 1], row=2, col=1)
    return fig


st.subheader("Mean trajectories across scenarios")
st.caption("Scenario-averaged price path and mobility (both shared across policies), with "
           "±1 SEM bands (std/√N, faint) — the uncertainty of the mean.")
st.plotly_chart(mean_trajectory_figure(), use_container_width=True)

st.divider()

# ── Trip-duration distribution by mobility model ──────────────────────────────

st.subheader("Trip-duration distribution by mobility model")
st.caption("Samples each mobility model's trips (default params; independent of the solved "
           "policy). Baseline → geometric (decaying); NegBin → peaked; Poisson-k → wider.")


_compute_trip_durations = st.cache_data(show_spinner="Sampling trip durations…")(compute_trip_durations)


st.plotly_chart(
    trip_duration_figure(_compute_trip_durations()),
    use_container_width=True,
    config={"displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": "trip_duration_by_model", "scale": 4}},
)
