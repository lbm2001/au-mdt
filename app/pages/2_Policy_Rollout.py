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
        night_charging_policy, minimum_soc_policy, always_minimum_policy, random_policy,
    )
    from models.negative_binomial_trips.rollout import (
        generate_rollout_scenario, rollout_metrics, simulate_policy_rollout,
    )
else:
    from models.baseline import (
        mean_price,
        backward_induction_policy, maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy, random_policy,
        dp_heuristic_policy, expected_parking_policy,
    )
    from models.baseline.rollout import generate_rollout_scenario, rollout_metrics, simulate_policy_rollout

POLICY_COLORS = {
    "Backward induction":   "steelblue",
    "DP heuristic":         "teal",
    "Expected parking":     "darkviolet",
    "Maximal charging":     "seagreen",
    "Price-oriented":     "crimson",
    "Night charging":     "purple",
    "Minimum SoC":        "darkorange",
    "Always minimum":     "gray",
    "Random":             "pink",
}

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
    "Backward induction": simulate_policy_rollout(
        backward_induction_policy, scenario, float(e0), chi0_int, params,
        pi=pi, actions=actions, e_grid=e_grid),
    "DP heuristic": simulate_policy_rollout(
        dp_heuristic_policy, scenario, float(e0), chi0_int, params),
    "Expected parking": simulate_policy_rollout(
        expected_parking_policy, scenario, float(e0), chi0_int, params),
    "Maximal charging": simulate_policy_rollout(
        maximal_charging_policy, scenario, float(e0), chi0_int, params),
    "Price-oriented": simulate_policy_rollout(
        price_oriented_policy, scenario, float(e0), chi0_int, params,
        low_threshold=low_threshold, high_threshold=high_threshold),
    "Night charging": simulate_policy_rollout(
        night_charging_policy, scenario, float(e0), chi0_int, params),
    "Minimum SoC": simulate_policy_rollout(
        minimum_soc_policy, scenario, float(e0), chi0_int, params,
        soc_threshold=soc_threshold),
    "Always minimum": simulate_policy_rollout(
        always_minimum_policy, scenario, float(e0), chi0_int, params),
    "Random": simulate_policy_rollout(
        random_policy, scenario, float(e0), chi0_int, params,
        rng=np.random.default_rng(int(seed))),
}

hours        = np.arange(T) / 60
lam_traj     = scenario["lam_path"]
chi_traj_ref = single_rollouts["Backward induction"]["chi_traj"]

# Binarise for display: parked=0, any driving phase=1
driving_traj = (chi_traj_ref > 0).astype(int)


def sim_figure() -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("Battery level", "Mobility state", "Charge rate",
                        "Price and cumulative cost"),
        vertical_spacing=0.07,
    )
    for name, rollout in single_rollouts.items():
        color = POLICY_COLORS[name]
        fig.add_trace(go.Scatter(x=hours, y=rollout["e_traj"], mode="lines",
                                 line=dict(color=color, width=1.5), name=name,
                                 legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=rollout["u_traj"], mode="lines",
                                 line=dict(color=color, width=1.5, shape="hv"), name=name,
                                 legendgroup=name, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=hours, y=np.cumsum(rollout["cost_traj"]), mode="lines",
                                 line=dict(color=color, width=1.5), name=name,
                                 legendgroup=name, showlegend=False),
                      row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=hours, y=driving_traj, mode="lines", fill="tozeroy",
                             line=dict(color="orange", width=1.2, shape="hv"),
                             name="Driving state",
                             hovertemplate="Hour: %{x:.2f}<br>State: %{y}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=lam_traj, mode="lines",
                             line=dict(color="lightgray", width=1.0, shape="hv"),
                             name="λ_t sampled"), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=hours,
                             y=np.array([mean_price(t, params) for t in range(T)]),
                             mode="lines",
                             line=dict(color="black", width=1.4, dash="dash", shape="hv"),
                             name="λ̄_t mean"), row=4, col=1, secondary_y=False)
    fig.update_layout(height=1000, hovermode="x unified",
                      margin=dict(l=30, r=30, t=80, b=35))
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    fig.update_xaxes(title_text="Hour", row=4, col=1)
    fig.update_yaxes(title_text="Battery (kWh)",
                     range=[params.e_min - 0.5, params.e_max + 0.5], row=1, col=1)
    fig.update_yaxes(title_text="State", tickvals=[0, 1],
                     ticktext=["Parked", "Driving"], row=2, col=1)
    fig.update_yaxes(title_text="Charge rate (kW)",
                     range=[-0.2, params.u_max + 0.5], row=3, col=1)
    fig.update_yaxes(title_text="€ / kWh", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative cost (€)", row=4, col=1, secondary_y=True)
    return fig


st.plotly_chart(sim_figure(), use_container_width=True)

st.subheader("Policy Comparison")
comparison_df = pd.DataFrame(
    {name: rollout_metrics(rollout, params) for name, rollout in single_rollouts.items()}
).T
st.dataframe(
    comparison_df.style.format({
        "Total cost (€)":                     "{:.3f}",
        "Energy charged (kWh)":               "{:.3f}",
        "Penalty minutes":                    "{:.0f}",
        "Final battery (kWh)":                "{:.3f}",
        "Mean charge rate while parked (kW)": "{:.3f}",
    }),
    use_container_width=True,
)

st.divider()

# ── N-scenario simulation ─────────────────────────────────────────────────────

st.subheader("N-Scenario Policy Comparison")

nd_col1, nd_col2, nd_col3, nd_col4 = st.columns(4)
with nd_col1:
    n_days = st.slider("Number of scenarios N", 10, 500, 100, 10, key="n_days")
with nd_col2:
    nd_e0 = st.slider("Initial battery (kWh) ", float(params.e_min), float(params.e_max),
                      float(params.e_max / 2), 0.5, key="nd_e0")
with nd_col3:
    nd_chi0 = st.radio("Initial state ", ["Parked", "Driving"], horizontal=True, key="nd_chi0")
with nd_col4:
    nd_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="nd_seed")

nd_chi0_int = 0 if nd_chi0 == "Parked" else 1

with st.spinner(f"Rolling out all policies over {n_days} scenarios…"):
    rng_nd = np.random.default_rng(int(nd_seed))
    nd_scenarios = [
        generate_rollout_scenario(params, int(rng_nd.integers(0, 1_000_000)), horizon=T)
        for _ in range(n_days)
    ]
    nd_rollouts: dict[str, list] = {name: [] for name in POLICY_COLORS}
    for sc in nd_scenarios:
        rng_rand = np.random.default_rng(int(rng_nd.integers(0, 1_000_000)))
        nd_rollouts["Backward induction"].append(simulate_policy_rollout(
            backward_induction_policy, sc, float(nd_e0), nd_chi0_int, params,
            pi=pi, actions=actions, e_grid=e_grid))
        nd_rollouts["DP heuristic"].append(simulate_policy_rollout(
            dp_heuristic_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Expected parking"].append(simulate_policy_rollout(
            expected_parking_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Maximal charging"].append(simulate_policy_rollout(
            maximal_charging_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Price-oriented"].append(simulate_policy_rollout(
            price_oriented_policy, sc, float(nd_e0), nd_chi0_int, params,
            low_threshold=low_threshold, high_threshold=high_threshold))
        nd_rollouts["Night charging"].append(simulate_policy_rollout(
            night_charging_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Minimum SoC"].append(simulate_policy_rollout(
            minimum_soc_policy, sc, float(nd_e0), nd_chi0_int, params,
            soc_threshold=soc_threshold))
        nd_rollouts["Always minimum"].append(simulate_policy_rollout(
            always_minimum_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Random"].append(simulate_policy_rollout(
            random_policy, sc, float(nd_e0), nd_chi0_int, params, rng=rng_rand))


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


def cost_box_figure() -> go.Figure:
    fig = go.Figure()
    for name, rolls in nd_rollouts.items():
        costs = [r["cost_traj"].sum() for r in rolls]
        fig.add_trace(go.Box(y=costs, name=name, marker_color=POLICY_COLORS[name]))
    fig.update_layout(
        title=f"Cost distribution across {n_days} scenarios",
        yaxis_title="Total cost (€)",
        height=500,
        margin=dict(l=30, r=30, t=55, b=35),
        showlegend=False,
    )
    return fig


st.plotly_chart(cost_box_figure(), use_container_width=True)
