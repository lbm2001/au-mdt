import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models.baseline import (
    BaselineParams, mean_price, transition_probs, consumption, price_bin_probs,
    backward_induction_policy, maximal_charging_policy, price_oriented_policy,
    night_charging_policy, minimum_soc_policy, always_minimum_policy,
    random_policy, plan_perfect_foresight, perfect_foresight_policy,
)
from models.baseline.rollout import (generate_rollout_scenario, rollout_metrics,
                                     simulate_policy_rollout)
from utils.backward_induction import backward_induction

st.set_page_config(page_title="EV Charging MDP", layout="wide")
st.title("EV Charging MDP — Baseline Policy Explorer")

_DEFAULTS = dict(
    u_max=11.0, u_min=1.4, e_max=40.0, e_min=0.0,
    eta_c=0.95, phi=1000.0, beta=0.999,
    v=50.0, mu=0.20,
    price_night=0.30, price_morning=0.48, price_midday=0.39,
    price_evening=0.55, price_late=0.34, sigma_lambda=0.05,
    p_pd_morning=0.08, p_pd_lunch=0.03, p_pd_evening=0.07, p_pd_default=0.005,
    p_dp_morning=0.15, p_dp_lunch=0.20, p_dp_evening=0.15, p_dp_default=0.25,
    N_e=200,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

if st.sidebar.button("Reset to defaults"):
    for key in _DEFAULTS:
        st.session_state.pop(key, None)
    st.rerun()

st.sidebar.header("Battery")
u_max = st.sidebar.slider("Max charge rate u_max (kW)", 1.0, 22.0, 11.0, 0.5, key="u_max")
u_min = st.sidebar.slider("Min charge rate u_min (kW)", 0.1, 5.0, 1.4, 0.1, key="u_min")
e_max = st.sidebar.slider("Battery capacity e_max (kWh)", 10.0, 100.0, 40.0, 1.0, key="e_max")
e_min = st.sidebar.slider("Min battery level e_min (kWh)", 0.0, 10.0, 0.0, 0.5, key="e_min")

st.sidebar.header("Charging & Cost")
eta_c = st.sidebar.slider("Charging efficiency η_c", 0.50, 1.00, 0.95, 0.01, key="eta_c")
phi = st.sidebar.slider("Unserved-driving penalty φ (€/h)", 0.0, 5000.0, 1000.0, 50.0, key="phi")
beta = st.sidebar.slider("Discount factor β", 0.900, 1.000, 0.999, 0.001, format="%.3f", key="beta")

st.sidebar.header("Vehicle")
v = st.sidebar.slider("Driving speed v (km/h)", 10.0, 150.0, 50.0, 5.0, key="v")
mu = st.sidebar.slider("Energy consumption μ (kWh/km)", 0.05, 0.50, 0.20, 0.01, key="mu")

st.sidebar.header("Electricity Price (€/kWh)")
price_night   = st.sidebar.slider("Night (00–06 h)",         0.0, 1.0, 0.30, 0.01, key="price_night")
price_morning = st.sidebar.slider("Morning peak (06–09 h)",  0.0, 1.0, 0.48, 0.01, key="price_morning")
price_midday  = st.sidebar.slider("Midday (09–16 h)",        0.0, 1.0, 0.39, 0.01, key="price_midday")
price_evening = st.sidebar.slider("Evening peak (16–21 h)",  0.0, 1.0, 0.55, 0.01, key="price_evening")
price_late    = st.sidebar.slider("Late night (21–24 h)",    0.0, 1.0, 0.34, 0.01, key="price_late")
sigma_lambda  = st.sidebar.slider("Price std dev σ_λ (€/kWh)", 0.0, 0.20, 0.05, 0.01, key="sigma_lambda")

st.sidebar.header("Transition Probabilities (per minute)")
st.sidebar.subheader("Parked → Driving")
p_pd_morning = st.sidebar.slider("Morning  (07–09 h)",  0.0, 0.50, 0.08,  0.005, format="%.3f", key="p_pd_morning")
p_pd_lunch   = st.sidebar.slider("Lunch    (12–14 h)",  0.0, 0.50, 0.03,  0.005, format="%.3f", key="p_pd_lunch")
p_pd_evening = st.sidebar.slider("Evening  (16–18 h)",  0.0, 0.50, 0.07,  0.005, format="%.3f", key="p_pd_evening")
p_pd_default = st.sidebar.slider("Default",             0.0, 0.10, 0.005, 0.001, format="%.3f", key="p_pd_default")
st.sidebar.subheader("Driving → Parked")
p_dp_morning = st.sidebar.slider("Morning  (07:30–09:30 h)", 0.0, 1.0, 0.15, 0.01, key="p_dp_morning")
p_dp_lunch   = st.sidebar.slider("Lunch    (12:15–14:15 h)", 0.0, 1.0, 0.20, 0.01, key="p_dp_lunch")
p_dp_evening = st.sidebar.slider("Evening  (16:30–18:30 h)", 0.0, 1.0, 0.15, 0.01, key="p_dp_evening")
p_dp_default = st.sidebar.slider("Default",                  0.0, 1.0, 0.25, 0.01, key="p_dp_default")

st.sidebar.header("Solver")
N_e = st.sidebar.select_slider("Battery grid points N_e", [25, 50, 100, 200, 500, 1000, 2000], value=200, key="N_e")

# ── Assemble params and solve ─────────────────────────────────────────────────

params = BaselineParams(
    u_max=u_max, u_min=u_min, e_max=e_max, e_min=e_min,
    eta_c=eta_c, phi=phi, beta=beta,
    v=v, mu=mu,
    price_night=price_night, price_morning=price_morning,
    price_midday=price_midday, price_evening=price_evening,
    price_late=price_late, sigma_lambda=sigma_lambda,
    p_pd_morning=p_pd_morning, p_pd_lunch=p_pd_lunch,
    p_pd_evening=p_pd_evening, p_pd_default=p_pd_default,
    p_dp_morning=p_dp_morning, p_dp_lunch=p_dp_lunch,
    p_dp_evening=p_dp_evening, p_dp_default=p_dp_default,
)

with st.spinner("Running backward induction…"):
    V, pi, actions, e_grid, lam_grid = backward_induction(
        params,
        transition_probs_fn=lambda t: transition_probs(t, params),
        consumption_fn=lambda chi: consumption(chi, params),
        price_bin_probs_fn=lambda t: price_bin_probs(t, params),
        T=1440,
        N_e=N_e,
    )

st.session_state["V"] = V
st.session_state["pi"] = pi
st.session_state["actions"] = actions
st.session_state["e_grid"] = e_grid
st.session_state["lam_grid"] = lam_grid
st.session_state["params"] = params


# ── Helpers ───────────────────────────────────────────────────────────────────

HOURS = np.arange(1440) / 60


def _binned_policy_rates(rates: np.ndarray, time_bin_minutes: int,
                         battery_bin_kwh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time_bins = 1440 // time_bin_minutes
    usable = n_time_bins * time_bin_minutes
    rates_time = rates[:usable].reshape(n_time_bins, time_bin_minutes, rates.shape[1]).mean(axis=1)

    bin_edges = np.arange(params.e_min, params.e_max + battery_bin_kwh, battery_bin_kwh)
    n_bins = len(bin_edges) - 1
    rates_binned = np.zeros((n_time_bins, n_bins))
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (e_grid >= bin_edges[i]) & (e_grid < bin_edges[i + 1])
        else:
            mask = (e_grid >= bin_edges[i]) & (e_grid <= bin_edges[i + 1])
        if mask.any():
            rates_binned[:, i] = rates_time[:, mask].mean(axis=1)

    time_centers = (np.arange(n_time_bins) + 0.5) * time_bin_minutes / 60
    battery_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return time_centers, battery_centers, rates_binned


def _policy_heatmap_figure(rates: np.ndarray, title: str, time_bin_minutes: int,
                           battery_bin_kwh: float) -> go.Figure:
    time_centers, battery_centers, rates_binned = _binned_policy_rates(
        rates, time_bin_minutes, battery_bin_kwh,
    )
    fig = go.Figure(data=go.Heatmap(
        x=time_centers,
        y=battery_centers,
        z=rates_binned.T,
        zmin=0,
        zmax=params.u_max,
        colorscale="RdYlBu_r",
        colorbar=dict(title="kW"),
        hovertemplate="Hour: %{x:.2f}<br>Battery: %{y:.2f} kWh<br>Charge: %{z:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Hour of day",
        yaxis_title="Battery level (kWh)",
        height=430,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    fig.update_xaxes(range=[0, 24], dtick=3)
    return fig


def optimal_policy_rates(chi: int, lam_bin: int) -> np.ndarray:
    return effective_policy_rates(chi, actions[pi[:, chi, :, lam_bin]])


def benchmark_policy_rates(policy_fn, chi: int, **policy_kwargs) -> np.ndarray:
    """Compute desired charge rates for a benchmark policy over (1440, N_e)."""
    t_arr = np.arange(1440)
    if policy_fn is maximal_charging_policy:
        desired_rates = np.full((1440, len(e_grid)), params.u_max)
    elif policy_fn is price_oriented_policy:
        lam_path = np.array([mean_price(t, params) for t in t_arr])
        low, high = policy_kwargs["low_threshold"], policy_kwargs["high_threshold"]
        per_min = np.where(lam_path <= low, params.u_max,
                           np.where(lam_path <= high, params.u_max / 2, 0.0))
        desired_rates = np.repeat(per_min[:, np.newaxis], len(e_grid), axis=1)
    elif policy_fn is night_charging_policy:
        per_min = np.where(t_arr < 360, params.u_max, 0.0)
        desired_rates = np.repeat(per_min[:, np.newaxis], len(e_grid), axis=1)
    elif policy_fn is always_minimum_policy:
        desired_rates = np.full((1440, len(e_grid)), params.u_min)
    elif policy_fn is minimum_soc_policy:
        threshold = policy_kwargs["soc_threshold"]
        desired_rates = np.where(e_grid[np.newaxis, :] < threshold, params.u_max, 0.0)
        desired_rates = np.broadcast_to(desired_rates, (1440, len(e_grid))).copy()
    else:
        desired_rates = np.zeros((1440, len(e_grid)))
        for t in t_arr:
            lam = mean_price(t, params)
            for i, e in enumerate(e_grid):
                desired_rates[t, i] = policy_fn(
                    t=t, chi=chi, e=float(e), lam=lam, params=params, **policy_kwargs,
                )
    return effective_policy_rates(chi, desired_rates)


def effective_policy_rates(chi: int, desired_rates: np.ndarray) -> np.ndarray:
    rates = np.clip(desired_rates, 0.0, params.u_max).astype(float, copy=True)
    if chi == 1:
        rates[:, e_grid > params.e_min] = 0.0
    return rates


def policy_heatmap(chi: int, title: str, time_bin_minutes: int, battery_bin_kwh: float,
                   lam_bin: int) -> go.Figure:
    return _policy_heatmap_figure(
        optimal_policy_rates(chi, lam_bin),
        title,
        time_bin_minutes,
        battery_bin_kwh,
    )


def price_figure() -> go.Figure:
    minutes = np.arange(1440)
    prices  = np.array([mean_price(t, params) for t in minutes])
    sigma   = params.sigma_lambda
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([HOURS, HOURS[::-1]]),
        y=np.concatenate([prices - sigma, (prices + sigma)[::-1]]),
        fill="toself",
        fillcolor="rgba(70, 130, 180, 0.2)",
        line=dict(color="rgba(70, 130, 180, 0)"),
        name="±σ_λ",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=HOURS,
        y=prices,
        mode="lines",
        line=dict(color="steelblue", width=2, shape="hv"),
        name="λ̄_t",
        hovertemplate="Hour: %{x:.2f}<br>Price: %{y:.3f} €/kWh<extra></extra>",
    ))
    fig.update_layout(
        title="Mean electricity price",
        xaxis_title="Hour of day",
        yaxis_title="€ / kWh",
        height=320,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    fig.update_xaxes(range=[0, 24], dtick=3)
    return fig


def transition_figure() -> go.Figure:
    probs = np.array([transition_probs(t, params) for t in range(1440)])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Parked → Driving", "Driving → Parked"),
        vertical_spacing=0.13,
    )
    fig.add_trace(go.Scatter(x=HOURS, y=probs[:, 0], mode="lines",
                             line=dict(color="orange", width=2, shape="hv"), name="p_PD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=HOURS, y=probs[:, 1], mode="lines",
                             line=dict(color="green", width=2, shape="hv"), name="p_DP"), row=2, col=1)
    fig.update_layout(height=430, margin=dict(l=30, r=30, t=60, b=35))
    fig.update_xaxes(range=[0, 24], dtick=3)
    fig.update_yaxes(title_text="Prob. / min", row=1, col=1)
    fig.update_yaxes(title_text="Prob. / min", row=2, col=1)
    fig.update_xaxes(title_text="Hour of day", row=2, col=1)
    return fig


def charge_vs_price_figure(low_threshold: float, high_threshold: float,
                           soc_threshold: float) -> go.Figure:
    prices   = np.array([mean_price(t, params) for t in range(1440)])
    lam_bin  = st.session_state.get("lam_bin_sel", params.K // 2)
    policy_rates = {
        "Backward induction":      optimal_policy_rates(0, lam_bin),
        "Maximal charging":        benchmark_policy_rates(maximal_charging_policy, 0),
        "Price-oriented":          benchmark_policy_rates(price_oriented_policy, 0,
                                       low_threshold=low_threshold, high_threshold=high_threshold),
        "Night charging":          benchmark_policy_rates(night_charging_policy, 0),
        "Minimum SoC":             benchmark_policy_rates(minimum_soc_policy, 0,
                                       soc_threshold=soc_threshold),
        "Always minimum":          benchmark_policy_rates(always_minimum_policy, 0),
    }
    colors = {
        "Backward induction": "steelblue",
        "Maximal charging": "seagreen",
        "Price-oriented": "crimson",
        "Night charging": "purple",
        "Minimum SoC": "darkorange",
        "Always minimum": "gray",
    }
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=np.concatenate([HOURS, HOURS[::-1]]),
        y=np.concatenate([prices - params.sigma_lambda, (prices + params.sigma_lambda)[::-1]]),
        fill="toself", fillcolor="rgba(70, 130, 180, 0.2)",
        line=dict(color="rgba(70, 130, 180, 0)"), name="±σ_λ", hoverinfo="skip",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(x=HOURS, y=prices, mode="lines",
                             line=dict(color="steelblue", width=2, shape="hv"), name="λ̄_t"),
                  secondary_y=False)
    for name, rates in policy_rates.items():
        mean_per_hour = rates.mean(axis=1).reshape(24, 60).mean(axis=1)
        fig.add_trace(go.Scatter(x=np.arange(24), y=mean_per_hour, mode="lines",
                                 line=dict(color=colors[name], width=2, shape="hv"),
                                 name=f"{name}"), secondary_y=True)
    fig.update_layout(title="Mean charge rate vs. electricity price (parked, averaged over battery grid)",
                      height=430, margin=dict(l=30, r=30, t=55, b=35))
    fig.update_xaxes(title_text="Hour of day", range=[0, 24], dtick=3)
    fig.update_yaxes(title_text="€ / kWh", secondary_y=False)
    fig.update_yaxes(title_text="Mean charge rate (kW)", secondary_y=True, rangemode="tozero")
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

default_low = float(params.price_night)
default_high = float(params.price_evening)
previous_defaults = st.session_state.get("benchmark_threshold_defaults")
if "benchmark_low_threshold" not in st.session_state:
    st.session_state["benchmark_low_threshold"] = default_low
elif previous_defaults and float(st.session_state["benchmark_low_threshold"]) == previous_defaults[0]:
    st.session_state["benchmark_low_threshold"] = default_low
if "benchmark_high_threshold" not in st.session_state:
    st.session_state["benchmark_high_threshold"] = default_high
elif previous_defaults and float(st.session_state["benchmark_high_threshold"]) == previous_defaults[1]:
    st.session_state["benchmark_high_threshold"] = default_high
st.session_state["benchmark_threshold_defaults"] = (default_low, default_high)
st.session_state["benchmark_low_threshold"] = min(
    float(st.session_state["benchmark_low_threshold"]),
    float(st.session_state["benchmark_high_threshold"]),
)

# ── Optimal policy heatmap ────────────────────────────────────────────────────

st.subheader("Optimal Policy")

lam_bin_sel = st.session_state.get("lam_bin_sel", params.K // 2)
lam_bin_label = f"λ̂ = {lam_grid[lam_bin_sel]:.3f} €/kWh (bin {lam_bin_sel})"

st.plotly_chart(policy_heatmap(0, f"Optimal policy — Parked  |  {lam_bin_label}",
                                st.session_state.get("time_bin", 10),
                                st.session_state.get("bat_bin", 1.0),
                                lam_bin_sel), use_container_width=True)

col_tb, col_bb, col_lb, col_low, col_high, col_soc = st.columns(6)
with col_tb:
    st.select_slider("Time bin (minutes)", [1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60], value=10, key="time_bin")
with col_bb:
    st.slider("Battery bin (kWh)", 0.5, 10.0, 1.0, 0.5, key="bat_bin")
with col_lb:
    st.slider("Price bin λ̂", 0, params.K - 1, params.K // 2, 1, key="lam_bin_sel",
              help=f"Bin centre: {lam_grid[lam_bin_sel]:.3f} €/kWh")
with col_low:
    low_threshold = st.slider("Low price threshold (€/kWh)", 0.0, 1.0,
                              key="benchmark_low_threshold", step=0.01)
if float(st.session_state["benchmark_high_threshold"]) < float(low_threshold):
    st.session_state["benchmark_high_threshold"] = float(low_threshold)
with col_high:
    high_threshold = st.slider("High price threshold (€/kWh)", low_threshold, 1.0,
                               key="benchmark_high_threshold", step=0.01)
with col_soc:
    soc_threshold = st.slider("Min SoC threshold (kWh)", float(params.e_min),
                              float(params.e_max), float(params.e_max * 0.25), 0.5,
                              key="soc_threshold")

# ── Benchmark heatmaps ────────────────────────────────────────────────────────

st.subheader("Benchmark Policy Heatmaps")
bench_tabs = st.tabs([
    "Maximal charging",
    "Price-oriented",
    "Night charging",
    "Minimum SoC",
    "Always minimum",
])
time_bin = st.session_state.get("time_bin", 10)
bat_bin  = st.session_state.get("bat_bin", 1.0)

with bench_tabs[0]:
    st.plotly_chart(_policy_heatmap_figure(
        benchmark_policy_rates(maximal_charging_policy, 0),
        "Maximal charging — Parked", time_bin, bat_bin,
    ), use_container_width=True)
with bench_tabs[1]:
    st.plotly_chart(_policy_heatmap_figure(
        benchmark_policy_rates(price_oriented_policy, 0,
                               low_threshold=float(low_threshold),
                               high_threshold=float(high_threshold)),
        "Price-oriented charging — Parked", time_bin, bat_bin,
    ), use_container_width=True)
with bench_tabs[2]:
    st.plotly_chart(_policy_heatmap_figure(
        benchmark_policy_rates(night_charging_policy, 0),
        "Night charging — Parked", time_bin, bat_bin,
    ), use_container_width=True)
with bench_tabs[3]:
    st.plotly_chart(_policy_heatmap_figure(
        benchmark_policy_rates(minimum_soc_policy, 0, soc_threshold=float(soc_threshold)),
        f"Minimum SoC — Parked  |  threshold = {soc_threshold:.1f} kWh", time_bin, bat_bin,
    ), use_container_width=True)
with bench_tabs[4]:
    st.plotly_chart(_policy_heatmap_figure(
        benchmark_policy_rates(always_minimum_policy, 0),
        "Always minimum rate — Parked", time_bin, bat_bin,
    ), use_container_width=True)

# ── Charge rate vs price ──────────────────────────────────────────────────────

st.subheader("Charge Rate vs. Price")
st.plotly_chart(charge_vs_price_figure(float(low_threshold), float(high_threshold),
                                       float(soc_threshold)), use_container_width=True)

# ── Input schedules ───────────────────────────────────────────────────────────

st.subheader("Input Schedules")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(price_figure(), use_container_width=True)
with col2:
    st.plotly_chart(transition_figure(), use_container_width=True)

# ── Single-day simulation ─────────────────────────────────────────────────────

st.subheader("Policy Rollout Simulation")

sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
with sim_col1:
    e0 = st.slider("Initial battery (kWh)", float(params.e_min), float(params.e_max),
                   float(params.e_max / 2), float((params.e_max - params.e_min) / (len(e_grid) - 1)))
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

chi0_int = 0 if chi0 == "Parked" else 1
scenario = generate_rollout_scenario(params, int(seed))

u_plan = plan_perfect_foresight(scenario, float(e0), chi0_int, params)

single_day_rollouts = {
    "Backward induction": simulate_policy_rollout(
        backward_induction_policy, scenario, float(e0), chi0_int, params,
        pi=pi, actions=actions, e_grid=e_grid),
    "Maximal charging": simulate_policy_rollout(
        maximal_charging_policy, scenario, float(e0), chi0_int, params),
    "Price-oriented": simulate_policy_rollout(
        price_oriented_policy, scenario, float(e0), chi0_int, params,
        low_threshold=float(low_threshold), high_threshold=float(high_threshold)),
    "Night charging": simulate_policy_rollout(
        night_charging_policy, scenario, float(e0), chi0_int, params),
    "Minimum SoC": simulate_policy_rollout(
        minimum_soc_policy, scenario, float(e0), chi0_int, params,
        soc_threshold=float(soc_threshold)),
    "Always minimum": simulate_policy_rollout(
        always_minimum_policy, scenario, float(e0), chi0_int, params),
    "Random": simulate_policy_rollout(
        random_policy, scenario, float(e0), chi0_int, params,
        rng=np.random.default_rng(int(seed))),
    "Perfect foresight": simulate_policy_rollout(
        perfect_foresight_policy, scenario, float(e0), chi0_int, params,
        u_plan=u_plan),
}

POLICY_COLORS = {
    "Backward induction": "steelblue",
    "Maximal charging":   "seagreen",
    "Price-oriented":     "crimson",
    "Night charging":     "purple",
    "Minimum SoC":        "darkorange",
    "Always minimum":     "gray",
    "Random":             "pink",
    "Perfect foresight":  "black",
}

hours = np.arange(1440) / 60
lam_traj = scenario["lam_path"]
chi_traj_ref = single_day_rollouts["Backward induction"]["chi_traj"]

def sim_figure() -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("Battery level", "Mobility state", "Charge rate",
                        "Price and cumulative cost"),
        vertical_spacing=0.07,
    )
    for name, rollout in single_day_rollouts.items():
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
    fig.add_trace(go.Scatter(x=hours, y=chi_traj_ref, mode="lines", fill="tozeroy",
                             line=dict(color="orange", width=1.2, shape="hv"),
                             name="Driving state",
                             hovertemplate="Hour: %{x:.2f}<br>State: %{y}<extra></extra>"),
                 row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=lam_traj, mode="lines",
                             line=dict(color="lightgray", width=1.0, shape="hv"),
                             name="λ_t sampled"), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=hours,
                             y=np.array([mean_price(t, params) for t in range(1440)]),
                             mode="lines",
                             line=dict(color="black", width=1.4, dash="dash", shape="hv"),
                             name="λ̄_t mean"), row=4, col=1, secondary_y=False)
    fig.update_layout(height=1000, hovermode="x unified",
                      margin=dict(l=30, r=30, t=80, b=35))
    fig.update_xaxes(range=[0, 24], dtick=3)
    fig.update_xaxes(title_text="Hour of day", row=4, col=1)
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

st.subheader("Single-Day Policy Comparison")
comparison_df = pd.DataFrame(
    {name: rollout_metrics(rollout, params) for name, rollout in single_day_rollouts.items()}
).T
st.dataframe(
    comparison_df.style.format({
        "Total cost (€)": "{:.3f}",
        "Energy charged (kWh)": "{:.3f}",
        "Penalty minutes": "{:.0f}",
        "Final battery (kWh)": "{:.3f}",
        "Mean charge rate while parked (kW)": "{:.3f}",
    }),
    use_container_width=True,
)

# ── N-day simulation ──────────────────────────────────────────────────────────

st.subheader("N-Day Policy Comparison")

nd_col1, nd_col2, nd_col3, nd_col4 = st.columns(4)
with nd_col1:
    n_days = st.slider("Number of days N", 10, 500, 100, 10, key="n_days")
with nd_col2:
    nd_e0 = st.slider("Initial battery (kWh) ", float(params.e_min), float(params.e_max),
                      float(params.e_max / 2), 0.5, key="nd_e0")
with nd_col3:
    nd_chi0 = st.radio("Initial state ", ["Parked", "Driving"], horizontal=True, key="nd_chi0")
with nd_col4:
    nd_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="nd_seed")

nd_chi0_int = 0 if nd_chi0 == "Parked" else 1

with st.spinner(f"Rolling out all policies over {n_days} days…"):
    rng_nd = np.random.default_rng(int(nd_seed))
    nd_scenarios = [generate_rollout_scenario(params, int(rng_nd.integers(0, 1_000_000)))
                    for _ in range(n_days)]

    nd_rollouts: dict[str, list] = {name: [] for name in POLICY_COLORS}
    for sc in nd_scenarios:
        u_plan_nd = plan_perfect_foresight(sc, float(nd_e0), nd_chi0_int, params)
        rng_rand = np.random.default_rng(int(rng_nd.integers(0, 1_000_000)))
        nd_rollouts["Backward induction"].append(simulate_policy_rollout(
            backward_induction_policy, sc, float(nd_e0), nd_chi0_int, params,
            pi=pi, actions=actions, e_grid=e_grid))
        nd_rollouts["Maximal charging"].append(simulate_policy_rollout(
            maximal_charging_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Price-oriented"].append(simulate_policy_rollout(
            price_oriented_policy, sc, float(nd_e0), nd_chi0_int, params,
            low_threshold=float(low_threshold), high_threshold=float(high_threshold)))
        nd_rollouts["Night charging"].append(simulate_policy_rollout(
            night_charging_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Minimum SoC"].append(simulate_policy_rollout(
            minimum_soc_policy, sc, float(nd_e0), nd_chi0_int, params,
            soc_threshold=float(soc_threshold)))
        nd_rollouts["Always minimum"].append(simulate_policy_rollout(
            always_minimum_policy, sc, float(nd_e0), nd_chi0_int, params))
        nd_rollouts["Random"].append(simulate_policy_rollout(
            random_policy, sc, float(nd_e0), nd_chi0_int, params, rng=rng_rand))
        nd_rollouts["Perfect foresight"].append(simulate_policy_rollout(
            perfect_foresight_policy, sc, float(nd_e0), nd_chi0_int, params, u_plan=u_plan_nd))

# Build statistics table
def _nd_stats(rollout_list: list) -> dict:
    costs    = np.array([r["cost_traj"].sum() for r in rollout_list])
    pen_mins = np.array([int(((r["chi_traj"] == 1) & (r["e_traj"] <= params.e_min)).sum())
                         for r in rollout_list])
    energy   = np.array([(r["u_traj"] * params.omega).sum() for r in rollout_list])
    final_e  = np.array([r["final_e"] for r in rollout_list])
    return {
        "Mean cost (€)":        costs.mean(),
        "Std cost (€)":         costs.std(),
        "Median cost (€)":      float(np.median(costs)),
        "Mean penalty min":     pen_mins.mean(),
        "% days with penalty":  float((pen_mins > 0).mean() * 100),
        "Mean energy charged (kWh)": energy.mean(),
        "Mean final battery (kWh)": final_e.mean(),
    }

stats_df = pd.DataFrame({name: _nd_stats(rolls) for name, rolls in nd_rollouts.items()}).T
st.dataframe(
    stats_df.style.format({
        "Mean cost (€)":             "{:.3f}",
        "Std cost (€)":              "{:.3f}",
        "Median cost (€)":           "{:.3f}",
        "Mean penalty min":          "{:.1f}",
        "% days with penalty":       "{:.1f}%",
        "Mean energy charged (kWh)": "{:.2f}",
        "Mean final battery (kWh)":  "{:.2f}",
    }),
    use_container_width=True,
)

# Cost distribution box plot
def cost_box_figure() -> go.Figure:
    fig = go.Figure()
    for name, rolls in nd_rollouts.items():
        costs = [r["cost_traj"].sum() for r in rolls]
        fig.add_trace(go.Box(y=costs, name=name, marker_color=POLICY_COLORS[name],
                             boxmean="sd", whiskerwidth=0))
    fig.update_layout(
        title=f"Daily cost distribution across {n_days} days",
        yaxis_title="Total cost (€)",
        height=500,
        margin=dict(l=30, r=30, t=55, b=35),
        showlegend=False,
    )
    return fig

st.plotly_chart(cost_box_figure(), use_container_width=True)
