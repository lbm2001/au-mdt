import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from models.baseline import BaselineParams, mean_price, transition_probs, consumption
from utils.backward_induction import backward_induction

st.set_page_config(page_title="EV Charging MDP", layout="wide")
st.title("EV Charging MDP — Baseline Policy Explorer")

_DEFAULTS = dict(
    u_max=11.0, u_min=1.4, e_max=40.0, e_min=0.0,
    eta_c=0.95, phi=1000.0, beta=0.999,
    v=50.0, mu=0.20,
    price_night=70.0, price_morning=150.0, price_midday=110.0,
    price_evening=170.0, price_late=100.0, sigma_lambda=20.0,
    p_pd_morning=0.08, p_pd_lunch=0.03, p_pd_evening=0.07, p_pd_default=0.005,
    p_dp_morning=0.15, p_dp_lunch=0.20, p_dp_evening=0.15, p_dp_default=0.25,
    N_e=200,  # must be a value present in the select_slider list
)

# ── Sidebar: parameter controls ───────────────────────────────────────────────

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

st.sidebar.header("Electricity Price (€/MWh)")
price_night   = st.sidebar.slider("Night (00–06 h)",         0.0, 300.0, 70.0,  5.0, key="price_night")
price_morning = st.sidebar.slider("Morning peak (06–09 h)",  0.0, 300.0, 150.0, 5.0, key="price_morning")
price_midday  = st.sidebar.slider("Midday (09–16 h)",        0.0, 300.0, 110.0, 5.0, key="price_midday")
price_evening = st.sidebar.slider("Evening peak (16–21 h)",  0.0, 300.0, 170.0, 5.0, key="price_evening")
price_late    = st.sidebar.slider("Late night (21–24 h)",    0.0, 300.0, 100.0, 5.0, key="price_late")
sigma_lambda  = st.sidebar.slider("Price std dev σ_λ (€/MWh)", 0.0, 100.0, 20.0, 1.0, key="sigma_lambda")

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

st.sidebar.header("Custom Transition Windows")
st.sidebar.caption("Override built-in peaks for any hour range. Last matching row wins.")
_empty_windows = pd.DataFrame({
    "start_h": pd.Series(dtype=float),
    "end_h":   pd.Series(dtype=float),
    "p_PD":    pd.Series(dtype=float),
    "p_DP":    pd.Series(dtype=float),
})
custom_df = st.sidebar.data_editor(
    st.session_state.get("custom_windows_df", _empty_windows),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "start_h": st.column_config.NumberColumn("Start (h)", min_value=0.0, max_value=24.0, step=0.25, format="%.2f"),
        "end_h":   st.column_config.NumberColumn("End (h)",   min_value=0.0, max_value=24.0, step=0.25, format="%.2f"),
        "p_PD":    st.column_config.NumberColumn("p_PD",      min_value=0.0, max_value=1.0,  step=0.005, format="%.3f"),
        "p_DP":    st.column_config.NumberColumn("p_DP",      min_value=0.0, max_value=1.0,  step=0.005, format="%.3f"),
    },
    key="custom_windows_editor",
)
st.session_state["custom_windows_df"] = custom_df
custom_windows = custom_df.dropna().to_dict("records")


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
    custom_windows=custom_windows,
)

with st.spinner("Running backward induction…"):
    V, pi, actions, e_grid = backward_induction(
        params,
        mean_price_fn=lambda t: mean_price(t, params),
        transition_probs_fn=lambda t: transition_probs(t, params),
        consumption_fn=lambda chi: consumption(chi, params),
        T=1440,
        N_e=N_e,
    )

st.session_state["V"] = V
st.session_state["pi"] = pi
st.session_state["actions"] = actions
st.session_state["e_grid"] = e_grid
st.session_state["params"] = params


# ── Helpers ───────────────────────────────────────────────────────────────────

HOURS = np.arange(1440) / 60


def policy_heatmap(chi: int, title: str, time_bin_minutes: int, battery_bin_kwh: float) -> plt.Figure:
    # Convert action indices → charge rates (kW): shape (1440, N_e)
    rates = actions[pi[:, chi, :]]

    # Average over time windows (truncate to nearest full bin)
    n_time_bins = 1440 // time_bin_minutes
    usable = n_time_bins * time_bin_minutes
    rates_time = rates[:usable].reshape(n_time_bins, time_bin_minutes, rates.shape[1]).mean(axis=1)

    # Average over battery bins
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

    time_edges = np.linspace(0, 24, n_time_bins + 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.pcolormesh(
        time_edges, bin_edges, rates_binned.T,
        cmap="RdYlBu_r", vmin=0, vmax=params.u_max,
    )
    plt.colorbar(im, ax=ax, label="Mean charge rate (kW)")
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Battery level (kWh)")
    ax.set_xticks(range(0, 25, 3))
    fig.tight_layout()
    return fig


def price_figure() -> plt.Figure:
    minutes = np.arange(1440)
    prices  = np.array([mean_price(t, params) for t in minutes])
    sigma   = params.sigma_lambda

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(HOURS, prices - sigma, prices + sigma,
                    step="post", alpha=0.2, color="steelblue", label=r"±σ_λ")
    ax.step(HOURS, prices, where="post", color="steelblue", linewidth=2, label=r"λ̄_t")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("€ / MWh")
    ax.set_title("Mean electricity price")
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def transition_figure() -> plt.Figure:
    probs = np.array([transition_probs(t, params) for t in range(1440)])
    fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
    axes[0].step(HOURS, probs[:, 0], where="post", color="tab:orange", linewidth=2)
    axes[0].set_ylabel("Prob. / min")
    axes[0].set_title("Parked → Driving")
    axes[0].grid(True, alpha=0.3)
    axes[1].step(HOURS, probs[:, 1], where="post", color="tab:green", linewidth=2)
    axes[1].set_ylabel("Prob. / min")
    axes[1].set_title("Driving → Parked")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_xlim(0, 24)
    axes[1].xaxis.set_major_locator(mticker.MultipleLocator(3))
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def charge_vs_price_figure() -> plt.Figure:
    prices          = np.array([mean_price(t, params) for t in range(1440)])
    charge_rates    = actions[pi[:, 0, :]]
    mean_per_minute = charge_rates.mean(axis=1)
    mean_per_hour   = mean_per_minute.reshape(24, 60).mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(11, 4))
    c_price = "steelblue"
    ax1.fill_between(HOURS, prices - params.sigma_lambda, prices + params.sigma_lambda,
                     step="post", alpha=0.2, color=c_price)
    ax1.step(HOURS, prices, where="post", color=c_price, linewidth=2, label="λ̄_t")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("€ / MWh", color=c_price)
    ax1.tick_params(axis="y", labelcolor=c_price)
    ax1.set_xlim(0, 24)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(3))
    ax1.grid(True, alpha=0.2)

    c_rate = "tab:red"
    ax2 = ax1.twinx()
    ax2.step(np.arange(24), mean_per_hour, where="post",
             color=c_rate, linewidth=2, label="Mean charge rate")
    ax2.set_ylabel("Mean charge rate (kW)", color=c_rate)
    ax2.tick_params(axis="y", labelcolor=c_rate)
    ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)
    ax1.set_title("Mean optimal charge rate vs. electricity price (parked, averaged over battery grid)")
    fig.tight_layout()
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.subheader("Optimal Policy")
tab_parked, tab_driving = st.tabs(["Parked", "Driving"])
with tab_parked:
    st.pyplot(policy_heatmap(0, "",
                             st.session_state.get("time_bin", 10),
                             st.session_state.get("bat_bin", 1.0)))
with tab_driving:
    st.pyplot(policy_heatmap(1, "Optimal policy — Driving",
                             st.session_state.get("time_bin", 10),
                             st.session_state.get("bat_bin", 1.0)))

col_tb, col_bb = st.columns(2)
with col_tb:
    st.select_slider("Time bin (minutes)", [1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60], value=10, key="time_bin")
with col_bb:
    st.slider("Battery bin (kWh)", 0.5, 10.0, 1.0, 0.5, key="bat_bin")

st.subheader("Charge Rate vs. Price")
st.pyplot(charge_vs_price_figure())

st.subheader("Input Schedules")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(price_figure())
with col2:
    st.pyplot(transition_figure())

# ── Simulation ────────────────────────────────────────────────────────────────

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

# re-run button just changes seed by 1 to trigger a fresh draw
if rerun_sim:
    st.session_state["sim_seed"] = int(seed) + 1
    seed = st.session_state["sim_seed"]
else:
    seed = st.session_state.get("sim_seed", int(seed))

rng    = np.random.default_rng(int(seed))
chi    = 0 if chi0 == "Parked" else 1
e      = float(e0)
N_e_   = len(e_grid)

e_traj    = np.zeros(1440)
chi_traj  = np.zeros(1440, dtype=int)
u_traj    = np.zeros(1440)
lam_traj  = np.zeros(1440)
cost_traj = np.zeros(1440)

for t in range(1440):
    e_traj[t]   = e
    chi_traj[t] = chi

    # look up policy
    e_idx = int(np.argmin(np.abs(e_grid - e)))
    a_idx = pi[t, chi, e_idx]
    u     = float(actions[a_idx])
    u_a   = 0.0 if (chi == 1 and e > params.e_min) else u

    # sample price
    lam = float(np.maximum(0.0, rng.normal(mean_price(t, params), params.sigma_lambda)))
    lam_traj[t] = lam
    u_traj[t]   = u_a

    # cost
    cost = lam / 1000 * params.omega * u_a
    if chi == 1 and e <= params.e_min:
        cost += params.omega * params.phi
    cost_traj[t] = cost

    # next battery
    cons = consumption(chi, params)
    e    = float(np.clip(e + params.eta_c * params.omega * u_a - cons, params.e_min, params.e_max))

    # next driving state
    p_PD, p_DP = transition_probs(t, params)
    if chi == 0:
        chi = 1 if rng.random() < p_PD else 0
    else:
        chi = 0 if rng.random() < p_DP else 1

hours = np.arange(1440) / 60

def sim_figure() -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Battery level
    axes[0].plot(hours, e_traj, color="steelblue", linewidth=1.2)
    axes[0].set_ylabel("Battery (kWh)")
    axes[0].set_ylim(params.e_min - 0.5, params.e_max + 0.5)
    axes[0].axhline(params.e_min, color="red", linewidth=0.8, linestyle="--", label="e_min")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Driving state
    axes[1].fill_between(hours, chi_traj, step="post", alpha=0.6, color="tab:orange", label="Driving")
    axes[1].set_ylabel("State")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Parked", "Driving"])
    axes[1].grid(True, alpha=0.3)

    # Charge rate
    axes[2].step(hours, u_traj, where="post", color="tab:green", linewidth=1.2)
    axes[2].set_ylabel("Charge rate (kW)")
    axes[2].set_ylim(-0.2, params.u_max + 0.5)
    axes[2].grid(True, alpha=0.3)

    # Price and cumulative cost
    ax_lam = axes[3]
    ax_lam.step(hours, lam_traj, where="post", color="steelblue", linewidth=1, alpha=0.7, label="λ_t (sampled)")
    ax_lam.step(hours, np.array([mean_price(t, params) for t in range(1440)]),
                where="post", color="steelblue", linewidth=1.5, linestyle="--", label="λ̄_t (mean)")
    ax_lam.set_ylabel("€ / MWh", color="steelblue")
    ax_lam.tick_params(axis="y", labelcolor="steelblue")
    ax_lam.legend(fontsize=8, loc="upper left")
    ax_lam.grid(True, alpha=0.3)

    ax_cost = ax_lam.twinx()
    ax_cost.plot(hours, np.cumsum(cost_traj), color="tab:red", linewidth=1.5, label="Cumulative cost (€)")
    ax_cost.set_ylabel("Cumulative cost (€)", color="tab:red")
    ax_cost.tick_params(axis="y", labelcolor="tab:red")
    ax_cost.legend(fontsize=8, loc="upper right")

    axes[3].set_xlabel("Hour of day")
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(3))

    total = cost_traj.sum()
    penalty = (chi_traj == 1) & (e_traj <= params.e_min)
    fig.suptitle(
        f"Simulated day — total cost: {total:.3f} € "
        f"(penalty minutes: {penalty.sum()})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig

st.pyplot(sim_figure())
