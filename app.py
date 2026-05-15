import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import streamlit as st

from models.baseline import BaselineParams, mean_price, transition_probs, consumption
from utils.backward_induction import backward_induction

st.set_page_config(page_title="EV Charging MDP", layout="wide")
st.title("EV Charging MDP — Baseline Policy Explorer")


# ── Sidebar: parameter controls ───────────────────────────────────────────────

st.sidebar.header("Battery")
u_max = st.sidebar.slider("Max charge rate u_max (kW)", 1.0, 22.0, 11.0, 0.5)
u_min = st.sidebar.slider("Min charge rate u_min (kW)", 0.1, 5.0, 1.4, 0.1)
e_max = st.sidebar.slider("Battery capacity e_max (kWh)", 10.0, 100.0, 40.0, 1.0)
e_min = st.sidebar.slider("Min battery level e_min (kWh)", 0.0, 10.0, 0.0, 0.5)

st.sidebar.header("Charging & Cost")
eta_c = st.sidebar.slider("Charging efficiency η_c", 0.50, 1.00, 0.95, 0.01)
phi = st.sidebar.slider("Unserved-driving penalty φ (€/h)", 0.0, 5000.0, 1000.0, 50.0)
beta = st.sidebar.slider("Discount factor β", 0.900, 1.000, 0.999, 0.001, format="%.3f")

st.sidebar.header("Vehicle")
v = st.sidebar.slider("Driving speed v (km/h)", 10.0, 150.0, 50.0, 5.0)
mu = st.sidebar.slider("Energy consumption μ (kWh/km)", 0.05, 0.50, 0.20, 0.01)

st.sidebar.header("Electricity Price (€/MWh)")
price_night   = st.sidebar.slider("Night (00–06 h)",    0.0, 300.0, 70.0,  5.0)
price_morning = st.sidebar.slider("Morning peak (06–09 h)", 0.0, 300.0, 150.0, 5.0)
price_midday  = st.sidebar.slider("Midday (09–16 h)",   0.0, 300.0, 110.0, 5.0)
price_evening = st.sidebar.slider("Evening peak (16–21 h)", 0.0, 300.0, 170.0, 5.0)
price_late    = st.sidebar.slider("Late night (21–24 h)", 0.0, 300.0, 100.0, 5.0)
sigma_lambda  = st.sidebar.slider("Price std dev σ_λ (€/MWh)", 0.0, 100.0, 20.0, 1.0)

st.sidebar.header("Transition Probabilities (per minute)")
st.sidebar.subheader("Parked → Driving")
p_pd_morning = st.sidebar.slider("Morning  (07–09 h)", 0.0, 0.50, 0.08, 0.005, format="%.3f")
p_pd_lunch   = st.sidebar.slider("Lunch    (12–14 h)", 0.0, 0.50, 0.03, 0.005, format="%.3f")
p_pd_evening = st.sidebar.slider("Evening  (16–18 h)", 0.0, 0.50, 0.07, 0.005, format="%.3f")
p_pd_default = st.sidebar.slider("Default",            0.0, 0.10, 0.005, 0.001, format="%.3f")
st.sidebar.subheader("Driving → Parked")
p_dp_morning = st.sidebar.slider("Morning  (07:30–09:30 h)", 0.0, 1.0, 0.15, 0.01)
p_dp_lunch   = st.sidebar.slider("Lunch    (12:15–14:15 h)", 0.0, 1.0, 0.20, 0.01)
p_dp_evening = st.sidebar.slider("Evening  (16:30–18:30 h)", 0.0, 1.0, 0.15, 0.01)
p_dp_default = st.sidebar.slider("Default",                  0.0, 1.0, 0.25, 0.01)

st.sidebar.header("Solver")
N_e = st.sidebar.select_slider("Battery grid points N_e", [50, 100, 200, 500], value=200)


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
    V, pi, actions, e_grid = backward_induction(
        params,
        mean_price_fn=lambda t: mean_price(t, params),
        transition_probs_fn=lambda t: transition_probs(t, params),
        consumption_fn=lambda chi: consumption(chi, params),
        T=1440,
        N_e=N_e,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

PALETTE = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]
CMAP    = mcolors.ListedColormap(PALETTE)
BOUNDS  = np.arange(len(actions) + 1) - 0.5
NORM    = mcolors.BoundaryNorm(BOUNDS, CMAP.N)
HOURS   = np.arange(1440) / 60


def policy_heatmap(chi: int, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.pcolormesh(HOURS, e_grid, pi[:, chi, :].T, cmap=CMAP, norm=NORM)
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Battery level (kWh)")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    legend_handles = [
        mpatches.Patch(color=PALETTE[i], label=f"{actions[i]:.1f} kW")
        for i in range(len(actions))
    ]
    fig.legend(
        handles=legend_handles, title="Charge rate",
        loc="lower center", ncol=len(actions),
        bbox_to_anchor=(0.5, -0.08), frameon=False,
    )
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
    st.pyplot(policy_heatmap(0, "Optimal policy — Parked"))
with tab_driving:
    st.pyplot(policy_heatmap(1, "Optimal policy — Driving"))

st.subheader("Charge Rate vs. Price")
st.pyplot(charge_vs_price_figure())

st.subheader("Input Schedules")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(price_figure())
with col2:
    st.pyplot(transition_figure())
