import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Policy Explorer — EV Charging MDP", layout="wide")
st.title("Policy Explorer")

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

HOURS = np.arange(T) / 60

# ── Model-specific imports ────────────────────────────────────────────────────

from ev_mdt.models.common.model_utils import price_bin_probs
from ev_mdt.plots.viz import POLICY_COLORS

if is_negbin:
    from ev_mdt.models.negbin import (
        mean_price, transition_probs, p_pd,
        maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy,
        dp_heuristic_policy, backward_induction_policy,
        simulate_policy_rollout, generate_rollout_scenario,
    )
else:
    from ev_mdt.models.baseline import (
        mean_price, transition_probs,
        maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy,
        dp_heuristic_policy, backward_induction_policy,
        simulate_policy_rollout, generate_rollout_scenario,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _paper_config(filename: str) -> dict:
    """Plotly config so the modebar download button exports a clean, paper-ready vector SVG."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 4},
    }


def _chart(fig, filename: str):
    """Render a chart full-width with a paper-ready SVG download configured."""
    st.plotly_chart(fig, use_container_width=True, config=_paper_config(filename))


def _binned_policy_rates(rates: np.ndarray, time_bin_minutes: int,
                         battery_bin_kwh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time_bins = T // time_bin_minutes
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

    time_centers    = (np.arange(n_time_bins) + 0.5) * time_bin_minutes / 60
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
        colorbar=dict(title="u (kW)"),
        hovertemplate="Hour: %{x:.2f}<br>Battery: %{y:.2f} kWh<br>Charge: %{z:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        #title=title,
        xaxis_title="Hour (h)",
        yaxis_title="Battery (kWh)",
        height=430,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    return fig


def effective_policy_rates(chi: int, desired_rates: np.ndarray) -> np.ndarray:
    rates = np.clip(desired_rates, 0.0, params.u_max).astype(float, copy=True)
    if chi > 0:  # any driving phase
        rates[:, e_grid > params.e_min] = 0.0
    return rates


def optimal_policy_rates(chi: int, lam_bin: int) -> np.ndarray:
    return effective_policy_rates(chi, actions[pi[:, chi, :, lam_bin]])


def optimal_policy_rates_averaged(chi: int) -> np.ndarray:
    """E_λ[actions[π(t,χ,e,k)]] weighted by the time-dependent price distribution."""
    desired = actions[pi[:, chi, :, :]]                                    # (T, N_e, K)
    weights = np.array([price_bin_probs(t, params) for t in range(T)])     # (T, K)
    averaged = (desired * weights[:, np.newaxis, :]).sum(axis=2)           # (T, N_e)
    return effective_policy_rates(chi, averaged)


def policy_price_map_figure(chi: int, t: int, title: str) -> go.Figure:
    """u*(battery × price bin) at a fixed minute t.

    Unlike the time–battery heatmap (which averages over price), this keeps the price
    axis, so the policy's price response — the charge-vs-defer threshold per SoC — is
    directly visible.
    """
    rates = actions[pi[t, chi, :, :]].astype(float, copy=True)   # (N_e, K)
    if chi > 0:  # driving: can only charge when essentially empty
        rates[e_grid > params.e_min, :] = 0.0
    fig = go.Figure(data=go.Heatmap(
        x=lam_grid,
        y=e_grid,
        z=rates,
        zmin=0,
        zmax=params.u_max,
        colorscale="RdYlBu_r",
        colorbar=dict(title="u (kW)"),
        hovertemplate="Price: %{x:.3f} €/kWh<br>Battery: %{y:.2f} kWh<br>Charge: %{z:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Price (€/kWh)",
        yaxis_title="Battery (kWh)",
        height=430,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

default_low  = float(params.price_night)
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

avg_bins = st.toggle("Average over all price bins", value=False, key="opt_avg_bins",
                     help="Weight each price-bin policy by the time-dependent price distribution p_t(k).")

lam_bin_sel   = st.session_state.get("lam_bin_sel", params.K // 2)
lam_bin_label = f"λ̂ = {lam_grid[lam_bin_sel]:.3f} €/kWh (bin {lam_bin_sel})"

if avg_bins:
    _opt_rates = optimal_policy_rates_averaged(0)
    _opt_title = "Optimal policy — Parked  |  price-averaged"
else:
    _opt_rates = optimal_policy_rates(0, lam_bin_sel)
    _opt_title = f"Optimal policy — Parked  |  {lam_bin_label}"

_chart(
    _policy_heatmap_figure(_opt_rates, _opt_title,
                           st.session_state.get("time_bin", 10),
                           st.session_state.get("bat_bin", 1.0)),
    "explorer_optimal_policy",
)

col_tb, col_bb, col_lb = st.columns(3)
with col_tb:
    st.select_slider("Time bin (min)", [1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60],
                     value=10, key="time_bin")
with col_bb:
    st.slider("Battery bin (kWh)", 0.5, 10.0, 1.0, 0.5, key="bat_bin")
with col_lb:
    if not avg_bins:
        st.slider("Price bin λ̂", 0, params.K - 1, params.K // 2, 1, key="lam_bin_sel",
                  help=f"Bin centre: {lam_grid[lam_bin_sel]:.3f} €/kWh")
    else:
        st.empty()

# ── Optimal policy vs price (fixed hour) ──────────────────────────────────────

st.subheader("Optimal Policy vs Price")
st.caption(
    "u*(battery × price) at a fixed hour — keeps the price axis instead of averaging it out, "
    "so the policy's charge-vs-defer price response per SoC is visible."
)
ppmap_hour = st.slider("Hour of day", 0, T_hours - 1, min(12, T_hours - 1), key="ppmap_hour")
_chart(
    policy_price_map_figure(
        0, ppmap_hour * 60,
        f"Optimal u*(battery × price) — Parked  |  hour {ppmap_hour:02d}:00",
    ),
    "explorer_policy_vs_price",
)
