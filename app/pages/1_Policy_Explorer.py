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

from models.model_utils import price_bin_probs
from utils.viz import POLICY_COLORS    # shared canonical colour map

if is_negbin:
    from models.negative_binomial_trips import (
        mean_price, transition_probs, p_pd,
        maximal_charging_policy, price_oriented_policy,
        night_charging_policy, minimum_soc_policy, always_minimum_policy,
        dp_heuristic_policy, backward_induction_policy,
        simulate_policy_rollout, generate_rollout_scenario,
    )
else:
    from models.baseline import (
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


def policy_threshold_figure(chi: int, time_bin_minutes: int, title: str) -> go.Figure:
    """Charging price-threshold map: colour = highest price at which the policy still charges.

    Collapses the price dimension to a single interpretable surface — charge while the current
    price ≤ colour; blank where it never charges at that (t, e).
    """
    import warnings
    u = actions[pi[:, chi, :, :]].astype(float)        # (T, N_e, K)
    if chi > 0:
        u[:, e_grid > params.e_min, :] = 0.0
    charging = u > 0
    last_k   = (charging * (np.arange(len(lam_grid)) + 1)).max(axis=2) - 1     # (T, N_e)
    thr = np.where(charging.any(axis=2),
                   lam_grid[np.clip(last_k, 0, len(lam_grid) - 1)], np.nan)    # (T, N_e)
    n_t    = max(1, T // time_bin_minutes)
    usable = n_t * time_bin_minutes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thr_b = np.nanmean(thr[:usable].reshape(n_t, time_bin_minutes, -1), axis=1)   # (n_t, N_e)
    t_centers = (np.arange(n_t) + 0.5) * time_bin_minutes / 60
    fig = go.Figure(data=go.Heatmap(
        x=t_centers, y=e_grid, z=thr_b.T,
        colorscale="Viridis", colorbar=dict(title="€/kWh"),
        hovertemplate="Hour: %{x:.2f}<br>Battery: %{y:.1f} kWh<br>charge if price ≤ %{z:.3f} €/kWh<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Hour (h)", yaxis_title="Battery (kWh)",
                      height=430, margin=dict(l=30, r=30, t=55, b=35))
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    return fig


def rollout_trajectory_figure(seed: int) -> go.Figure:
    """One simulated optimal-policy trajectory: price, battery SoC (driving shaded), charge rate."""
    scenario = generate_rollout_scenario(params, seed, horizon=T)
    ro = simulate_policy_rollout(
        backward_induction_policy, scenario, float(params.e_max / 2), 0, params,
        pi=pi, actions=actions, e_grid=e_grid,
    )
    hours   = np.arange(T) / 60
    driving = ro["chi_traj"] > 0
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.3, 0.4, 0.3],
        subplot_titles=("Electricity price λ_t", "Battery SoC  (grey = driving)", "Charge rate u_t"),
    )
    fig.add_trace(go.Scatter(x=hours, y=ro["lam_traj"], mode="lines",
                             line=dict(color="steelblue", width=1.5, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=np.where(driving, params.e_max, 0.0), fill="tozeroy",
                             fillcolor="rgba(150,150,150,0.25)", line=dict(width=0),
                             hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=ro["e_traj"], mode="lines",
                             line=dict(color="seagreen", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=ro["u_traj"], mode="lines",
                             line=dict(color="crimson", width=1.5, shape="hv")), row=3, col=1)
    fig.update_layout(height=560, margin=dict(l=30, r=30, t=55, b=35), showlegend=False)
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", range=[0, params.e_max], row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_xaxes(title_text="Hour (h)", range=[0, T_hours], dtick=T_hours // 8, row=3, col=1)
    return fig


def benchmark_policy_rates(policy_fn, chi: int, **policy_kwargs) -> np.ndarray:
    """Compute desired charge rates for a benchmark policy over (T, N_e)."""
    t_arr = np.arange(T)
    if policy_fn is maximal_charging_policy:
        desired_rates = np.full((T, len(e_grid)), params.u_max)
    elif policy_fn is price_oriented_policy:
        lam_path = np.array([mean_price(t, params) for t in t_arr])
        low, high = policy_kwargs["low_threshold"], policy_kwargs["high_threshold"]
        per_min = np.where(lam_path <= low, params.u_max,
                           np.where(lam_path <= high, params.u_max / 2, 0.0))
        desired_rates = np.repeat(per_min[:, np.newaxis], len(e_grid), axis=1)
    elif policy_fn is night_charging_policy:
        per_min = np.where(t_arr % 1440 < 360, params.u_max, 0.0)
        desired_rates = np.repeat(per_min[:, np.newaxis], len(e_grid), axis=1)
    elif policy_fn is always_minimum_policy:
        desired_rates = np.full((T, len(e_grid)), params.u_min)
    elif policy_fn is minimum_soc_policy:
        threshold = policy_kwargs["soc_threshold"]
        desired_rates = np.where(e_grid[np.newaxis, :] < threshold, params.u_max, 0.0)
        desired_rates = np.broadcast_to(desired_rates, (T, len(e_grid))).copy()
    elif policy_fn is dp_heuristic_policy:
        # E[u(t,e)] = u_max × Σ_k prob_k × 1[F_cdf(k) ≤ 1 − e/e_max]
        desired_rates = np.zeros((T, len(e_grid)))
        thresh = 1.0 - e_grid / params.e_max          # (N_e,) — higher when emptier
        for t in t_arr:
            probs = price_bin_probs(t, params)         # (K,)
            F_cdf = np.cumsum(probs)                   # (K,)
            # charge mask: (N_e, K) — True where price bin qualifies
            mask = F_cdf[np.newaxis, :] <= thresh[:, np.newaxis]
            desired_rates[t] = params.u_max * (probs[np.newaxis, :] * mask).sum(axis=1)
    else:
        desired_rates = np.zeros((T, len(e_grid)))
        for t in t_arr:
            lam = mean_price(t, params)
            for i, e in enumerate(e_grid):
                desired_rates[t, i] = policy_fn(
                    t=t, chi=chi, e=float(e), lam=lam, params=params, **policy_kwargs,
                )
    return effective_policy_rates(chi, desired_rates)


def policy_heatmap(chi: int, title: str, time_bin_minutes: int, battery_bin_kwh: float,
                   lam_bin: int) -> go.Figure:
    return _policy_heatmap_figure(
        optimal_policy_rates(chi, lam_bin), title, time_bin_minutes, battery_bin_kwh,
    )


def price_figure() -> go.Figure:
    minutes = np.arange(T)
    prices  = np.array([mean_price(t, params) for t in minutes])
    sigma   = params.sigma_lambda
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([HOURS, HOURS[::-1]]),
        y=np.concatenate([prices - sigma, (prices + sigma)[::-1]]),
        fill="toself", fillcolor="rgba(70, 130, 180, 0.2)",
        line=dict(color="rgba(70, 130, 180, 0)"), name="±σ_λ", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=HOURS, y=prices, mode="lines",
        line=dict(color="steelblue", width=2, shape="hv"), name="λ̄_t",
        hovertemplate="Hour: %{x:.2f}<br>Price: %{y:.3f} €/kWh<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Hour (h)",
                      yaxis_title="€/kWh", height=320,
                      margin=dict(l=30, r=30, t=55, b=35))
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    return fig


def transition_figure() -> go.Figure:
    if is_negbin:
        ppd = np.array([p_pd(t, params) for t in range(T)])
        q_line = np.full(T, params.q)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Parked → Driving (p_PD)",
                                            f"Phase D_i → D_{{i+1}} / P  (q = {params.q:.2f})"),
                            vertical_spacing=0.13)
        fig.add_trace(go.Scatter(x=HOURS, y=ppd, mode="lines",
                                 line=dict(color="orange", width=2, shape="hv"), name="p_PD"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=HOURS, y=q_line, mode="lines",
                                 line=dict(color="green", width=2, shape="hv"), name="q"),
                      row=2, col=1)
    else:
        probs = np.array([transition_probs(t, params) for t in range(T)])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Parked → Driving", "Driving → Parked"),
                            vertical_spacing=0.13)
        fig.add_trace(go.Scatter(x=HOURS, y=probs[:, 0], mode="lines",
                                 line=dict(color="orange", width=2, shape="hv"), name="p_PD"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=HOURS, y=probs[:, 1], mode="lines",
                                 line=dict(color="green", width=2, shape="hv"), name="p_DP"),
                      row=2, col=1)
    fig.update_layout(height=430, margin=dict(l=30, r=30, t=60, b=35))
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    fig.update_yaxes(title_text="Prob. / min", row=1, col=1)
    fig.update_yaxes(title_text="Prob. / min", row=2, col=1)
    fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
    return fig


def charge_vs_price_figure(low_threshold: float, high_threshold: float,
                           soc_threshold: float) -> go.Figure:
    prices  = np.array([mean_price(t, params) for t in range(T)])
    lam_bin = st.session_state.get("lam_bin_sel", params.K // 2)
    policy_rates = {
        "Backward induction": optimal_policy_rates(0, lam_bin),
        "DP heuristic":       benchmark_policy_rates(dp_heuristic_policy, 0),
        "Maximal charging":   benchmark_policy_rates(maximal_charging_policy, 0),
        "Price-oriented":     benchmark_policy_rates(price_oriented_policy, 0,
                                  low_threshold=low_threshold, high_threshold=high_threshold),
        "Night charging":     benchmark_policy_rates(night_charging_policy, 0),
        "Minimum SoC":        benchmark_policy_rates(minimum_soc_policy, 0,
                                  soc_threshold=soc_threshold),
        "Always minimum":     benchmark_policy_rates(always_minimum_policy, 0),
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
        mean_per_hour = rates.mean(axis=1).reshape(T_hours, 60).mean(axis=1)
        fig.add_trace(go.Scatter(x=np.arange(T_hours), y=mean_per_hour, mode="lines",
                                 line=dict(color=POLICY_COLORS[name], width=2, shape="hv"),
                                 name=name), secondary_y=True)
    fig.update_layout(height=430, margin=dict(l=30, r=30, t=55, b=35))
    fig.update_xaxes(title_text="Hour (h)", range=[0, T_hours], dtick=T_hours // 8)
    fig.update_yaxes(title_text="€/kWh", secondary_y=False)
    fig.update_yaxes(title_text="u (kW)", secondary_y=True, rangemode="tozero")
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

col_tb, col_bb, col_lb, col_low, col_high, col_soc = st.columns(6)
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

# ── Charging price threshold ──────────────────────────────────────────────────

st.subheader("Charging Price Threshold")
st.caption(
    "Colour = the highest price at which the parked policy still charges (charge while price ≤ "
    "colour). Collapses the price response to one surface; blank where it never charges."
)
_chart(
    policy_threshold_figure(0, st.session_state.get("time_bin", 10),
                            "Charging price threshold — Parked"),
    "explorer_price_threshold",
)

# ── Benchmark heatmaps ────────────────────────────────────────────────────────

st.subheader("Benchmark Policy Heatmaps")
bench_tabs = st.tabs([
    "Maximal charging", "Price-oriented", "Night charging", "Minimum SoC", "Always minimum",
    "DP heuristic",
])
time_bin = st.session_state.get("time_bin", 10)
bat_bin  = st.session_state.get("bat_bin", 1.0)

with bench_tabs[0]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(maximal_charging_policy, 0),
        "Maximal charging — Parked", time_bin, bat_bin), "explorer_benchmark_maximal")
with bench_tabs[1]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(price_oriented_policy, 0,
                               low_threshold=float(low_threshold),
                               high_threshold=float(high_threshold)),
        "Price-oriented charging — Parked", time_bin, bat_bin), "explorer_benchmark_price_oriented")
with bench_tabs[2]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(night_charging_policy, 0),
        "Night charging — Parked", time_bin, bat_bin), "explorer_benchmark_night")
with bench_tabs[3]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(minimum_soc_policy, 0, soc_threshold=float(soc_threshold)),
        f"Minimum SoC — Parked  |  threshold = {soc_threshold:.1f} kWh",
        time_bin, bat_bin), "explorer_benchmark_minimum_soc")
with bench_tabs[4]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(always_minimum_policy, 0),
        "Always minimum rate — Parked", time_bin, bat_bin), "explorer_benchmark_always_minimum")
with bench_tabs[5]:
    _chart(_policy_heatmap_figure(
        benchmark_policy_rates(dp_heuristic_policy, 0),
        "DP heuristic — Parked", time_bin, bat_bin), "explorer_benchmark_dp_heuristic")

# ── Charge rate vs price ──────────────────────────────────────────────────────

st.subheader("Charge Rate vs. Price")
_chart(charge_vs_price_figure(float(low_threshold), float(high_threshold),
                              float(soc_threshold)), "explorer_charge_vs_price")

# ── Sample rollout ────────────────────────────────────────────────────────────

st.subheader("Sample Rollout (Optimal Policy)")
st.caption(
    "One simulated trajectory under the optimal policy: the price path it faced, the resulting "
    "battery SoC (driving periods shaded), and the charge rate. Shows what the policy actually does."
)
rollout_seed = st.number_input("Rollout seed", 0, 9999, 0, key="explorer_rollout_seed")
_chart(rollout_trajectory_figure(int(rollout_seed)), "explorer_sample_rollout")

# ── Input schedules ───────────────────────────────────────────────────────────

st.subheader("Input Schedules")
col1, col2 = st.columns(2)
with col1:
    _chart(price_figure(), "explorer_price_schedule")
with col2:
    _chart(transition_figure(), "explorer_transition_schedule")
