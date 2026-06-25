import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from ev_mdt.models.common.model_utils import price_bin_probs
from ev_mdt.plots.sensitivity import (
    fig_baseline_policy_heatmaps,
    fig_policy_heatmap,
    fig_policy_price_map,
)

st.set_page_config(page_title="Policy Explorer — EV Charging MDP", layout="wide")
st.title("Policy Explorer")

# ── Guard ─────────────────────────────────────────────────────────────────────

if "pi" not in st.session_state:
    st.warning("No solution found. Please go to **Settings** and click **Run Backward Induction** first.")
    st.stop()

pi       = st.session_state["pi"]
actions  = st.session_state["actions"]
e_grid   = st.session_state["e_grid"]
lam_grid = st.session_state["lam_grid"]
params   = st.session_state["params"]
T        = st.session_state["T"]
T_hours  = T // 60

# Price world the policy was solved in (Settings) — used to weight the
# price-averaged optimal-policy heatmap (None → Gaussian parametric).
_price_sampler = st.session_state.get("price_sampler")
if _price_sampler is not None:
    from ev_mdt.pricing.samplers import make_price_bin_probs_fn
    _pbp_fn = make_price_bin_probs_fn(
        _price_sampler, params,
        st.session_state.get("price_season") or "winter",
        bool(st.session_state.get("price_is_weekend", False)),
    )
else:
    _pbp_fn = lambda t: price_bin_probs(t, params)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _paper_config(filename: str) -> dict:
    """Plotly config so the modebar download button exports a clean, paper-ready PNG."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 4},
    }


def _chart(fig, filename: str):
    st.plotly_chart(fig, use_container_width=True, config=_paper_config(filename))


# ── Benchmark thresholds (shared with the Policy Rollout page) ─────────────────

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

lam_bin_sel = st.session_state.get("lam_bin_sel", params.K // 2)

_chart(
    fig_policy_heatmap(
        pi, actions, e_grid, params, T, chi=0,
        lam_bin=None if avg_bins else lam_bin_sel, pbp_fn=_pbp_fn,
        time_bin_min=st.session_state.get("time_bin", 10),
        battery_bin_kwh=st.session_state.get("bat_bin", 1.0),
    ),
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
    fig_policy_price_map(pi, actions, e_grid, lam_grid, params, chi=0, t=ppmap_hour * 60),
    "explorer_policy_vs_price",
)

# ── Baseline policy heatmaps ──────────────────────────────────────────────────

st.subheader("Baseline Policy Heatmaps")
st.caption(
    "Price-averaged charge rate u(hour × battery) for each benchmark policy. "
    "Uses the same time / battery bin resolution as the optimal-policy heatmap above."
)
_chart(
    fig_baseline_policy_heatmaps(
        params, e_grid, lam_grid, T, _pbp_fn,
        pi=pi, actions=actions,
        low_threshold=st.session_state["benchmark_low_threshold"],
        high_threshold=st.session_state["benchmark_high_threshold"],
        soc_threshold=params.e_max * 0.25,
        du_target_mode=st.session_state.get("du_target_mode", "fixed"),
        du_target_frac=st.session_state.get("du_target_frac", 1.0),

        du_reserve_frac=st.session_state.get("du_reserve_frac", 0.25),
        du_use_reserve=st.session_state.get("du_use_reserve", True),
        du_alpha=st.session_state.get("du_alpha", 0.5),
        time_bin_min=st.session_state.get("time_bin", 10),
        battery_bin_kwh=st.session_state.get("bat_bin", 1.0),
    ),
    "explorer_baseline_policy_heatmaps",
)
