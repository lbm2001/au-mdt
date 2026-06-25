import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from ev_mdt.params import BASELINE_MODEL, NEGBIN_SAMPLED_MODEL
from ev_mdt.models.common.model_utils import price_bin_probs
from ev_mdt.models.common.policies import policy_registry
from ev_mdt.models.common.rollout_utils import generate_rollout_scenario, run_policies
from ev_mdt.analysis.sensitivity import rollout_fn, rollout_stats_table
from ev_mdt.plots.sensitivity import (
    fig_baseline_cost, fig_rollout_trajectories, SUMMARY_METRIC_FORMATS,
)
from ev_mdt.plots.viz import MODEL_COLORS
from ev_mdt.plots.trip_duration import compute_trip_durations, trip_duration_figure

st.set_page_config(page_title="Policy Rollout — EV Charging MDP", layout="wide")
st.title("Policy Rollout")

# ── Guard ─────────────────────────────────────────────────────────────────────

if "pi" not in st.session_state:
    st.warning("No solution found. Please go to **Settings** and click **Run Backward Induction** first.")
    st.stop()

pi       = st.session_state["pi"]
actions  = st.session_state["actions"]
e_grid   = st.session_state["e_grid"]
params   = st.session_state["params"]
T        = st.session_state["T"]
T_hours  = T // 60

_solved_model = st.session_state.get("solved_model", "")
is_negbin    = _solved_model != BASELINE_MODEL
is_poisson_k = _solved_model == NEGBIN_SAMPLED_MODEL
_rf = rollout_fn(_solved_model)

# Benchmark thresholds set in Policy Explorer (with sensible defaults).
low_threshold  = float(st.session_state.get("benchmark_low_threshold",  params.price_night))
high_threshold = float(st.session_state.get("benchmark_high_threshold", params.price_evening))
soc_threshold  = float(st.session_state.get("soc_threshold",            params.e_max * 0.25))

# Price world the policy was solved in (Settings) — draw rollout prices and the
# Battery Level Urgency's price distribution from it (None → Gaussian parametric).
_price_sampler   = st.session_state.get("price_sampler")
_price_season    = st.session_state.get("price_season") or "winter"
_price_isweekend = bool(st.session_state.get("price_is_weekend", False))
if _price_sampler is not None:
    from ev_mdt.pricing.samplers import make_price_bin_probs_fn
    _pbp_fn = make_price_bin_probs_fn(_price_sampler, params, _price_season, _price_isweekend)
else:
    _pbp_fn = lambda t: price_bin_probs(t, params)
_scen_kw = dict(sampler=_price_sampler, season=_price_season, is_weekend=_price_isweekend)

# ── N-scenario simulation ─────────────────────────────────────────────────────

st.subheader("N-Scenario Policy Comparison")

nd_col1, nd_col2, nd_col3 = st.columns(3)
with nd_col1:
    n_days = st.slider("Number of scenarios N", 10, 500, 500, 10, key="n_days")
with nd_col2:
    nd_chi0 = st.radio("Initial state", ["Parked", "Driving"], horizontal=True, key="nd_chi0")
with nd_col3:
    nd_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="nd_seed")
st.caption("Initial battery is randomised uniformly over [e_min, e_max] per scenario (seeded; "
           "same start for all policies within a scenario).")

nd_chi0_int = 0 if nd_chi0 == "Parked" else 1

# Cache the rollouts so re-renders (e.g. toggling a plot option) don't recompute them.
_nd_key = (n_days, nd_chi0_int, int(nd_seed),
           low_threshold, high_threshold, soc_threshold, T, id(pi),
           st.session_state.get("du_target_mode", "fixed"),
           st.session_state.get("du_target_frac", 1.0),
           st.session_state.get("du_reserve_frac", 0.25),
           st.session_state.get("du_use_reserve", True),
           st.session_state.get("du_alpha", 0.5))
if st.session_state.get("_nd_key") != _nd_key:
    with st.spinner(f"Rolling out all policies over {n_days} scenarios…"):
        rng_nd = np.random.default_rng(int(nd_seed))
        nd_scenarios = [
            generate_rollout_scenario(params, int(rng_nd.integers(0, 1_000_000)), horizon=T, **_scen_kw)
            for _ in range(n_days)
        ]
        nd_e0s = [float(rng_nd.uniform(params.e_min, params.e_max)) for _ in range(n_days)]
        registry = policy_registry(
            params, _pbp_fn, pi=pi, actions=actions, e_grid=e_grid,
            low_threshold=low_threshold, high_threshold=high_threshold, soc_threshold=soc_threshold,
            du_target_mode=st.session_state.get("du_target_mode", "fixed"),
            du_target_frac=st.session_state.get("du_target_frac", 1.0),
            du_reserve_frac=st.session_state.get("du_reserve_frac", 0.25),
            du_use_reserve=st.session_state.get("du_use_reserve", True),
            du_alpha=st.session_state.get("du_alpha", 0.5),
        )
        nd_rollouts = run_policies(registry, nd_scenarios, nd_e0s, nd_chi0_int, params, _rf)

        # Mobility of the "other" NegBin variant (same scenarios, sibling k model).
        nd_mob_other = None
        if is_negbin:
            from ev_mdt.models.negbin.rollout import sibling_variant_mobility
            nd_mob_other = sibling_variant_mobility(params, nd_scenarios, nd_e0s, nd_chi0_int)

    st.session_state["_nd_rollouts"]  = nd_rollouts
    st.session_state["_nd_scenarios"] = nd_scenarios
    st.session_state["_nd_mob_other"] = nd_mob_other
    st.session_state["_nd_key"]       = _nd_key

nd_rollouts  = st.session_state["_nd_rollouts"]
nd_scenarios = st.session_state["_nd_scenarios"]
nd_mob_other = st.session_state["_nd_mob_other"]

stats_df = rollout_stats_table(nd_rollouts, params)
st.dataframe(stats_df.style.format(SUMMARY_METRIC_FORMATS), use_container_width=True)

_TABLES_DIR = Path(__file__).parent.parent.parent / "export" / "tables"
_tc1, _tc2 = st.columns(2)
with _tc1:
    st.download_button("Download CSV", stats_df.to_csv().encode(),
                       "policy_rollout.csv", "text/csv")
with _tc2:
    if st.button("💾 Export → export/tables/policy_rollout.csv"):
        _TABLES_DIR.mkdir(parents=True, exist_ok=True)
        _out = _TABLES_DIR / "policy_rollout.csv"
        stats_df.to_csv(_out)
        st.success(f"Saved `{_out.relative_to(Path(__file__).parent.parent.parent)}`")

# ── Mean cost ─────────────────────────────────────────────────────────────────

st.subheader("Mean cost")
st.caption("Mean total cost per scenario — **including the unserved-driving penalty** — "
           "one bar per policy. SEM = uncertainty of the mean (std/√N); Std = spread across "
           "scenarios. Lower bar clamped at 0.")
cc1, cc2 = st.columns(2)
with cc1:
    nd_cost_axis = st.radio("Cost axis", ["Log", "Linear"], horizontal=True, key="nd_cost_axis")
with cc2:
    nd_err = st.radio("Error bars", ["SEM", "Std"], horizontal=True, key="nd_cost_err")
st.plotly_chart(
    fig_baseline_cost(nd_rollouts, error=nd_err.lower(), log_y=(nd_cost_axis == "Log")),
    use_container_width=True,
)

# ── Mean trajectories ─────────────────────────────────────────────────────────

st.subheader("Mean trajectories across scenarios")
st.caption("Scenario-averaged price path and mobility (both shared across policies), with "
           "±1 SEM bands (std/√N, faint) — the uncertainty of the mean.")

_bi_chi = [r["chi_traj"] for r in nd_rollouts["Backward Induction"]]
if is_negbin and nd_mob_other is not None:
    label_cur   = "Negative Binomial (Poisson k)" if is_poisson_k else "Negative Binomial (fixed k)"
    label_other = "Negative Binomial (fixed k)"   if is_poisson_k else "Negative Binomial (Poisson k)"
    mobility_bands = [
        (label_cur,   MODEL_COLORS[label_cur],   _bi_chi,      True),
        (label_other, MODEL_COLORS[label_other], nd_mob_other, True),
    ]
else:
    mobility_bands = [("driving", "orange", _bi_chi, False)]
st.plotly_chart(fig_rollout_trajectories(nd_scenarios, T, mobility_bands), use_container_width=True)

st.divider()

# ── Trip-duration distribution by mobility model ──────────────────────────────

st.subheader("Trip-duration distribution by mobility model")
st.caption("Samples each mobility model's trips (default params; independent of the solved "
           "policy). Baseline → geometric (decaying); Negative Binomial → peaked; Poisson-k → wider.")

_compute_trip_durations = st.cache_data(show_spinner="Sampling trip durations…")(compute_trip_durations)

st.plotly_chart(
    trip_duration_figure(_compute_trip_durations()),
    use_container_width=True,
    config={"displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": "trip_duration_by_model", "scale": 4}},
)
