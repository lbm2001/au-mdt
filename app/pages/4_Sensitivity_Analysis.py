"""
=============================================================================
Sensitivity Analysis — EV Charging MDP  (app layer)
=============================================================================
All sweep logic and figure factories live in the ev_mdt package. This file only
does Streamlit UI: sliders, buttons, progress bars, and calling ev_mdt.

Costs are exact (analytical); the figure/table export for the paper is done via
the CLI (`python -m ev_mdt sensitivity --all`), not from this page.
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from ev_mdt.analysis.sensitivity import (
    sweep_pricing_model,
    sweep_pricing_season,
    sweep_pricing_daytype,
    sweep_pricing_crisis,
    sweep_penalty,
    sweep_horizon,
    sweep_departure_profiles,
    sweep_mobility_models,
    PHI_VALUES,
    HORIZON_HOURS,
    DEPARTURE_PROFILES,
    CRISIS_YEARS,
    BASELINE_MODEL,
)
from ev_mdt.plots.sensitivity import (
    fig_policy_heatmap_grid,
    fig_policy_charge_border_grid,
    fig_cost_distribution,
    build_summary_df,
    SUMMARY_METRIC_FORMATS,
    PAPER_POLICIES,
)
from ev_mdt.pricing.samplers import GaussianBinnedSampler, GMMSampler, MDNSampler
from ev_mdt.pricing.entsoe import load_prices


# ── Streamlit helpers ─────────────────────────────────────────────────────────

def _get_gbins(mode: str) -> GaussianBinnedSampler:
    """Fitted Gaussian-bins sampler, cached per crisis setting.

    mode: "excl" (all years except 2021–23), "incl" (all years),
          "only" (2021–23 crisis years only).
    """
    key = {"excl": "sa_gbins_excl", "incl": "sa_gbins_incl", "only": "sa_gbins_only"}[mode]
    if key not in st.session_state:
        if "sa_price_df" not in st.session_state:
            with st.spinner("Loading ENTSO-E price data…"):
                st.session_state["sa_price_df"] = load_prices()
        df = st.session_state["sa_price_df"]
        is_crisis = df["timestamp"].dt.year.isin(CRISIS_YEARS)
        if mode == "excl":
            df = df[~is_crisis]
        elif mode == "only":
            df = df[is_crisis]
        label = {"excl": "excl. crisis", "incl": "incl. crisis", "only": "crisis only"}[mode]
        with st.spinner(f"Fitting Gaussian-bins price model ({label})…"):
            st.session_state[key] = GaussianBinnedSampler().fit(df)
    return st.session_state[key]


_PRICE_MODEL_CLASSES = {"Gaussian Bins": GaussianBinnedSampler, "GMM": GMMSampler, "MDN": MDNSampler}


def _get_price_model(model_name: str):
    """Fitted price sampler of the given type (crisis-excluded data), cached."""
    if model_name == "Gaussian Bins":
        return _get_gbins("excl")
    key = f"sa_pmodel_{model_name}"
    if key not in st.session_state:
        if "sa_price_df" not in st.session_state:
            with st.spinner("Loading ENTSO-E price data…"):
                st.session_state["sa_price_df"] = load_prices()
        df = st.session_state["sa_price_df"]
        df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
        with st.spinner(f"Fitting {model_name} price model…"):
            st.session_state[key] = _PRICE_MODEL_CLASSES[model_name]().fit(df)
    return st.session_state[key]


def _paper_config(filename: str) -> dict:
    return {"displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 4}}


def _chart(fig, filename: str):
    st.plotly_chart(fig, use_container_width=True, config=_paper_config(filename))


def _show_results(results: list[dict], sweep_label: str):
    """Render the policy heatmaps, charge borders, exact cost bar and summary table."""
    if results:
        models = {r.get("model", "?") for r in results}
        st.caption("Mobility model: **varies by panel**" if len(models) > 1
                   else f"Mobility model: **{next(iter(models))}**")

    policy = st.radio("Policy", list(PAPER_POLICIES), horizontal=True,
                      key=f"sa_policy_{sweep_label}")

    st.subheader("Policy heatmaps")
    st.caption("Price-averaged charge rate u(hour × battery), one panel per swept value.")
    _chart(fig_policy_heatmap_grid(results, policy), f"{sweep_label}_{policy}_heatmaps")

    st.subheader("Charge / no-charge border (all hours)")
    st.caption(
        "The charge-vs-defer boundary in the price × battery plane, one curve per hour "
        "(colour = hour). Charge below each curve, defer above."
    )
    _chart(fig_policy_charge_border_grid(results, policy), f"{sweep_label}_{policy}_charge_border")

    st.subheader("Expected cost")
    st.caption("Exact expected total cost per policy — **including the unserved-driving penalty** — "
               "grouped by swept value, one bar per policy (charging/penalty split). No Monte-Carlo.")
    cost_axis = st.radio("Cost axis", ["Log", "Linear"], horizontal=True, key=f"sa_cost_axis_{sweep_label}")
    _chart(fig_cost_distribution(results, log_y=(cost_axis == "Log")), f"{sweep_label}_cost")

    st.subheader("Summary table")
    df = build_summary_df(results)
    st.dataframe(df.style.format(SUMMARY_METRIC_FORMATS),
                 use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV", df.to_csv(index=False).encode(),
        f"sensitivity_{sweep_label.replace(' ', '_')}.csv", "text/csv",
    )


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sensitivity Analysis — EV Charging MDP", layout="wide")
st.title("Sensitivity Analysis")
with st.expander("About this page", expanded=False):
    st.markdown(f"""
**Baseline configuration** (SharedParams / BaselineParams defaults):
battery e_max = 40 kWh · η_c = 0.95 · u_max = 11 kW · φ = 1000 €/h · K = 100 bins · λ_max = 0.25 €/kWh

**Prices** are wholesale DK1 day-ahead levels (€/kWh). The Gaussian-parametric means are fitted to
ENTSO-E data **excluding** the 2021–23 crisis; the data-driven models (bins/GMM/MDN) train on
crisis-excluded data. Negative wholesale prices are floored to 0.

**Policies** (exact expected cost): Backward Induction · Departure Urgency · Battery Level Urgency ·
Price-Oriented · Night Charging · Always-Maximum · Minimum Battery Level · Always-Minimum.

**Sweep dimensions** (others held at baseline):
1. **Pricing model / season / day-type / crisis** — ENTSO-E DK1 (Gaussian Bins · GMM · MDN)
2. **Penalty** — φ ∈ {PHI_VALUES} €/h
3. **Horizon T** — {HORIZON_HOURS} h
4. **Departure profile** — {list(DEPARTURE_PROFILES)}
5. **Mobility model** — NegBin {{fixed-k, Poisson-k}} × {{k=5, k=10}}

> Exact per-policy cost runs one backward pass per policy per panel — lower **N_e** (sidebar) if a
> sweep drags. Re-run a single sweep with its **Run** button.
    """)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sweep settings")
    N_e = st.select_slider("Battery grid points N_e", [50, 100, 200, 500], value=500, key="sa_N_e")
    st.caption("Figures + tables for the paper are exported via the CLI: "
               "`python -m ev_mdt sensitivity --all`.")


def _run(fn, key: str, label: str):
    """Run a sweep with a progress bar and store its results in session state."""
    st.session_state.pop(key, None)
    bar = st.progress(0.0, text="Starting…")
    with st.spinner(f"Running {label}…"):
        st.session_state[key] = fn(lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_price, tab_phi, tab_T, tab_departure, tab_mobility = st.tabs(
    ["Pricing Model", "Penalty", "Horizon T", "Departure Profile", "Mobility Model"]
)

# ─── Tab 1: Pricing model ─────────────────────────────────────────────────────
with tab_price:
    st.markdown(
        "Real-data pricing on ENTSO-E DK1. Four sub-sweeps vary one factor at a time; the others "
        "are held at baseline (Gaussian Bins · spring · weekday · crisis-excluded). Baseline mobility."
    )
    sub_model, sub_season, sub_daytype, sub_crisis = st.tabs(
        ["Pricing model", "Season", "Weekday/Weekend", "Energy crisis"])

    with sub_model:
        st.caption("Vary the price model (Gaussian Bins · GMM · MDN) — spring · weekday · crisis-excluded.")
        if st.button("Run pricing-model sweep", key="sa_run_pmodel"):
            samplers = {m: _get_price_model(m) for m in ("Gaussian Bins", "GMM", "MDN")}
            _run(lambda cb: sweep_pricing_model(samplers, N_e, cb),
                 "sa_pricing_model_results", "pricing-model sweep")
        if "sa_pricing_model_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_model_results"], "pricing_model")
        else:
            st.info("Click **Run pricing-model sweep** to compute results.")

    with sub_season:
        st.caption("Vary season — held at weekday, crisis-excluded.")
        if st.button("Run season sweep", key="sa_run_pseason"):
            _run(lambda cb: sweep_pricing_season(_get_gbins("excl"), N_e, cb),
                 "sa_pricing_season_results", "season sweep")
        if "sa_pricing_season_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_season_results"], "pricing_season")
        else:
            st.info("Click **Run season sweep** to compute results.")

    with sub_daytype:
        st.caption("Vary weekday vs weekend — held at spring, crisis-excluded.")
        if st.button("Run weekday/weekend sweep", key="sa_run_pdaytype"):
            _run(lambda cb: sweep_pricing_daytype(_get_gbins("excl"), N_e, cb),
                 "sa_pricing_daytype_results", "weekday/weekend sweep")
        if "sa_pricing_daytype_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_daytype_results"], "pricing_daytype")
        else:
            st.info("Click **Run weekday/weekend sweep** to compute results.")

    with sub_crisis:
        st.caption("Compare price models fitted on three slices of ENTSO-E data — excluding, "
                   "including, and the 2021–23 crisis years only — held at spring, weekday.")
        if st.button("Run energy-crisis sweep", key="sa_run_pcrisis"):
            _run(lambda cb: sweep_pricing_crisis(_get_gbins("excl"), _get_gbins("incl"),
                                                 _get_gbins("only"), N_e, cb),
                 "sa_pricing_crisis_results", "energy-crisis sweep")
        if "sa_pricing_crisis_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_crisis_results"], "pricing_crisis")
        else:
            st.info("Click **Run energy-crisis sweep** to compute results.")

# ─── Tab 2: Penalty ───────────────────────────────────────────────────────────
with tab_phi:
    st.markdown(f"Sweeps the unserved-driving penalty φ ∈ {PHI_VALUES} €/h over 24 h. "
                "Gaussian parametric pricing; other params at baseline.")
    if st.button("Run penalty sweep", key="sa_run_phi"):
        _run(lambda cb: sweep_penalty(BASELINE_MODEL, N_e, cb), "sa_phi_results", "penalty sweep")
    if "sa_phi_results" in st.session_state:
        _show_results(st.session_state["sa_phi_results"], "penalty")
    else:
        st.info("Click **Run penalty sweep** to compute results.")

# ─── Tab 3: Horizon T ─────────────────────────────────────────────────────────
with tab_T:
    st.markdown(f"Compares horizon lengths T ∈ {HORIZON_HOURS} h. Gaussian parametric pricing; "
                "other params at baseline. The 168 h solve is the slow one — lower **N_e** if it drags.")
    if st.button("Run horizon sweep", key="sa_run_T"):
        _run(lambda cb: sweep_horizon(BASELINE_MODEL, N_e, cb), "sa_horizon_results", "horizon sweep")
    if "sa_horizon_results" in st.session_state:
        _show_results(st.session_state["sa_horizon_results"], "horizon")
    else:
        st.info("Click **Run horizon sweep** to compute results.")

# ─── Tab 4: Departure profile ─────────────────────────────────────────────────
with tab_departure:
    st.markdown(f"Compares departure profiles {list(DEPARTURE_PROFILES)} over 24 h. Each overrides "
                "only the **p_PD_*** departure probabilities — all other params held at baseline.")
    if st.button("Run departure-profile sweep", key="sa_run_departure"):
        _run(lambda cb: sweep_departure_profiles(BASELINE_MODEL, N_e, cb),
             "sa_departure_results", "departure-profile sweep")
    if "sa_departure_results" in st.session_state:
        _show_results(st.session_state["sa_departure_results"], "departure_profile")
    else:
        st.info("Click **Run departure-profile sweep** to compute results.")

# ─── Tab 5: Mobility model ────────────────────────────────────────────────────
with tab_mobility:
    st.markdown("Compares Negative Binomial mobility models over 24 h: **{fixed-k, Poisson-k} × "
                "{k=5, k=10}** (4 configs). Gaussian parametric pricing; other params at baseline.")
    if st.button("Run mobility-model sweep", key="sa_run_mobility"):
        _run(lambda cb: sweep_mobility_models(N_e, cb), "sa_mobility_results", "mobility-model sweep")
    if "sa_mobility_results" in st.session_state:
        _show_results(st.session_state["sa_mobility_results"], "mobility_model")
    else:
        st.info("Click **Run mobility-model sweep** to compute results.")
