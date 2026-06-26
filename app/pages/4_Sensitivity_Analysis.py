"""
=============================================================================
Sensitivity Analysis — EV Charging MDP  (app layer)
=============================================================================
All sweep logic and figure factories live in the ev_mdt package.
This file only does Streamlit UI: sliders, buttons, progress bars,
and calling ev_mdt.analysis / ev_mdt.plots.
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# ── Package imports ───────────────────────────────────────────────────────────

from ev_mdt.analysis.sensitivity import (
    sweep_pricing_model,
    sweep_pricing_season,
    sweep_pricing_daytype,
    sweep_pricing_crisis,
    sweep_penalty,
    sweep_horizon,
    sweep_departure_profiles,
    sweep_mobility_models,
    save_tables,
    baseline_optimal_result,
    baseline_model_figures,
    HEATMAP_NCOLS,
    PHI_VALUES,
    HORIZON_HOURS,
    DEPARTURE_PROFILES,
    CRISIS_YEARS,
    BASELINE_MODEL,
)
from ev_mdt.plots.sensitivity import (
    fig_heatmap_grid,
    fig_charge_boundary_grid,
    fig_cost_distribution,
    build_summary_df,
    figure_to_png,
    SUMMARY_METRIC_FORMATS,
)
from ev_mdt.plots.viz import SWEEP_AXIS_LABEL
from ev_mdt.plots.trip_duration import compute_trip_durations, trip_duration_figure
from ev_mdt.pricing.samplers import GaussianBinnedSampler, GMMSampler, MDNSampler
from ev_mdt.pricing.entsoe import load_prices


# ── Streamlit helpers ─────────────────────────────────────────────────────────

_EXPORT_DIR = Path(__file__).parent.parent.parent / "export"
FIGURES_DIR = _EXPORT_DIR / "figures_app"
TABLES_DIR  = _EXPORT_DIR / "tables"


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


_SWEEP_RESULT_KEYS = [
    ("sa_pricing_model_results",   "pricing_model"),
    ("sa_pricing_season_results",  "pricing_season"),
    ("sa_pricing_daytype_results", "pricing_daytype"),
    ("sa_pricing_crisis_results",  "pricing_crisis"),
    ("sa_phi_results",             "penalty"),
    ("sa_horizon_results",         "horizon"),
    ("sa_departure_results",       "departure_profile"),
    ("sa_mobility_results",        "mobility_model"),
]

_MODEL_PREFIX = {
    "Baseline":                              "baseline",
    "Negative Binomial trips (fixed k)":    "negbin",
    "Negative Binomial trips (sampled k)":  "negbin_poisson",
}


def _available_export_figures() -> list[dict]:
    items = [
        {"id": "baseline:Baseline:cost",            "path": "baseline_models/baseline_cost.png",
         "label": "baseline_models / baseline_cost"},
        {"id": "baseline:Baseline:optimal_policy",  "path": "baseline_models/baseline_optimal_policy.png",
         "label": "baseline_models / baseline_optimal_policy"},
        {"id": "baseline:Baseline:trajectories",    "path": "baseline_models/baseline_trajectories.png",
         "label": "baseline_models / baseline_trajectories"},
        {"id": "baseline:Negative Binomial trips (fixed k):trajectories",
         "path": "baseline_models/negbin_trajectories.png",
         "label": "baseline_models / negbin_trajectories"},
        {"id": "baseline:Negative Binomial trips (sampled k):trajectories",
         "path": "baseline_models/negbin_poisson_trajectories.png",
         "label": "baseline_models / negbin_poisson_trajectories"},
        {"id": "trip_duration", "path": "baseline_models/trip_duration_by_model.png",
         "label": "baseline_models / trip_duration_by_model"},
    ]
    for key, folder in _SWEEP_RESULT_KEYS:
        if not st.session_state.get(key):
            continue
        for name in ("policy_heatmaps", "charge_border", "cost"):
            items.append({"id": f"sweep:{key}:{name}",
                          "path": f"sensitivity_figures/{folder}/{name}.png",
                          "label": f"sensitivity_figures / {folder} / {name}"})
    return items


def _render_export_figure(export_id: str) -> tuple[str, bytes]:
    """Render a single export figure and return (relative path, PNG bytes)."""
    N_e_val = int(st.session_state.get("sa_N_e", 500))
    N_r_val = int(st.session_state.get("sa_N_rollouts", 200))
    seed_val = int(st.session_state.get("sa_seed", 42))

    if export_id == "trip_duration":
        return "baseline_models/trip_duration_by_model.png", \
               figure_to_png(trip_duration_figure(compute_trip_durations()))

    kind, *parts = export_id.split(":")
    if kind == "baseline":
        model, figure_name = parts
        result = baseline_optimal_result(model, N_e_val)
        if figure_name == "optimal_policy":
            fig = fig_heatmap_grid([result], show_titles=False)
        else:
            figs = baseline_model_figures(result, N_r_val, seed_val)
            fig = figs[f"baseline_{figure_name}"]
        prefix = _MODEL_PREFIX[model]
        path = (f"baseline_models/{prefix}_{figure_name}.png"
                if model != "Baseline" else f"baseline_models/baseline_{figure_name}.png")
        return path, figure_to_png(fig)

    if kind == "sweep":
        key, figure_name = parts
        results = st.session_state.get(key)
        if not results:
            raise ValueError("Selected sweep has no results.")
        folder = dict(_SWEEP_RESULT_KEYS)[key]
        if figure_name == "policy_heatmaps":
            fig = fig_heatmap_grid(results, ncols=HEATMAP_NCOLS.get(folder, 1))
        elif figure_name == "charge_border":
            fig = fig_charge_boundary_grid(results)
        elif figure_name == "cost":
            fig = fig_cost_distribution(results)
        else:
            raise ValueError(f"Unknown export figure: {figure_name}")
        return f"sensitivity_figures/{folder}/{figure_name}.png", figure_to_png(fig)

    raise ValueError(f"Unknown export id: {export_id}")


def _sweep_export_ids(key: str) -> list[str]:
    return [f"sweep:{key}:{name}" for name in ("policy_heatmaps", "charge_border", "cost")]


def _show_results(results: list[dict], sweep_label: str):
    """Render all output plots and tables for a completed sweep."""
    if results:
        models = {r.get("model", "?") for r in results}
        st.caption("Mobility model: **varies by panel**" if len(models) > 1
                   else f"Mobility model: **{next(iter(models))}**")

    st.subheader("Policy heatmaps")
    _chart(fig_heatmap_grid(results, ncols=HEATMAP_NCOLS.get(sweep_label, 1)),
           f"{sweep_label}_policy_heatmaps")

    st.subheader("Charge / no-charge border (all hours)")
    st.caption(
        "Just the charge-vs-defer boundary of the map above, drawn in the price × battery plane, "
        "with one curve per hour of the day (colour = hour). Charge below each curve, defer above — "
        "so you can see the border move across the day without picking a single hour."
    )
    _chart(fig_charge_boundary_grid(results), f"{sweep_label}_charge_border")

    st.subheader("Mean cost")
    st.caption("Mean total cost per sampled trip — **including the unserved-driving penalty** — "
               "grouped by swept value, one bar per policy. Error bars: **SEM** = uncertainty of "
               "the mean (std/√N); **Std** = spread of individual trips. Lower bar clamped at 0.")
    cc1, cc2 = st.columns(2)
    with cc1:
        cost_axis = st.radio("Cost axis", ["Log", "Linear"], horizontal=True,
                             key=f"sa_cost_axis_{sweep_label}")
    with cc2:
        err_mode = st.radio("Error bars", ["SEM", "Std"], horizontal=True,
                            key=f"sa_cost_err_{sweep_label}")
    x_label = SWEEP_AXIS_LABEL.get(sweep_label, "Swept value")
    _chart(fig_cost_distribution(results, log_y=(cost_axis == "Log"), x_label=x_label,
                                 error=err_mode.lower()), f"{sweep_label}_cost")

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
ENTSO-E data **excluding** the 2021–23 crisis; the data-driven models (bins/GMM/MDN) train on **all**
years. Negative wholesale prices (~2.6% of hours) are floored to 0.

**Policies compared:** Optimal (Backward Induction) · Night Charging · Battery Level Urgency · Always-Maximum · Always-Minimum

**Mobility model** (sidebar — applies to every sweep):
- **Baseline** — trip ~ Geom(p_DP); 2-state chain; default E[T] ≈ 11 min.
- **Negative Binomial (fixed k)** — trip ~ NB(k, q); k-phase chain; default E[T] = k/q = 25 min.
- **Negative Binomial (sampled k)** — k ~ Poisson(λ_k) drawn at each trip start; default E[T] ≈ 25 min.

> Default trip durations differ across models, so switching the model is **not** a controlled
> comparison — read each model's sweeps on their own.

**Five independent sweep dimensions** (others held at baseline):
1. **Pricing model / season / day-type / crisis** — on ENTSO-E DK1 data (Gaussian Bins · GMM · MDN)
2. **Penalty** — φ ∈ {PHI_VALUES} €/h
3. **Horizon T** — {HORIZON_HOURS} h
4. **Departure profile** — {list(DEPARTURE_PROFILES)}
5. **Mobility model** — NegBin {{fixed-k, Poisson-k}} × {{k=5, k=10}}

> **Reading the Pricing tab:** each pricing model is solved *and* evaluated in its **own** price
> world. Compare policies *within* a column (which policy wins, optimality gap, feasibility) — not
> absolute costs *across* columns. The Battery Level Urgency uses each world's own price distribution.
> Negative Binomial models have more mobility states → slower solves; lower **N_e** if needed.
> Re-run a single sweep with its **Run** button.
    """)

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sweep settings")
    N_rollouts = st.slider("Rollouts per config", 10, 500, 500, 10, key="sa_N_rollouts")
    N_e        = st.select_slider("Battery grid points N_e", [50, 100, 200, 500], value=500, key="sa_N_e")
    seed       = st.number_input("Random seed", 0, 9999, 42, key="sa_seed")
    st.divider()
    if st.button("▶ Run all sweeps", type="primary", use_container_width=True, key="sa_run_all"):
        st.session_state["sa_run_all_triggered"] = True
        st.rerun()
    st.caption("Computes every sweep once; results persist across tabs. Export PNGs are saved "
               "under `figures/` as each sweep completes.")

# ── Run-all orchestration ─────────────────────────────────────────────────────

if st.session_state.pop("sa_run_all_triggered", False):
    bar = st.progress(0.0, text="Starting…")
    st.session_state["sa_run_all_export_errors"] = []
    _s_excl   = _get_gbins("excl")
    _s_incl   = _get_gbins("incl")
    _s_crisis = _get_gbins("only")
    _du_kw = dict(
        du_gamma=st.session_state.get("du_gamma", 0.5),
        du_use_reserve=st.session_state.get("du_use_reserve", True),
    )
    _steps = [
        ("Pricing · model",    "sa_pricing_model_results",
         lambda cb: sweep_pricing_model(
             {m: _get_price_model(m) for m in ("Gaussian Bins", "GMM", "MDN")},
             N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Pricing · season",   "sa_pricing_season_results",
         lambda cb: sweep_pricing_season(_s_excl, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Pricing · day-type", "sa_pricing_daytype_results",
         lambda cb: sweep_pricing_daytype(_s_excl, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Pricing · crisis",   "sa_pricing_crisis_results",
         lambda cb: sweep_pricing_crisis(_s_excl, _s_incl, _s_crisis, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Penalty",            "sa_phi_results",
         lambda cb: sweep_penalty(BASELINE_MODEL, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Horizon",            "sa_horizon_results",
         lambda cb: sweep_horizon(BASELINE_MODEL, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Departure",          "sa_departure_results",
         lambda cb: sweep_departure_profiles(BASELINE_MODEL, N_rollouts, N_e, seed, cb, **_du_kw)),
        ("Mobility",           "sa_mobility_results",
         lambda cb: sweep_mobility_models(N_rollouts, N_e, seed, cb, **_du_kw)),
    ]
    n = len(_steps)

    _sweep_name_by_key = dict(_SWEEP_RESULT_KEYS)

    def _emit_export(export_id: str) -> None:
        labels_by_id = {item["id"]: item["label"] for item in _available_export_figures()}
        label = labels_by_id.get(export_id, export_id)
        bar.progress(min((i + 0.9) / n, 1.0), text=f"Rendering export: {label}")
        try:
            rel_path, data = _render_export_figure(export_id)
            out_path = FIGURES_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
        except Exception as exc:
            st.session_state["sa_run_all_export_errors"].append(f"{label}: {exc}")

    def _emit_table(key: str) -> None:
        sweep_name = _sweep_name_by_key.get(key, key)
        try:
            out_path = TABLES_DIR / "sensitivity_figures" / sweep_name / "summary.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            build_summary_df(st.session_state[key]).to_csv(out_path, index=False)
        except Exception as exc:
            st.session_state["sa_run_all_export_errors"].append(f"table {sweep_name}: {exc}")

    for i, (name, key, fn) in enumerate(_steps):
        st.session_state[key] = fn(
            lambda f, m, i=i, name=name: bar.progress((i + f) / n, text=f"{name}: {m}"))
        for export_id in _sweep_export_ids(key):
            _emit_export(export_id)
        _emit_table(key)

    for export_id in [
        "baseline:Baseline:cost", "baseline:Baseline:optimal_policy",
        "baseline:Baseline:trajectories",
        "baseline:Negative Binomial trips (fixed k):trajectories",
        "baseline:Negative Binomial trips (sampled k):trajectories",
        "trip_duration",
    ]:
        _emit_export(export_id)

    # Combined baseline/NegBin model summary table.
    bar.progress(0.97, text="Writing baseline-model tables…")
    try:
        save_tables({}, out_dir=TABLES_DIR, N_rollouts=N_rollouts, seed=seed,
                    N_e=N_e, include_baseline=True)
    except Exception as exc:
        st.session_state["sa_run_all_export_errors"].append(f"baseline tables: {exc}")

    bar.progress(1.0, text="Done.")
    bar.empty()
    errors = st.session_state.get("sa_run_all_export_errors", [])
    _root = Path(__file__).parent.parent.parent
    if errors:
        st.warning("Run-all complete with export errors:\n" + "\n".join(f"- {e}" for e in errors))
    else:
        st.success(f"Run-all complete. Figures under `{FIGURES_DIR.relative_to(_root)}/`, "
                   f"tables under `{TABLES_DIR.relative_to(_root)}/`.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_price, tab_phi, tab_T, tab_departure, tab_mobility = st.tabs(
    ["Pricing Model", "Penalty", "Horizon T", "Departure Profile", "Mobility Model"]
)

# ─── Tab 1: Pricing model ─────────────────────────────────────────────────────
with tab_price:
    st.markdown(
        "Real-data pricing on ENTSO-E DK1 data. Four sub-sweeps vary one factor at a time; the "
        "others are held at baseline (Gaussian Bins · spring · weekday · crisis-excluded). "
        "All use the Baseline mobility model."
    )
    sub_model, sub_season, sub_daytype, sub_crisis = st.tabs(
        ["Pricing model", "Season", "Weekday/Weekend", "Energy crisis"])

    with sub_model:
        st.caption("Vary the price model (Gaussian Bins · GMM · MDN) — held at spring · weekday · "
                   "crisis-excluded. Each model is fitted on the same ENTSO-E data and drives its "
                   "own sampled price world. (MDN fitting trains a neural net — first run is slower.)")
        if st.button("Run pricing-model sweep", key="sa_run_pmodel"):
            st.session_state.pop("sa_pricing_model_results", None)
            samplers = {m: _get_price_model(m) for m in ("Gaussian Bins", "GMM", "MDN")}
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running pricing-model sweep…"):
                st.session_state["sa_pricing_model_results"] = sweep_pricing_model(
                    samplers, N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_model_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_model_results"], "pricing_model")
        else:
            st.info("Click **Run pricing-model sweep** to compute results.")

    with sub_season:
        st.caption("Vary season — held at weekday, crisis-excluded.")
        if st.button("Run season sweep", key="sa_run_pseason"):
            st.session_state.pop("sa_pricing_season_results", None)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running season sweep…"):
                st.session_state["sa_pricing_season_results"] = sweep_pricing_season(
                    _get_gbins("excl"), N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_season_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_season_results"], "pricing_season")
        else:
            st.info("Click **Run season sweep** to compute results.")

    with sub_daytype:
        st.caption("Vary weekday vs weekend — held at spring, crisis-excluded.")
        if st.button("Run weekday/weekend sweep", key="sa_run_pdaytype"):
            st.session_state.pop("sa_pricing_daytype_results", None)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running weekday/weekend sweep…"):
                st.session_state["sa_pricing_daytype_results"] = sweep_pricing_daytype(
                    _get_gbins("excl"), N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_daytype_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_daytype_results"], "pricing_daytype")
        else:
            st.info("Click **Run weekday/weekend sweep** to compute results.")

    with sub_crisis:
        st.caption("Compare price models fitted on three slices of ENTSO-E data — all years "
                   "**excluding** 2021–23, **including** them, and the 2021–23 crisis years "
                   "**only** — held at spring, weekday.")
        if st.button("Run energy-crisis sweep", key="sa_run_pcrisis"):
            st.session_state.pop("sa_pricing_crisis_results", None)
            bar = st.progress(0.0, text="Starting…")
            with st.spinner("Running energy-crisis sweep…"):
                st.session_state["sa_pricing_crisis_results"] = sweep_pricing_crisis(
                    _get_gbins("excl"), _get_gbins("incl"), _get_gbins("only"),
                    N_rollouts, N_e, seed,
                    progress_cb=lambda f, m: bar.progress(f, text=m))
            bar.empty(); st.rerun()
        if "sa_pricing_crisis_results" in st.session_state:
            _show_results(st.session_state["sa_pricing_crisis_results"], "pricing_crisis")
        else:
            st.info("Click **Run energy-crisis sweep** to compute results.")

# ─── Tab 2: Penalty ───────────────────────────────────────────────────────────
with tab_phi:
    st.markdown(
        f"Sweeps the unserved-driving penalty φ ∈ {PHI_VALUES} €/h over a 24 h horizon.  "
        "Uses Gaussian parametric pricing.  All other params at baseline."
    )
    if st.button("Run penalty sweep", key="sa_run_phi"):
        st.session_state.pop("sa_phi_results", None)
        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running penalty sweep…"):
            st.session_state["sa_phi_results"] = sweep_penalty(
                BASELINE_MODEL, N_rollouts, N_e, seed,
                progress_cb=lambda f, m: bar.progress(f, text=m))
        bar.empty(); st.rerun()
    if "sa_phi_results" in st.session_state:
        _show_results(st.session_state["sa_phi_results"], "penalty")
    else:
        st.info("Click **Run penalty sweep** to compute results.")

# ─── Tab 3: Horizon T ─────────────────────────────────────────────────────────
with tab_T:
    st.markdown(
        f"Compares horizon lengths T ∈ {HORIZON_HOURS} h.  "
        "Uses Gaussian parametric pricing.  All other params at baseline.  "
        "Note: the 168 h solve is the slow one — and Negative Binomial models add mobility "
        "states on top, so lower **N_e** (sidebar) if it drags."
    )
    if st.button("Run horizon sweep", key="sa_run_T"):
        st.session_state.pop("sa_horizon_results", None)
        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running horizon sweep…"):
            st.session_state["sa_horizon_results"] = sweep_horizon(
                BASELINE_MODEL, N_rollouts, N_e, seed,
                progress_cb=lambda f, m: bar.progress(f, text=m))
        bar.empty(); st.rerun()
    if "sa_horizon_results" in st.session_state:
        _show_results(st.session_state["sa_horizon_results"], "horizon")
    else:
        st.info("Click **Run horizon sweep** to compute results.")

# ─── Tab 5: Departure profile ─────────────────────────────────────────────────
with tab_departure:
    st.markdown(
        f"Compares departure profiles {list(DEPARTURE_PROFILES)} over a 24 h horizon.  "
        "Each profile overrides only the **p_PD_*** departure probabilities — trip duration, "
        "pricing (Gaussian parametric) and all other params are held at baseline — so the "
        "differences isolate the effect of *when/how often the car departs*."
    )
    if st.button("Run departure-profile sweep", key="sa_run_departure"):
        st.session_state.pop("sa_departure_results", None)
        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running departure-profile sweep…"):
            st.session_state["sa_departure_results"] = sweep_departure_profiles(
                BASELINE_MODEL, N_rollouts, N_e, seed,
                progress_cb=lambda f, m: bar.progress(f, text=m))
        bar.empty(); st.rerun()
    if "sa_departure_results" in st.session_state:
        _show_results(st.session_state["sa_departure_results"], "departure_profile")
    else:
        st.info("Click **Run departure-profile sweep** to compute results.")

# ─── Tab 6: Mobility model ────────────────────────────────────────────────────
with tab_mobility:
    st.markdown(
        "Compares Negative Binomial mobility models over a 24 h horizon: **{fixed-k, Poisson-k} × {k=5, k=10}** "
        "(4 configs).  Uses Gaussian parametric pricing; all other params at baseline — so the "
        "differences isolate the effect of the *trip-duration dynamics* (larger k → longer trips).  "
        "The Baseline (binomial) model is shown in the figure-export's `baseline_models/` instead."
    )
    if st.button("Run mobility-model sweep", key="sa_run_mobility"):
        st.session_state.pop("sa_mobility_results", None)
        bar = st.progress(0.0, text="Starting…")
        with st.spinner("Running mobility-model sweep…"):
            st.session_state["sa_mobility_results"] = sweep_mobility_models(
                N_rollouts, N_e, seed,
                progress_cb=lambda f, m: bar.progress(f, text=m))
        bar.empty(); st.rerun()
    if "sa_mobility_results" in st.session_state:
        _show_results(st.session_state["sa_mobility_results"], "mobility_model")
    else:
        st.info("Click **Run mobility-model sweep** to compute results.")
