import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from ev_mdt.params import (
    SharedParams as _SP, BaselineParams as _BP, NegBinParams as _NP,
    N_e as _N_E, T_hours as _T_HOURS,
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL, MODEL_LABELS,
)

st.set_page_config(page_title="Settings — EV Charging MDP", layout="wide")
st.title("Settings")
st.caption("Adjust all model parameters here, then click **Run Backward Induction** to compute the optimal policy.")

# ── Model selector ────────────────────────────────────────────────────────────

def _on_model_change():
    _sampler_keys = ["price_sampler_Gaussian Bins", "price_sampler_GMM", "price_sampler_MDN"]
    for key in ["V", "pi", "actions", "e_grid", "lam_grid", "params", "T"] + _sampler_keys:
        st.session_state.pop(key, None)

model = st.selectbox(
    "Mobility model",
    MODEL_LABELS,
    key="model",
    on_change=_on_model_change,
    help=(
        "**Baseline**: trip duration ~ Geom(p_DP), mean ≈ 1/p_DP min. "
        "**Negative Binomial (fixed k)**: trip duration ~ NB(k, q) via exactly k Erlang phases, mean = k/q min. "
        "**Negative Binomial (sampled k)**: k ~ Poisson(λ_k) drawn each trip, so mean = λ_k/q min."
    ),
)
is_negbin    = model != BASELINE_MODEL
is_poisson_k = model == NEGBIN_SAMPLED_MODEL

# ── Defaults (single source of truth: the dataclass defaults) ────────────────

_sp, _bp, _np = _SP(), _BP(), _NP()
_DEFAULTS = dict(
    u_max=_sp.u_max, u_min=_sp.u_min, e_max=_sp.e_max, e_min=_sp.e_min,
    eta_c=_sp.eta_c, phi=_sp.phi, beta=_sp.beta,
    v=_sp.v, mu=_sp.mu,
    price_night=_sp.price_night, price_morning=_sp.price_morning,
    price_midday=_sp.price_midday, price_evening=_sp.price_evening,
    price_late=_sp.price_late, sigma_lambda=_sp.sigma_lambda,
    lambda_max=_sp.lambda_max,
    p_pd_morning=_sp.p_pd_morning, p_pd_lunch=_sp.p_pd_lunch,
    p_pd_evening=_sp.p_pd_evening, p_pd_default=_sp.p_pd_default,
    p_dp_morning=_bp.p_dp_morning, p_dp_lunch=_bp.p_dp_lunch,
    p_dp_evening=_bp.p_dp_evening, p_dp_default=_bp.p_dp_default,
    nb_k=_np.k, nb_q=_np.q,
    N_e=_N_E, T_hours=_T_HOURS,
)

if st.button("Reset to defaults"):
    for key in _DEFAULTS:
        st.session_state.pop(key, None)
    st.rerun()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Battery")
    u_max = st.slider("Max charge rate u_max (kW)", 1.0, 22.0, _DEFAULTS["u_max"], 0.5, key="u_max")
    u_min = st.slider("Min charge rate u_min (kW)", 0.1, 5.0, _DEFAULTS["u_min"], 0.1, key="u_min")
    e_max = st.slider("Battery capacity e_max (kWh)", 10.0, 100.0, _DEFAULTS["e_max"], 1.0, key="e_max")
    e_min = st.slider("Min battery level e_min (kWh)", 0.0, 10.0, _DEFAULTS["e_min"], 0.5, key="e_min")

    st.subheader("Vehicle")
    v  = st.slider("Driving speed v (km/h)", 10.0, 150.0, _DEFAULTS["v"], 5.0, key="v")
    mu = st.slider("Energy consumption μ (kWh/km)", 0.05, 0.50, _DEFAULTS["mu"], 0.01, key="mu")

    st.subheader("Solver")
    N_e     = st.select_slider("Battery grid points N_e",
                               [25, 50, 100, 200, 500, 1000, 2000],
                               value=_DEFAULTS["N_e"], key="N_e")
    T_hours = st.select_slider("Time horizon (hours)", [24, 48, 72, 96],
                               value=_DEFAULTS["T_hours"], key="T_hours")
    N_a_sel = st.select_slider(
        "Action space (non-zero rates)",
        options=["Default", 5, 10, 20, 50],
        value="Default", key="N_a_sel",
        help="**Default**: [0, u_min, u_max/2, u_max]. Otherwise N evenly spaced rates from u_min to u_max, plus 0.",
    )

with col2:
    st.subheader("Charging & Cost")
    eta_c = st.slider("Charging efficiency η_c", 0.50, 1.00, _DEFAULTS["eta_c"], 0.01, key="eta_c")
    phi   = st.number_input("Unserved-driving penalty φ (€/h)", min_value=0.0, max_value=5000.0,
                            value=_DEFAULTS["phi"], step=0.5, key="phi")
    beta  = st.slider("Discount factor β", 0.900, 1.000, _DEFAULTS["beta"], 0.001,
                      format="%.3f", key="beta")

    st.subheader("Electricity Price")
    price_source = st.radio(
        "Price source",
        ["Gaussian (parametric)", "Gaussian Bins", "GMM", "MDN"],
        key="price_source",
        horizontal=True,
        help=(
            "**Gaussian (parametric)**: manual time-of-day mean prices. "
            "**Gaussian Bins / GMM**: per-(weekend, hour, season) bin fitted from ENTSO-E data. "
            "**MDN**: single neural network conditioned on context, trained on all data jointly."
        ),
    )
    use_sampler = price_source != "Gaussian (parametric)"

    if use_sampler:
        from ev_mdt.pricing.samplers import SEASONS as _SEASONS
        price_season = st.radio(
            "Season", [s.capitalize() for s in _SEASONS],
            horizontal=True, key="price_season",
        ).lower()
        price_is_weekend = st.toggle("Weekend", key="price_is_weekend")
        lambda_max = _DEFAULTS["lambda_max"]  # wholesale range; matches the Sensitivity page
        st.caption(f"λ_max = {lambda_max} EUR/kWh (wholesale range; rare crisis peaks clip to the top bin)")
        # Dummy values — not used by the solver when sampler is active
        price_night = price_morning = price_midday = price_evening = price_late = _DEFAULTS["price_night"]
        sigma_lambda = _DEFAULTS["sigma_lambda"]

        if price_source == "MDN":
            st.divider()
            use_wandb = st.checkbox("Log MDN training to W&B", key="use_wandb")
            if use_wandb:
                _wc1, _wc2 = st.columns(2)
                with _wc1:
                    wandb_project = st.text_input("W&B project", value="au-mdt", key="wandb_project")
                with _wc2:
                    wandb_run_name = st.text_input("Run name", placeholder="auto", value="", key="wandb_run_name")
            else:
                wandb_project = wandb_run_name = None
        else:
            use_wandb = False
            wandb_project = wandb_run_name = None
    else:
        use_wandb = False
        wandb_project = wandb_run_name = None
        price_is_weekend = price_season = None
        lambda_max = _DEFAULTS["lambda_max"]
        st.markdown("*Mean prices (€/kWh)*")
        price_night   = st.slider("Night (00–06 h)",        0.0, 1.0, _DEFAULTS["price_night"],   0.01, key="price_night")
        price_morning = st.slider("Morning peak (06–09 h)", 0.0, 1.0, _DEFAULTS["price_morning"], 0.01, key="price_morning")
        price_midday  = st.slider("Midday (09–16 h)",       0.0, 1.0, _DEFAULTS["price_midday"],  0.01, key="price_midday")
        price_evening = st.slider("Evening peak (16–21 h)", 0.0, 1.0, _DEFAULTS["price_evening"], 0.01, key="price_evening")
        price_late    = st.slider("Late night (21–24 h)",   0.0, 1.0, _DEFAULTS["price_late"],    0.01, key="price_late")
        sigma_lambda  = st.slider("Price std dev σ_λ (€/kWh)", 0.0, 0.20, _DEFAULTS["sigma_lambda"],
                                  0.01, key="sigma_lambda")

with col3:
    st.subheader("Transition Probabilities (per minute)")
    st.markdown("**Parked → Driving**")
    p_pd_morning = st.slider("Morning  (07–09 h)",  0.0, 0.50, _DEFAULTS["p_pd_morning"],
                             0.005, format="%.3f", key="p_pd_morning")
    p_pd_lunch   = st.slider("Lunch    (12–14 h)",  0.0, 0.50, _DEFAULTS["p_pd_lunch"],
                             0.005, format="%.3f", key="p_pd_lunch")
    p_pd_evening = st.slider("Evening  (16–18 h)",  0.0, 0.50, _DEFAULTS["p_pd_evening"],
                             0.005, format="%.3f", key="p_pd_evening")
    p_pd_default = st.slider("Default",             0.0, 0.10, _DEFAULTS["p_pd_default"],
                             0.001, format="%.3f", key="p_pd_default")

    if is_negbin:
        st.markdown("**Negative Binomial trip duration: T ~ NB(k, q)**")
        nb_q = st.slider("Phase prob q  (timescale)",
                         0.01, 1.0, _DEFAULTS["nb_q"], 0.01,
                         format="%.2f", key="nb_q",
                         help="Per-phase transition probability. E[T] = k / q.")
        if is_poisson_k:
            nb_lambda_k = st.slider("Mean phases λ_k",
                                    0.5, 20.0, float(_DEFAULTS["nb_k"]), 0.5, key="nb_lambda_k",
                                    help="Mean of Poisson distribution for k. E[T] = λ_k / q.")
            import math as _math
            # k_max = 99.9th percentile of Poisson(lambda_k); start CDF from k=0
            _pmf_r = _math.exp(-nb_lambda_k)
            _cdf   = _pmf_r
            nb_k   = 0
            while _cdf < 0.999:
                nb_k  += 1
                _pmf_r *= nb_lambda_k / nb_k
                _cdf   += _pmf_r
            nb_k = max(nb_k, 1)
            st.caption(f"λ_k = {nb_lambda_k:.1f},  k_max = {nb_k},  "
                       f"E[T] = {nb_lambda_k:.1f}/{nb_q:.2f} = **{nb_lambda_k / nb_q:.1f} min**")
        else:
            nb_k = st.slider("Phases k",
                             1, 20, _DEFAULTS["nb_k"], 1, key="nb_k",
                             help="Fixed number of driving phases.")
            nb_lambda_k = None
            st.caption(f"E[T] = {nb_k}/{nb_q:.2f} = **{nb_k / nb_q:.1f} min**,  "
                       f"Var[T] = {nb_k * (1 - nb_q) / nb_q**2:.1f} min²")
    else:
        st.markdown("**Driving → Parked**")
        p_dp_morning = st.slider("Morning  (07:30–09:30 h)", 0.0, 1.0, _DEFAULTS["p_dp_morning"], 0.01, key="p_dp_morning")
        p_dp_lunch   = st.slider("Lunch    (12:15–14:15 h)", 0.0, 1.0, _DEFAULTS["p_dp_lunch"],   0.01, key="p_dp_lunch")
        p_dp_evening = st.slider("Evening  (16:30–18:30 h)", 0.0, 1.0, _DEFAULTS["p_dp_evening"], 0.01, key="p_dp_evening")
        p_dp_default = st.slider("Default",                  0.0, 1.0, _DEFAULTS["p_dp_default"], 0.01, key="p_dp_default")

# ── Summary statistics (steady-state approximation) ───────────────────────────
st.subheader("Expected statistics")

# Day-weighted average p_PD: three 2-hour peak windows + 18 h default
_w_peak = 120 / 1440          # each 2-hour window / 24 h
_w_def  = 1080 / 1440         # remaining 18 h
_avg_p_pd = (_w_peak * (p_pd_morning + p_pd_lunch + p_pd_evening)
             + _w_def * p_pd_default)

if is_negbin:
    # E[k]: use lambda_k when Poisson-sampled, fixed k otherwise
    _exp_k      = float(nb_lambda_k) if nb_lambda_k is not None else float(nb_k)
    _E_trip     = _exp_k / nb_q          # E[T] = E[k] / q
    # Var[T] for fixed k: k(1-q)/q²; for Poisson k: (λ_k + λ_k(1-q)) / q² = λ_k(2-q)/q²
    if nb_lambda_k is None:
        _std_trip = (float(nb_k) * (1 - nb_q)) ** 0.5 / nb_q
    else:
        _std_trip = (nb_lambda_k * (2 - nb_q)) ** 0.5 / nb_q
    _avg_p_dp   = None
else:
    _avg_p_dp   = (_w_peak * (p_dp_morning + p_dp_lunch + p_dp_evening)
                   + _w_def * p_dp_default)
    _E_trip     = (1.0 / _avg_p_dp) if _avg_p_dp > 0 else float("inf")
    _std_trip   = _E_trip  # Geom: std = sqrt(1-p)/p ≈ 1/p for small p

# Steady-state: π_D / π_P = avg_p_pd * E_trip  (flow balance)
if _avg_p_pd > 0 and _E_trip < 1e9:
    _rate_ratio = _avg_p_pd * _E_trip   # π_D / π_P
    _pi_d = _rate_ratio / (1 + _rate_ratio)
else:
    _pi_d = 0.0
_pi_p = 1.0 - _pi_d
_trips_per_day = _pi_p * 1440 * _avg_p_pd

_E_dist   = v * (_E_trip / 60)   # km  (v in km/h, E_trip in min)
_E_energy = mu * _E_dist          # kWh

_sc = st.columns(5)
with _sc[0]:
    st.metric("Avg p_PD (per min)", f"{_avg_p_pd:.4f}",
              help="Day-weighted average parked→driving probability.")
with _sc[1]:
    st.metric("E[trip duration]", f"{_E_trip:.1f} min",
              delta=f"σ = {_std_trip:.1f} min", delta_color="off",
              help="Expected trip length. For Baseline: 1/avg_p_DP. For Negative Binomial: E[k]/q.")
with _sc[2]:
    if _avg_p_dp is not None:
        st.metric("Avg p_DP (per min)", f"{_avg_p_dp:.4f}",
                  help="Day-weighted average driving→parked probability (Baseline only).")
    else:
        st.metric("Phase prob q", f"{nb_q:.2f}",
                  help="Per-phase transition probability; E[T] = E[k]/q.")
with _sc[3]:
    st.metric("% time driving", f"{_pi_d * 100:.1f} %",
              delta=f"{_trips_per_day:.1f} trips/day", delta_color="off",
              help="Steady-state fraction of time in driving state.")
with _sc[4]:
    st.metric("E[energy / trip]", f"{_E_energy:.2f} kWh",
              delta=f"{_E_dist:.1f} km", delta_color="off",
              help="Expected distance and energy consumed per trip, given v and μ.")

st.divider()

col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_btn = st.button("▶ Run Backward Induction", type="primary", use_container_width=True)
with col_status:
    if "e_grid" in st.session_state:
        stored_T      = st.session_state.get("T", 0)
        stored_N_e    = len(st.session_state["e_grid"])
        stored_N_a    = len(st.session_state["actions"])
        stored_model  = st.session_state.get("solved_model", "unknown")
        st.success(f"Solution ready — model: {stored_model}, N_e = {stored_N_e}, N_a = {stored_N_a}, T = {stored_T} min ({stored_T // 60} h)")
    else:
        st.info("No solution computed yet. Click **Run Backward Induction** to start.")

if run_btn:
    T   = int(T_hours) * 60
    N_a = None if N_a_sel == "Default" else int(N_a_sel)
    common_kwargs = dict(
        u_max=u_max, u_min=u_min, e_max=e_max, e_min=e_min,
        eta_c=eta_c, phi=phi, beta=beta,
        v=v, mu=mu,
        price_night=price_night, price_morning=price_morning,
        price_midday=price_midday, price_evening=price_evening,
        price_late=price_late, sigma_lambda=sigma_lambda,
        lambda_max=lambda_max,
        p_pd_morning=p_pd_morning, p_pd_lunch=p_pd_lunch,
        p_pd_evening=p_pd_evening, p_pd_default=p_pd_default,
    )

    # ── Build price_bin_probs_fn ──────────────────────────────────────────────
    _pbp_fn = None  # None → models fall back to Gaussian parametric path
    if use_sampler:
        from ev_mdt.pricing.entsoe import load_prices
        from ev_mdt.pricing.samplers import (
            GaussianBinnedSampler, GMMSampler, MDNSampler, make_price_bin_probs_fn,
        )

        _sampler_classes = {
            "Gaussian Bins": GaussianBinnedSampler,
            "GMM": GMMSampler,
            "MDN": MDNSampler,
        }
        _cache_key = f"price_sampler_{price_source}"

        if _cache_key not in st.session_state:
            _label = f"Fitting {price_source} price model on ENTSO-E data…"
            with st.status(_label, expanded=True) as _fit_status:
                _log_lines = []
                _log_area  = st.empty()
                _prog      = st.progress(0.0)
                _detail    = st.empty()

                def _loader_log(msg: str) -> None:
                    _log_lines.append(msg)
                    _log_area.caption("  \n".join(_log_lines))

                _detail.caption("Loading ENTSO-E price data…")
                _df = load_prices(_log=_loader_log)

                def _fit_progress(fraction: float, message: str) -> None:
                    _prog.progress(min(fraction, 1.0))
                    _detail.caption(message)

                _wandb_run = None
                if use_wandb and price_source == "MDN":
                    try:
                        import wandb
                        _wandb_run = wandb.init(
                            project=wandb_project,
                            name=wandb_run_name or None,
                            config={
                                "n_components": 3,
                                "epochs": 200,
                                "batch_size": 1024,
                                "lr": 1e-3,
                                "n_samples": len(_df),
                            },
                            reinit="create_new",
                        )
                    except ImportError:
                        st.warning("wandb is not installed — run `uv add wandb` to enable logging.")

                _sampler = _sampler_classes[price_source]()
                if _wandb_run is not None:
                    st.session_state[_cache_key] = _sampler.fit(
                        _df, _progress=_fit_progress, _wandb_run=_wandb_run,
                    )
                    _wandb_run.finish()
                else:
                    st.session_state[_cache_key] = _sampler.fit(_df, _progress=_fit_progress)
                _fit_status.update(label=f"{price_source} model ready.", state="complete", expanded=False)

        from ev_mdt.params import SharedParams as _SP
        _sp_tmp = _SP(**{k: v for k, v in common_kwargs.items() if k in _SP.__dataclass_fields__})
        _pbp_fn = make_price_bin_probs_fn(st.session_state[_cache_key], _sp_tmp, price_season, price_is_weekend)

    if is_negbin:
        from ev_mdt.params import NegBinParams
        from ev_mdt.models.negbin.backward_induction import backward_induction as _bi
        params = NegBinParams(**common_kwargs, k=int(nb_k), q=float(nb_q),
                              lambda_k=float(nb_lambda_k) if nb_lambda_k is not None else None)
        mode_label = f"λ_k={nb_lambda_k:.1f}" if nb_lambda_k is not None else f"k={nb_k}"
        with st.spinner(f"Running backward induction (Negative Binomial {mode_label}, q={nb_q:.2f}, T={T} min, N_e={N_e})…"):
            V, pi, actions, e_grid, lam_grid = _bi(params, price_bin_probs_fn=_pbp_fn, T=T, N_e=N_e, N_a=N_a)
    else:
        from ev_mdt.params import BaselineParams
        from ev_mdt.models.baseline.model import transition_probs
        from ev_mdt.models.common.model_utils import consumption, price_bin_probs as _gaussian_pbp
        from ev_mdt.models.baseline.backward_induction import backward_induction as _bi
        params = BaselineParams(
            **common_kwargs,
            p_dp_morning=p_dp_morning, p_dp_lunch=p_dp_lunch,
            p_dp_evening=p_dp_evening, p_dp_default=p_dp_default,
        )
        _pbp_fn_baseline = _pbp_fn if _pbp_fn is not None else (lambda t: _gaussian_pbp(t, params))
        n_a_label = "Default" if N_a is None else str(N_a)
        with st.spinner(f"Running backward induction (Baseline, T={T} min, N_e={N_e}, N_a={n_a_label})…"):
            V, pi, actions, e_grid, lam_grid = _bi(
                params,
                transition_probs_fn=lambda t: transition_probs(t, params),
                consumption_fn=lambda chi: consumption(chi, params),
                price_bin_probs_fn=_pbp_fn_baseline,
                T=T,
                N_e=N_e,
                N_a=N_a,
            )

    st.session_state["V"]            = V
    st.session_state["pi"]           = pi
    st.session_state["actions"]      = actions
    st.session_state["e_grid"]       = e_grid
    st.session_state["lam_grid"]     = lam_grid
    st.session_state["params"]       = params
    st.session_state["T"]            = T
    st.session_state["solved_model"] = model
    st.rerun()

# ── Parameter summary (copy-paste) ────────────────────────────────────────────
with st.expander("Parameter summary (copy-paste)"):
    def _fmt(x, d=2):
        return f"{x:.{d}f}"

    lines = [f"Model: {model}",
             f"Solver: N_e = {N_e}, T = {T_hours} h",
             "",
             "Battery & vehicle:",
             f"  u_max = {_fmt(u_max,1)} kW,  u_min = {_fmt(u_min,1)} kW",
             f"  e_max = {_fmt(e_max,1)} kWh,  e_min = {_fmt(e_min,1)} kWh",
             f"  eta_c = {_fmt(eta_c,2)},  beta = {_fmt(beta,3)}",
             f"  v = {_fmt(v,1)} km/h,  mu = {_fmt(mu,2)} kWh/km",
             "",
             "Cost:",
             f"  phi = {_fmt(phi,1)} EUR/h",
             "",
             "Electricity price (mean EUR/kWh):",
             f"  Night   00-06h: {_fmt(price_night,2)}",
             f"  Morning 06-09h: {_fmt(price_morning,2)}",
             f"  Midday  09-16h: {_fmt(price_midday,2)}",
             f"  Evening 16-21h: {_fmt(price_evening,2)}",
             f"  Late    21-24h: {_fmt(price_late,2)}",
             f"  sigma_lambda = {_fmt(sigma_lambda,2)} EUR/kWh",
             "",
             "Mobility — Parked -> Driving (per min):",
             f"  Morning  07-09h: {_fmt(p_pd_morning,4)}",
             f"  Lunch    12-14h: {_fmt(p_pd_lunch,4)}",
             f"  Evening  16-18h: {_fmt(p_pd_evening,4)}",
             f"  Default:         {_fmt(p_pd_default,4)}",
             ]

    if is_negbin:
        exp_k = float(nb_lambda_k) if nb_lambda_k is not None else float(nb_k)
        mode_str = (f"Poisson-sampled, lambda_k = {_fmt(nb_lambda_k,1)}, k_max = {nb_k}"
                    if nb_lambda_k is not None else f"fixed k = {nb_k}")
        lines += ["",
                  "Trip duration — Negative Binomial NB(k, q):",
                  f"  Phases: {mode_str}",
                  f"  q = {_fmt(nb_q,2)}",
                  f"  E[T] = {_fmt(exp_k / nb_q, 1)} min"]
    else:
        lines += ["",
                  "Mobility — Driving -> Parked (per min):",
                  f"  Morning  07:30-09:30h: {_fmt(p_dp_morning,3)}",
                  f"  Lunch    12:15-14:15h: {_fmt(p_dp_lunch,3)}",
                  f"  Evening  16:30-18:30h: {_fmt(p_dp_evening,3)}",
                  f"  Default:               {_fmt(p_dp_default,3)}"]

    st.code("\n".join(lines), language=None)
