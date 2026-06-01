import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Settings — EV Charging MDP", layout="wide")
st.title("Settings")
st.caption("Adjust all model parameters here, then click **Run Backward Induction** to compute the optimal policy.")

# ── Model selector ────────────────────────────────────────────────────────────

def _on_model_change():
    for key in ["V", "pi", "actions", "e_grid", "lam_grid", "params", "T"]:
        st.session_state.pop(key, None)

model = st.selectbox(
    "Mobility model",
    ["Baseline — Geometric trips", "NegBin — Erlang trips"],
    key="model",
    on_change=_on_model_change,
    help=(
        "**Baseline**: trip duration ~ Geom(p_DP), mean ≈ 1/p_DP min. "
        "**NegBin**: trip duration ~ NegBin(k, q) via a k-phase Markov chain, "
        "mean = k/q min."
    ),
)
is_negbin = "NegBin" in model

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    u_max=11.0, u_min=1.4, e_max=40.0, e_min=0.0,
    eta_c=0.95, phi=1000.0, beta=0.999,
    v=50.0, mu=0.20,
    price_night=0.30, price_morning=0.48, price_midday=0.39,
    price_evening=0.55, price_late=0.34, sigma_lambda=0.05,
    p_pd_morning=0.08, p_pd_lunch=0.03, p_pd_evening=0.07, p_pd_default=0.005,
    p_dp_morning=0.15, p_dp_lunch=0.20, p_dp_evening=0.15, p_dp_default=0.25,
    nb_k=5, nb_q=0.20,
    N_e=200, T_hours=48,
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
    phi   = st.slider("Unserved-driving penalty φ (€/h)", 0.0, 5000.0, _DEFAULTS["phi"], 50.0, key="phi")
    beta  = st.slider("Discount factor β", 0.900, 1.000, _DEFAULTS["beta"], 0.001,
                      format="%.3f", key="beta")

    st.subheader("Electricity Price (€/kWh)")
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
        st.markdown("**NegBin trip duration: T ~ NegBin(k, q)**")
        nb_k = st.slider("Phases k  (concentration)",
                         1, 20, _DEFAULTS["nb_k"], 1, key="nb_k",
                         help="Number of driving phases. E[T] = k / q.")
        nb_q = st.slider("Phase prob q  (timescale)",
                         0.01, 1.0, _DEFAULTS["nb_q"], 0.01,
                         format="%.2f", key="nb_q",
                         help="Per-phase transition probability. E[T] = k / q.")
        st.caption(f"E[T] = {nb_k}/{nb_q:.2f} = **{nb_k / nb_q:.1f} min**,  "
                   f"Var[T] = {nb_k * (1 - nb_q) / nb_q**2:.1f} min²")
    else:
        st.markdown("**Driving → Parked**")
        p_dp_morning = st.slider("Morning  (07:30–09:30 h)", 0.0, 1.0, _DEFAULTS["p_dp_morning"], 0.01, key="p_dp_morning")
        p_dp_lunch   = st.slider("Lunch    (12:15–14:15 h)", 0.0, 1.0, _DEFAULTS["p_dp_lunch"],   0.01, key="p_dp_lunch")
        p_dp_evening = st.slider("Evening  (16:30–18:30 h)", 0.0, 1.0, _DEFAULTS["p_dp_evening"], 0.01, key="p_dp_evening")
        p_dp_default = st.slider("Default",                  0.0, 1.0, _DEFAULTS["p_dp_default"], 0.01, key="p_dp_default")

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
        p_pd_morning=p_pd_morning, p_pd_lunch=p_pd_lunch,
        p_pd_evening=p_pd_evening, p_pd_default=p_pd_default,
    )

    if is_negbin:
        from models.negative_binomial_trips import NegBinParams
        from models.negative_binomial_trips.backward_induction import backward_induction as _bi
        params = NegBinParams(**common_kwargs, k=int(nb_k), q=float(nb_q))
        with st.spinner(f"Running backward induction (NegBin k={nb_k}, q={nb_q:.2f}, T={T} min, N_e={N_e})…"):
            V, pi, actions, e_grid, lam_grid = _bi(params, T=T, N_e=N_e)
    else:
        from models.baseline import BaselineParams
        from models.baseline.model import transition_probs, consumption, price_bin_probs
        from utils.backward_induction import backward_induction as _bi
        params = BaselineParams(
            **common_kwargs,
            p_dp_morning=p_dp_morning, p_dp_lunch=p_dp_lunch,
            p_dp_evening=p_dp_evening, p_dp_default=p_dp_default,
        )
        n_a_label = "Default" if N_a is None else str(N_a)
        with st.spinner(f"Running backward induction (Baseline, T={T} min, N_e={N_e}, N_a={n_a_label})…"):
            V, pi, actions, e_grid, lam_grid = _bi(
                params,
                transition_probs_fn=lambda t: transition_probs(t, params),
                consumption_fn=lambda chi: consumption(chi, params),
                price_bin_probs_fn=lambda t: price_bin_probs(t, params),
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
