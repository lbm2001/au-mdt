import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from models.baseline import BaselineParams, mean_price, transition_probs, consumption, price_bin_probs
from utils.backward_induction import backward_induction

st.set_page_config(page_title="Backward Induction Steps", layout="wide")
st.title("Backward Induction — Step-by-Step Verification")

_DEFAULTS = dict(
    u_max=11.0, u_min=1.4, e_max=40.0, e_min=0.0,
    eta_c=0.95, phi=1000.0, beta=0.999,
    v=50.0, mu=0.20,
    price_night=70.0, price_morning=150.0, price_midday=110.0,
    price_evening=170.0, price_late=100.0, sigma_lambda=20.0,
    p_pd_morning=0.08, p_pd_lunch=0.03, p_pd_evening=0.07, p_pd_default=0.005,
    p_dp_morning=0.15, p_dp_lunch=0.20, p_dp_evening=0.15, p_dp_default=0.25,
    N_e=200,
)


def _session_value(key: str):
    return st.session_state.get(key, _DEFAULTS[key])

def _params_from_session() -> tuple[BaselineParams, int]:
    params = BaselineParams(
        u_max=_session_value("u_max"),
        u_min=_session_value("u_min"),
        e_max=_session_value("e_max"),
        e_min=_session_value("e_min"),
        eta_c=_session_value("eta_c"),
        phi=_session_value("phi"),
        beta=_session_value("beta"),
        v=_session_value("v"),
        mu=_session_value("mu"),
        price_night=_session_value("price_night"),
        price_morning=_session_value("price_morning"),
        price_midday=_session_value("price_midday"),
        price_evening=_session_value("price_evening"),
        price_late=_session_value("price_late"),
        sigma_lambda=_session_value("sigma_lambda"),
        p_pd_morning=_session_value("p_pd_morning"),
        p_pd_lunch=_session_value("p_pd_lunch"),
        p_pd_evening=_session_value("p_pd_evening"),
        p_pd_default=_session_value("p_pd_default"),
        p_dp_morning=_session_value("p_dp_morning"),
        p_dp_lunch=_session_value("p_dp_lunch"),
        p_dp_evening=_session_value("p_dp_evening"),
        p_dp_default=_session_value("p_dp_default"),
    )
    return params, int(_session_value("N_e"))


def _ensure_solution() -> None:
    required = {"V", "pi", "actions", "e_grid", "lam_grid", "params"}
    if required.issubset(st.session_state):
        return

    params, N_e = _params_from_session()
    with st.spinner("Rebuilding solution from Policy Explorer parameters..."):
        V, pi, actions, e_grid, lam_grid = backward_induction(
            params,
            transition_probs_fn=lambda t: transition_probs(t, params),
            consumption_fn=lambda chi: consumption(chi, params),
            price_bin_probs_fn=lambda t: price_bin_probs(t, params),
            T=1440,
            N_e=N_e,
        )

    st.session_state["V"] = V
    st.session_state["pi"] = pi
    st.session_state["actions"] = actions
    st.session_state["e_grid"] = e_grid
    st.session_state["lam_grid"] = lam_grid
    st.session_state["params"] = params


_ensure_solution()

V        = st.session_state["V"]
actions  = st.session_state["actions"]
e_grid   = st.session_state["e_grid"]
lam_grid = st.session_state["lam_grid"]
params   = st.session_state["params"]

T   = V.shape[0] - 1   # 1440
N_e = len(e_grid)
K   = len(lam_grid)

PARKED  = 0
DRIVING = 1
STATE_LABELS = {PARKED: "Parked", DRIVING: "Driving"}

# ── Controls ──────────────────────────────────────────────────────────────────

st.caption(
    f"Using Policy Explorer parameters: N_e={N_e}, "
    f"battery {params.e_min:.1f}-{params.e_max:.1f} kWh, "
    f"u_max={params.u_max:.1f} kW."
)

n_steps = st.slider("Number of steps to show", 1, 10, 3)
col_e, col_k = st.columns(2)
with col_e:
    if "step_e_sel" in st.session_state:
        st.session_state["step_e_sel"] = min(
            max(float(st.session_state["step_e_sel"]), float(params.e_min)),
            float(params.e_max),
        )
    e_sel = st.slider(
        "Battery level to inspect (kWh)",
        float(params.e_min), float(params.e_max), float(params.e_max / 2),
        step=float((params.e_max - params.e_min) / (N_e - 1)),
        key="step_e_sel",
    )
with col_k:
    if "lam_bin_sel" in st.session_state:
        st.session_state["lam_bin_sel"] = min(max(int(st.session_state["lam_bin_sel"]), 0), K - 1)
    else:
        st.session_state["lam_bin_sel"] = K // 2
    k_sel = st.slider(
        "Price bin λ̂", 0, K - 1, key="lam_bin_sel",
        help="Bin-centre price shown in the Bellman table",
    )

e_idx    = int(np.argmin(np.abs(e_grid - e_sel)))
e_actual = float(e_grid[e_idx])
lam_sel  = float(lam_grid[k_sel])

st.caption(
    f"Nearest grid point: **{e_actual:.3f} kWh** (index {e_idx} of {N_e - 1})  |  "
    f"Price bin: **{k_sel}** → centre **{lam_sel:.1f} €/MWh**"
)
st.divider()


def bellman_table(t: int, chi: int, e: float, e_i: int, k: int) -> pd.DataFrame:
    """Full Bellman backup for one (t, chi, e, price-bin k) tuple."""
    lam          = lam_grid[k]                   # bin-centre price (€/MWh)
    p_PD, p_DP   = transition_probs(t, params)
    P            = np.array([[1 - p_PD, p_PD], [p_DP, 1 - p_DP]])
    cons         = consumption(chi, params)

    # Price-averaged continuation: E_{λ̂'}[V_{t+1}(chi', e', λ̂')]
    # = V[t+1] @ p_next, shape (2, N_e)
    p_next = price_bin_probs(t + 1, params)
    V_bar  = V[t + 1] @ p_next               # (2, N_e)

    rows = []
    for a_idx, u in enumerate(actions):
        u_a = 0.0 if (chi == DRIVING and e > params.e_min) else u

        # immediate reward uses bin-centre price
        r = -(lam / 1000 * params.omega * u_a)
        if chi == DRIVING and e <= params.e_min:
            r -= params.omega * params.phi

        # next battery with linear interpolation
        e_next = float(np.clip(
            e + params.eta_c * params.omega * u_a - cons,
            params.e_min, params.e_max,
        ))
        e_next_f = (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
        e_lo = int(np.floor(e_next_f))
        e_hi = min(e_lo + 1, N_e - 1)
        w_hi = e_next_f - e_lo
        w_lo = 1.0 - w_hi

        V_next = (P[chi, PARKED]  * (w_lo * V_bar[PARKED,  e_lo] + w_hi * V_bar[PARKED,  e_hi])
                + P[chi, DRIVING] * (w_lo * V_bar[DRIVING, e_lo] + w_hi * V_bar[DRIVING, e_hi]))

        Q = r + params.beta * V_next

        rows.append({
            "u (kW)":       f"{u:.1f}",
            "u_a (kW)":     f"{u_a:.1f}",
            "e_next (kWh)": f"{e_next:.3f}",
            "r":            f"{r:.6f}",
            "E[V_{t+1}]":   f"{V_next:.6f}",
            "Q = r + β·V":  f"{Q:.6f}",
            "optimal":      "",
        })

    q_vals  = [float(row["Q = r + β·V"]) for row in rows]
    opt_idx = int(np.argmax(q_vals))
    rows[opt_idx]["optimal"] = "✓"

    return pd.DataFrame(rows)


# ── Show each step ────────────────────────────────────────────────────────────

for step in range(n_steps):
    t      = T - 1 - step
    h, m   = divmod(t, 60)
    lam_bar    = mean_price(t, params)
    p_PD, p_DP = transition_probs(t, params)

    with st.expander(f"Step {step + 1}  —  t = {t}  ({h:02d}:{m:02d})", expanded=(step == 0)):
        st.markdown(
            f"**λ̄_t** = {lam_bar:.1f} €/MWh &nbsp;|&nbsp; "
            f"**p_PD** = {p_PD:.4f} &nbsp;|&nbsp; "
            f"**p_DP** = {p_DP:.4f} &nbsp;|&nbsp; "
            f"showing price bin **{k_sel}** (centre {lam_sel:.1f} €/MWh)"
        )

        # Price-averaged V[t+1] at the selected battery level
        p_next = price_bin_probs(t + 1, params)
        V_bar_t1 = V[t + 1] @ p_next   # (2, N_e)
        st.markdown(
            f"**E[V[{t+1}]]** at e = {e_actual:.3f} kWh &nbsp;→&nbsp; "
            f"Parked: `{V_bar_t1[PARKED, e_idx]:.6f}` &nbsp;|&nbsp; "
            f"Driving: `{V_bar_t1[DRIVING, e_idx]:.6f}`"
        )

        col_p, col_d = st.columns(2)
        for chi, col in ((PARKED, col_p), (DRIVING, col_d)):
            with col:
                st.markdown(f"**{STATE_LABELS[chi]}**")
                df = bellman_table(t, chi, e_actual, e_idx, k_sel)
                st.dataframe(df, hide_index=True, use_container_width=True)

        # V[t] at selected (e, k) as stored by the solver
        V_t_parked  = V[t, PARKED,  e_idx, k_sel]
        V_t_driving = V[t, DRIVING, e_idx, k_sel]
        st.markdown(
            f"**V[{t}]** stored by solver at bin {k_sel} &nbsp;→&nbsp; "
            f"Parked: `{V_t_parked:.6f}` &nbsp;|&nbsp; "
            f"Driving: `{V_t_driving:.6f}`"
        )
