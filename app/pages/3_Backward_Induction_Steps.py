import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from ev_mdt.models.baseline import mean_price, transition_probs, consumption, price_bin_probs

st.set_page_config(page_title="Backward Induction — EV Charging MDP", layout="wide")
st.title("Backward Induction — Step-by-Step Verification")

# ── Guard ─────────────────────────────────────────────────────────────────────

if "pi" not in st.session_state:
    st.warning("No solution found. Please go to **Settings** and click **Run Backward Induction** first.")
    st.stop()

if st.session_state.get("solved_model", "").startswith("NegBin"):
    st.info(
        "Step-by-step verification is only available for the **Baseline** model. "
        "The NegBin model uses a (k+1)×(k+1) transition matrix; "
        "switch to Baseline in **Settings** to use this page."
    )
    st.stop()

V        = st.session_state["V"]
actions  = st.session_state["actions"]
e_grid   = st.session_state["e_grid"]
lam_grid = st.session_state["lam_grid"]
params   = st.session_state["params"]
T        = V.shape[0] - 1
N_e      = len(e_grid)
K        = len(lam_grid)

PARKED        = 0
DRIVING       = 1
STATE_LABELS  = {PARKED: "Parked", DRIVING: "Driving"}

# ── Controls ──────────────────────────────────────────────────────────────────

st.caption(
    f"Using Settings parameters: N_e = {N_e}, "
    f"battery {params.e_min:.1f}–{params.e_max:.1f} kWh, "
    f"u_max = {params.u_max:.1f} kW, "
    f"T = {T} min ({T // 60} h)."
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
    k_sel = st.slider("Price bin λ̂", 0, K - 1, key="lam_bin_sel",
                      help="Bin-centre price shown in the Bellman table")

e_idx    = int(np.argmin(np.abs(e_grid - e_sel)))
e_actual = float(e_grid[e_idx])
lam_sel  = float(lam_grid[k_sel])

st.caption(
    f"Nearest grid point: **{e_actual:.3f} kWh** (index {e_idx} of {N_e - 1})  |  "
    f"Price bin: **{k_sel}** → centre **{lam_sel:.3f} €/kWh**"
)
st.divider()


def bellman_table(t: int, chi: int, e: float, k: int) -> pd.DataFrame:
    """Full Bellman backup for one (t, chi, e, price-bin k) tuple."""
    lam          = lam_grid[k]
    p_PD, p_DP   = transition_probs(t, params)
    P            = np.array([[1 - p_PD, p_PD], [p_DP, 1 - p_DP]])
    cons         = consumption(chi, params)

    p_next = price_bin_probs(t + 1, params)
    V_bar  = V[t + 1] @ p_next   # (2, N_e)

    rows = []
    for u in actions:
        u_a = 0.0 if (chi == DRIVING and e > params.e_min) else u

        r = -(lam * params.omega * u_a)
        if chi == DRIVING and e <= params.e_min:
            r -= params.omega * params.phi

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
            f"**λ̄_t** = {lam_bar:.3f} €/kWh &nbsp;|&nbsp; "
            f"**p_PD** = {p_PD:.4f} &nbsp;|&nbsp; "
            f"**p_DP** = {p_DP:.4f} &nbsp;|&nbsp; "
            f"showing price bin **{k_sel}** (centre {lam_sel:.3f} €/kWh)"
        )

        p_next   = price_bin_probs(t + 1, params)
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
                df = bellman_table(t, chi, e_actual, k_sel)
                st.dataframe(df, hide_index=True, use_container_width=True)

        V_t_parked  = V[t, PARKED,  e_idx, k_sel]
        V_t_driving = V[t, DRIVING, e_idx, k_sel]
        st.markdown(
            f"**V[{t}]** stored by solver at bin {k_sel} &nbsp;→&nbsp; "
            f"Parked: `{V_t_parked:.6f}` &nbsp;|&nbsp; "
            f"Driving: `{V_t_driving:.6f}`"
        )
