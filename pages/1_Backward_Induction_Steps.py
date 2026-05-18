import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from models.baseline import mean_price, transition_probs, consumption

st.set_page_config(page_title="Backward Induction Steps", layout="wide")
st.title("Backward Induction — Step-by-Step Verification")

if "V" not in st.session_state:
    st.warning("Run the **Policy Explorer** page first to compute the solution.")
    st.stop()

V       = st.session_state["V"]
actions = st.session_state["actions"]
e_grid  = st.session_state["e_grid"]
params  = st.session_state["params"]

T   = V.shape[0] - 1   # 1440
N_e = len(e_grid)

PARKED  = 0
DRIVING = 1
STATE_LABELS = {PARKED: "Parked", DRIVING: "Driving"}

# ── Controls ──────────────────────────────────────────────────────────────────

n_steps = st.slider("Number of steps to show", 1, 10, 3)
e_sel   = st.slider(
    "Battery level to inspect (kWh)",
    float(params.e_min), float(params.e_max), float(params.e_max / 2),
    step=float((params.e_max - params.e_min) / (N_e - 1)),
)
e_idx = int(np.argmin(np.abs(e_grid - e_sel)))
e_actual = float(e_grid[e_idx])

st.caption(f"Nearest grid point: **{e_actual:.3f} kWh** (index {e_idx} of {N_e - 1})")
st.divider()


def bellman_table(t: int, chi: int, e: float, e_i: int) -> pd.DataFrame:
    """Full Bellman backup for one (t, chi, e) triple."""
    lam_bar      = mean_price(t, params)
    p_PD, p_DP   = transition_probs(t, params)
    P = np.array([[1 - p_PD, p_PD], [p_DP, 1 - p_DP]])
    cons         = consumption(chi, params)

    rows = []
    for a_idx, u in enumerate(actions):
        # actual charging rate
        u_a = 0.0 if (chi == DRIVING and e > params.e_min) else u

        # immediate reward
        r = -(lam_bar / 1000 * params.omega * u_a)
        if chi == DRIVING and e <= params.e_min:
            r -= params.omega * params.phi

        # next battery level
        e_next = float(np.clip(
            e + params.eta_c * params.omega * u_a - cons,
            params.e_min, params.e_max,
        ))
        e_next_i = int(np.round(
            (e_next - params.e_min) / (params.e_max - params.e_min) * (N_e - 1)
        ))

        # expected continuation value
        V_next = (P[chi, PARKED]  * V[t + 1, PARKED,  e_next_i]
                + P[chi, DRIVING] * V[t + 1, DRIVING, e_next_i])

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

    # mark optimal action
    q_vals  = [float(row["Q = r + β·V"]) for row in rows]
    opt_idx = int(np.argmax(q_vals))
    rows[opt_idx]["optimal"] = "✓"

    return pd.DataFrame(rows)


# ── Show each step ────────────────────────────────────────────────────────────

for step in range(n_steps):
    t      = T - 1 - step
    h, m   = divmod(t, 60)
    lam_bar          = mean_price(t, params)
    p_PD, p_DP       = transition_probs(t, params)

    with st.expander(f"Step {step + 1}  —  t = {t}  ({h:02d}:{m:02d})", expanded=(step == 0)):
        st.markdown(
            f"**λ̄_t** = {lam_bar:.1f} €/MWh &nbsp;|&nbsp; "
            f"**p_PD** = {p_PD:.4f} &nbsp;|&nbsp; "
            f"**p_DP** = {p_DP:.4f}"
        )

        # V[t+1] at the selected battery level (context for V_next)
        st.markdown(
            f"**V[{t+1}]** at e = {e_actual:.3f} kWh &nbsp;→&nbsp; "
            f"Parked: `{V[t + 1, PARKED, e_idx]:.6f}` &nbsp;|&nbsp; "
            f"Driving: `{V[t + 1, DRIVING, e_idx]:.6f}`"
        )

        col_p, col_d = st.columns(2)
        for chi, col in ((PARKED, col_p), (DRIVING, col_d)):
            with col:
                st.markdown(f"**{STATE_LABELS[chi]}**")
                df = bellman_table(t, chi, e_actual, e_idx)
                st.dataframe(df, hide_index=True, use_container_width=True)

        # V[t] at selected e as computed by the solver (ground truth check)
        V_t_parked  = V[t, PARKED,  e_idx]
        V_t_driving = V[t, DRIVING, e_idx]
        st.markdown(
            f"**V[{t}]** stored by solver &nbsp;→&nbsp; "
            f"Parked: `{V_t_parked:.6f}` &nbsp;|&nbsp; "
            f"Driving: `{V_t_driving:.6f}`"
        )
