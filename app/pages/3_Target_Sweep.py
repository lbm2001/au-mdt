import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from ev_mdt.analysis.sensitivity import sweep_target_ceiling_exact
from ev_mdt.plots.calibration import fig_target_sweep
from ev_mdt.models.common.policies import E_CEIL_BASE
from ev_mdt.models.common.model_utils import expected_trip_minutes as _etm

st.set_page_config(page_title="Target Sweep — EV Charging MDP", layout="wide")
st.title("Departure Urgency — Target Ceiling (e_base) Calibration")
st.caption(
    "Sweeps the DU target ceiling from e_min to e_max in fixed kWh steps and plots the "
    "**exact** expected cost (no Monte-Carlo). Use it to find the cost-minimising ceiling."
)

# ── Guard ─────────────────────────────────────────────────────────────────────

if "params" not in st.session_state:
    st.warning("No parameters found. Please go to **Settings** first.")
    st.stop()

params        = st.session_state["params"]
_solved_model = st.session_state.get("solved_model", "Baseline")
use_reserve   = bool(st.session_state.get("du_use_reserve", True))

# ── Controls ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    step_kwh = st.select_slider("Step size (kWh)", [1, 2, 5, 10], value=5)
with col2:
    N_e = st.select_slider("Battery grid points N_e", [100, 200, 500], value=500)

_e_trip_kwh = _etm(params) * params.mu * params.v * params.omega
st.caption(
    f"Reserve floor = e_trip = **{_e_trip_kwh:.3f} kWh** "
    f"({'active' if use_reserve else 'disabled'}) — configure in Settings."
)

# ── Run sweep (cached on its inputs) ───────────────────────────────────────────

_sweep_key = (step_kwh, N_e, use_reserve, _solved_model)
if st.session_state.get("_tsweep_key") != _sweep_key:
    _prog = st.progress(0.0, text="Running exact sweep…")
    rows = sweep_target_ceiling_exact(
        model_label=_solved_model, step_kwh=step_kwh, N_e=N_e, use_reserve=use_reserve,
        progress_cb=lambda f, m: _prog.progress(min(f, 1.0), text=m),
    )
    _prog.empty()
    st.session_state["_tsweep_key"]  = _sweep_key
    st.session_state["_tsweep_rows"] = rows

rows = st.session_state.get("_tsweep_rows", [])
if not rows:
    st.stop()

df   = pd.DataFrame(rows)
best = df.loc[df["mean_cost"].idxmin()]

# ── Plot + table ───────────────────────────────────────────────────────────────

st.plotly_chart(
    fig_target_sweep(rows), use_container_width=True,
    config={"toImageButtonOptions": {"format": "png", "filename": "target_sweep", "scale": 4},
            "displaylogo": False},
)

st.caption(
    f"**Best ceiling**: {best['target_kwh']:.0f} kWh ({best['target_frac']:.0%} of e_max) — "
    f"expected cost {best['mean_cost']:.4f} €. Set it as `E_CEIL_BASE` in "
    f"`ev_mdt/models/common/policies.py` (currently {E_CEIL_BASE:.1f} kWh)."
)

st.subheader("Full results")
st.dataframe(
    df.rename(columns={
        "target_kwh":        "Target ceiling (kWh)",
        "target_frac":       "Target ceiling (%)",
        "mean_cost":         "Expected cost (€)",
        "mean_charge_cost":  "Expected charging (€)",
        "mean_penalty_cost": "Expected penalty (€)",
        "mean_penalty_min":  "Expected penalty (min)",
        "mean_charged":      "Expected charged (kWh)",
    }).style
      .highlight_min(subset=["Expected cost (€)"], color="#d4edda")
      .format({
          "Target ceiling (kWh)":  "{:.0f}",
          "Target ceiling (%)":    "{:.0%}",
          "Expected cost (€)":     "{:.4f}",
          "Expected charging (€)": "{:.4f}",
          "Expected penalty (€)":  "{:.4f}",
          "Expected penalty (min)": "{:.1f}",
          "Expected charged (kWh)": "{:.2f}",
      }),
    use_container_width=True, hide_index=True,
)
