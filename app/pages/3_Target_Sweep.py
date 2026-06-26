import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ev_mdt.analysis.sensitivity import sweep_target_ceiling

st.set_page_config(page_title="Target Sweep — EV Charging MDP", layout="wide")
st.title("Departure Urgency — Target Ceiling Sweep")
st.caption(
    "Sweeps the target ceiling from e_min to e_max in fixed kWh steps, "
    "runs N rollouts per value, and plots mean cost. "
    "Use this to empirically find the best ceiling for the current params."
)

# ── Guard ─────────────────────────────────────────────────────────────────────

if "params" not in st.session_state:
    st.warning("No parameters found. Please go to **Settings** first.")
    st.stop()

params        = st.session_state["params"]
_solved_model = st.session_state.get("solved_model", "Baseline")

# ── Sweep controls ─────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_rollouts = st.select_slider("Rollouts per value", [50, 100, 200, 500], value=500)
with col2:
    step_kwh = st.select_slider("Step size (kWh)", [1, 2, 5, 10], value=5)
with col3:
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
with col4:
    chi0_label = st.radio("Initial state", ["Parked", "Driving"], horizontal=True)
chi0 = 0 if chi0_label == "Parked" else 1

st.divider()
st.subheader("Fixed policy settings")
st.caption("Reserve and α are held constant across the sweep — only the ceiling moves.")

use_reserve = bool(st.session_state.get("du_use_reserve", True))

from ev_mdt.models.common.model_utils import expected_trip_minutes as _etm
_e_trip_kwh = _etm(params) * params.mu * params.v * params.omega

st.caption(
    f"Reserve floor = e_trip = **{_e_trip_kwh:.3f} kWh** "
    f"({'active' if use_reserve else 'disabled'}) — configure in Settings."
)

# ── Run sweep ─────────────────────────────────────────────────────────────────

_sweep_key = (n_rollouts, step_kwh, int(seed), chi0, use_reserve, _solved_model)

if st.session_state.get("_sweep_key") != _sweep_key:
    _prog = st.progress(0.0, text="Running sweep…")

    def _progress_cb(frac: float, msg: str) -> None:
        _prog.progress(frac, text=msg)

    rows = sweep_target_ceiling(
        model_label=_solved_model,
        N_rollouts=n_rollouts,
        seed=int(seed),
        step_kwh=step_kwh,
        progress_cb=_progress_cb,
    )
    _prog.empty()

    # Normalise keys for display
    for r in rows:
        r["Target ceiling (kWh)"]    = r.pop("target_kwh")
        r["Target ceiling (%)"]      = int(round(r.pop("target_frac") * 100))
        r["Mean cost (€)"]           = r.pop("mean_cost")
        r["Std cost (€)"]            = r.pop("std_cost")
        r["Mean penalty (min)"]      = r.pop("mean_penalty")
        r["Mean charged (kWh)"]      = r.pop("mean_charged")
        r["Mean charging cost (€)"]  = r.pop("mean_charge_cost")
        r["Mean penalty cost (€)"]   = r.pop("mean_penalty_cost")

    st.session_state["_sweep_key"]     = _sweep_key
    st.session_state["_sweep_results"] = rows

rows = st.session_state.get("_sweep_results", [])
if not rows:
    st.stop()

df = pd.DataFrame(rows)
best_idx = int(df["Mean cost (€)"].idxmin())
best_row = df.iloc[best_idx]

# ── Plot ───────────────────────────────────────────────────────────────────────

fig = go.Figure()

x = df["Target ceiling (kWh)"]

# error band (left axis)
fig.add_trace(go.Scatter(
    x=pd.concat([x, x.iloc[::-1]]),
    y=pd.concat([
        df["Mean cost (€)"] + df["Std cost (€)"],
        (df["Mean cost (€)"] - df["Std cost (€)"]).iloc[::-1],
    ]),
    fill="toself", fillcolor="rgba(68,119,170,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Total ±1 std", hoverinfo="skip", yaxis="y1",
))

fig.add_trace(go.Scatter(
    x=x, y=df["Mean cost (€)"],
    mode="lines+markers",
    line=dict(color="#4477AA", width=2),
    marker=dict(size=7),
    name="Total cost", yaxis="y1",
))

fig.add_trace(go.Scatter(
    x=x, y=df["Mean penalty cost (€)"],
    mode="lines+markers",
    line=dict(color="#CC3311", width=2, dash="dash"),
    marker=dict(size=6),
    name="Penalty cost", yaxis="y1",
))

fig.add_trace(go.Scatter(
    x=[best_row["Target ceiling (kWh)"]],
    y=[best_row["Mean cost (€)"]],
    mode="markers",
    marker=dict(color="#EE6677", size=12, symbol="star"),
    name=f"Best: {best_row['Target ceiling (kWh)']:.0f} kWh ({best_row['Target ceiling (%)']:.0f}%)",
    yaxis="y1",
))

# charging cost — right axis
fig.add_trace(go.Scatter(
    x=x, y=df["Mean charging cost (€)"],
    mode="lines+markers",
    line=dict(color="#228833", width=2, dash="dot"),
    marker=dict(size=6),
    name="Charging cost (right)", yaxis="y2",
))

# current settings line (show where E_CEIL_BASE sits, since that's the baseline-calibrated value)
from ev_mdt.models.common.policies import E_CEIL_BASE
current_ceil = E_CEIL_BASE
fig.add_vline(
    x=current_ceil, line_dash="dash", line_color="#009988",
    annotation_text=f"Current setting ({current_ceil:.0f} kWh)",
    annotation_position="top right",
)

fig.update_layout(
    xaxis_title="Target ceiling (kWh)",
    yaxis=dict(title="Total / penalty cost (€)"),
    yaxis2=dict(title="Charging cost (€)", overlaying="y", side="right",
                showgrid=False),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    height=420,
    margin=dict(l=60, r=70, t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True,
                config={"toImageButtonOptions": {"format": "png", "filename": "target_sweep", "scale": 4},
                        "displaylogo": False})

st.caption(
    f"**Best ceiling**: {best_row['Target ceiling (kWh)']:.0f} kWh "
    f"({best_row['Target ceiling (%)']:.0f}% of e_max) — "
    f"mean cost {best_row['Mean cost (€)']:.4f} €"
)

# ── Apply best button ──────────────────────────────────────────────────────────

st.caption(
    f"The best ceiling found here should be set as `E_CEIL_BASE` in `ev_mdt/models/common/policies.py` "
    f"(currently {E_CEIL_BASE:.1f} kWh). The γ and α sliders in Settings let you scale it across "
    f"mobility models without re-running this sweep."
)

# ── Table ─────────────────────────────────────────────────────────────────────

st.subheader("Full results")
st.dataframe(
    df.style
      .highlight_min(subset=["Mean cost (€)"], color="#d4edda")
      .format({
          "Target ceiling (kWh)":   "{:.0f}",
          "Target ceiling (%)":     "{:.0f}%",
          "Mean cost (€)":          "{:.4f}",
          "Std cost (€)":           "{:.4f}",
          "Mean charging cost (€)": "{:.4f}",
          "Mean penalty cost (€)":  "{:.4f}",
          "Mean penalty (min)":     "{:.1f}",
          "Mean charged (kWh)":     "{:.2f}",
      }),
    use_container_width=True,
    hide_index=True,
)
