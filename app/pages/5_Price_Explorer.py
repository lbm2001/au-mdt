import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from ev_mdt.pricing.entsoe import load_prices
from ev_mdt.pricing.samplers import SEASONS
from ev_mdt.analysis.prices import fit_samplers, simulate_price_paths, price_figures

st.set_page_config(page_title="Price Explorer — EV Charging MDP", layout="wide")
st.title("Price Explorer")
st.caption(
    "Fit all four pricing models on ENTSO-E data, simulate N days of hourly prices, "
    "and compare their mean diurnal profile."
)

# ── Fit ───────────────────────────────────────────────────────────────────────

st.subheader("Pricing models")

_SAMPLER_NAMES = ["Gaussian Bins", "GMM", "MDN"]
_ALL_FITTED    = all(f"price_sampler_{n}" in st.session_state for n in _SAMPLER_NAMES)

if st.button("Load ENTSO-E data & fit all models", type="primary", disabled=_ALL_FITTED):
    with st.status("Fitting pricing models…", expanded=True) as _status:
        _log_lines = []
        _log_area  = st.empty()
        _prog      = st.progress(0.0)
        _detail    = st.empty()

        def _loader_log(msg: str) -> None:
            _log_lines.append(msg)
            _log_area.caption("  \n".join(_log_lines[-6:]))

        _detail.caption("Loading ENTSO-E price data…")
        _df = load_prices(_log=_loader_log)
        n_samples = len(_df)
        y0 = int(_df["timestamp"].dt.year.min())
        y1 = int(_df["timestamp"].dt.year.max())
        st.session_state["_pe_fit_meta"] = {"n_samples": n_samples, "y0": y0, "y1": y1}

        def _progress(model: str, frac: float, msg: str) -> None:
            _prog.progress(frac)
            _detail.caption(f"{model}: {msg}")

        fitted = fit_samplers(_df, progress_cb=_progress)
        for name, sampler in fitted.items():
            st.session_state[f"price_sampler_{name}"] = sampler

        _prog.progress(1.0)
        _status.update(label="All models fitted.", state="complete", expanded=False)
    st.rerun()

if st.button("Re-fit models", disabled=not _ALL_FITTED):
    for n in _SAMPLER_NAMES:
        st.session_state.pop(f"price_sampler_{n}", None)
    st.session_state.pop("_pe_fit_meta", None)
    st.session_state.pop("_pe_sim", None)
    st.rerun()

if not _ALL_FITTED:
    st.info("Click **Load ENTSO-E data & fit all models** to begin.")
    st.stop()

meta = st.session_state.get("_pe_fit_meta", {})
if meta:
    st.success(
        f"Fitted on **{meta['n_samples']:,}** measurements ({meta['y0']}–{meta['y1']})."
    )

st.divider()

# ── Simulation controls ───────────────────────────────────────────────────────

st.subheader("Simulation")

c1, c2, c3, c4 = st.columns(4)
with c1:
    season_sel  = st.selectbox("Season",   ["All"] + [s.capitalize() for s in SEASONS])
with c2:
    daytype_sel = st.selectbox("Day type", ["All", "Weekday", "Weekend"])
with c3:
    n_days = st.select_slider("Simulated days N", [100, 250, 500, 1000, 2000], value=1000)
with c4:
    sim_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

_sim_key = (season_sel, daytype_sel, n_days, int(sim_seed))
if st.session_state.get("_pe_sim_key") != _sim_key:
    st.session_state.pop("_pe_sim", None)

if "_pe_sim" not in st.session_state:
    samplers = {n: st.session_state[f"price_sampler_{n}"] for n in _SAMPLER_NAMES}
    season   = None if season_sel == "All" else season_sel.lower()

    with st.spinner(f"Simulating {n_days} days across all models…"):
        results = simulate_price_paths(
            samplers, n_days=n_days, season=season,
            daytype=daytype_sel.lower(), seed=int(sim_seed),
        )
    st.session_state["_pe_sim"]     = results
    st.session_state["_pe_sim_key"] = _sim_key

results = st.session_state["_pe_sim"]

# ── Plots ─────────────────────────────────────────────────────────────────────

fig_mean, fig_std = price_figures(results)

st.subheader("Mean diurnal price profile")
st.caption(
    f"Mean price ± 1 SEM across {n_days} simulated days — one line per pricing model.  "
    "SEM = std/√N (uncertainty of the mean, not spread across days)."
)
st.plotly_chart(fig_mean, use_container_width=True, config={"displaylogo": False})

st.subheader("Price spread (std) by hour")
st.caption(
    "Standard deviation of sampled prices across all simulated days — "
    "captures how volatile each model is at each hour."
)
st.plotly_chart(fig_std, use_container_width=True, config={"displaylogo": False})
