import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from itertools import product

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ev_mdt.params import SharedParams
from ev_mdt.pricing.entsoe import load_prices
from ev_mdt.pricing.samplers import GaussianBinnedSampler, GMMSampler, MDNSampler, SEASONS
from ev_mdt.models.common.model_utils import mean_price

st.set_page_config(page_title="Price Explorer — EV Charging MDP", layout="wide")
st.title("Price Explorer")
st.caption(
    "Simulate price paths from all four pricing models over N days and compare "
    "their mean diurnal profile."
)

# ── Colors ────────────────────────────────────────────────────────────────────

_COLORS = {
    "Gaussian (parametric)": "#4477AA",
    "Gaussian Bins":          "#EE7733",
    "GMM":                    "#228833",
    "MDN":                    "#EE6677",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


# ── Fit data-driven models ────────────────────────────────────────────────────

st.subheader("Fit pricing models")

_SAMPLER_CLASSES = {
    "Gaussian Bins": GaussianBinnedSampler,
    "GMM":           GMMSampler,
    "MDN":           MDNSampler,
}

if st.button("Load ENTSO-E data & fit all models", type="primary"):
    st.session_state.pop("_pe_fit_info", None)
    for key in [f"price_sampler_{n}" for n in _SAMPLER_CLASSES]:
        st.session_state.pop(key, None)

_fit_info = st.session_state.get("_pe_fit_info")
_df_prices = None

if not _fit_info or any(f"price_sampler_{n}" not in st.session_state for n in _SAMPLER_CLASSES):
    _need_fit = True
else:
    _need_fit = False

if st.session_state.get("_pe_load_triggered"):
    st.session_state.pop("_pe_load_triggered")
    _need_fit = True

# Auto-fit on first button press (handled by pop above)
if "Load ENTSO-E data & fit all models" and _need_fit and "_pe_fit_info" not in st.session_state:
    pass  # wait for user to click

# Check if fit button was just clicked
_fit_requested = "_pe_fit_info" not in st.session_state and any(
    f"price_sampler_{n}" not in st.session_state for n in _SAMPLER_CLASSES
)

if _fit_requested and st.session_state.get("_pe_fit_requested"):
    with st.status("Fitting pricing models…", expanded=True) as _status:
        _log_lines = []
        _log_area  = st.empty()
        _prog      = st.progress(0.0)
        _detail    = st.empty()

        def _loader_log(msg: str) -> None:
            _log_lines.append(msg)
            _log_area.caption("  \n".join(_log_lines[-6:]))

        _detail.caption("Loading ENTSO-E price data…")
        _df_prices = load_prices(_log=_loader_log)
        n_samples  = len(_df_prices)
        y0 = int(_df_prices["timestamp"].dt.year.min())
        y1 = int(_df_prices["timestamp"].dt.year.max())
        st.session_state["_pe_fit_info"] = {"n_samples": n_samples, "year_min": y0, "year_max": y1}

        for i, (name, cls) in enumerate(_SAMPLER_CLASSES.items()):
            frac = (i + 0.1) / len(_SAMPLER_CLASSES)
            _prog.progress(frac)
            _detail.caption(f"Fitting {name}…")

            def _fit_progress(fraction: float, message: str, _n=name) -> None:
                _prog.progress((i + fraction) / len(_SAMPLER_CLASSES))
                _detail.caption(f"{_n}: {message}")

            cache_key = f"price_sampler_{name}"
            if cache_key not in st.session_state:
                sampler = cls()
                st.session_state[cache_key] = sampler.fit(_df_prices, _progress=_fit_progress)

        _prog.progress(1.0)
        _status.update(label="All models fitted.", state="complete", expanded=False)
    st.rerun()

# Simpler flow: button sets a flag and reruns
if st.button("", key="_pe_fit_btn_hidden", help=""):
    pass  # dummy — actual logic below

# Cleaner trigger mechanism
_ALL_FITTED = all(f"price_sampler_{n}" in st.session_state for n in _SAMPLER_CLASSES)

if not _ALL_FITTED:
    if st.session_state.get("_trigger_fit"):
        st.session_state.pop("_trigger_fit")
        with st.status("Fitting pricing models…", expanded=True) as _status:
            _log_lines = []
            _log_area  = st.empty()
            _prog      = st.progress(0.0)
            _detail    = st.empty()

            def _loader_log(msg: str) -> None:
                _log_lines.append(msg)
                _log_area.caption("  \n".join(_log_lines[-6:]))

            _detail.caption("Loading ENTSO-E price data…")
            _df_prices = load_prices(_log=_loader_log)
            n_samples  = len(_df_prices)
            y0 = int(_df_prices["timestamp"].dt.year.min())
            y1 = int(_df_prices["timestamp"].dt.year.max())
            st.session_state["_pe_fit_info"] = {
                "n_samples": n_samples, "year_min": y0, "year_max": y1,
            }

            for i, (name, cls) in enumerate(_SAMPLER_CLASSES.items()):
                def _fit_progress(fraction: float, message: str, _i=i, _n=name) -> None:
                    _prog.progress((_i + fraction) / len(_SAMPLER_CLASSES))
                    _detail.caption(f"{_n}: {message}")

                _detail.caption(f"Fitting {name}…")
                cache_key = f"price_sampler_{name}"
                sampler = cls()
                st.session_state[cache_key] = sampler.fit(_df_prices, _progress=_fit_progress)

            _prog.progress(1.0)
            _status.update(label="All models fitted.", state="complete", expanded=False)
            _ALL_FITTED = True
            st.rerun()
    else:
        st.info("Click **Load ENTSO-E data & fit all models** to begin.")
        st.stop()

# Show fit summary
_fit_info = st.session_state.get("_pe_fit_info", {})
if _fit_info:
    st.success(
        f"Models fitted on **{_fit_info['n_samples']:,}** hourly samples "
        f"({_fit_info['year_min']}–{_fit_info['year_max']})."
    )

# Wire up the button properly
if st.session_state.get("_pe_fit_btn_clicked"):
    st.session_state.pop("_pe_fit_btn_clicked")
    for key in [f"price_sampler_{n}" for n in _SAMPLER_CLASSES]:
        st.session_state.pop(key, None)
    st.session_state.pop("_pe_fit_info", None)
    st.session_state["_trigger_fit"] = True
    st.rerun()

st.divider()

# ── Simulation controls ───────────────────────────────────────────────────────

st.subheader("Simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    season_sel = st.selectbox(
        "Season", ["All"] + [s.capitalize() for s in SEASONS], key="pe_season",
    )
with col2:
    daytype_sel = st.selectbox(
        "Day type", ["All", "Weekday", "Weekend"], key="pe_daytype",
    )
with col3:
    n_days = st.select_slider(
        "Simulated days N", [100, 250, 500, 1000, 2000], value=1000, key="pe_n_days",
    )
with col4:
    sim_seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, key="pe_seed")

# ── Run simulation ────────────────────────────────────────────────────────────

_sim_key = (season_sel, daytype_sel, n_days, int(sim_seed))
if st.session_state.get("_pe_sim_key") != _sim_key:
    st.session_state.pop("_pe_sim_results", None)

if "_pe_sim_results" not in st.session_state:
    with st.spinner(f"Simulating {n_days} days across all models…"):
        rng = np.random.default_rng(int(sim_seed))
        params = SharedParams()

        # Draw per-day contexts
        if season_sel == "All":
            day_seasons = [SEASONS[i] for i in rng.integers(0, 4, n_days)]
        else:
            day_seasons = [season_sel.lower()] * n_days

        if daytype_sel == "Weekday":
            day_dows = rng.integers(0, 5, n_days).tolist()
        elif daytype_sel == "Weekend":
            day_dows = (rng.integers(0, 2, n_days) + 5).tolist()
        else:
            day_dows = rng.integers(0, 7, n_days).tolist()

        day_is_weekend = [dow >= 5 for dow in day_dows]
        HOURS = np.arange(24)

        # Gaussian parametric — fully vectorised
        hour_means = np.array([mean_price(h * 60, params) for h in range(24)])
        param_prices = np.clip(
            rng.normal(hour_means[np.newaxis, :], params.sigma_lambda, size=(n_days, 24)),
            0.0, None,
        )

        # Data-driven models — group days by (is_weekend, season) to batch draws
        data_prices = {name: np.zeros((n_days, 24)) for name in _SAMPLER_CLASSES}
        for name in _SAMPLER_CLASSES:
            sampler = st.session_state[f"price_sampler_{name}"]
            prices  = data_prices[name]
            for (is_wknd, season) in product([False, True], SEASONS):
                idx = [d for d in range(n_days)
                       if day_is_weekend[d] == is_wknd and day_seasons[d] == season]
                if not idx:
                    continue
                dow = 5 if is_wknd else 0
                for h in range(24):
                    draws = [max(0.0, sampler.sample(dow, h, season, rng))
                             for _ in idx]
                    for j, d in enumerate(idx):
                        prices[d, h] = draws[j]

        st.session_state["_pe_sim_results"] = {
            "Gaussian (parametric)": param_prices,
            **{name: data_prices[name] for name in _SAMPLER_CLASSES},
        }
        st.session_state["_pe_sim_key"] = _sim_key

sim_results = st.session_state["_pe_sim_results"]

# ── Plot ──────────────────────────────────────────────────────────────────────

st.subheader("Mean diurnal price profile")
st.caption(
    f"Mean price ± 1 SEM across {n_days} simulated days — one line per pricing model.  "
    "SEM = std/√N (uncertainty of the mean, not spread across days)."
)

hours = np.arange(24)
n = n_days
fig = go.Figure()

for name, prices in sim_results.items():
    col  = _COLORS[name]
    mu   = prices.mean(axis=0)          # (24,)
    sem  = prices.std(axis=0) / np.sqrt(n)

    fig.add_trace(go.Scatter(
        x=np.concatenate([hours, hours[::-1]]),
        y=np.concatenate([mu + sem, (mu - sem)[::-1]]),
        fill="toself", fillcolor=_rgba(col, 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
        legendgroup=name,
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=mu, mode="lines",
        line=dict(color=col, width=2),
        name=name, legendgroup=name,
        hovertemplate=f"<b>{name}</b><br>Hour %{{x:02d}}:00<br>%{{y:.4f}} €/kWh<extra></extra>",
    ))

fig.update_layout(
    height=480,
    hovermode="x unified",
    margin=dict(l=50, r=20, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(title="Hour of day", dtick=2, range=[0, 23]),
    yaxis=dict(title="Price (€/kWh)"),
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ── Per-model std ─────────────────────────────────────────────────────────────

st.subheader("Price spread (std) by hour")
st.caption(
    "Standard deviation of sampled prices across all days — captures how volatile each "
    "model is at each hour, not the uncertainty of the mean."
)

fig2 = go.Figure()
for name, prices in sim_results.items():
    col = _COLORS[name]
    std = prices.std(axis=0)
    fig2.add_trace(go.Scatter(
        x=hours, y=std, mode="lines",
        line=dict(color=col, width=2),
        name=name, legendgroup=name,
        hovertemplate=f"<b>{name}</b><br>Hour %{{x:02d}}:00<br>σ = %{{y:.4f}} €/kWh<extra></extra>",
    ))

fig2.update_layout(
    height=380,
    hovermode="x unified",
    margin=dict(l=50, r=20, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(title="Hour of day", dtick=2, range=[0, 23]),
    yaxis=dict(title="Std (€/kWh)"),
)
st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
