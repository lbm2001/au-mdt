"""Figure factories for sensitivity analysis results.

Each function takes a list of sweep-step result dicts (as returned by
ev_mdt.analysis.sensitivity sweep functions) and returns a Plotly Figure.
They are framework-agnostic: usable from the CLI, scripts, or the Streamlit app.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ev_mdt.models.common.model_utils import (
    expected_trip_minutes, minutes_to_departure,
)
from ev_mdt.models.common.policies import (
    E_CEIL_BASE, _du_e_daily, _e_daily_ref, du_gamma_for_params,
)
from ev_mdt.plots.viz import POLICY_COLORS, POLICY_ORDER, rgba as _rgba


# ── Internal helpers ───────────────────────────────────────────────────────────

def _opt_rates_averaged(pi, actions, params, pbp_fn, T: int, chi: int = 0) -> np.ndarray:
    """u*(t, e) for state `chi`, averaged over price bins by the price distribution."""
    desired = actions[pi[:, chi, :, :]]                           # (T, N_e, K)
    weights = np.array([pbp_fn(t) for t in range(T)])             # (T, K)
    avg = (desired * weights[:, np.newaxis, :]).sum(axis=2)        # (T, N_e)
    return np.clip(avg, 0.0, params.u_max)


def _effective_rates(rates: np.ndarray, chi: int, e_grid, params) -> np.ndarray:
    """Clip to [0, u_max] and zero out charging while driving above the floor.

    `rates` is (T, N_e) (one row per time) or (N_e,) — masking applies on the
    battery axis (the last axis).
    """
    rates = np.clip(rates, 0.0, params.u_max).astype(float, copy=True)
    if chi > 0:
        rates[..., e_grid > params.e_min] = 0.0
    return rates


def _bin_heatmap(rates, e_grid, T: int, time_bin_min: int, battery_bin_kwh: float,
                 e_min: float, e_max: float):
    """Aggregate (T, N_e) charge rates into time × battery bins.

    Returns (z, t_centers, b_centers) with z shaped (n_battery_bins, n_time_bins).
    """
    n_t    = max(1, T // time_bin_min)
    usable = n_t * time_bin_min
    rt     = rates[:usable].reshape(n_t, time_bin_min, rates.shape[1]).mean(axis=1)

    edges = np.arange(e_min, e_max + battery_bin_kwh, battery_bin_kwh)
    n_b   = len(edges) - 1
    z     = np.full((n_t, n_b), np.nan)
    for i in range(n_b):
        lo, hi = edges[i], edges[i + 1]
        mask = (e_grid >= lo) & (e_grid < hi) if i < n_b - 1 else (e_grid >= lo) & (e_grid <= hi)
        if mask.any():
            z[:, i] = rt[:, mask].mean(axis=1)
    t_centers = (np.arange(n_t) + 0.5) * time_bin_min / 60
    b_centers = (edges[:-1] + edges[1:]) / 2
    return z.T, t_centers, b_centers


def _charge_battery_ceiling(pi, actions, e_grid, t: int) -> np.ndarray:
    """(K,) highest battery level at which the parked policy still charges at each price bin."""
    charging = actions[pi[t, 0, :, :]] > 0                        # (N_e, K)
    e_rank   = (np.arange(len(e_grid)) + 1)[:, np.newaxis]        # (N_e, 1)
    top_e    = (charging * e_rank).max(axis=0) - 1                # (K,) highest charging e-index
    return np.where(charging.any(axis=0),
                    e_grid[np.clip(top_e, 0, len(e_grid) - 1)], np.nan)


def _du_e_ceil(params, gamma=None) -> float:
    """Demand-scaled DU ceiling for the given params and gamma (None → per-model γ)."""
    if gamma is None:
        gamma = du_gamma_for_params(params)
    e_daily = _du_e_daily(params)
    ref     = _e_daily_ref()
    ratio   = e_daily / ref if ref > 0 else 1.0
    return min(params.e_max, E_CEIL_BASE * ratio ** gamma)


def _du_charge_battery_ceiling(params, pbp_fn, e_grid, t: int,
                                gamma=None, use_reserve: bool = True) -> np.ndarray:
    """(K,) highest battery at which DU still charges per price bin at t (gamma None → per-model γ)."""
    probs   = np.asarray(pbp_fn(t))                                # (K,)
    tau     = minutes_to_departure(t, params)
    e_trip  = expected_trip_minutes(params) * params.mu * params.v * params.omega
    e_ceil  = _du_e_ceil(params, gamma)

    deliverable = params.u_max * params.eta_c * params.omega * tau
    rho = np.clip((e_ceil - e_grid) / deliverable, 0.0, 1.0) if deliverable > 0 else np.ones(len(e_grid))

    cumprobs = np.cumsum(probs)                                    # (K,)
    extra    = 1.0 / tau if tau > 0 else 0.0

    rho_2d = rho[:, np.newaxis]                                    # (N_e, 1)
    cum_2d = cumprobs[np.newaxis, :]                               # (1, K)

    charging = (cum_2d <= rho_2d + extra) & (e_grid[:, np.newaxis] < params.e_max)  # (N_e, K)
    if use_reserve:
        charging[e_grid < e_trip, :] = True
    charging[e_grid >= params.e_max, :] = False

    e_rank = (np.arange(len(e_grid)) + 1)[:, np.newaxis]
    top_e  = (charging * e_rank).max(axis=0) - 1                  # (K,)
    return np.where(charging.any(axis=0),
                    e_grid[np.clip(top_e, 0, len(e_grid) - 1)], np.nan)


def _blu_charge_battery_ceiling(params, pbp_fn, t: int) -> np.ndarray:
    """(K,) highest battery at which Battery Level Urgency still charges per price bin at time t.

    BLU charges u_max iff F_t(λ) ≤ 1 − e/e_max, i.e. e ≤ e_max·(1 − F_t(λ)).
    """
    cumprobs = np.cumsum(np.asarray(pbp_fn(t)))                   # F_t at each bin
    return np.clip(params.e_max * (1.0 - cumprobs), 0.0, params.e_max)


# ── Public figure factories ────────────────────────────────────────────────────

# ── Policy figure factories (BI / DU / BLU) ─────────────────────────────────────
#
# The three policies the paper figures compare. ``policy`` arguments below accept
# any of these names.
BI_POLICY  = "Backward Induction"
DU_POLICY  = "Departure Urgency"
BLU_POLICY = "Battery Level Urgency"
PAPER_POLICIES = (BI_POLICY, DU_POLICY, BLU_POLICY)

_PAPER_BATTERY_BIN_KWH = 0.5
_T_DAY = 24 * 60


def _heatmap_dims(n: int) -> tuple[int, int]:
    cols = 1 if n <= 3 else 2
    return int(np.ceil(n / cols)), cols


def _border_dims(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    return int(np.ceil(n / 3)), 3


def _policy_heatmap_rates(policy: str, r: dict) -> np.ndarray:
    """Price-averaged charge rate u(t, e) for one policy on a solved-config dict."""
    params, T = r["params"], r["T"]
    if policy == BI_POLICY:
        rates = _opt_rates_averaged(r["pi"], r["actions"], params, r["pbp_fn"], T)
    else:
        probs  = np.array([r["pbp_fn"](t) for t in range(T)])
        cumsum = probs.cumsum(axis=1)
        rates  = _baseline_policy_rates(policy, {}, r["e_grid"], r["lam_grid"], params,
                                        T, probs, cumsum)
    return np.clip(rates, 0.0, params.u_max)


def _policy_border_ceiling(policy: str, r: dict, t: int) -> np.ndarray:
    """(K,) highest battery at which `policy` still charges per price bin at minute t."""
    if policy == BI_POLICY:
        return _charge_battery_ceiling(r["pi"], r["actions"], r["e_grid"], t)
    if policy == DU_POLICY:
        return _du_charge_battery_ceiling(r["params"], r["pbp_fn"], r["e_grid"], t)
    return _blu_charge_battery_ceiling(r["params"], r["pbp_fn"], t)


def fig_policy_heatmap_grid(results: list[dict], policy: str = BI_POLICY) -> go.Figure:
    """Price-averaged heatmaps for one ``policy``, one subplot panel per sweep value.

    ``results`` is a list of solved-config dicts (one per swept value). Layout:
    ≤3 panels → one per row; >3 → two per row.
    """
    n = len(results)
    rows, cols = _heatmap_dims(n)
    titles = None if n == 1 else [r["label"] for r in results]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=0.10, vertical_spacing=0.06 if rows > 1 else 0.0,
    )
    for i, r in enumerate(results):
        row, col = i // cols + 1, i % cols + 1
        params, T = r["params"], r["T"]
        rates = _policy_heatmap_rates(policy, r)
        time_bin = max(1, T // _T_DAY)           # ~1440 columns regardless of horizon
        z, t_c, b_c = _bin_heatmap(rates, r["e_grid"], T, time_bin, _PAPER_BATTERY_BIN_KWH,
                                   params.e_min, params.e_max)
        fig.add_trace(go.Heatmap(
            x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max, colorscale="RdYlBu_r",
            showscale=(i == 0),
            colorbar=dict(title="u (kW)", x=1.02, len=0.9) if i == 0 else None,
            hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Hour (h)" if row == rows else "", range=[0, T // 60],
                         showticklabels=(row == rows), title_standoff=12, row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         title_standoff=16, row=row, col=col)
    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=300 * rows + 80, width=480 * cols + 120,
                      margin=dict(l=70, r=90, t=50, b=55))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def fig_policy_charge_border_grid(results: list[dict], policy: str = BI_POLICY) -> go.Figure:
    """Charge/no-charge border (price × battery) for one ``policy``, one curve per
    hour of the day, one subplot panel per sweep value.

    Layout: ≤3 panels → one row; 4 → 2×2; 5-6 → three per row.
    """
    from plotly.colors import sample_colorscale
    n = len(results)
    rows, cols = _border_dims(n)
    titles = None if n == 1 else [r["label"] for r in results]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=0.10, vertical_spacing=0.16 if rows > 1 else 0.0,
    )
    for i, r in enumerate(results):
        row, col = i // cols + 1, i % cols + 1
        n_h = min(24, r["T"] // 60)
        for h in range(n_h):
            ceil_e = _policy_border_ceiling(policy, r, h * 60)
            color  = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
            fig.add_trace(go.Scatter(
                x=r["lam_grid"], y=ceil_e, mode="lines", line=dict(color=color, width=1.3),
                showlegend=False,
                hovertemplate=f"Hour {h:02d}:00<br>Price %{{x:.3f}} €/kWh<br>"
                              "charge if battery ≤ %{y:.1f} kWh<extra></extra>",
            ), row=row, col=col)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == rows else "",
                         showticklabels=(row == rows), title_standoff=12, row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         range=[0, r["params"].e_max], title_standoff=12, row=row, col=col)
    # Viridis hour colorbar (carried by a dummy trace in the top-right panel).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.02)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=cols)
    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=300 * rows + 80, width=480 * cols + 120,
                      margin=dict(l=70, r=90, t=50, b=55))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def fig_all_policy_heatmaps(result: dict, policies=PAPER_POLICIES, *,
                            time_bin_min: int = 1,
                            battery_bin_kwh: float = _PAPER_BATTERY_BIN_KWH) -> go.Figure:
    """All-policies-in-one heatmap figure (one panel per policy) for a single config."""
    params, T = result["params"], result["T"]
    T_hours = T // 60
    n = len(policies)
    fig = make_subplots(rows=n, cols=1, subplot_titles=list(policies), vertical_spacing=0.08)
    for idx, name in enumerate(policies):
        row = idx + 1
        rates = _policy_heatmap_rates(name, result)
        z, t_c, b_c = _bin_heatmap(rates, result["e_grid"], T, time_bin_min, battery_bin_kwh,
                                   params.e_min, params.e_max)
        fig.add_trace(go.Heatmap(
            x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max, colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.02, len=0.9) if idx == 0 else None,
            hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u: %{z:.2f} kW<extra></extra>",
        ), row=row, col=1)
        fig.update_xaxes(title_text="Hour (h)" if row == n else "", range=[0, T_hours],
                         title_standoff=12, showticklabels=(row == n), row=row, col=1)
        fig.update_yaxes(title_text="Battery (kWh)", title_standoff=16, row=row, col=1)
    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=300 * n + 80, margin=dict(l=70, r=90, t=50, b=55))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def fig_all_policy_charge_borders(result: dict, policies=PAPER_POLICIES) -> go.Figure:
    """All-policies-in-one charge-border figure (one panel per policy) for a single config."""
    from plotly.colors import sample_colorscale
    params, T = result["params"], result["T"]
    T_hours = T // 60
    n = len(policies)
    n_h = min(24, T_hours)
    fig = make_subplots(rows=n, cols=1, subplot_titles=list(policies), vertical_spacing=0.08)
    for idx, name in enumerate(policies):
        row = idx + 1
        for h in range(n_h):
            color  = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
            ceil_e = _policy_border_ceiling(name, result, h * 60)
            fig.add_trace(go.Scatter(
                x=result["lam_grid"], y=ceil_e, mode="lines",
                line=dict(color=color, width=1.3), showlegend=False,
                hovertemplate=f"{name}<br>Hour {h:02d}:00<br>"
                              "Price %{x:.3f} €/kWh<br>charge if battery ≤ %{y:.1f} kWh<extra></extra>",
            ), row=row, col=1)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == n else "",
                         title_standoff=12, showticklabels=(row == n), row=row, col=1)
        fig.update_yaxes(title_text="Battery (kWh)", title_standoff=12,
                         range=[0, params.e_max], row=row, col=1)
    # Viridis hour colorbar.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.02)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=300 * n + 80, margin=dict(l=70, r=90, t=50, b=55))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def _cost_floor(totals: np.ndarray) -> float:
    """Common log-axis floor below all positive totals (anchors the proportional split)."""
    pos = totals[totals > 0]
    return float(pos.min() / 3.0) if pos.size else 1e-3


def _log_err_arms(totals, errs):
    """Log-symmetric error arms for ±err around totals (delta method).

    A symmetric linear ±SEM looks lopsided on a log axis (the lower arm plunges and
    can cross zero). Converting to a half-width in log10 units, d = (err/total)/ln10,
    gives caps at total·10^±d — balanced on the log axis and always positive. Returns
    (array_up, array_minus_down) suitable for a plotly error_y with symmetric=False.
    """
    totals = np.asarray(totals, float)
    errs   = np.asarray(errs,   float)
    safe   = np.where(totals > 0, totals, 1.0)
    d  = (errs / safe) / np.log(10.0)
    up = totals * (10.0 ** d - 1.0)
    dn = totals * (1.0 - 10.0 ** (-d))
    return up, dn


def _add_stacked_cost_bars(fig, xs, totals, charge, penalty, color, name, errs, *,
                           log_y: bool, y0: float, offsetgroup=None,
                           showlegend: bool = True, row=None, col=None) -> None:
    """Stacked charging+penalty bar per x: total height = total cost, split by cost %.

    On a log axis the segment boundary is placed so the *visual* split equals the
    charging/penalty proportion (e.g. 25/75 → bottom ¼ charging, top ¾ penalty);
    charging is the full solid policy colour, penalty is a 50%-saturated fill of the
    same colour hatched (gestreift) with diagonal stripes in the full colour.
    """
    totals  = np.asarray(totals,  float)
    charge  = np.asarray(charge,  float)
    penalty = np.asarray(penalty, float)
    safe    = np.where(totals > 0, totals, 1.0)
    frac_c  = np.clip(charge / safe, 0.0, 1.0)                       # charging share
    pen_fill = _rgba(color, 0.50)                                   # penalty fill (50%)

    if log_y:
        tot = np.where(totals > y0, totals, y0)
        B   = y0 * (tot / y0) ** frac_c                             # split boundary value
        charge_base, charge_len = np.full_like(tot, y0), B - y0
        pen_base,    pen_len    = B, tot - B
    else:
        charge_base, charge_len = np.zeros_like(totals), charge
        pen_base,    pen_len    = charge, penalty

    pct_c = (frac_c * 100)
    pct_p = (100 - pct_c)
    pos = dict(row=row, col=col) if row is not None else {}
    # Charging share: full solid policy colour.
    fig.add_trace(go.Bar(
        x=xs, base=charge_base, y=charge_len, name=name, legendgroup=name,
        marker_color=color, showlegend=showlegend, offsetgroup=offsetgroup,
        customdata=np.stack([charge, pct_c], axis=-1),
        hovertemplate="%{x}<br>charging %{customdata[0]:.3f} € (%{customdata[1]:.0f}%)"
                      f"<extra>{name}</extra>",
    ), **pos)
    # Penalty share: 50%-saturated fill, diagonal hatch (gestreift) in the full colour.
    fig.add_trace(go.Bar(
        x=xs, base=pen_base, y=pen_len, name=name, legendgroup=name,
        marker=dict(color=pen_fill, pattern=dict(shape="/", fgcolor=color,
                                                 size=8, solidity=0.42)),
        showlegend=False, offsetgroup=offsetgroup,
        customdata=np.stack([totals, penalty, pct_p], axis=-1),
        hovertemplate="%{x}<br>total %{customdata[0]:.3f} €<br>"
                      "penalty %{customdata[1]:.3f} € (%{customdata[2]:.0f}%)"
                      f"<extra>{name}</extra>",
    ), **pos)
    # Error bars (±SEM/Std on total): carried by a transparent full-height bar so the
    # whisker is centred at the *total* (Plotly centres error_y at the bar's y-length,
    # which for the based segments above would land mid-bar). Same offsetgroup keeps it
    # aligned within the policy's grouped slot. On a log axis use log-symmetric arms so
    # the whisker is balanced and never plunges below zero. ``errs=None`` (exact mode)
    # draws no whisker at all.
    if errs is None:
        return
    if log_y:
        up, dn = _log_err_arms(totals, errs)
        err_kw = dict(type="data", symmetric=False, array=up, arrayminus=dn)
    else:
        err_kw = dict(type="data", array=np.asarray(errs, float))
    fig.add_trace(go.Bar(
        x=xs, y=totals, offsetgroup=offsetgroup, legendgroup=name,
        marker=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        error_y=dict(visible=True, thickness=1.2, width=4, color="#333333", **err_kw),
    ), **pos)


def fig_cost_distribution(results: list[dict], *, log_y: bool = True,
                          x_label: str = "Swept value") -> go.Figure:
    """Exact expected-cost bar per policy, grouped by swept value (charging/penalty split).

    Each result must carry an ``exact_breakdown`` ({policy: {total, charging, penalty}});
    it is computed on demand if missing.
    """
    from ev_mdt.analysis.sensitivity import compute_all_exact_costs_breakdown
    labels = [r["label"] for r in results]
    exact = [r.get("exact_breakdown") or compute_all_exact_costs_breakdown(r) for r in results]

    series = []   # (policy, totals, charge, penalty)
    for policy in POLICY_ORDER:
        if not all(policy in e for e in exact):
            continue
        totals  = np.array([e[policy]["total"]    for e in exact])
        charge  = np.array([e[policy]["charging"] for e in exact])
        penalty = np.array([e[policy]["penalty"]  for e in exact])
        series.append((policy, totals, charge, penalty))

    all_totals = np.concatenate([s[1] for s in series]) if series else np.array([1.0])
    y0 = _cost_floor(all_totals)

    fig = go.Figure()
    for policy, totals, charge, penalty in series:
        _add_stacked_cost_bars(fig, labels, totals, charge, penalty,
                               POLICY_COLORS[policy], policy, None,
                               log_y=log_y, y0=y0, offsetgroup=policy)

    yaxis = dict(title="Expected cost (€)" + ("  [log]" if log_y else ""),
                 type="log" if log_y else "linear")
    if log_y:
        yaxis["range"] = [np.log10(y0), np.log10(all_totals.max() * 1.1)]
        yaxis["dtick"] = 1          # decade ticks only: 0.1, 1, 10, …
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="group", yaxis=yaxis, height=440,
        margin=dict(l=80, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11), itemsizing="constant"),
    )
    return fig


# Metric columns shared with the Policy-Rollout table (excluding the
# Swept value / Policy identifier columns). Used for display formatting.
SUMMARY_METRIC_FORMATS = {
    "Mean cost (€)":             "{:.3f}",
    "Mean charging (€)":         "{:.3f}",
    "Mean penalty (€)":          "{:.3f}",
    "Penalty %":                 "{:.1f}%",
    "Optimality gap %":          "{:.2f}%",
    "Mean penalty min":          "{:.1f}",
    "Mean energy charged (kWh)": "{:.2f}",
}


def _penalty_pct(charge: float, penalty: float) -> float:
    """Penalty's share of total cost (charging + penalty), in percent."""
    tot = charge + penalty
    return float(penalty / tot * 100) if tot > 0 else 0.0


def _exact_summary(bd: dict) -> dict:
    """Per-policy summary from an exact cost breakdown ``{total,charging,penalty,energy_kwh}``.

    The distributional metrics (SEM / Std / penalty-minutes / % scenarios) have no
    exact analogue and are omitted; the cost columns are the noise-free expectations.
    """
    return {
        "Mean cost (€)":             bd["total"],
        "Mean charging (€)":         bd["charging"],
        "Mean penalty (€)":          bd["penalty"],
        "Penalty %":                 _penalty_pct(bd["charging"], bd["penalty"]),
        "Mean penalty min":          bd["penalty_min"],
        "Mean energy charged (kWh)": bd["energy_kwh"],
    }


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """One row per (swept_value, policy) with the exact expected-cost metrics.

    Includes the per-policy optimality gap vs Backward Induction (exact).
    """
    from ev_mdt.analysis.sensitivity import compute_all_exact_costs_breakdown
    rows = []
    for r in results:
        bd = r.get("exact_breakdown") or compute_all_exact_costs_breakdown(r)
        bi_cost = bd.get("Backward Induction", {}).get("total", None)
        for policy in POLICY_ORDER:
            if policy not in bd:
                continue
            row = {"Swept value": r["label"], "Policy": policy, **_exact_summary(bd[policy])}
            if bi_cost is not None and bi_cost > 0:
                row["Optimality gap %"] = (bd[policy]["total"] - bi_cost) / bi_cost * 100
            rows.append(row)
    return pd.DataFrame(rows)


def fig_baseline_cost(result: dict, *, log_y: bool = True) -> go.Figure:
    """Per-policy bar: exact expected total cost split into charging/penalty shares.

    ``result`` is a solved-config dict; its ``exact_breakdown`` is used (computed on
    demand if missing). One bar per policy, ordered by POLICY_ORDER.
    """
    from ev_mdt.analysis.sensitivity import compute_all_exact_costs_breakdown
    bd = result.get("exact_breakdown") or compute_all_exact_costs_breakdown(result)
    names = [p for p in POLICY_ORDER if p in bd]
    totals  = [bd[n]["total"]    for n in names]
    charge  = [bd[n]["charging"] for n in names]
    penalty = [bd[n]["penalty"]  for n in names]

    totals_arr = np.array(totals) if totals else np.array([1.0])
    y0 = _cost_floor(totals_arr)

    fig = go.Figure()
    for i, name in enumerate(names):
        # Shared offsetgroup → one slot per policy category so bars stay full-width
        # (a unique offsetgroup per policy would split each category into len(names) slots).
        _add_stacked_cost_bars(fig, [name], [totals[i]], [charge[i]], [penalty[i]],
                               POLICY_COLORS[name], name, None,
                               log_y=log_y, y0=y0, offsetgroup="cost")
    yaxis = dict(title="Expected cost (€)" + ("  [log]" if log_y else ""),
                 type="log" if log_y else "linear")
    if log_y:
        yaxis["range"] = [np.log10(y0), np.log10(totals_arr.max() * 1.1)]
        yaxis["dtick"] = 1          # decade ticks only: 0.1, 1, 10, …
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=yaxis,
        height=460, margin=dict(l=40, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11), itemsizing="constant"),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=POLICY_ORDER,
                     showticklabels=False, title_text="")
    return fig


def _hourly_mean(arr2d: np.ndarray, T_hours: int) -> np.ndarray:
    """Average a (N, T_minutes) array down to (N, T_hours)."""
    n, t = arr2d.shape
    return arr2d.reshape(n, T_hours, t // T_hours).mean(axis=2)


def _traj_band(fig, x, mean, half, color, name, row, showlegend=False) -> None:
    """Add a mean line + ±half SEM ribbon to subplot `row` (shared by trajectory figs)."""
    fill = _rgba(color, 0.12)
    fig.add_trace(go.Scatter(x=x, y=mean + half, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", legendgroup=name), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=mean - half, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=fill, showlegend=False,
                             hoverinfo="skip", legendgroup=name), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", line=dict(color=color, width=1.6),
                             name=name, legendgroup=name, showlegend=showlegend), row=row, col=1)


def fig_rollout_trajectories(scenarios: list, T: int, mobility_bands: list) -> go.Figure:
    """Scenario-averaged price + mobility trajectories for the Policy-Rollout page.

    mobility_bands : list of (label, color, [chi_traj arrays], showlegend) — one
    band per mobility series (one for Baseline; two for a NegBin model and its
    sibling variant). Price is always a single light-grey band.
    """
    T_hours = T // 60
    h_axis = np.arange(T_hours)
    m_axis = np.arange(T) / 60

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Mean sampled price", "Mean sampled mobility (0 parked, 1 driving)"))

    P = _hourly_mean(np.array([sc["lam_path"] for sc in scenarios]), T_hours)
    n_scen = max(P.shape[0], 1)
    sem = lambda a: a.std(axis=0) / np.sqrt(n_scen)
    _traj_band(fig, h_axis, P.mean(0), sem(P), "lightgray", "λ̄<sub>t</sub>", row=1)

    show_legend = False
    for label, color, chi_list, sl in mobility_bands:
        Mob = np.array([(chi > 0).astype(float) for chi in chi_list])
        _traj_band(fig, m_axis, Mob.mean(0), sem(Mob), color, label, row=2, showlegend=sl)
        show_legend = show_legend or sl

    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=560, hovermode="x unified",
                      margin=dict(l=50, r=30, t=50, b=40), showlegend=show_legend,
                      legend=dict(x=1.01, y=0.2, xanchor="left"))
    fig.update_xaxes(range=[0, T_hours], dtick=max(1, T_hours // 8))
    fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Fraction driving", tickvals=[0, 0.5, 1], row=2, col=1)
    return fig


def fig_policy_heatmap(pi, actions, e_grid, params, T: int, chi: int = 0, *,
                       lam_bin: int | None = None, pbp_fn=None,
                       time_bin_min: int = 10, battery_bin_kwh: float = 1.0) -> go.Figure:
    """Single optimal-policy heatmap over (hour × battery) for state `chi`.

    Either a single price bin (`lam_bin`) or price-averaged (`pbp_fn`). Used by
    the Policy-Explorer page.
    """
    if lam_bin is not None:
        rates = actions[pi[:, chi, :, lam_bin]]                    # (T, N_e)
    else:
        rates = _opt_rates_averaged(pi, actions, params, pbp_fn, T, chi)
    rates = _effective_rates(rates, chi, e_grid, params)
    z, t_centers, b_centers = _bin_heatmap(
        rates, e_grid, T, time_bin_min, battery_bin_kwh, params.e_min, params.e_max)
    T_hours = T // 60
    fig = go.Figure(data=go.Heatmap(
        x=t_centers, y=b_centers, z=z, zmin=0, zmax=params.u_max,
        colorscale="RdYlBu_r", colorbar=dict(title="u (kW)"),
        hovertemplate="Hour: %{x:.2f}<br>Battery: %{y:.2f} kWh<br>Charge: %{z:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Hour (h)", yaxis_title="Battery (kWh)", height=430,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    fig.update_xaxes(range=[0, T_hours], dtick=T_hours // 8)
    return fig


def fig_policy_price_map(pi, actions, e_grid, lam_grid, params, chi: int = 0, t: int = 0) -> go.Figure:
    """u*(battery × price bin) at a fixed minute `t` — keeps the price axis."""
    rates = actions[pi[t, chi, :, :]].astype(float, copy=True)     # (N_e, K)
    if chi > 0:
        rates[e_grid > params.e_min, :] = 0.0
    fig = go.Figure(data=go.Heatmap(
        x=lam_grid, y=e_grid, z=rates, zmin=0, zmax=params.u_max,
        colorscale="RdYlBu_r", colorbar=dict(title="u (kW)"),
        hovertemplate="Price: %{x:.3f} €/kWh<br>Battery: %{y:.2f} kWh<br>Charge: %{z:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Price (€/kWh)", yaxis_title="Battery (kWh)", height=430,
        margin=dict(l=30, r=30, t=55, b=35),
    )
    return fig


def _price_charge_prob(
    cumsum: np.ndarray, probs: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """P(F_t(λ) ≤ rho[t,e]) for each (t,e) pair.

    cumsum : (T, K) — cumulative price-bin probabilities
    probs  : (T, K) — price-bin probabilities
    rho    : (T, N_e)
    returns: (T, N_e)
    """
    mask = cumsum[:, :, np.newaxis] <= rho[:, np.newaxis, :]   # (T, K, N_e)
    return (probs[:, :, np.newaxis] * mask).sum(axis=1)         # (T, N_e)


def _baseline_policy_rates(
    name: str, kwargs: dict,
    e_grid: np.ndarray, lam_grid: np.ndarray, params,
    T: int, probs: np.ndarray, cumsum: np.ndarray,
) -> np.ndarray:
    """(T, N_e) price-averaged parked-state (chi=0) charge rates for one baseline policy.

    Vectorised over the grid so the whole batch is fast enough to compute
    without a Python loop over the T × N_e × K state space.
    """
    N_e = len(e_grid)

    if name == "Always-Maximum":
        return np.full((T, N_e), params.u_max)

    if name == "Always-Minimum":
        return np.full((T, N_e), params.u_min)

    if name == "Night Charging":
        rate_t = np.where(np.arange(T) % 1440 < 360, params.u_max, 0.0)
        return np.broadcast_to(rate_t[:, np.newaxis], (T, N_e)).copy()

    if name == "Minimum Battery Level":
        soc = kwargs["soc_threshold"]
        rate_e = np.where(e_grid < soc, params.u_max, 0.0)
        return np.broadcast_to(rate_e[np.newaxis, :], (T, N_e)).copy()

    if name == "Price-Oriented":
        low, high = kwargs["low_threshold"], kwargs["high_threshold"]
        mask_low = lam_grid[np.newaxis, :] <= low                            # (1, K)
        mask_mid = (lam_grid[np.newaxis, :] > low) & (lam_grid[np.newaxis, :] <= high)
        rate_t = (params.u_max       * (probs * mask_low).sum(axis=1)
                  + params.u_max / 2 * (probs * mask_mid).sum(axis=1))       # (T,)
        return np.broadcast_to(rate_t[:, np.newaxis], (T, N_e)).copy()

    if name == "Battery Level Urgency":
        # Charge u_max when F_t(λ) ≤ 1 − e/e_max (price is cheap relative to urgency).
        thresh = np.clip(1.0 - e_grid / params.e_max, 0.0, 1.0)             # (N_e,)
        mask = cumsum[:, :, np.newaxis] <= thresh[np.newaxis, np.newaxis, :] # (T, K, N_e)
        rate = params.u_max * (probs[:, :, np.newaxis] * mask).sum(axis=1)  # (T, N_e)
        return np.where(e_grid[np.newaxis, :] >= params.e_max, 0.0, rate)

    if name == "Departure Urgency":
        gamma       = kwargs.get("gamma", None)        # None → per-model γ
        use_reserve = kwargs.get("use_reserve", True)
        e_trip      = expected_trip_minutes(params) * params.mu * params.v * params.omega
        e_ceil      = _du_e_ceil(params, gamma)

        slots = np.array([minutes_to_departure(t, params) for t in range(T)])  # (T,)

        deliverable = params.u_max * params.eta_c * params.omega * slots        # (T,)
        e_diff      = np.maximum(0.0, e_ceil - e_grid[np.newaxis, :])           # (T, N_e)
        safe_del    = np.where(deliverable > 0, deliverable, 1.0)
        rho = np.where(
            deliverable[:, np.newaxis] > 0,
            e_diff / safe_del[:, np.newaxis],
            np.inf,
        )                                                                        # (T, N_e)
        band = np.where(slots > 0, 1.0 / slots, 0.0)                           # (T,)

        p1  = _price_charge_prob(cumsum, probs, rho)
        p12 = _price_charge_prob(cumsum, probs, rho + band[:, np.newaxis])
        rate = params.u_max * p1 + (params.u_max / 2) * (p12 - p1)

        if use_reserve:
            rate = np.where(e_grid[np.newaxis, :] < e_trip, params.u_max, rate)
        rate = np.where(e_grid[np.newaxis, :] >= params.e_max, 0.0, rate)
        return rate

    raise ValueError(f"No vectorised implementation for policy '{name}'")


def fig_baseline_policy_heatmaps(
    params, e_grid: np.ndarray, lam_grid: np.ndarray, T: int, pbp_fn, *,
    pi=None, actions=None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    soc_threshold: float | None = None,
    du_gamma=None,
    du_use_reserve: bool = True,
    time_bin_min: int = 10,
    battery_bin_kwh: float = 1.0,
) -> go.Figure:
    """Price-averaged policy heatmaps for all baseline (non-BI) policies.

    Accepts the same grid/param args as the other sensitivity figure factories
    so the app can pass session-state data directly. ``pi`` and ``actions`` are
    accepted (and forwarded to ``policy_registry``) but only used to generate
    the registry — BI is dropped from the grid. ``du_gamma=None`` → per-model γ.
    """
    from ev_mdt.models.common.policies import policy_registry

    if du_gamma is None:
        du_gamma = du_gamma_for_params(params)
    registry = policy_registry(
        params, pbp_fn,
        pi=pi, actions=actions, e_grid=e_grid,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        soc_threshold=soc_threshold,
        du_gamma=du_gamma,
        du_use_reserve=du_use_reserve,
    )
    policies = [(name, fn, kw) for name, fn, kw in registry if name != "Backward Induction"]
    du = [(n, f, k) for n, f, k in policies if n == "Departure Urgency"]
    rest = [(n, f, k) for n, f, k in policies if n != "Departure Urgency"]
    policies = du + rest
    bi_entry = next(((name, fn, kw) for name, fn, kw in registry if name == "Backward Induction"), None)
    if bi_entry is not None and pi is not None and actions is not None:
        policies.insert(0, bi_entry)

    probs  = np.array([pbp_fn(t) for t in range(T)])   # (T, K)
    cumsum = probs.cumsum(axis=1)                        # (T, K)

    ncols  = 2
    nrows  = int(np.ceil(len(policies) / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[name for name, _, _ in policies],
        horizontal_spacing=0.06,
        vertical_spacing=0.05,
    )
    T_hours = T // 60
    for idx, (name, _fn, kw) in enumerate(policies):
        row = idx // ncols + 1
        col = idx %  ncols + 1
        if name == "Backward Induction":
            rates = _opt_rates_averaged(pi, actions, params, pbp_fn, T)
        else:
            rates = _baseline_policy_rates(name, kw, e_grid, lam_grid, params, T, probs, cumsum)
        rates = np.clip(rates, 0.0, params.u_max)
        z, t_c, b_c = _bin_heatmap(
            rates, e_grid, T, time_bin_min, battery_bin_kwh, params.e_min, params.e_max,
        )
        fig.add_trace(go.Heatmap(
            x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.02) if idx == 0 else None,
            hovertemplate="Hour: %{x:.1f} h<br>Battery: %{y:.1f} kWh<br>u: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(
            title_text="Hour (h)" if row == nrows else "",
            range=[0, T_hours], title_standoff=10,
            showticklabels=(row == nrows),
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="Battery (kWh)" if col == 1 else "",
            title_standoff=14, row=row, col=col,
        )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=260 * nrows + 80,
        margin=dict(l=70, r=60, t=55, b=50),
    )
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def figure_to_png(fig: go.Figure, width: int = 1400, scale: int = 3,
                  top: int | None = None) -> bytes:
    """Tight high-res PNG (requires kaleido): margins shrink (via automargin) to
    exactly fit the axis labels and colorbar, so the plot content fills the frame
    edge-to-edge with no extra whitespace on the sides.

    ``top`` overrides the top margin — needed for the cost figures, whose
    horizontal legend sits above the plot (automargin does not reserve for it).
    """
    import copy
    fig = copy.deepcopy(fig)
    fig.update_layout(template="plotly_white", plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(size=16))
    # Subplot titles are annotations — bump them to match the larger font.
    has_titles = bool(fig.layout.annotations)
    for ann in fig.layout.annotations:
        if ann.font and ann.font.size:
            ann.font.size = max(ann.font.size, 18)
        else:
            ann.update(font=dict(size=18))
    # automargin reserves room only for what's actually drawn (tick labels, axis
    # titles, colorbar); the small base margins are the *minimum* padding.
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    t = top if top is not None else (34 if has_titles else 10)
    fig.update_layout(margin=dict(l=8, r=8, t=t, b=8))
    h = int(fig.layout.height or 500)
    return fig.to_image(format="png", width=width, height=h, scale=scale)
