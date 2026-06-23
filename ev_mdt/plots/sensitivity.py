"""Figure factories for sensitivity analysis results.

Each function takes a list of sweep-step result dicts (as returned by
ev_mdt.analysis.sensitivity sweep functions) and returns a Plotly Figure.
They are framework-agnostic: usable from the CLI, scripts, or the Streamlit app.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ev_mdt.plots.viz import POLICY_COLORS, POLICY_ORDER


# ── Internal helpers ───────────────────────────────────────────────────────────

def _costs(rollout_results: dict, policy: str) -> np.ndarray:
    return np.array([m["Total cost (€)"] for m in rollout_results[policy]])


def _mean_u(rollout_results: dict, policy: str) -> float:
    return float(np.mean([m["Mean charge rate while parked (kW)"] for m in rollout_results[policy]]))


def _opt_rates_averaged(pi, actions, params, pbp_fn, T: int) -> np.ndarray:
    """u*(t, e) averaged over price bins for parked state (chi=0)."""
    desired = actions[pi[:, 0, :, :]]                              # (T, N_e, K)
    weights = np.array([pbp_fn(t) for t in range(T)])             # (T, K)
    avg = (desired * weights[:, np.newaxis, :]).sum(axis=2)        # (T, N_e)
    return np.clip(avg, 0.0, params.u_max)


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


def _grid_dims(n: int) -> tuple[int, int]:
    cols = 2 if n == 4 else min(n, 3)
    rows = int(np.ceil(n / cols))
    return rows, cols


def _charge_battery_ceiling(pi, actions, e_grid, t: int) -> np.ndarray:
    """(K,) highest battery level at which the parked policy still charges at each price bin."""
    charging = actions[pi[t, 0, :, :]] > 0                        # (N_e, K)
    e_rank   = (np.arange(len(e_grid)) + 1)[:, np.newaxis]        # (N_e, 1)
    top_e    = (charging * e_rank).max(axis=0) - 1                # (K,) highest charging e-index
    return np.where(charging.any(axis=0),
                    e_grid[np.clip(top_e, 0, len(e_grid) - 1)], np.nan)


def _rgba(hex_color: str, alpha: float) -> str:
    """rgba() string for a hex (#RRGGBB) or named colour, at the given opacity."""
    named = {"orange": "255,165,0", "lightgray": "211,211,211"}
    if hex_color.startswith("#"):
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
    return f"rgba({named.get(hex_color, '128,128,128')},{alpha})"


# ── Public figure factories ────────────────────────────────────────────────────

def fig_heatmap_grid(results: list[dict], ncols: int = 1, time_bin_min: int = 1,
                     battery_bin_kwh: float = 0.5, show_titles: bool = True) -> go.Figure:
    """Optimal-policy heatmaps (price-averaged). ncols=1 → one per row; ncols>1 → grid."""
    n    = len(results)
    rows = int(np.ceil(n / ncols))
    fig = make_subplots(
        rows=rows, cols=ncols,
        subplot_titles=[r["label"] for r in results] if show_titles else None,
        horizontal_spacing=0.08 if ncols > 1 else 0.0,
        vertical_spacing=0.14 if rows > 1 else 0.0,
    )
    for idx, r in enumerate(results):
        row = idx // ncols + 1
        col = idx %  ncols + 1
        T   = r["T"]
        rates = _opt_rates_averaged(r["pi"], r["actions"], r["params"], r["pbp_fn"], T)
        z, t_centers, b_centers = _bin_heatmap(
            rates, r["e_grid"], T, time_bin_min, battery_bin_kwh,
            r["params"].e_min, r["params"].e_max,
        )
        fig.add_trace(go.Heatmap(
            x=t_centers, y=b_centers, z=z,
            zmin=0, zmax=r["params"].u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.01) if idx == 0 else None,
            hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u*: %{z:.2f} kW<extra></extra>",
        ), row=row, col=col)
        fig.update_xaxes(title_text="Hour (h)" if row == rows else "", range=[0, T // 60],
                         title_standoff=12, showticklabels=(row == rows), row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         title_standoff=16, row=row, col=col)
    fig.update_layout(height=280 * rows + 70, margin=dict(l=70, r=60, t=55, b=50))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


def fig_charge_boundary_grid(results: list[dict]) -> go.Figure:
    """Charge/no-charge border in the (price, battery) plane, one curve per hour of the day."""
    from plotly.colors import sample_colorscale
    n = len(results)
    rows, cols = _grid_dims(n)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[r["label"] for r in results],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for idx, r in enumerate(results):
        row = idx // cols + 1
        col = idx % cols  + 1
        n_h = min(24, r["T"] // 60)
        for h in range(n_h):
            ceil  = _charge_battery_ceiling(r["pi"], r["actions"], r["e_grid"], h * 60)
            color = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
            fig.add_trace(go.Scatter(
                x=r["lam_grid"], y=ceil, mode="lines", line=dict(color=color, width=1.3),
                showlegend=False,
                hovertemplate=f"Hour {h:02d}:00<br>Price %{{x:.3f}} €/kWh<br>charge if battery ≤ %{{y:.1f}} kWh<extra></extra>",
            ), row=row, col=col)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == rows else "",
                         title_standoff=12, row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         title_standoff=12,
                         range=[0, r["params"].e_max], row=row, col=col)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.01)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.update_layout(height=350 * rows + 60, margin=dict(l=60, r=60, t=40, b=60))
    return fig


def fig_cost_distribution(results: list[dict], log_y: bool = True,
                           x_label: str = "Swept value", error: str = "sem") -> go.Figure:
    """Mean total cost (incl. penalty) over sampled rollouts, grouped bars per swept value."""
    labels = [r["label"] for r in results]
    fig = go.Figure()
    for policy in POLICY_ORDER:
        means, errs = [], []
        for r in results:
            costs = _costs(r["rollouts"], policy)
            n = len(costs)
            sd = float(np.std(costs, ddof=1)) if n > 1 else 0.0
            means.append(float(np.mean(costs)))
            errs.append(sd / np.sqrt(n) if (error == "sem" and n > 0) else sd)
        minus = [min(e, m) for m, e in zip(means, errs)]
        fig.add_trace(go.Bar(
            x=labels, y=means, name=policy, marker_color=POLICY_COLORS[policy],
            error_y=dict(type="data", symmetric=False, array=errs, arrayminus=minus,
                         visible=True, thickness=1.2, width=4),
            hovertemplate="%{x}<br>mean %{y:.3f} € (± %{error_y.array:.3f})<extra>" + policy + "</extra>",
        ))
    yaxis = dict(title="Mean total cost incl. penalty (€)" + ("  [log]" if log_y else ""),
                 type="log" if log_y else "linear")
    if log_y:
        yaxis["dtick"] = 1
    fig.update_layout(
        barmode="group", yaxis=yaxis, height=440,
        margin=dict(l=80, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """One row per (swept_value, policy) with key metrics."""
    rows = []
    for r in results:
        for policy in POLICY_ORDER:
            costs = _costs(r["rollouts"], policy)
            rows.append({
                "Swept value":         r["label"],
                "Policy":              policy,
                "Mean cost (€)":       round(float(np.mean(costs)), 4),
                "Std cost (€)":        round(float(np.std(costs, ddof=1)), 4),
                "Mean u parked (kW)":  round(_mean_u(r["rollouts"], policy), 3),
            })
    return pd.DataFrame(rows)


def fig_baseline_cost(full: dict) -> go.Figure:
    """Per-policy mean total cost (incl. penalty), log axis, ordered by POLICY_ORDER, ±SEM."""
    names = [p for p in POLICY_ORDER if p in full]
    means, errs = [], []
    for name in names:
        costs = np.array([r["cost_traj"].sum() for r in full[name]])
        m  = len(costs)
        sd = float(costs.std(ddof=1)) if m > 1 else 0.0
        means.append(float(costs.mean()))
        errs.append(sd / np.sqrt(m) if m > 0 else 0.0)
    minus = [min(e, mu) for mu, e in zip(means, errs)]
    fig = go.Figure(go.Bar(
        x=names, y=means, marker_color=[POLICY_COLORS[n] for n in names],
        error_y=dict(type="data", symmetric=False, array=errs, arrayminus=minus,
                     visible=True, thickness=1.2, width=4)))
    fig.update_layout(
        yaxis=dict(title="Mean total cost incl. penalty (€)  [log]", type="log", dtick=1),
        xaxis_title="Policy", height=460, margin=dict(l=40, r=20, t=20, b=110),
        showlegend=False,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=POLICY_ORDER)
    return fig


def fig_baseline_trajectories(full: dict, scenarios: list, T: int, params) -> go.Figure:
    """Scenario-averaged trajectories: price and mobility (±SEM bands)."""
    hours, T_hours = np.arange(T) / 60, T // 60
    n = max(len(scenarios), 1)
    sem = lambda a: a.std(axis=0) / np.sqrt(n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Mean price", "Mean mobility (0 parked, 1 driving)"))

    def band(mean, half, color, name, row, legend=False):
        fill = _rgba(color, 0.12)
        fig.add_trace(go.Scatter(x=hours, y=mean + half, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=name), row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean - half, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=fill, showlegend=False,
                                 hoverinfo="skip", legendgroup=name), row=row, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mean, mode="lines", line=dict(color=color, width=1.6),
                                 name=name, legendgroup=name, showlegend=legend), row=row, col=1)

    P = np.array([sc["lam_path"] for sc in scenarios])
    band(P.mean(0), sem(P), "lightgray", "λ̄<sub>t</sub>", row=1)
    Mob = np.array([(r["chi_traj"] > 0).astype(float) for r in full["Backward Induction"]])
    band(Mob.mean(0), sem(Mob), "orange", "driving", row=2)

    fig.update_layout(height=620, margin=dict(l=40, r=30, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(range=[0, T_hours], dtick=max(T_hours // 8, 1))
    fig.update_xaxes(title_text="Hour (h)", row=2, col=1)
    fig.update_yaxes(title_text="€/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Fraction driving", tickvals=[0, 0.5, 1], row=2, col=1)
    return fig


def figure_to_png(fig: go.Figure, width: int = 1400, scale: int = 3) -> bytes:
    """Render a Plotly figure to high-res PNG bytes (requires kaleido)."""
    import copy
    fig = copy.deepcopy(fig)
    fig.update_layout(font=dict(size=16))
    # Subplot titles are annotations — bump them separately
    for ann in fig.layout.annotations:
        if ann.font and ann.font.size:
            ann.font.size = max(ann.font.size, 18)
        else:
            ann.update(font=dict(size=18))
    # The per-figure margins were tuned for the default font; at 16pt the axis
    # titles/ticks no longer fit. Let Plotly reserve space automatically and
    # enforce generous minimum margins so nothing is clipped.
    fig.update_xaxes(automargin=True, title_standoff=12)
    fig.update_yaxes(automargin=True, title_standoff=12)
    m = fig.layout.margin
    fig.update_layout(margin=dict(
        l=max(m.l or 0, 100),
        r=max(m.r or 0, 40),
        t=max(m.t or 0, 50),
        b=max(m.b or 0, 80),
    ))
    h = int(fig.layout.height or 500)
    return fig.to_image(format="png", width=width, height=h, scale=scale)
