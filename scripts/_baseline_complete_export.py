"""One-off: 3×1 policy heatmaps and 3×1 charging borders for BI, DU, BLU."""
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from pathlib import Path

from ev_mdt.analysis.sensitivity import BASELINE_MODEL, baseline_optimal_result, _gaussian_pbp
from ev_mdt.plots.sensitivity import (
    _opt_rates_averaged, _baseline_policy_rates, _bin_heatmap,
    _charge_battery_ceiling, _du_charge_battery_ceiling, _blu_charge_battery_ceiling,
    figure_to_png,
)

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "baseline_complete"
N_E  = 500
TIME_BIN_MIN   = 1
BATTERY_BIN_KWH = 0.5
POLICIES = ["Backward Induction", "Departure Urgency", "Battery Level Urgency"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Solving Baseline model…", flush=True)
    result   = baseline_optimal_result(BASELINE_MODEL, N_e=N_E)
    params   = result["params"]
    pi       = result["pi"]
    actions  = result["actions"]
    e_grid   = result["e_grid"]
    lam_grid = result["lam_grid"]
    T        = result["T"]
    pbp_fn   = result["pbp_fn"]
    T_hours  = T // 60

    probs  = np.array([pbp_fn(t) for t in range(T)])
    cumsum = probs.cumsum(axis=1)

    # ── Policy heatmaps (3 × 1) ───────────────────────────────────────────────
    print("Building heatmaps…", flush=True)
    fig_hm = make_subplots(
        rows=3, cols=1,
        subplot_titles=POLICIES,
        vertical_spacing=0.08,
    )
    for idx, name in enumerate(POLICIES):
        row = idx + 1
        if name == "Backward Induction":
            rates = _opt_rates_averaged(pi, actions, params, pbp_fn, T)
        else:
            rates = _baseline_policy_rates(name, {}, e_grid, lam_grid, params, T, probs, cumsum)
        rates = np.clip(rates, 0.0, params.u_max)
        z, t_c, b_c = _bin_heatmap(
            rates, e_grid, T, TIME_BIN_MIN, BATTERY_BIN_KWH, params.e_min, params.e_max,
        )
        fig_hm.add_trace(go.Heatmap(
            x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max,
            colorscale="RdYlBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="u (kW)", x=1.02, len=0.9) if idx == 0 else None,
            hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u: %{z:.2f} kW<extra></extra>",
        ), row=row, col=1)
        fig_hm.update_xaxes(
            title_text="Hour (h)" if row == 3 else "",
            range=[0, T_hours], title_standoff=12,
            showticklabels=(row == 3), row=row, col=1,
        )
        fig_hm.update_yaxes(title_text="Battery (kWh)", title_standoff=16, row=row, col=1)

    fig_hm.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        height=300 * 3 + 80, margin=dict(l=70, r=90, t=50, b=55),
    )
    for ann in fig_hm.layout.annotations:
        ann.yshift = 10

    hm_path = OUT / "policy_heatmaps.png"
    hm_path.write_bytes(figure_to_png(fig_hm))
    print(f"Saved: {hm_path}")

    # ── Charging borders (3 × 1) ──────────────────────────────────────────────
    print("Building charge borders…", flush=True)
    fig_bd = make_subplots(
        rows=3, cols=1,
        subplot_titles=POLICIES,
        vertical_spacing=0.08,
    )
    n_h = min(24, T_hours)
    for idx, name in enumerate(POLICIES):
        row = idx + 1
        for h in range(n_h):
            color = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
            if name == "Backward Induction":
                ceil = _charge_battery_ceiling(pi, actions, e_grid, h * 60)
            elif name == "Departure Urgency":
                ceil = _du_charge_battery_ceiling(params, pbp_fn, e_grid, h * 60)
            else:
                ceil = _blu_charge_battery_ceiling(params, pbp_fn, h * 60)
            fig_bd.add_trace(go.Scatter(
                x=lam_grid, y=ceil, mode="lines",
                line=dict(color=color, width=1.3), showlegend=False,
                hovertemplate=f"{name}<br>Hour {h:02d}:00<br>"
                              "Price %{x:.3f} €/kWh<br>charge if battery ≤ %{y:.1f} kWh<extra></extra>",
            ), row=row, col=1)
        fig_bd.update_xaxes(
            title_text="Price (€/kWh)" if row == 3 else "",
            title_standoff=12, showticklabels=(row == 3), row=row, col=1,
        )
        fig_bd.update_yaxes(
            title_text="Battery (kWh)", title_standoff=12,
            range=[0, params.e_max], row=row, col=1,
        )

    # Hour colorbar
    fig_bd.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.02)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig_bd.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        height=300 * 3 + 80, margin=dict(l=70, r=90, t=50, b=55),
    )
    for ann in fig_bd.layout.annotations:
        ann.yshift = 10

    bd_path = OUT / "charge_borders.png"
    bd_path.write_bytes(figure_to_png(fig_bd))
    print(f"Saved: {bd_path}")


if __name__ == "__main__":
    main()
