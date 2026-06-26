"""One-off: per-policy benchmark figures for the departure_profile and
mobility_model sweeps.

Six figures total (3 per sweep):
  - Departure Urgency  → heatmaps grid  (DU varies across the sweep)
  - Departure Urgency  → charge-border grid
  - Battery Level Urgency → combined heatmap + charge border, single
    representative panel (BLU is invariant across these sweeps).

Layout: departure_profile = 1x3, mobility_model = 2x2.

No BI solve is needed — the benchmark policies are closed-form in the state, so
we build lightweight result dicts directly.

Outputs -> changes/.
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL,
    DEPARTURE_PROFILES, build_params, _gaussian_pbp,
)
from ev_mdt.params import NegBinParams
from ev_mdt.plots.sensitivity import (
    _baseline_policy_rates, _bin_heatmap, figure_to_png,
    _du_charge_battery_ceiling, _blu_charge_battery_ceiling,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "changes"
N_E = 500
T = 24 * 60
TIME_BIN_MIN = 1
BATTERY_BIN_KWH = 0.5


def make_result(label: str, params) -> dict:
    K = params.K
    return dict(
        label=label,
        params=params,
        pbp_fn=(lambda t, p=params: _gaussian_pbp(t, p)),
        e_grid=np.linspace(params.e_min, params.e_max, N_E),
        lam_grid=np.array([(j + 0.5) * params.lambda_max / K for j in range(K)]),
        T=T,
    )


def departure_results() -> list[dict]:
    return [make_result(label, build_params(BASELINE_MODEL, **ov))
            for label, ov in DEPARTURE_PROFILES.items()]


def mobility_results() -> list[dict]:
    configs = [
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=5",    NegBinParams(k=5)),
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=10",   NegBinParams(k=10)),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=5",  NegBinParams(lambda_k=5.0,  k=NegBinParams.k_max_for_lambda(5.0))),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=10", NegBinParams(lambda_k=10.0, k=NegBinParams.k_max_for_lambda(10.0))),
    ]
    return [make_result(label, params) for _, label, params in configs]


def _grid_pos(i: int, cols: int) -> tuple[int, int]:
    return i // cols + 1, i % cols + 1


def _add_heatmap(fig, r, pname, row, col, show_scale):
    params, e_grid, lam_grid, pbp_fn = r["params"], r["e_grid"], r["lam_grid"], r["pbp_fn"]
    probs = np.array([pbp_fn(t) for t in range(T)])
    cumsum = probs.cumsum(axis=1)
    rates = _baseline_policy_rates(pname, {}, e_grid, lam_grid, params, T, probs, cumsum)
    rates = np.clip(rates, 0.0, params.u_max)
    z, t_c, b_c = _bin_heatmap(rates, e_grid, T, TIME_BIN_MIN, BATTERY_BIN_KWH,
                               params.e_min, params.e_max)
    fig.add_trace(go.Heatmap(
        x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max, colorscale="RdYlBu_r",
        showscale=show_scale,
        colorbar=dict(title="u (kW)", x=1.02, len=0.9) if show_scale else None,
        hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u: %{z:.2f} kW<extra></extra>",
    ), row=row, col=col)


def _add_border(fig, r, pname, row, col):
    params, e_grid, lam_grid, pbp_fn = r["params"], r["e_grid"], r["lam_grid"], r["pbp_fn"]
    n_h = min(24, T // 60)
    for h in range(n_h):
        if pname == "Battery Level Urgency":
            ceil = _blu_charge_battery_ceiling(params, pbp_fn, h * 60)
        else:
            ceil = _du_charge_battery_ceiling(params, pbp_fn, e_grid, h * 60)
        color = sample_colorscale("Viridis", [h / max(1, n_h - 1)])[0]
        fig.add_trace(go.Scatter(
            x=lam_grid, y=ceil, mode="lines", line=dict(color=color, width=1.3),
            showlegend=False,
            hovertemplate=f"{pname}<br>Hour {h:02d}:00<br>Price %{{x:.3f}} €/kWh<br>"
                          "charge if battery ≤ %{y:.1f} kWh<extra></extra>",
        ), row=row, col=col)


def _hour_colorbar(fig, row, col, x=1.02):
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=x)),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)


# ── DU: heatmaps-only grid ───────────────────────────────────────────────────
def fig_heatmaps_grid(results, pname, rows, cols) -> go.Figure:
    titles = [r["label"] for r in results]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.16 if rows > 1 else 0.0)
    for i, r in enumerate(results):
        row, col = _grid_pos(i, cols)
        _add_heatmap(fig, r, pname, row, col, show_scale=(i == 0))
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


# ── DU: charge-border-only grid ──────────────────────────────────────────────
def fig_borders_grid(results, pname, rows, cols) -> go.Figure:
    titles = [r["label"] for r in results]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.16 if rows > 1 else 0.0)
    for i, r in enumerate(results):
        row, col = _grid_pos(i, cols)
        _add_border(fig, r, pname, row, col)
        fig.update_xaxes(title_text="Price (€/kWh)" if row == rows else "",
                         showticklabels=(row == rows), title_standoff=12, row=row, col=col)
        fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                         range=[0, r["params"].e_max], title_standoff=12, row=row, col=col)
    _hour_colorbar(fig, row=1, col=cols)
    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=300 * rows + 80, width=480 * cols + 120,
                      margin=dict(l=70, r=90, t=50, b=55))
    for ann in fig.layout.annotations:
        ann.yshift = 10
    return fig


# ── BL: combined heatmap + charge border, no title, single Battery y-axis ─────
def fig_bl_combined(result, pname="Battery Level Urgency") -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
    _add_heatmap(fig, result, pname, row=1, col=1, show_scale=True)
    # override the heatmap colorbar to sit between the two panels
    fig.data[0].colorbar = dict(title="u (kW)", x=0.46, len=0.9)
    fig.update_xaxes(title_text="Hour (h)", range=[0, T // 60], title_standoff=12, row=1, col=1)
    fig.update_yaxes(title_text="Battery (kWh)", title_standoff=16, row=1, col=1)

    _add_border(fig, result, pname, row=1, col=2)
    fig.update_xaxes(title_text="Price (€/kWh)", title_standoff=12, row=1, col=2)
    fig.update_yaxes(title_text="", range=[0, result["params"].e_max], row=1, col=2)
    _hour_colorbar(fig, row=1, col=2, x=1.01)

    fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                      height=380, margin=dict(l=70, r=70, t=30, b=55))
    return fig


def export(sweep: str, results: list[dict], rows: int, cols: int) -> None:
    hm = fig_heatmaps_grid(results, "Departure Urgency", rows, cols)
    (OUT / f"{sweep}_du_heatmaps.png").write_bytes(figure_to_png(hm))

    bd = fig_borders_grid(results, "Departure Urgency", rows, cols)
    (OUT / f"{sweep}_du_charge_border.png").write_bytes(figure_to_png(bd))

    blu = fig_bl_combined(results[0])
    (OUT / f"{sweep}_battery_level_urgency.png").write_bytes(figure_to_png(blu))
    print(f"[{sweep}] saved DU heatmaps + DU borders + BLU combined")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export("departure_profile", departure_results(), rows=1, cols=3)
    export("mobility_model", mobility_results(), rows=2, cols=2)
    print(f"All figures → {OUT}")


if __name__ == "__main__":
    main()
