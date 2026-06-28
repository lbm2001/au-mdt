"""One-off figures: selected panels from the sensitivity sweep, in custom layouts.

Outputs to figures_appendix/oneoff/:
  penalty_bi_border_1x3.png        – BI charge borders for φ ∈ {1, 500, 5000}    (1×3)
  mobility_bi_2x2.png              – BI heatmap + border for NegBin fixed k=5/10  (2×2)
  departure_bi_du_heatmap_2x3.png  – BI + DU heatmaps for all departure profiles  (2×3)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import copy
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from ev_mdt.params import NegBinParams
from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, NEGBIN_FIXED_MODEL,
    DEPARTURE_PROFILES,
    build_params, solve, _gaussian_pbp,
)
from ev_mdt.plots.sensitivity import (
    _opt_rates_averaged, _charge_battery_ceiling,
    _baseline_policy_rates, _bin_heatmap,
    _du_charge_battery_ceiling,
)

OUT = Path(__file__).resolve().parents[1] / "figures_appendix" / "oneoff"
N_E = 500
T_DAY = 24 * 60
BATTERY_BIN_KWH = 0.5
_BENCH_NAME = {"DU": "Departure Urgency", "BL": "Battery Level Urgency"}


# ── Export helper ──────────────────────────────────────────────────────────────

def figure_to_png(fig: go.Figure, width: int = 1400, scale: int = 3,
                  top: int | None = None) -> bytes:
    fig = copy.deepcopy(fig)
    fig.update_layout(template="plotly_white", plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(size=16))
    has_titles = bool(fig.layout.annotations)
    for ann in fig.layout.annotations:
        if ann.font and ann.font.size:
            ann.font.size = max(ann.font.size, 18)
        else:
            ann.update(font=dict(size=18))
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    t = top if top is not None else (34 if has_titles else 10)
    fig.update_layout(margin=dict(l=8, r=8, t=t, b=8))
    h = int(fig.layout.height or 500)
    return fig.to_image(format="png", width=width, height=h, scale=scale)


# ── Solving ────────────────────────────────────────────────────────────────────

def solve_panel(label: str, model_label: str, params, pbp_fn, T: int = T_DAY) -> dict:
    print(f"  [{label}] solving BI (T={T // 60}h)…", flush=True)
    pi, actions, e_grid, lam_grid = solve(model_label, params, pbp_fn, T, N_E)
    return dict(label=label, params=params, pbp_fn=pbp_fn, pi=pi, actions=actions,
                e_grid=e_grid, lam_grid=lam_grid, T=T)


def _gauss_panel(label: str, model_label: str, params, T: int = T_DAY) -> dict:
    return solve_panel(label, model_label, params,
                       (lambda t, p=params: _gaussian_pbp(t, p)), T)


# ── Data accessors ─────────────────────────────────────────────────────────────

def _heatmap_rates(policy: str, r: dict) -> np.ndarray:
    params, T = r["params"], r["T"]
    if policy == "BI":
        rates = _opt_rates_averaged(r["pi"], r["actions"], params, r["pbp_fn"], T)
    else:
        probs  = np.array([r["pbp_fn"](t) for t in range(T)])
        cumsum = probs.cumsum(axis=1)
        rates  = _baseline_policy_rates(_BENCH_NAME[policy], {}, r["e_grid"],
                                        r["lam_grid"], params, T, probs, cumsum)
    return np.clip(rates, 0.0, params.u_max)


def _border_ceiling(policy: str, r: dict, t: int) -> np.ndarray:
    if policy == "BI":
        return _charge_battery_ceiling(r["pi"], r["actions"], r["e_grid"], t)
    return _du_charge_battery_ceiling(r["params"], r["pbp_fn"], r["e_grid"], t)


# ── Shared subplot style ───────────────────────────────────────────────────────

def _std_layout(fig: go.Figure, rows: int, cols: int) -> None:
    fig.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        height=300 * rows + 80, width=480 * cols + 120,
        margin=dict(l=70, r=90, t=50, b=55),
    )
    for ann in fig.layout.annotations:
        ann.yshift = 10


def _add_heatmap_panel(fig, r: dict, policy: str, row: int, col: int,
                       rows: int, show_colorbar: bool) -> None:
    params, T = r["params"], r["T"]
    rates = _heatmap_rates(policy, r)
    time_bin = max(1, T // T_DAY)
    z, t_c, b_c = _bin_heatmap(rates, r["e_grid"], T, time_bin, BATTERY_BIN_KWH,
                                params.e_min, params.e_max)
    fig.add_trace(go.Heatmap(
        x=t_c, y=b_c, z=z, zmin=0, zmax=params.u_max, colorscale="RdYlBu_r",
        showscale=show_colorbar,
        colorbar=dict(title="u (kW)", x=1.02, len=0.9) if show_colorbar else None,
        hovertemplate="Hour: %{x:.2f} h<br>Battery: %{y:.2f} kWh<br>u: %{z:.2f} kW<extra></extra>",
    ), row=row, col=col)
    fig.update_xaxes(title_text="Hour (h)" if row == rows else "",
                     range=[0, T // 60], showticklabels=(row == rows),
                     title_standoff=12, row=row, col=col)
    fig.update_yaxes(title_text="Battery (kWh)" if col == 1 else "",
                     title_standoff=16, row=row, col=col)


def _add_border_panel(fig, r: dict, policy: str, row: int, col: int,
                      rows: int) -> None:
    n_h = min(24, r["T"] // 60)
    for h in range(n_h):
        ceil_e = _border_ceiling(policy, r, h * 60)
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


# ── Figure 1: Penalty BI charge borders 1×3 ────────────────────────────────────

def export_penalty_border() -> None:
    """BI charge borders for φ ∈ {1, 500, 5000} in a 1×3 panel."""
    phi_vals = [1, 500, 5000]
    results = [
        _gauss_panel(f"{phi} €/h", BASELINE_MODEL,
                     build_params(BASELINE_MODEL, phi=float(phi)))
        for phi in phi_vals
    ]

    titles = [r["label"] for r in results]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.10)

    for i, r in enumerate(results):
        col = i + 1
        _add_border_panel(fig, r, "BI", row=1, col=col, rows=1)

    # Viridis hour colorbar via dummy trace
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.02)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=3)

    _std_layout(fig, rows=1, cols=3)
    dest = OUT / "penalty_bi_border_1x3.png"
    dest.write_bytes(figure_to_png(fig))
    print(f"[penalty_border] saved → {dest}")


# ── Figure 2: Mobility BI+DU charge borders 2×2 ───────────────────────────────

def export_mobility_2x2() -> None:
    """Charge borders for NegBin fixed k=5/10 × BI (top) and DU (bottom)."""
    configs = [
        ("NegBin fixed k=5",  NegBinParams(k=5)),
        ("NegBin fixed k=10", NegBinParams(k=10)),
    ]
    results = [
        _gauss_panel(label, NEGBIN_FIXED_MODEL, params)
        for label, params in configs
    ]

    # Column titles (top row); row 2 titles are blank.
    titles = [r["label"] for r in results] + ["", ""]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.18)

    for i, r in enumerate(results):
        col = i + 1
        _add_border_panel(fig, r, "BI", row=1, col=col, rows=2)
        _add_border_panel(fig, r, "DU", row=2, col=col, rows=2)

    # Viridis hour colorbar (shared across both rows)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale="Viridis", cmin=0, cmax=23, color=[0], showscale=True,
                    colorbar=dict(title="Hour", x=1.02)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)

    _std_layout(fig, rows=2, cols=2)
    dest = OUT / "mobility_bi_du_border_2x2.png"
    dest.write_bytes(figure_to_png(fig))
    print(f"[mobility_2x2] saved → {dest}")


# ── Figure 3: Departure profile BI+DU heatmaps 2×3 ────────────────────────────

def export_departure_heatmap_2x3() -> None:
    """Heatmaps for all 3 departure profiles × BI and DU policies in a 2×3 grid."""
    profiles = list(DEPARTURE_PROFILES.items())          # 3 entries
    results = [
        _gauss_panel(label, BASELINE_MODEL, build_params(BASELINE_MODEL, **ov))
        for label, ov in profiles
    ]

    # Subplot titles: row 1 = "BI: <profile>", row 2 = "DU: <profile>"
    titles = [f"BI – {r['label']}" for r in results] + \
             [f"DU – {r['label']}" for r in results]

    fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.14)

    for i, r in enumerate(results):
        col = i + 1
        _add_heatmap_panel(fig, r, "BI", row=1, col=col,
                           rows=2, show_colorbar=(col == 3))
        _add_heatmap_panel(fig, r, "DU", row=2, col=col,
                           rows=2, show_colorbar=False)

    _std_layout(fig, rows=2, cols=3)
    dest = OUT / "departure_bi_du_heatmap_2x3.png"
    dest.write_bytes(figure_to_png(fig))
    print(f"[departure_bi_du] saved → {dest}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_penalty_border()
    export_mobility_2x2()
    export_departure_heatmap_2x3()
    print(f"\nAll figures → {OUT}")


if __name__ == "__main__":
    main()
