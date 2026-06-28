"""One-off: export policy-rule figures (heatmaps + charge borders) for every
sweep dimension, in one consistent format for the sensitivity-analysis section.

For each sweep dimension (plus a standalone ``baseline`` folder) we emit six PNGs:

    figures_app/<sweep>/
        bi_heatmap.png          bi_charge_border.png
        du_heatmap.png          du_charge_border.png
        bl_heatmap.png          bl_charge_border.png

i.e. policy heatmap + charge border for each of Backward Induction (BI),
Departure Urgency (DU) and Battery Level Urgency (BL). One subplot panel per
sweep value.

Panel layouts (identical across BI/DU/BL within a folder):
    heatmaps : <=3 -> one per row (N x 1); >3 -> two per row.
    borders  : <=3 -> one row (1 x N); 4 -> 2 x 2; 5-6 -> three per row.

Only the optimal policy (BI) needs a backward-induction solve; DU and BL are
closed-form in the state. No rollouts and no exact-cost/optimality computation
are run — we only solve for the BI policy and read the closed-form rules.
"""
import argparse
import sys
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from ev_mdt.params import NegBinParams
from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL,
    CRISIS_YEARS, DEPARTURE_PROFILES, HORIZON_HOURS, PHI_VALUES,
    build_params, solve, _gaussian_pbp,
)
from ev_mdt.plots.sensitivity import (
    _opt_rates_averaged, _charge_battery_ceiling,
    _baseline_policy_rates, _bin_heatmap,
    _du_charge_battery_ceiling, _blu_charge_battery_ceiling,
    fig_cost_distribution, fig_baseline_cost,
)

OUT = Path(__file__).resolve().parents[1] / "figures_appendix"
TABLES = ROOT / "data" / "tables_final" / "tables"
# Sweep folders that have a cost summary table (same names as the figure folders).
SWEEP_FOLDERS = [
    "pricing_model", "pricing_season", "pricing_daytype", "pricing_crisis",
    "penalty", "horizon", "departure_profile", "mobility_model",
]
N_E = 500
T_DAY = 24 * 60
BATTERY_BIN_KWH = 0.5

# Short code -> (file stem, benchmark-policy name for DU/BL).
POLICIES = [
    ("BI", "bi"),
    ("DU", "du"),
    ("BL", "bl"),
]
_BENCH_NAME = {"DU": "Departure Urgency", "BL": "Battery Level Urgency"}


# ── Solving one panel ──────────────────────────────────────────────────────────

def solve_panel(label: str, model_label: str, params, pbp_fn, T: int) -> dict:
    """Backward-induction solve for one config; returns a render-ready result dict."""
    print(f"  [{label}] solving BI (T={T // 60}h)…", flush=True)
    pi, actions, e_grid, lam_grid = solve(model_label, params, pbp_fn, T, N_E)
    return dict(label=label, params=params, pbp_fn=pbp_fn, pi=pi, actions=actions,
                e_grid=e_grid, lam_grid=lam_grid, T=T)


def _gauss_panel(label: str, model_label: str, params, T: int = T_DAY) -> dict:
    return solve_panel(label, model_label, params,
                       (lambda t, p=params: _gaussian_pbp(t, p)), T)


# ── Per-policy data accessors ──────────────────────────────────────────────────

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
    if policy == "DU":
        return _du_charge_battery_ceiling(r["params"], r["pbp_fn"], r["e_grid"], t)
    return _blu_charge_battery_ceiling(r["params"], r["pbp_fn"], t)


# ── Layouts ─────────────────────────────────────────────────────────────────────

def _heatmap_dims(n: int) -> tuple[int, int]:
    cols = 1 if n <= 3 else 2
    return ceil(n / cols), cols


def _border_dims(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    cols = 3
    return ceil(n / cols), cols


# ── Figure factories ────────────────────────────────────────────────────────────

def heatmap_grid(results: list[dict], policy: str) -> go.Figure:
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
        rates = _heatmap_rates(policy, r)
        time_bin = max(1, T // T_DAY)           # ~1440 columns regardless of horizon
        z, t_c, b_c = _bin_heatmap(rates, r["e_grid"], T, time_bin, BATTERY_BIN_KWH,
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


def border_grid(results: list[dict], policy: str) -> go.Figure:
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


def figure_to_png(fig: go.Figure, width: int = 1400, scale: int = 3,
                  top: int | None = None) -> bytes:
    """Tight high-res PNG: margins shrink (via automargin) to exactly fit the
    axis labels and colorbar, so the plot content fills the frame edge-to-edge
    with no extra whitespace on the sides.

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


def export_folder(folder: str, results: list[dict]) -> None:
    dest = OUT / folder
    dest.mkdir(parents=True, exist_ok=True)
    for code, stem in POLICIES:
        (dest / f"{stem}_heatmap.png").write_bytes(figure_to_png(heatmap_grid(results, code)))
        (dest / f"{stem}_charge_border.png").write_bytes(figure_to_png(border_grid(results, code)))
    print(f"[{folder}] saved {len(POLICIES) * 2} figures → {dest}")


# ── Cost figures (re-rendered from saved exact-cost tables, tighter margins) ─────

def _breakdown_from_rows(sub: pd.DataFrame) -> dict:
    """{policy: {total, charging, penalty}} from summary-table rows (exact source)."""
    return {
        row["Policy"]: {
            "total":    row["Mean cost (€)"],
            "charging": row["Mean charging (€)"],
            "penalty":  row["Mean penalty (€)"],
        }
        for _, row in sub.iterrows()
    }


def export_sweep_cost(folder: str) -> None:
    """Re-render a sweep's cost-distribution figure from its summary.csv.

    Feeds the saved exact breakdown straight into the production
    ``fig_cost_distribution`` factory, so the figure is identical to before —
    only the (tighter) export margins differ.
    """
    df = pd.read_csv(TABLES / "sensitivity_figures" / folder / "summary.csv")
    labels = list(dict.fromkeys(df["Swept value"]))
    results = [{"label": str(v), "exact_breakdown": _breakdown_from_rows(df[df["Swept value"] == v])}
               for v in labels]
    fig = fig_cost_distribution(results, source="exact")
    dest = OUT / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cost.png").write_bytes(figure_to_png(fig, top=52))
    print(f"[{folder}] saved cost.png → {dest}")


def export_baseline_cost() -> None:
    """Re-render the baseline-model per-policy cost bar from optimality_gap.csv."""
    df = pd.read_csv(TABLES / "baseline_models" / "optimality_gap.csv")
    result = {"exact_breakdown": _breakdown_from_rows(df)}
    fig = fig_baseline_cost({}, source="exact", result=result)
    dest = OUT / "baseline"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cost.png").write_bytes(figure_to_png(fig, top=52))
    print(f"[baseline] saved cost.png → {dest}")


def export_all_costs() -> None:
    export_baseline_cost()
    for folder in SWEEP_FOLDERS:
        export_sweep_cost(folder)


# ── Price-model mean profile (std error bands) ──────────────────────────────────

def export_price_profile() -> None:
    """One-off: mean hourly price profile of all four pricing models over one day,
    with ±1 SEM as shaded error bands. Same look as the production price figure,
    re-rendered with the new tight-margin export.
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.analysis.prices import (
        fit_samplers, simulate_price_paths, PRICE_MODEL_COLORS,
    )
    from ev_mdt.plots.viz import rgba as _rgba

    print("Loading price data…", flush=True)
    df = load_prices()
    print("Fitting samplers (Gaussian Bins, GMM, MDN)…", flush=True)
    samplers = fit_samplers(df)
    print("Simulating price paths…", flush=True)
    results = simulate_price_paths(samplers)

    hours = np.arange(24)
    fig = go.Figure()
    for name, prices in results.items():
        col = PRICE_MODEL_COLORS.get(name, "#888888")
        mu  = prices.mean(axis=0)
        sem = prices.std(axis=0) / np.sqrt(prices.shape[0])
        fig.add_trace(go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([mu + sem, (mu - sem)[::-1]]),
            fill="toself", fillcolor=_rgba(col, 0.12),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=name,
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=mu, mode="lines", line=dict(color=col, width=1.6),
            name=name, legendgroup=name,
            hovertemplate=f"<b>{name}</b><br>Hour %{{x:02d}}:00<br>%{{y:.4f}} €/kWh<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
        height=480, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Hour of day", dtick=2, range=[0, 23]),
        yaxis=dict(title="Mean price (€/kWh)"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "price_profile.png").write_bytes(figure_to_png(fig, top=40))
    print(f"[price_profile] saved → {OUT / 'price_profile.png'}")


# ── Sweep panel builders ────────────────────────────────────────────────────────

def baseline_results() -> list[dict]:
    return [_gauss_panel("Baseline", BASELINE_MODEL, build_params(BASELINE_MODEL))]


def penalty_results() -> list[dict]:
    return [_gauss_panel(f"{phi} €/h", BASELINE_MODEL, build_params(BASELINE_MODEL, phi=float(phi)))
            for phi in PHI_VALUES]


def horizon_results() -> list[dict]:
    return [_gauss_panel(f"{T_h} h", BASELINE_MODEL, build_params(BASELINE_MODEL), T_h * 60)
            for T_h in HORIZON_HOURS]


def departure_results() -> list[dict]:
    return [_gauss_panel(label, BASELINE_MODEL, build_params(BASELINE_MODEL, **ov))
            for label, ov in DEPARTURE_PROFILES.items()]


def mobility_results() -> list[dict]:
    configs = [
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=5",    NegBinParams(k=5)),
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=10",   NegBinParams(k=10)),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=5",  NegBinParams(lambda_k=5.0,  k=NegBinParams.k_max_for_lambda(5.0))),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=10", NegBinParams(lambda_k=10.0, k=NegBinParams.k_max_for_lambda(10.0))),
    ]
    return [_gauss_panel(label, model, params) for model, label, params in configs]


def _gbins_panel(label: str, sampler, season: str, is_weekend: bool) -> dict:
    from ev_mdt.pricing.samplers import make_price_bin_probs_fn
    params = build_params(BASELINE_MODEL)
    pbp_fn = make_price_bin_probs_fn(sampler, params, season, is_weekend)
    return solve_panel(label, BASELINE_MODEL, params, pbp_fn, T_DAY)


def pricing_results(samplers: dict) -> dict[str, list[dict]]:
    excl, incl, crisis = samplers["excl"], samplers["incl"], samplers["crisis"]
    return {
        "pricing_model": [
            _gbins_panel("Gaussian Bins", excl, "spring", False),
            _gbins_panel("GMM",           samplers["gmm"], "spring", False),
            _gbins_panel("MDN",           samplers["mdn"], "spring", False),
        ],
        "pricing_season": [
            _gbins_panel(s.capitalize(), excl, s, False)
            for s in ("winter", "spring", "summer", "autumn")
        ],
        "pricing_daytype": [
            _gbins_panel("Weekday", excl, "spring", False),
            _gbins_panel("Weekend", excl, "spring", True),
        ],
        "pricing_crisis": [
            _gbins_panel("Excluding crisis", excl,   "spring", False),
            _gbins_panel("Including crisis", incl,   "spring", False),
            _gbins_panel("Crisis only",      crisis, "spring", False),
        ],
    }


def fit_pricing_samplers() -> dict:
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.pricing.samplers import GaussianBinnedSampler, GMMSampler, MDNSampler
    print("Loading price data…", flush=True)
    df = load_prices()
    df_excl = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
    df_crisis = df[df["timestamp"].dt.year.isin(CRISIS_YEARS)]
    print("Fitting samplers (Gaussian Bins ×3, GMM, MDN)…", flush=True)
    return {
        "excl":   GaussianBinnedSampler().fit(df_excl),
        "incl":   GaussianBinnedSampler().fit(df),
        "crisis": GaussianBinnedSampler().fit(df_crisis),
        "gmm":    GMMSampler().fit(df_excl),
        "mdn":    MDNSampler().fit(df_excl),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-only", action="store_true",
                    help="Only re-render the cost figures from the saved tables "
                         "(no backward-induction solves).")
    ap.add_argument("--price-profile", action="store_true",
                    help="Only render the price-model mean profile (std bands).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.price_profile:
        export_price_profile()
        print(f"All figures → {OUT}")
        return

    if not args.cost_only:
        export_folder("baseline", baseline_results())
        export_folder("penalty", penalty_results())
        export_folder("horizon", horizon_results())
        export_folder("departure_profile", departure_results())
        export_folder("mobility_model", mobility_results())

        samplers = fit_pricing_samplers()
        for folder, results in pricing_results(samplers).items():
            export_folder(folder, results)

    export_all_costs()

    print(f"All figures → {OUT}")


if __name__ == "__main__":
    main()
