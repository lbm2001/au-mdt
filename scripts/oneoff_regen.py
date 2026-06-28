"""One-off: regenerate three existing figures with tight-margin export settings.

Outputs to figures_appendix/regen/:
  policy_heatmaps.png       – baseline policy heatmap grid (all policies)
  optimality_gap.png        – cost comparison from export/changes/optimality_gap.csv
  trip_duration_by_model.png – trip-duration density + survival (right y: 0.001/0.01/0.1 only)
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go

from ev_mdt.analysis.sensitivity import BASELINE_MODEL, baseline_optimal_result
from ev_mdt.plots.sensitivity import fig_baseline_policy_heatmaps, fig_baseline_cost
from ev_mdt.plots.trip_duration import compute_trip_durations, trip_duration_figure

OUT = ROOT / "figures_appendix" / "regen"
OPT_GAP_CSV = ROOT / "export" / "changes" / "optimality_gap.csv"


# ── Tight-margin export (same as policy_rules_figures.py) ─────────────────────

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


# ── Figure 1: Policy heatmaps ──────────────────────────────────────────────────

def export_policy_heatmaps() -> None:
    print("Solving Baseline model…", flush=True)
    result = baseline_optimal_result(BASELINE_MODEL, N_e=500)
    fig = fig_baseline_policy_heatmaps(
        result["params"], result["e_grid"], result["lam_grid"],
        result["T"], result["pbp_fn"],
        pi=result["pi"], actions=result["actions"],
    )
    dest = OUT / "policy_heatmaps.png"
    dest.write_bytes(figure_to_png(fig))
    print(f"[policy_heatmaps] saved → {dest}")


# ── Figure 2: Optimality gap cost figure ──────────────────────────────────────

def _breakdown_from_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        row["Policy"]: {
            "total":    row["Mean cost (€)"],
            "charging": row["Mean charging (€)"],
            "penalty":  row["Mean penalty (€)"],
        }
        for _, row in df.iterrows()
    }


def export_optimality_gap() -> None:
    print("Building optimality gap figure from CSV…", flush=True)
    result = {"exact_breakdown": _breakdown_from_csv(OPT_GAP_CSV)}
    fig = fig_baseline_cost({}, source="exact", result=result)
    dest = OUT / "optimality_gap.png"
    dest.write_bytes(figure_to_png(fig, top=52))
    print(f"[optimality_gap] saved → {dest}")


# ── Figure 3: Trip duration ────────────────────────────────────────────────────

def export_trip_duration() -> None:
    print("Simulating trip durations…", flush=True)
    durs = compute_trip_durations()
    fig = trip_duration_figure(durs)
    # Right panel (yaxis2): show only three reference ticks on the log scale
    fig.update_layout(
        yaxis2=dict(
            tickmode="array",
            tickvals=[0.001, 0.01, 0.1],
            ticktext=["0.001", "0.01", "0.1"],
        )
    )
    dest = OUT / "trip_duration_by_model.png"
    dest.write_bytes(figure_to_png(fig, top=40))
    print(f"[trip_duration] saved → {dest}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_optimality_gap()
    export_trip_duration()
    export_policy_heatmaps()   # slow — solve last so fast figures appear first
    print(f"\nAll figures → {OUT}")


if __name__ == "__main__":
    main()
