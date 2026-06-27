"""One-off: solve Baseline model, export the baseline-policy heatmap grid to PNG."""
from pathlib import Path

from ev_mdt.analysis.sensitivity import BASELINE_MODEL, baseline_optimal_result
from ev_mdt.plots.sensitivity import fig_baseline_policy_heatmaps, figure_to_png

ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("Solving Baseline model…", flush=True)
    result = baseline_optimal_result(BASELINE_MODEL, N_e=500)
    params  = result["params"]
    e_grid  = result["e_grid"]
    lam_grid = result["lam_grid"]
    T       = result["T"]
    pbp_fn  = result["pbp_fn"]
    pi      = result["pi"]
    actions = result["actions"]

    fig = fig_baseline_policy_heatmaps(
        params, e_grid, lam_grid, T, pbp_fn,
        pi=pi, actions=actions,
    )

    out = ROOT / "policy_heatmaps.png"
    out.write_bytes(figure_to_png(fig))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
