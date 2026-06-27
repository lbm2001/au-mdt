"""One-off: rerun the Baseline model + rollouts, save the summary CSV and the
baseline cost figure (legend kept, no per-bar captions) to the repo root."""
import argparse
from pathlib import Path

from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, baseline_optimal_result, make_scenario,
    run_rollouts_full, rollout_fn,
)
from ev_mdt.models.common.rollout_utils import rollout_metrics
from ev_mdt.plots.sensitivity import build_summary_df, fig_baseline_cost, figure_to_png

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-rollouts", type=int, default=1000)
    ap.add_argument("--N-e", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Solving Baseline (N_e={args.N_e}) + {args.N_rollouts} rollouts…", flush=True)
    result = baseline_optimal_result(BASELINE_MODEL, args.N_e)
    params, T, pbp_fn = result["params"], result["T"], result["pbp_fn"]
    scenarios = [make_scenario(params, args.seed + i, T) for i in range(args.N_rollouts)]
    full = run_rollouts_full(result["pi"], result["actions"], result["e_grid"],
                             params, scenarios, rollout_fn(BASELINE_MODEL), pbp_fn,
                             desc="Baseline rollouts")

    # ── table ───────────────────────────────────────────────────────────────
    metrics = {name: [rollout_metrics(r, params) for r in rolls]
               for name, rolls in full.items()}
    df = build_summary_df([{"rollouts": metrics, "label": "Baseline"}])
    csv_path = ROOT / "baseline_cost.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table:  {csv_path}")
    print(df.to_string(index=False))

    # ── figure (keep legend, drop per-bar captions) ──────────────────────────
    fig = fig_baseline_cost(full)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=40, r=20, t=40, b=20),
    )
    png_path = ROOT / "baseline_cost.png"
    png_path.write_bytes(figure_to_png(fig))
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
