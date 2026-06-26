"""Exact expected optimality gap of each benchmark heuristic vs. Backward
Induction, for the plain baseline model.

By the suboptimality identity, integrated over the rollout initial-state
distribution with beta=1,

    E[J_0^pi] - E[J_0^*]  =  E_pi[ sum_t delta_t(S_t) ],

and the left side is computable exactly from the same backward recursion that
yields E[J_0] — no sampling, hence none of the SEM noise of the rollouts. So the
gap for each policy is simply

    gap(pi) = exact_cost(pi) - exact_cost(Backward Induction),

both produced by `compute_all_exact_costs`. This is the exact per-configuration
suboptimality that can be reported alongside the exact optimal costs.

This is slow: each non-BI policy runs a full backward-pass evaluation that
rebuilds the (N_e x K) action grid via a pure-Python scalar-policy call at every
(t, chi) cell, so a per-policy progress bar is shown.

Outputs -> optimality_gap.csv (repo root by default).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from ev_mdt.analysis.sensitivity import (
    BASELINE_MODEL, baseline_optimal_result, compute_all_exact_costs_breakdown,
)
from ev_mdt.plots.sensitivity import _exact_summary, _penalty_pct, fig_baseline_cost, figure_to_png
from ev_mdt.plots.viz import POLICY_ORDER

BI = "Backward Induction"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N-e", type=int, default=500, help="battery grid resolution")
    ap.add_argument("--out", type=Path, default=ROOT / "optimality_gap.csv",
                    help="output CSV path")
    ap.add_argument("--fig", type=Path, default=ROOT / "optimality_gap.png",
                    help="output figure path")
    args = ap.parse_args()

    tqdm.write(f"[1/2] Solving baseline model (N_e={args.N_e})…")
    result = baseline_optimal_result(BASELINE_MODEL, args.N_e)

    tqdm.write("[2/2] Computing exact cost for every policy (backward-pass evaluation)…")
    bd = compute_all_exact_costs_breakdown(result, desc="exact costs")

    j_star = bd[BI]["total"]
    rows = []
    for policy in POLICY_ORDER:
        if policy not in bd:
            continue
        vals = bd[policy]
        row = {"Policy": policy, **_exact_summary(vals)}
        row["Optimality gap %"] = (vals["total"] - j_star) / j_star * 100 if j_star else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).reset_index(drop=True)

    df.to_csv(args.out, index=False)
    tqdm.write(f"Saved: {args.out}")

    result["exact_breakdown"] = bd
    fig = fig_baseline_cost({}, source="exact", result=result)
    args.fig.write_bytes(figure_to_png(fig))
    tqdm.write(f"Saved: {args.fig}")


if __name__ == "__main__":
    main()
