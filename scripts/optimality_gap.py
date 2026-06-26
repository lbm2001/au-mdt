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
    BASELINE_MODEL, baseline_optimal_result, compute_all_exact_costs,
)

BI = "Backward Induction"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N-e", type=int, default=500, help="battery grid resolution")
    ap.add_argument("--out", type=Path, default=ROOT / "optimality_gap.csv",
                    help="output CSV path")
    args = ap.parse_args()

    tqdm.write(f"[1/2] Solving baseline model (N_e={args.N_e})…")
    result = baseline_optimal_result(BASELINE_MODEL, args.N_e)

    tqdm.write("[2/2] Computing exact cost for every policy (backward-pass evaluation)…")
    costs = compute_all_exact_costs(result, beta=1.0, desc="exact costs")

    j_star = costs[BI]
    rows = [{
        "Policy": policy,
        "Exact cost (€)": cost,
        "Optimality gap (€)": cost - j_star,
        "Gap (%)": 100.0 * (cost - j_star) / j_star if j_star else float("nan"),
    } for policy, cost in costs.items()]
    df = pd.DataFrame(rows).sort_values("Optimality gap (€)").reset_index(drop=True)

    df.to_csv(args.out, index=False)
    tqdm.write(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
