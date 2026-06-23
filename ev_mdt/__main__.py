"""CLI entry point for ev_mdt.

Usage
-----
    python -m ev_mdt solve [--model BASELINE] [--N-e 500] [--hours 24] [--phi 1.0] ...
    python -m ev_mdt rollout --n 200 --seed 42 [same solve flags]
    python -m ev_mdt sensitivity --sweep penalty
    python -m ev_mdt sensitivity --sweep all [--N-rollouts 500] [--N-e 500] [--seed 42]
                                              [--out-dir figures/]
"""
import argparse
import sys


def _add_solve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model",   default="Baseline",
                        choices=["Baseline", "NegBin trips (fixed k)", "NegBin trips (sampled k)"],
                        help="Mobility model")
    parser.add_argument("--N-e",    type=int,   default=500,  metavar="N",  help="Battery grid points")
    parser.add_argument("--hours",  type=int,   default=24,   metavar="H",  help="Horizon in hours")
    parser.add_argument("--phi",    type=float, default=None, metavar="Φ",  help="Penalty (€/h)")
    parser.add_argument("--beta",   type=float, default=None, metavar="β",  help="Discount factor")


def cmd_solve(args: argparse.Namespace) -> None:
    from ev_mdt import solve
    overrides = {}
    if args.phi  is not None: overrides["phi"]  = args.phi
    if args.beta is not None: overrides["beta"] = args.beta
    print(f"Solving {args.model} model (T={args.hours}h, N_e={args.N_e}) …", flush=True)
    result = solve(model=args.model, N_e_override=args.N_e,
                   T_hours_override=args.hours, **overrides)
    print(f"Done. Policy shape: {result['pi'].shape}, actions: {result['actions'].shape}")


def cmd_rollout(args: argparse.Namespace) -> None:
    from ev_mdt import solve, rollout
    from ev_mdt.plots.sensitivity import build_summary_df
    overrides = {}
    if args.phi  is not None: overrides["phi"]  = args.phi
    if args.beta is not None: overrides["beta"] = args.beta
    print(f"Solving + running {args.n} rollouts for {args.model} model …", flush=True)
    result  = solve(model=args.model, N_e_override=args.N_e,
                    T_hours_override=args.hours, **overrides)
    full    = rollout(result, n=args.n, seed=args.seed)
    df      = build_summary_df([{**full, "label": args.model}])
    print(df.to_string(index=False))


def cmd_sensitivity(args: argparse.Namespace) -> None:
    from ev_mdt.analysis.sensitivity import run_all_sweeps, save_figures, ALL_SWEEP_NAMES

    sweeps = ALL_SWEEP_NAMES if args.sweep == "all" else [args.sweep]

    n_total = len(sweeps)
    def progress(f: float, msg: str) -> None:
        pct = int(f * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {msg:<50}", end="", flush=True)

    for i, sw in enumerate(sweeps):
        print(f"\n[{i+1}/{n_total}] Sweep: {sw}")
        def cb(f, m, _sw=sw): progress(f, m)
        results = run_all_sweeps(
            N_rollouts=args.N_rollouts, N_e=args.N_e, seed=args.seed,
            sweeps=[sw], progress_cb=cb,
        )
        print()
        saved = save_figures(results, out_dir=args.out_dir,
                             N_rollouts=args.N_rollouts, seed=args.seed, N_e=args.N_e,
                             include_baseline=(i == 0))
        for p in saved:
            print(f"  Saved: {p}")

    print(f"\nAll sweeps complete.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ev_mdt",
                                     description="EV Charging MDP CLI")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # solve
    p_solve = sub.add_parser("solve", help="Run backward induction and print policy stats")
    _add_solve_args(p_solve)

    # rollout
    p_rollout = sub.add_parser("rollout", help="Run backward induction then simulate rollouts")
    _add_solve_args(p_rollout)
    p_rollout.add_argument("--n",    type=int, default=200, help="Number of rollout scenarios")
    p_rollout.add_argument("--seed", type=int, default=42,  help="Base random seed")

    # sensitivity
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES
    p_sens = sub.add_parser("sensitivity", help="Run sensitivity analysis sweeps")
    p_sens.add_argument("--sweep", default="all",
                        choices=["all"] + ALL_SWEEP_NAMES, metavar="SWEEP",
                        help=f"Which sweep to run (or 'all'). Options: {', '.join(ALL_SWEEP_NAMES)}")
    p_sens.add_argument("--N-rollouts", type=int, default=500, metavar="N",
                        help="Rollouts per swept value")
    p_sens.add_argument("--N-e",        type=int, default=500, metavar="N",
                        help="Battery grid points")
    p_sens.add_argument("--seed",       type=int, default=42,  help="Base random seed")
    p_sens.add_argument("--out-dir",    default="figures/",    help="Output directory for PNGs")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "rollout":
        cmd_rollout(args)
    elif args.command == "sensitivity":
        cmd_sensitivity(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
