"""CLI entry point for ev_mdt.

Usage
-----
    python -m ev_mdt solve [--model BASELINE] [--N-e 500] [--hours 24] [--phi 1.0] ...
    python -m ev_mdt rollout --n 200 --seed 42 [same solve flags]
    python -m ev_mdt sensitivity --sweep penalty
    python -m ev_mdt sensitivity --sweep all [--N-rollouts 500] [--N-e 500] [--seed 42]
                                              [--out-dir figures/]
    python -m ev_mdt prices [--n-days 1000] [--season all] [--daytype all]
                            [--seed 42] [--out-dir figures/]
"""
import argparse
import sys


def _add_solve_args(parser: argparse.ArgumentParser) -> None:
    from ev_mdt.params import MODEL_LABELS
    parser.add_argument("--model", default="Baseline", choices=MODEL_LABELS,
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
    from tqdm import tqdm
    from ev_mdt.analysis.sensitivity import run_all_sweeps, save_figures, ALL_SWEEP_NAMES

    sweeps = ALL_SWEEP_NAMES if args.sweep == "all" else [args.sweep]

    outer = tqdm(sweeps, desc="Sweeps", unit="sweep", position=0)
    for i, sw in enumerate(outer):
        outer.set_description(f"Sweep: {sw}")
        inner = tqdm(total=100, desc="  progress", unit="%", position=1,
                     leave=False, bar_format="{l_bar}{bar}| {n:.0f}%  {postfix}")

        def cb(f: float, msg: str, _bar: tqdm = inner) -> None:
            _bar.n = int(f * 100)
            _bar.set_postfix_str(msg[:50])
            _bar.refresh()

        results = run_all_sweeps(
            N_rollouts=args.N_rollouts, N_e=args.N_e, seed=args.seed,
            sweeps=[sw], progress_cb=cb,
        )
        inner.close()

        saved = save_figures(results, out_dir=args.out_dir,
                             N_rollouts=args.N_rollouts, seed=args.seed, N_e=args.N_e,
                             include_baseline=(i == 0))
        for p in saved:
            tqdm.write(f"  Saved: {p}")

    outer.close()
    print("\nAll sweeps complete.")


def cmd_prices(args: argparse.Namespace) -> None:
    from pathlib import Path
    from tqdm import tqdm
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.analysis.prices import fit_samplers, simulate_price_paths, price_figures

    print("Loading ENTSO-E price data…", flush=True)
    df = load_prices(_log=tqdm.write)
    tqdm.write(f"Loaded {len(df):,} measurements "
               f"({df['timestamp'].dt.year.min()}–{df['timestamp'].dt.year.max()})\n")

    fit_bar = tqdm(total=100, desc="Fitting samplers", unit="%",
                   bar_format="{l_bar}{bar}| {n:.0f}%  {postfix}")

    def fit_progress(model: str, frac: float, msg: str) -> None:
        fit_bar.n = int(frac * 100)
        fit_bar.set_postfix_str(f"{model}: {msg[:40]}")
        fit_bar.refresh()

    samplers = fit_samplers(df, progress_cb=fit_progress)
    fit_bar.n = 100
    fit_bar.refresh()
    fit_bar.close()

    season  = None if args.season == "all" else args.season
    daytype = args.daytype

    print(f"\nSimulating {args.n_days} days "
          f"(season={args.season}, daytype={daytype}, seed={args.seed})…", flush=True)
    results = simulate_price_paths(samplers, n_days=args.n_days,
                                   season=season, daytype=daytype, seed=args.seed)
    for name, prices in results.items():
        print(f"  {name:<30}  mean = {prices.mean():.4f} €/kWh")

    out_dir = Path(args.out_dir) / "price_explorer"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ev_mdt.plots.sensitivity import figure_to_png
        save_bar = tqdm(["mean_profile", "std_profile"], desc="Saving figures", unit="fig")
        fig_mean, fig_std = price_figures(results)
        for name, fig in zip(save_bar, [fig_mean, fig_std]):
            save_bar.set_postfix_str(name)
            (out_dir / f"{name}.png").write_bytes(figure_to_png(fig))
        save_bar.close()
        print(f"Saved figures to {out_dir}/")
    except Exception as e:
        print(f"\nCould not save figures (kaleido missing?): {e}")


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

    # prices
    p_prices = sub.add_parser("prices", help="Fit price models and simulate diurnal profiles")
    p_prices.add_argument("--n-days",  type=int,   default=1000, help="Simulated days")
    p_prices.add_argument("--season",  default="all",
                          choices=["all", "spring", "summer", "autumn", "winter"])
    p_prices.add_argument("--daytype", default="all",
                          choices=["all", "weekday", "weekend"])
    p_prices.add_argument("--seed",    type=int,   default=42,   help="Random seed")
    p_prices.add_argument("--out-dir", default="figures/",       help="Output directory for PNGs")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "rollout":
        cmd_rollout(args)
    elif args.command == "sensitivity":
        cmd_sensitivity(args)
    elif args.command == "prices":
        cmd_prices(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
