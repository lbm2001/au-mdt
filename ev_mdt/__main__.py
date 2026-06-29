"""CLI entry point for ev_mdt.

Every command writes figures (and, where applicable, CSV tables) under
``<out-dir>/<timestamp>/<command>/`` (pass ``--no-timestamp`` to write straight
into ``<out-dir>/<command>/``). All figure and computation logic lives in the
``ev_mdt`` package — this module only parses arguments, calls the package, and
writes the returned figures/tables to disk.

Commands
--------
    baseline             Baseline all-policy heatmaps, charge borders, exact cost bar (+ table)
    sensitivity          Per-sweep BI/DU/BLU heatmaps + charge borders + exact cost bars (+ tables)
    calibrate-du         Departure-Urgency e_base (target-ceiling) and γ calibration sweeps
    trip-duration        Trip-duration distribution figure
    model-trajectories   Mean sampled price + fraction-driving (swap price / mobility model)
    price-models         Mean/std diurnal price comparison across price models
    fit-mdn              Fit the MDN price sampler + its two training-curve figures
    run --all            Run every command (full regeneration)
    run --all-paper      Fast paper figures only (heatmaps/borders/trajectories/prices; no exact cost)
"""
import argparse
import sys
from pathlib import Path


def _timestamped_dir(base, *, enabled: bool = True) -> Path:
    """Append a ``YYYY-MM-DD_HHMMSS`` run folder to ``base`` (unless disabled)."""
    base = Path(base)
    if not enabled:
        return base
    from datetime import datetime
    return base / datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _save_fig(fig, path: Path, *, top: int | None = None, _log=print) -> Path:
    from ev_mdt.plots.sensitivity import figure_to_png
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(figure_to_png(fig, top=top))
    _log(f"  Saved {path}")
    return path


def _save_table(df, path: Path, _log=print) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _log(f"  Saved {path}")
    return path


# Cost bars carry a horizontal legend above the plot → reserve top margin on export.
_COST_TOP = 52


# ── Per-command export workers (also reused by `run`) ──────────────────────────

def _export_baseline(out_base: Path, *, N_e: int, compute_cost: bool, _log=print) -> None:
    from ev_mdt.analysis.sensitivity import baseline_figures
    from ev_mdt.plots.sensitivity import build_summary_df
    dest = out_base / "baseline"
    result, figs = baseline_figures(N_e=N_e, compute_cost=compute_cost, _log=_log)
    for name, fig in figs.items():
        _save_fig(fig, dest / f"{name}.png", top=_COST_TOP if name == "cost" else None, _log=_log)
    if compute_cost and "exact_breakdown" in result:
        _save_table(build_summary_df([result]), dest / "summary.csv", _log)


def _export_sensitivity(out_base: Path, *, dims, N_e: int, compute_cost: bool, _log=print) -> None:
    from ev_mdt.analysis.sensitivity import sensitivity_figures
    from ev_mdt.plots.sensitivity import build_summary_df
    out = sensitivity_figures(dims=dims, N_e=N_e, compute_cost=compute_cost, _log=_log)
    for dim, payload in out.items():
        dest = out_base / "sensitivity" / dim
        for name, fig in payload["figures"].items():
            _save_fig(fig, dest / f"{name}.png", top=_COST_TOP if name == "cost" else None, _log=_log)
        if compute_cost:
            _save_table(build_summary_df(payload["panels"]), dest / "summary.csv", _log)


def _export_calibrate_du(out_base: Path, *, N_e: int, gamma_step: float, step_kwh: float,
                         use_reserve: bool, _log=print) -> None:
    import pandas as pd
    from ev_mdt.analysis.sensitivity import calibrate_du_figures
    dest = out_base / "calibrate_du"
    res = calibrate_du_figures(N_e=N_e, gamma_step=gamma_step, step_kwh=step_kwh,
                               use_reserve=use_reserve, _log=_log)
    target_rows, target_fig = res["target"]
    gamma_results, gamma_fig = res["gamma"]
    _save_fig(target_fig, dest / "target_sweep.png", _log=_log)
    _save_table(pd.DataFrame(target_rows), dest / "target_sweep.csv", _log)
    _save_fig(gamma_fig, dest / "gamma_sweep.png", _log=_log)
    for model_name, rows in gamma_results.items():
        slug = model_name.lower().replace(" ", "_").replace("=", "")
        _save_table(pd.DataFrame(rows), dest / f"gamma_sweep_{slug}.csv", _log)


# CLI short keys → trip-duration / mobility-model display names.
_MOBILITY_KEYS = {
    "baseline":       "Baseline",
    "negbin_fixed":   "Negative Binomial (fixed k)",
    "negbin_poisson": "Negative Binomial (Poisson k)",
}
_MOBILITY_MODEL_LABEL = {
    "baseline":       "Baseline",
    "negbin_fixed":   "Negative Binomial trips (fixed k)",
    "negbin_poisson": "Negative Binomial trips (sampled k)",
}


def _export_trip_duration(out_base: Path, *, models=None, _log=print) -> None:
    from ev_mdt.plots.trip_duration import compute_trip_durations, trip_duration_figure
    _log("Sampling trip durations…")
    durs = compute_trip_durations(models=models)
    _save_fig(trip_duration_figure(durs), out_base / "trip_duration" / "trip_duration_by_model.png", _log=_log)


def _export_model_trajectories(out_base: Path, *, mobility_model: str, price_model: str,
                               n: int, seed: int, _log=print) -> None:
    from ev_mdt.analysis.sensitivity import model_trajectory_figure
    fig = model_trajectory_figure(mobility_model=mobility_model, price_model=price_model,
                                  n=n, seed=seed, _log=_log)
    _save_fig(fig, out_base / "model_trajectories" / "model_trajectories.png", _log=_log)


def _export_price_models(out_base: Path, *, n_days: int, season, daytype: str, seed: int,
                         _log=print) -> None:
    from ev_mdt.analysis.sensitivity import price_model_figures
    _, figs = price_model_figures(n_days=n_days, season=season, daytype=daytype, seed=seed, _log=_log)
    for name, fig in figs.items():
        _save_fig(fig, out_base / "price_models" / f"{name}.png", _log=_log)


def _export_fit_mdn(out_base: Path, *, n_components: int, hidden, epochs: int,
                    batch_size: int, lr: float) -> None:
    from tqdm import tqdm
    from ev_mdt.analysis.sensitivity import fit_mdn
    bar = tqdm(total=100, desc="Training MDN", unit="%", bar_format="{l_bar}{bar}| {n:.0f}%  {postfix}")

    def _progress(frac: float, msg: str) -> None:
        bar.n = int(frac * 100)
        bar.set_postfix_str(msg[:50])
        bar.refresh()

    _, _history, figs = fit_mdn(n_components=n_components, hidden_dims=hidden, epochs=epochs,
                                batch_size=batch_size, lr=lr, _log=tqdm.write, _progress=_progress)
    bar.n = 100; bar.refresh(); bar.close()
    for name, fig in figs.items():
        _save_fig(fig, out_base / "fit_mdn" / f"{name}.png", _log=tqdm.write)


# ── Command handlers ───────────────────────────────────────────────────────────

def cmd_baseline(args) -> None:
    from tqdm import tqdm
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_baseline(out_base, N_e=args.N_e, compute_cost=not args.no_cost, _log=tqdm.write)
    print(f"\nDone → {out_base}/baseline/")


def cmd_sensitivity(args) -> None:
    from tqdm import tqdm
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES
    dims = list(ALL_SWEEP_NAMES) if args.all else args.dims
    if not dims:
        print("Nothing to do: pass --all or --dims <a,b,…>.", file=sys.stderr)
        sys.exit(1)
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_sensitivity(out_base, dims=dims, N_e=args.N_e, compute_cost=not args.no_cost, _log=tqdm.write)
    print(f"\nDone → {out_base}/sensitivity/")


def cmd_calibrate_du(args) -> None:
    from tqdm import tqdm
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_calibrate_du(out_base, N_e=args.N_e, gamma_step=args.gamma_step, step_kwh=args.step,
                         use_reserve=not args.no_reserve, _log=tqdm.write)
    print(f"\nDone → {out_base}/calibrate_du/")


def cmd_trip_duration(args) -> None:
    from tqdm import tqdm
    models = [_MOBILITY_KEYS[k] for k in args.models] if args.models else None
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_trip_duration(out_base, models=models, _log=tqdm.write)
    print(f"\nDone → {out_base}/trip_duration/")


def cmd_model_trajectories(args) -> None:
    from tqdm import tqdm
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_model_trajectories(out_base, mobility_model=_MOBILITY_MODEL_LABEL[args.mobility_model],
                               price_model=args.price_model, n=args.n, seed=args.seed, _log=tqdm.write)
    print(f"\nDone → {out_base}/model_trajectories/")


def cmd_price_models(args) -> None:
    from tqdm import tqdm
    season = None if args.season == "all" else args.season
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_price_models(out_base, n_days=args.n_days, season=season, daytype=args.daytype,
                         seed=args.seed, _log=tqdm.write)
    print(f"\nDone → {out_base}/price_models/")


def cmd_fit_mdn(args) -> None:
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    _export_fit_mdn(out_base, n_components=args.n_components, hidden=args.hidden, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr)
    print(f"\nDone → {out_base}/fit_mdn/")


def cmd_run(args) -> None:
    """Aggregate export. ``--all`` runs every command; ``--all-paper`` runs the fast
    paper-figure subset (heatmaps/borders/trajectories/prices, no exact cost bars,
    no calibration/training)."""
    from tqdm import tqdm
    if not (args.all or args.all_paper):
        print("Nothing to do: pass --all or --all-paper.", file=sys.stderr)
        sys.exit(1)
    compute_cost = bool(args.all)
    out_base = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    log = tqdm.write

    log("[1] Baseline figures…")
    _export_baseline(out_base, N_e=args.N_e, compute_cost=compute_cost, _log=log)
    log("[2] Sensitivity figures (all sweeps)…")
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES
    _export_sensitivity(out_base, dims=list(ALL_SWEEP_NAMES), N_e=args.N_e,
                        compute_cost=compute_cost, _log=log)
    log("[3] Trip-duration figure…")
    _export_trip_duration(out_base, _log=log)
    log("[4] Model rollout trajectories…")
    _export_model_trajectories(out_base, mobility_model=_MOBILITY_MODEL_LABEL["baseline"],
                               price_model="Gaussian (parametric)", n=args.n, seed=args.seed, _log=log)
    log("[5] Price-model comparison…")
    _export_price_models(out_base, n_days=args.n_days, season=None, daytype="all", seed=args.seed, _log=log)

    if args.all:
        log("[6] Departure-Urgency calibration…")
        _export_calibrate_du(out_base, N_e=args.N_e, gamma_step=0.1, step_kwh=5.0,
                             use_reserve=True, _log=log)
        log("[7] Fit MDN + training curves…")
        _export_fit_mdn(out_base, n_components=3, hidden=None, epochs=200, batch_size=1024, lr=1e-3)

    print(f"\nRun complete → {out_base}/")


# ── Argument parsing ────────────────────────────────────────────────────────────

def _add_out_args(p: argparse.ArgumentParser, default: str = "export") -> None:
    p.add_argument("--out-dir", default=default,
                   help="Export base dir; outputs go under <dir>/<timestamp>/<command>/")
    p.add_argument("--no-timestamp", action="store_true",
                   help="Write directly to --out-dir instead of a timestamped subfolder")


def _dims_type(value: str) -> list[str]:
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES
    dims = [d.strip() for d in value.split(",") if d.strip()]
    bad = [d for d in dims if d not in ALL_SWEEP_NAMES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown sweep(s) {bad}; choose from {', '.join(ALL_SWEEP_NAMES)}")
    return dims


def main() -> None:
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES

    parser = argparse.ArgumentParser(prog="ev_mdt", description="EV Charging MDP CLI")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # baseline
    p = sub.add_parser("baseline", help="Baseline all-policy heatmaps, charge borders, exact cost bar")
    p.add_argument("--N-e", type=int, default=500, metavar="N", help="Battery grid points")
    p.add_argument("--no-cost", action="store_true",
                   help="Skip the (slow) exact expected-cost bar + table")
    _add_out_args(p)

    # sensitivity
    p = sub.add_parser("sensitivity",
                       help="Per-sweep BI/DU/BLU heatmaps + charge borders + exact cost bars")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run every sweep dimension")
    g.add_argument("--dims", type=_dims_type, default=None, metavar="A,B,…",
                   help=f"Comma-separated sweep dimensions: {', '.join(ALL_SWEEP_NAMES)}")
    p.add_argument("--N-e", type=int, default=500, metavar="N", help="Battery grid points")
    p.add_argument("--no-cost", action="store_true",
                   help="Skip the (slow) exact expected-cost bars + tables (heatmaps/borders only)")
    _add_out_args(p)

    # calibrate-du
    p = sub.add_parser("calibrate-du", help="Departure-Urgency e_base + γ calibration sweeps (exact)")
    p.add_argument("--N-e", type=int, default=500, metavar="N", help="Battery grid points")
    p.add_argument("--step", type=float, default=5.0, metavar="kWh",
                   help="Target-ceiling (e_base) step size in kWh")
    p.add_argument("--gamma-step", type=float, default=0.1, metavar="STEP", help="γ step size")
    p.add_argument("--no-reserve", action="store_true", help="Disable the DU reserve floor")
    _add_out_args(p)

    # trip-duration
    p = sub.add_parser("trip-duration", help="Trip-duration distribution figure")
    p.add_argument("--models", nargs="+", choices=list(_MOBILITY_KEYS), default=None,
                   help="Mobility models to include (default: all three)")
    _add_out_args(p)

    # model-trajectories
    p = sub.add_parser("model-trajectories",
                       help="Mean sampled price + fraction-driving (swap price / mobility model)")
    p.add_argument("--mobility-model", choices=list(_MOBILITY_MODEL_LABEL), default="baseline",
                   help="Mobility model (default: baseline)")
    p.add_argument("--price-model",
                   choices=["Gaussian (parametric)", "Gaussian Bins", "GMM", "MDN"],
                   default="Gaussian (parametric)", help="Price model (default: Gaussian parametric)")
    p.add_argument("--n", type=int, default=1000, help="Number of sampled scenarios")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    _add_out_args(p)

    # price-models
    p = sub.add_parser("price-models", help="Mean/std diurnal price comparison across price models")
    p.add_argument("--n-days", type=int, default=1000, help="Simulated days")
    p.add_argument("--season", default="all",
                   choices=["all", "spring", "summer", "autumn", "winter"])
    p.add_argument("--daytype", default="all", choices=["all", "weekday", "weekend"])
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    _add_out_args(p)

    # fit-mdn
    p = sub.add_parser("fit-mdn", help="Fit the MDN price sampler + two training-curve figures")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="+", default=None, metavar="DIM",
                   help="Hidden layer sizes, e.g. --hidden 128 128 (default: model default)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    _add_out_args(p)

    # run (aggregator)
    p = sub.add_parser("run", help="Run every command (--all) or the fast paper subset (--all-paper)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run all seven commands (full regeneration)")
    g.add_argument("--all-paper", action="store_true",
                   help="Fast paper figures only: heatmaps/borders/trajectories/prices (no exact cost bars, "
                        "no calibration/training)")
    p.add_argument("--N-e", type=int, default=500, metavar="N", help="Battery grid points")
    p.add_argument("--n", type=int, default=1000, help="Trajectory scenarios")
    p.add_argument("--n-days", type=int, default=1000, help="Price-model simulated days")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    _add_out_args(p)

    args = parser.parse_args()

    handlers = {
        "baseline": cmd_baseline,
        "sensitivity": cmd_sensitivity,
        "calibrate-du": cmd_calibrate_du,
        "trip-duration": cmd_trip_duration,
        "model-trajectories": cmd_model_trajectories,
        "price-models": cmd_price_models,
        "fit-mdn": cmd_fit_mdn,
        "run": cmd_run,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
