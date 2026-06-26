"""CLI entry point for ev_mdt.

Usage
-----
    python -m ev_mdt solve [--model BASELINE] [--N-e 500] [--hours 24] [--phi 1.0] ...
    python -m ev_mdt rollout --n 200 --seed 42 [same solve flags]
    python -m ev_mdt run --all [--N-rollouts 500] [--N-e 500] [--seed 42] [--out-dir export]
    python -m ev_mdt run --sweep penalty        # single sweep, no baseline models
    python -m ev_mdt prices [--n-days 1000] [--season all] [--daytype all]
                            [--seed 42] [--out-dir figures/]

`run --all` writes figures to <out-dir>/<timestamp>/figures_app/ and summary tables to
<out-dir>/<timestamp>/tables/. Pass --no-timestamp to write straight into <out-dir>.
"""
import argparse
import sys


def _timestamped_dir(base, *, enabled: bool = True):
    """Append a ``YYYY-MM-DD_HHMMSS`` run folder to ``base`` (unless disabled)."""
    from pathlib import Path
    base = Path(base)
    if not enabled:
        return base
    from datetime import datetime
    return base / datetime.now().strftime("%Y-%m-%d_%H%M%S")


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


def cmd_run(args: argparse.Namespace) -> None:
    """Run sweeps (and, with --all, the baseline/NegBin models) → export figures + tables."""
    import itertools
    import threading
    import time
    from tqdm import tqdm
    from ev_mdt.analysis.sensitivity import (
        run_all_sweeps, save_figures, save_tables, ALL_SWEEP_NAMES,
    )

    base_dir    = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    figures_dir = base_dir / "figures_app"
    tables_dir  = base_dir / "tables"

    # ── Exact-cost mode: analytical expected BI cost per scenario (no rollouts) ─
    if args.exact_cost:
        from ev_mdt.analysis.sensitivity import exact_bi_cost_table
        print("Computing exact (analytical) BI cost per scenario…", flush=True)
        df = exact_bi_cost_table(N_e=args.N_e, seed=args.seed, _log=tqdm.write)
        tables_dir.mkdir(parents=True, exist_ok=True)
        out = tables_dir / "exact_bi_cost.csv"
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        print(f"\nSaved: {out}")
        return

    # ── Baseline-only mode: just the baseline/NegBin model figures ─────────────
    if args.baseline_only:
        print("Rendering baseline-model figures only…", flush=True)
        saved = save_figures({}, out_dir=figures_dir, N_rollouts=args.N_rollouts,
                             seed=args.seed, N_e=args.N_e, include_baseline=True)
        for p in saved:
            print(f"  Saved: {p}")
        print(f"\nDone. Figures → {figures_dir}/baseline_models/")
        return

    if args.all:
        sweeps = list(ALL_SWEEP_NAMES)
    elif args.sweep:
        sweeps = [args.sweep]
    else:
        print("Nothing to do: pass --all, --baseline-only, or --sweep <name>.", file=sys.stderr)
        sys.exit(1)

    do_baseline = args.all

    # ── W&B setup ─────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or None,
                config={
                    "sweeps":     sweeps,
                    "N_rollouts": args.N_rollouts,
                    "N_e":        args.N_e,
                    "seed":       args.seed,
                },
            )
            tqdm.write(f"W&B run: {wandb_run.url}")
        except ImportError:
            tqdm.write("wandb not installed — run `uv add wandb`. Continuing without logging.")
            wandb_run = None

    outer = tqdm(sweeps, desc="Sweeps", unit="sweep", position=0)
    for i, sw in enumerate(outer):
        outer.set_description(f"Sweep: {sw}")
        inner = tqdm(total=100, desc="  progress", unit="%", position=1,
                     leave=False, bar_format="{l_bar}{bar}| {n:.0f}%  {postfix}")

        # Shared state updated by progress_cb; the heartbeat thread renders it so
        # the bar keeps animating (spinner + elapsed) even while solve() blocks.
        hb = {"msg": "starting…", "since": time.monotonic()}

        def cb(f: float, msg: str, _bar: tqdm = inner) -> None:
            _bar.n = int(f * 100)
            hb["msg"] = msg[:50]
            hb["since"] = time.monotonic()

        stop = threading.Event()

        def heartbeat(_bar: tqdm = inner) -> None:
            spinner = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            while not stop.wait(0.25):
                elapsed = time.monotonic() - hb["since"]
                _bar.set_postfix_str(f"{next(spinner)} {hb['msg']} ({elapsed:4.0f}s)")
                _bar.refresh()

        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()
        try:
            # W&B logs only the MDN fitting curve (during the pricing_model sweep);
            # no sweep metrics or figures are uploaded.
            results = run_all_sweeps(
                N_rollouts=args.N_rollouts, N_e=args.N_e, seed=args.seed,
                sweeps=[sw], progress_cb=cb, _log=tqdm.write, _wandb_run=wandb_run,
            )
        finally:
            stop.set()
            hb_thread.join()
        inner.close()

        # The MDN is fit (and logged) during the pricing_model sweep — finish W&B
        # right after so the run contains only the MDN curve and nothing else.
        if wandb_run is not None and sw == "pricing_model":
            wandb_run.finish()
            wandb_run = None
            tqdm.write("W&B: MDN fitting logged — run finished.")

        saved = save_figures(results, out_dir=figures_dir,
                             N_rollouts=args.N_rollouts, seed=args.seed, N_e=args.N_e,
                             include_baseline=(do_baseline and i == 0))
        saved += save_tables(results, out_dir=tables_dir,
                             N_rollouts=args.N_rollouts, seed=args.seed, N_e=args.N_e,
                             include_baseline=False, _log=tqdm.write)
        for p in saved:
            tqdm.write(f"  Saved: {p}")

    outer.close()

    # Baseline/NegBin model tables (figures already emitted with the first sweep).
    if do_baseline:
        tqdm.write("Writing baseline-model tables…")
        for p in save_tables({}, out_dir=tables_dir, N_rollouts=args.N_rollouts,
                             seed=args.seed, N_e=args.N_e, include_baseline=True,
                             _log=tqdm.write):
            tqdm.write(f"  Saved: {p}")

        # Price-model comparison figures (mean diurnal profile + std), as on the
        # Price Explorer page.
        tqdm.write("Fitting price models + rendering comparison…")
        from ev_mdt.pricing.entsoe import load_prices
        from ev_mdt.analysis.prices import fit_samplers, simulate_price_paths, price_figures
        from ev_mdt.plots.sensitivity import figure_to_png
        _df = load_prices(_log=tqdm.write)
        _samplers = fit_samplers(_df)
        _px = simulate_price_paths(_samplers, n_days=1000, seed=args.seed)
        _fig_mean, _fig_std = price_figures(_px)
        px_dir = figures_dir / "price_explorer"
        px_dir.mkdir(parents=True, exist_ok=True)
        for _nm, _fig in (("mean_profile", _fig_mean), ("std_profile", _fig_std)):
            _p = px_dir / f"{_nm}.png"
            _p.write_bytes(figure_to_png(_fig))
            tqdm.write(f"  Saved: {_p}")

    print("\nRun complete.")
    print(f"  Figures → {figures_dir}/")
    print(f"  Tables  → {tables_dir}/")
    if wandb_run is not None:
        wandb_run.finish()


def cmd_target_sweep(args: argparse.Namespace) -> None:
    from tqdm import tqdm
    from ev_mdt.analysis.sensitivity import sweep_target_ceiling
    from ev_mdt.plots.sensitivity import figure_to_png

    out_dir = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = sweep_target_ceiling(
        model_label=args.model,
        N_rollouts=args.N_rollouts,
        seed=args.seed,
        step_kwh=args.step,
    )

    import pandas as pd
    import numpy as np
    df = pd.DataFrame(rows)
    best = df.loc[df["mean_cost"].idxmin()]
    tqdm.write(f"\nBest ceiling: {best['target_kwh']:.0f} kWh "
               f"({best['target_frac']:.0%}) — mean cost {best['mean_cost']:.4f} €")

    # Save table
    csv_path = out_dir / "target_sweep.csv"
    df.rename(columns={
        "target_kwh":   "Target ceiling (kWh)",
        "target_frac":  "Target ceiling (%)",
        "mean_cost":    "Mean cost (€)",
        "std_cost":     "Std cost (€)",
        "mean_penalty": "Mean penalty (min)",
        "mean_charged": "Mean charged (kWh)",
    }).to_csv(csv_path, index=False)
    tqdm.write(f"Saved table → {csv_path}")

    # Save plot
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        x = df["target_kwh"]
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x.iloc[::-1]]),
            y=pd.concat([df["mean_cost"] + df["std_cost"],
                         (df["mean_cost"] - df["std_cost"]).iloc[::-1]]),
            fill="toself", fillcolor="rgba(68,119,170,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="Total ±1 std", hoverinfo="skip", yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_cost"], mode="lines+markers",
            line=dict(color="#4477AA", width=2), marker=dict(size=7), name="Total cost", yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_penalty_cost"], mode="lines+markers",
            line=dict(color="#CC3311", width=2, dash="dash"), marker=dict(size=6),
            name="Penalty cost", yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=[best["target_kwh"]], y=[best["mean_cost"]],
            mode="markers", marker=dict(color="#EE6677", size=12, symbol="star"),
            name=f"Best: {best['target_kwh']:.0f} kWh ({best['target_frac']:.0%})", yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_charge_cost"], mode="lines+markers",
            line=dict(color="#228833", width=2, dash="dot"), marker=dict(size=6),
            name="Charging cost (right)", yaxis="y2",
        ))
        fig.update_layout(
            xaxis_title="Target ceiling (kWh)",
            yaxis=dict(title="Total / penalty cost (€)"),
            yaxis2=dict(title="Charging cost (€)", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
            template="plotly_white", height=500,
        )
        png_path = out_dir / "target_sweep.png"
        png_path.write_bytes(figure_to_png(fig))
        tqdm.write(f"Saved plot  → {png_path}")
    except Exception as e:
        tqdm.write(f"Could not save plot (kaleido missing?): {e}")


def cmd_gamma_sweep(args: argparse.Namespace) -> None:
    import pandas as pd
    from tqdm import tqdm
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from ev_mdt.analysis.sensitivity import sweep_gamma
    from ev_mdt.plots.sensitivity import figure_to_png

    out_dir = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = sweep_gamma(
        N_rollouts=args.N_rollouts,
        seed=args.seed,
        use_reserve=not args.no_reserve,
    )

    # Save CSV per model
    for model_name, rows in results.items():
        slug = model_name.lower().replace(" ", "_").replace("=", "")
        csv_path = out_dir / f"gamma_sweep_{slug}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        tqdm.write(f"Saved {csv_path}")

    # Build stacked figure: one subplot per model, dual y-axes
    model_names = list(results.keys())
    n_models = len(model_names)
    fig = make_subplots(
        rows=n_models, cols=1,
        shared_xaxes=True,
        subplot_titles=model_names,
        vertical_spacing=0.10,
    )
    colors = {"total": "#4477AA", "charging": "#228833", "penalty": "#CC3311"}
    for row_i, model_name in enumerate(model_names, start=1):
        df   = pd.DataFrame(results[model_name])
        x    = df["gamma"]
        show = row_i == 1

        fig.add_trace(go.Scatter(
            x=x, y=df["mean_cost"], mode="lines+markers",
            line=dict(color=colors["total"], width=2), marker=dict(size=6),
            name="Total cost", legendgroup="total", showlegend=show,
        ), row=row_i, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_charge_cost"], mode="lines+markers",
            line=dict(color=colors["charging"], width=2, dash="dot"), marker=dict(size=5),
            name="Charging cost", legendgroup="charging", showlegend=show,
        ), row=row_i, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["mean_penalty_cost"], mode="lines+markers",
            line=dict(color=colors["penalty"], width=2, dash="dash"), marker=dict(size=5),
            name="Penalty cost", legendgroup="penalty", showlegend=show,
        ), row=row_i, col=1)

        best = df.loc[df["mean_cost"].idxmin()]
        fig.add_trace(go.Scatter(
            x=[best["gamma"]], y=[best["mean_cost"]],
            mode="markers", marker=dict(color="#EE6677", size=11, symbol="star"),
            name=f"Best γ={best['gamma']:.1f}", legendgroup=f"best_{row_i}",
            showlegend=show,
        ), row=row_i, col=1)

        fig.update_yaxes(title_text="Cost (€)", row=row_i, col=1)

        tqdm.write(f"{model_name}: best γ={best['gamma']:.1f}  "
                   f"(target={best['target_kwh']:.1f} kWh, cost={best['mean_cost']:.4f} €)")

    fig.update_xaxes(title_text="γ", row=n_models, col=1)
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=320 * n_models,
        margin=dict(l=70, r=80, t=60, b=40),
    )

    try:
        png_path = out_dir / "gamma_sweep.png"
        png_path.write_bytes(figure_to_png(fig))
        tqdm.write(f"Saved plot → {png_path}")
    except Exception as e:
        tqdm.write(f"Could not save plot: {e}")


def cmd_prices(args: argparse.Namespace) -> None:
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

    out_dir = _timestamped_dir(args.out_dir, enabled=not args.no_timestamp) / "price_explorer"
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

    # run (sweeps + baseline/NegBin models → figures + tables)
    from ev_mdt.analysis.sensitivity import ALL_SWEEP_NAMES
    p_run = sub.add_parser("run", help="Run sweeps + model rollouts, export figures and tables")
    p_run.add_argument("--all", action="store_true",
                       help="Run every sweep plus the baseline/NegBin models (full export)")
    p_run.add_argument("--baseline-only", action="store_true",
                       help="Only render the baseline/NegBin model figures (no sweeps)")
    p_run.add_argument("--exact-cost", action="store_true",
                       help="Analytical exact expected BI cost per scenario (no rollouts) → tables/exact_bi_cost.csv")
    p_run.add_argument("--sweep", default=None,
                       choices=ALL_SWEEP_NAMES, metavar="SWEEP",
                       help=f"Run a single sweep (no baseline models). Options: {', '.join(ALL_SWEEP_NAMES)}")
    p_run.add_argument("--N-rollouts",    type=int, default=500, metavar="N",
                       help="Rollouts per swept value")
    p_run.add_argument("--N-e",           type=int, default=500, metavar="N",
                       help="Battery grid points")
    p_run.add_argument("--seed",          type=int, default=42,  help="Base random seed")
    p_run.add_argument("--out-dir",       default="export",
                       help="Export base dir; outputs go under <dir>/<timestamp>/{figures_app,tables}")
    p_run.add_argument("--no-timestamp",  action="store_true",
                       help="Write directly to --out-dir instead of a timestamped subfolder")
    p_run.add_argument("--wandb",         action="store_true",   help="Log results and figures to Weights & Biases")
    p_run.add_argument("--wandb-project", default="au-mdt",      help="W&B project name")
    p_run.add_argument("--wandb-run",     default="",            help="W&B run name (auto if omitted)")

    # target-sweep
    from ev_mdt.params import MODEL_LABELS
    p_ts = sub.add_parser("target-sweep",
                           help="Sweep Departure Urgency target ceiling and export cost plot + CSV")
    p_ts.add_argument("--model",       default="Baseline", choices=MODEL_LABELS)
    p_ts.add_argument("--N-rollouts",  type=int,   default=500,   metavar="N")
    p_ts.add_argument("--seed",        type=int,   default=42)
    p_ts.add_argument("--step",        type=float, default=5.0,   metavar="kWh",
                      help="Target ceiling step size in kWh")
    p_ts.add_argument("--out-dir",     default="export/target_sweep",
                      help="Output base dir; outputs go under <dir>/<timestamp>/")
    p_ts.add_argument("--no-timestamp", action="store_true",
                      help="Write directly to --out-dir instead of a timestamped subfolder")

    # gamma-sweep
    p_gs = sub.add_parser("gamma-sweep",
                           help="Sweep ceiling scaling exponent γ across three mobility models")
    p_gs.add_argument("--N-rollouts",  type=int,   default=500,   metavar="N")
    p_gs.add_argument("--seed",        type=int,   default=42)
    p_gs.add_argument("--no-reserve",  action="store_true")
    p_gs.add_argument("--out-dir",     default="export/gamma_sweep",
                      help="Output base dir; outputs go under <dir>/<timestamp>/")
    p_gs.add_argument("--no-timestamp", action="store_true",
                      help="Write directly to --out-dir instead of a timestamped subfolder")

    # prices
    p_prices = sub.add_parser("prices", help="Fit price models and simulate diurnal profiles")
    p_prices.add_argument("--n-days",  type=int,   default=1000, help="Simulated days")
    p_prices.add_argument("--season",  default="all",
                          choices=["all", "spring", "summer", "autumn", "winter"])
    p_prices.add_argument("--daytype", default="all",
                          choices=["all", "weekday", "weekend"])
    p_prices.add_argument("--seed",    type=int,   default=42,   help="Random seed")
    p_prices.add_argument("--out-dir", default="figures/",
                          help="Output base dir; outputs go under <dir>/<timestamp>/price_explorer/")
    p_prices.add_argument("--no-timestamp", action="store_true",
                          help="Write directly to --out-dir instead of a timestamped subfolder")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "rollout":
        cmd_rollout(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "gamma-sweep":
        cmd_gamma_sweep(args)
    elif args.command == "target-sweep":
        cmd_target_sweep(args)
    elif args.command == "prices":
        cmd_prices(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
