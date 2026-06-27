"""Train the MDNSampler on ENTSO-E price data and optionally log to W&B.

Usage
-----
    uv run python scripts/train_mdn.py
    uv run python scripts/train_mdn.py --epochs 500 --n-components 8 --hidden 256 256
    uv run python scripts/train_mdn.py --wandb-project au-mdt --run-name my-run
"""

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ev_mdt.pricing.entsoe import load_prices
from ev_mdt.pricing.samplers import MDNSampler
from ev_mdt.plots.sensitivity import figure_to_png
from ev_mdt.plots.viz import SWEEP_PALETTE

ROOT = Path(__file__).parent.parent.parent


class _HistoryRecorder:
    """Drop-in W&B mock that records per-epoch metrics for local plotting."""

    def __init__(self):
        self._rows: list[dict] = []

    def log(self, d: dict, step: int | None = None) -> None:
        self._rows.append({"step": step, **d})

    def finish(self) -> None:
        pass

    @property
    def history(self) -> list[dict]:
        return self._rows


def _fig_nll(history: list[dict]) -> go.Figure:
    steps = [r["step"] for r in history]
    nll   = [r["loss_original_space"] for r in history]
    fig = go.Figure(go.Scatter(
        x=steps, y=nll, mode="lines",
        line=dict(color=SWEEP_PALETTE[0], width=1.8),
        hovertemplate="Epoch %{x}<br>NLL %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Epoch",
        yaxis_title="Neg. log-likelihood (original space)",
        height=420,
        margin=dict(l=70, r=30, t=40, b=50),
    )
    return fig


def _fig_components(history: list[dict], n_components: int) -> go.Figure:
    steps = [r["step"] for r in history]
    fig = go.Figure()
    for k in range(n_components):
        key = f"pi_{k}"
        weights = [r[key] for r in history if key in r]
        fig.add_trace(go.Scatter(
            x=steps, y=weights, mode="lines",
            name=f"Component {k}",
            line=dict(color=SWEEP_PALETTE[k % len(SWEEP_PALETTE)], width=1.8),
            hovertemplate=f"Component {k}<br>Epoch %{{x}}<br>π=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Epoch",
        yaxis_title="Mean mixture weight π_k",
        height=420,
        margin=dict(l=70, r=30, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="Train MDNSampler on ENTSO-E price data.")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--n-components", type=int,   default=3)
    p.add_argument("--hidden",       type=int,   nargs="+", default=[128, 128],
                   metavar="DIM", help="Hidden layer sizes, e.g. --hidden 256 256")
    p.add_argument("--batch-size",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--wandb-project", type=str,  default=None)
    p.add_argument("--run-name",      type=str,  default=None)
    p.add_argument("--out-dir",       type=Path, default=ROOT,
                   help="Directory to write training-curve PNGs (default: repo root)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading ENTSO-E price data…")
    df = load_prices(_log=print)
    print(f"Loaded {len(df):,} samples.\n")

    sampler = MDNSampler(
        n_components=args.n_components,
        hidden_dims=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    recorder = _HistoryRecorder()

    wandb_run = recorder  # always record locally
    if args.wandb_project:
        try:
            import wandb

            class _Both:
                """Forwards log/finish calls to both W&B and the local recorder."""
                def log(self, d, step=None):
                    recorder.log(d, step)
                    _wb.log(d, step=step)
                def finish(self):
                    _wb.finish()

            _wb = wandb.init(
                project=args.wandb_project,
                name=args.run_name or None,
                config=dict(
                    n_components=args.n_components,
                    hidden_dims=args.hidden,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                ),
            )
            wandb_run = _Both()
        except ImportError:
            print("wandb not installed — run `uv add wandb` to enable logging.")

    def progress(fraction, message):
        bar = "#" * int(fraction * 30)
        print(f"\r[{bar:<30}] {message}", end="", flush=True)

    print(f"Training MDN: {args.n_components} components, hidden={args.hidden}, "
          f"epochs={args.epochs}, lr={args.lr}")
    sampler.fit(df, _progress=progress, _wandb_run=wandb_run)
    print()  # newline after progress bar

    wandb_run.finish()

    # ── Export training curves ─────────────────────────────────────────────────
    history = recorder.history
    if history:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        nll_path = out_dir / "mdn_nll.png"
        nll_path.write_bytes(figure_to_png(_fig_nll(history)))
        print(f"Saved NLL curve:        {nll_path}")

        comp_path = out_dir / "mdn_components.png"
        comp_path.write_bytes(figure_to_png(_fig_components(history, args.n_components)))
        print(f"Saved component curve:  {comp_path}")

    print("Done.")


if __name__ == "__main__":
    main()
