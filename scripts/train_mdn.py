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

sys.path.insert(0, str(Path(__file__).parent.parent))

from pricing_models.entsoe_loader import load_prices
from pricing_models.pricing import MDNSampler


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

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
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
        except ImportError:
            print("wandb not installed — run `uv add wandb` to enable logging.")

    def progress(fraction, message):
        bar = "#" * int(fraction * 30)
        print(f"\r[{bar:<30}] {message}", end="", flush=True)

    print(f"Training MDN: {args.n_components} components, hidden={args.hidden}, "
          f"epochs={args.epochs}, lr={args.lr}")
    sampler.fit(df, _progress=progress, _wandb_run=wandb_run)
    print()  # newline after progress bar

    if wandb_run is not None:
        wandb_run.finish()

    print("Done.")


if __name__ == "__main__":
    main()
