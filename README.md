# Optimal EV Charging under Stochastic Departure Times and Prices

Backward-induction solver for an EV charging Markov Decision Process (MDP) with stochastic mobility and real electricity prices from ENTSO-E. All model logic and figure generation live in the `ev_mdt` package; the CLI and the Streamlit app are thin layers that call into it.

Policy costs are computed **analytically (exact)** by evaluating each solved policy — there is no Monte-Carlo cost estimation. Monte-Carlo sampling is used only to visualise price/mobility trajectories and trip-duration distributions.

## Setup

```bash
pip install -e .
```

Requires Python ≥ 3.13. An ENTSO-E API key is needed for live price data — set it via one of:

```bash
export ENTSOE_API_KEY=your_key          # environment variable
# or add to .streamlit/secrets.toml:
# [entsoe]
# api_key = "your_key"
```

## CLI

Quick reference:

| Command | Purpose |
|---------|---------|
| `baseline` | Baseline all-policy heatmaps, charge borders, exact cost bar (+ table) |
| `sensitivity` | Per-sweep BI/DU/BLU heatmaps + charge borders + exact cost bars (+ tables) |
| `calibrate-du` | Departure-Urgency `e_base` + `γ` calibration sweeps |
| `trip-duration` | Trip-duration distribution figure |
| `model-trajectories` | Mean sampled price + fraction-driving (swap price / mobility model) |
| `price-models` | Mean/std diurnal price comparison across price models |
| `fit-mdn` | Fit the MDN price sampler + two training-curve figures |
| `run --all` | Run every command (full regeneration) |
| `run --all-paper` | Fast paper figures only (no exact cost, calibration or training) |

All commands are available as `python -m ev_mdt <command>` or `ev_mdt <command>` after install. Each writes its figures (and CSV tables, where applicable) under `<out-dir>/<timestamp>/<command>/`. Pass `--no-timestamp` to write straight into `<out-dir>/<command>/`. The default `--out-dir` is `export`.

| Command | Output |
|---------|--------|
| `baseline` | All-policy heatmaps + charge borders (one figure each, all policies), exact expected-cost bar, and a summary table — for the baseline model |
| `sensitivity` | Per-sweep heatmaps and charge borders (one figure each for **Backward Induction / Departure Urgency / Battery Level Urgency**, subplots per swept value), exact cost bars, and summary tables |
| `calibrate-du` | Departure-Urgency calibration: the `e_base` (target-ceiling) and `γ` sweeps, with figures + CSVs |
| `trip-duration` | Trip-duration density + survival figure |
| `model-trajectories` | Two-subplot figure: mean sampled price + mean fraction-driving |
| `price-models` | Mean / std diurnal price-profile comparison across price models |
| `fit-mdn` | Fits the MDN price sampler and exports its two training-curve figures (NLL, mixture weights) |
| `run --all` | Runs every command above (full regeneration) |
| `run --all-paper` | Fast paper-figure subset only (heatmaps/borders/trajectories/prices; **no** exact cost bars, no calibration/training) |

### `baseline`

```bash
python -m ev_mdt baseline [--N-e 500] [--no-cost] [--out-dir export]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--N-e` | `500` | Battery grid points |
| `--no-cost` | — | Skip the (slow) exact expected-cost bar + table |

### `sensitivity`

```bash
python -m ev_mdt sensitivity --all [--N-e 500] [--no-cost]
python -m ev_mdt sensitivity --dims penalty,horizon [--N-e 500]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--all` | — | Run every sweep dimension |
| `--dims` | — | Comma-separated subset: `pricing_model`, `pricing_season`, `pricing_daytype`, `pricing_crisis`, `penalty`, `horizon`, `departure_profile`, `mobility_model` |
| `--N-e` | `500` | Battery grid points |
| `--no-cost` | — | Skip the (slow) exact cost bars + tables (heatmaps/borders only) |

### `calibrate-du`

```bash
python -m ev_mdt calibrate-du [--N-e 500] [--step 5.0] [--gamma-step 0.1] [--no-reserve]
```

Sweeps the Departure-Urgency target ceiling (`e_base`, the `target_sweep` figure) and the scaling exponent `γ` (`gamma_sweep`, one subplot per mobility model), both via exact backward-pass evaluation.

### `trip-duration`

```bash
python -m ev_mdt trip-duration [--models baseline negbin_fixed negbin_poisson]
```

`--models` selects which mobility models to include (default: all three).

### `model-trajectories`

```bash
python -m ev_mdt model-trajectories [--mobility-model baseline] \
    [--price-model "Gaussian (parametric)"] [--n 1000] [--seed 42]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mobility-model` | `baseline` | One of `baseline`, `negbin_fixed`, `negbin_poisson` |
| `--price-model` | `Gaussian (parametric)` | One of `Gaussian (parametric)`, `Gaussian Bins`, `GMM`, `MDN` |
| `--n` | `1000` | Number of sampled scenarios |

### `price-models`

```bash
python -m ev_mdt price-models [--n-days 1000] [--season all] [--daytype all] [--seed 42]
```

### `fit-mdn`

```bash
python -m ev_mdt fit-mdn [--epochs 200] [--n-components 3] [--hidden 128 128] \
    [--batch-size 1024] [--lr 1e-3]
```

### `run`

```bash
python -m ev_mdt run --all          # full regeneration of every figure (slow)
python -m ev_mdt run --all-paper    # fast paper figures only (no exact cost / calibration / training)
```

## Streamlit app

```bash
streamlit run app/app.py
```

| Page | Description |
|------|-------------|
| **Settings** | Configure all model parameters, choose mobility model and price source, run backward induction |
| **Policy Explorer** | Visualise the optimal and benchmark policies as heatmaps over battery level × time |
| **Model Rollout Trajectories** | Mean sampled price + fraction-driving across scenarios, and the trip-duration distribution |
| **Target Sweep** | Departure-Urgency target-ceiling (`e_base`) calibration — exact expected cost vs ceiling |
| **Sensitivity Analysis** | Per-sweep policy heatmaps, charge borders, exact expected-cost bars and summary tables |
| **Price Explorer** | Fit all pricing models on ENTSO-E data and compare simulated diurnal price profiles |

Figure/table export for the paper is done through the CLI (`run --all`), not the app.

## Models

### Mobility

| Model | Trip duration distribution |
|-------|---------------------------|
| **Baseline** | Geometric — constant per-minute return probability |
| **Negative Binomial (fixed k)** | NB(k, q) — exactly k Erlang phases, mean = k/q min |
| **Negative Binomial (sampled k)** | k ~ Poisson(λ_k) drawn each trip, mean = λ_k/q min |

### Pricing

| Model | Description |
|-------|-------------|
| **Gaussian (parametric)** | Manual time-of-day mean prices with Gaussian noise |
| **Gaussian Bins** | Per-(weekend, hour, season) empirical mean/std from ENTSO-E data |
| **GMM** | Gaussian mixture model conditioned on (weekend, hour, season) |
| **MDN** | Mixture density network trained jointly on all context variables |

Price data: ENTSO-E DK1 (West Denmark) day-ahead prices, 2015–present. Requires `ENTSOE_API_KEY`.

## Package layout

```
ev_mdt/
  params.py              single source of truth for all parameters & solver defaults
  models/                MDP dynamics, backward induction, policies, rollout engine
  pricing/               ENTSO-E loading + price samplers (Gaussian bins, GMM, MDN)
  analysis/
    sensitivity.py       exact-cost evaluation, sweeps, and the figure orchestrators
    prices.py            price-model fitting / simulation / comparison figures
  plots/                 figure factories (sensitivity, calibration, mdn, trip_duration)
  __main__.py            thin CLI that calls the orchestrators and writes figures/tables
app/                     thin Streamlit layer over the same package functions
```
