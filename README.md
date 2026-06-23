# Optimal EV Charging under Stochastic Departure Times and Prices

Backward-induction solver for an EV charging Markov Decision Process (MDP) with stochastic mobility and real electricity prices from ENTSO-E. Includes a Streamlit app for interactive exploration and a CLI for headless runs and figure export.

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

## Streamlit app

```bash
streamlit run app/app.py
```

| Page | Description |
|------|-------------|
| **Settings** | Configure all model parameters, choose mobility model and price source, run backward induction |
| **Policy Explorer** | Visualise the optimal charging policy as a heatmap over battery level × time |
| **Policy Rollout** | Simulate N scenarios and compare backward induction vs always-minimum policy |
| **Backward Induction Steps** | Step through the value function as it propagates backwards in time |
| **Sensitivity Analysis** | Sweep penalty, discount, horizon, departure profiles, pricing models and mobility models |
| **Price Explorer** | Fit all pricing models on ENTSO-E data and compare simulated diurnal price profiles |

## CLI

All commands are available as `python -m ev_mdt <command>` or `ev_mdt <command>` after install.

### `solve` — run backward induction

```bash
python -m ev_mdt solve [--model MODEL] [--N-e N] [--hours H] [--phi Φ] [--beta β]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Baseline` | Mobility model: `Baseline`, `"Negative Binomial trips (fixed k)"`, `"Negative Binomial trips (sampled k)"` |
| `--N-e` | `500` | Battery grid points |
| `--hours` | `24` | Planning horizon (hours) |
| `--phi` | — | Unserved-driving penalty (€/h) |
| `--beta` | — | Discount factor |

```bash
# Examples
python -m ev_mdt solve
python -m ev_mdt solve --model "Negative Binomial trips (fixed k)" --N-e 200 --hours 48
python -m ev_mdt solve --phi 50.0 --beta 0.999
```

### `rollout` — solve then simulate scenarios

```bash
python -m ev_mdt rollout [--model MODEL] [--N-e N] [--hours H] [--phi Φ] [--beta β]
                         [--n N] [--seed SEED]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `200` | Number of rollout scenarios |
| `--seed` | `42` | Random seed |
| *(+ all `solve` flags)* | | |

```bash
python -m ev_mdt rollout --n 500 --seed 0
python -m ev_mdt rollout --model "Negative Binomial trips (fixed k)" --n 200
```

### `sensitivity` — run parameter sweeps

```bash
python -m ev_mdt sensitivity [--sweep SWEEP] [--N-rollouts N] [--N-e N]
                              [--seed SEED] [--out-dir DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sweep` | `all` | Which sweep to run. One of: `pricing_model`, `pricing_season`, `pricing_daytype`, `pricing_crisis`, `penalty`, `beta`, `horizon`, `departure_profile`, `mobility_model`, or `all` |
| `--N-rollouts` | `500` | Rollouts per swept value |
| `--N-e` | `500` | Battery grid points |
| `--seed` | `42` | Random seed |
| `--out-dir` | `figures/` | Output directory for exported PNGs |

```bash
python -m ev_mdt sensitivity                          # all sweeps
python -m ev_mdt sensitivity --sweep penalty
python -m ev_mdt sensitivity --sweep mobility_model --N-rollouts 200 --out-dir out/
```

### `prices` — fit pricing models and plot diurnal profiles

```bash
python -m ev_mdt prices [--n-days N] [--season SEASON] [--daytype DAYTYPE]
                         [--seed SEED] [--out-dir DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-days` | `1000` | Simulated days |
| `--season` | `all` | Filter: `all`, `spring`, `summer`, `autumn`, `winter` |
| `--daytype` | `all` | Filter: `all`, `weekday`, `weekend` |
| `--seed` | `42` | Random seed |
| `--out-dir` | `figures/` | Output directory for exported PNGs |

```bash
python -m ev_mdt prices
python -m ev_mdt prices --season summer --daytype weekday --n-days 2000
python -m ev_mdt prices --out-dir out/price_explorer/
```

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
