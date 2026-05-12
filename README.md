# AU-MDT — EV Charging MDP

Optimal EV charging control via Markov Decision Process (MDP) solved with backward induction.

## Overview

The model decides how fast to charge an electric vehicle at each minute of the day, trading off electricity price against the risk of the car needing to drive with insufficient battery. The state is `(battery level, parking/driving status)` and the action is the charging rate `u ∈ [u_min, u_max]` (kW).

Key features:
- Stochastic driving-state transitions (parked ↔ driving)
- Time-varying electricity prices
- Penalty for unserved driving demand
- Backward induction over a 1440-minute (24 h) horizon

## Files

| File | Description |
|---|---|
| `backward_induction.py` | Core solver — backward induction algorithm |
| `baseline_mdp.py` | Marimo notebook — parameters, simulation, and visualisations |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install marimo numpy matplotlib
```

## Running

Open the interactive notebook:

```bash
marimo edit baseline_mdp.py
```

Or run the solver directly:

```python
from backward_induction import backward_induction
from baseline_mdp import MDPParams

params = MDPParams()
policy, value = backward_induction(params, mean_price_fn, transition_probs_fn, consumption_fn)
```
