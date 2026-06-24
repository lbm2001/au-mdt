"""Sensitivity analysis sweep logic for the EV charging MDP.

All computation lives here; no Streamlit dependency.  The app page and the CLI
both call these functions and pass the returned results to ev_mdt.plots.sensitivity.

Sweep functions
---------------
Each sweep_* function returns a list of "step result" dicts, one per swept value.
A step result is a plain dict and contains:

    model         str  — mobility model label (BASELINE_MODEL / NEGBIN_*)
    label         str  — human-readable swept value (e.g. "1000 €/h")
    params        params object
    pbp_fn        callable t -> (K,) price bin probs
    pi            ndarray (T, n_chi, N_e, K)
    actions       ndarray
    e_grid        ndarray
    lam_grid      ndarray
    T             int   — horizon in minutes
    rollouts      dict  — {policy_name: [metrics_dict, ...]}
    sample_rollout dict — single trajectory of the BI policy (first scenario)

Standalone end-to-end usage
----------------------------
    from ev_mdt.analysis.sensitivity import run_all_sweeps, save_figures
    results = run_all_sweeps(N_rollouts=200, N_e=200, seed=42)
    save_figures(results, out_dir="figures/")
"""
from pathlib import Path
from typing import Callable

import numpy as np

from ev_mdt.params import (
    BaselineParams, NegBinParams,
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL,
)
from ev_mdt.models.common.backward_induction import (
    backward_induction as _backward_induction, evaluate_policy as _evaluate_policy,
)
from ev_mdt.models.baseline.model import transition_matrix as _baseline_tm
from ev_mdt.models.baseline.rollout import simulate_policy_rollout as _baseline_rollout
from ev_mdt.models.negbin.model import transition_matrix as _negbin_tm
from ev_mdt.models.negbin.rollout import simulate_policy_rollout as _negbin_rollout
from ev_mdt.models.common.model_utils import price_bin_probs as _gaussian_pbp, mean_price
from ev_mdt.models.common.policies import backward_induction_policy, policy_registry
from ev_mdt.models.common.rollout_utils import rollout_metrics, run_policies
from ev_mdt.plots.viz import POLICY_ORDER


# ── Sweep constants (single source of truth) ───────────────────────────────────

PHI_VALUES       = [0, 0.05, 1, 50, 500, 5000]
BETA_VALUES      = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
HORIZON_HOURS    = [24, 48, 168]
CRISIS_YEARS     = (2021, 2022, 2023)
NEGBIN_LAMBDA_K  = 5.0

DEPARTURE_PROFILES = {
    "Single morning trip": dict(p_pd_morning=0.060, p_pd_lunch=0.000, p_pd_evening=0.000, p_pd_default=0.0005),
    "Stay-at-home":        dict(p_pd_morning=0.002, p_pd_lunch=0.001, p_pd_evening=0.002, p_pd_default=0.0005),
    "All-day errands":     dict(p_pd_morning=0.015, p_pd_lunch=0.015, p_pd_evening=0.015, p_pd_default=0.0150),
}

ALL_SWEEP_NAMES = [
    "pricing_model",
    "pricing_season",
    "pricing_daytype",
    "pricing_crisis",
    "penalty",
    "beta",
    "horizon",
    "departure_profile",
    "mobility_model",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def build_params(model_label: str, **overrides):
    """Build the params object for the selected mobility model, applying field overrides."""
    if model_label == BASELINE_MODEL:
        return BaselineParams(**overrides)
    if model_label == NEGBIN_SAMPLED_MODEL:
        return NegBinParams(**overrides, lambda_k=NEGBIN_LAMBDA_K,
                            k=NegBinParams.k_max_for_lambda(NEGBIN_LAMBDA_K))
    return NegBinParams(**overrides)


def rollout_fn(model_label: str):
    """Return the model-specific simulate_policy_rollout."""
    return _baseline_rollout if model_label == BASELINE_MODEL else _negbin_rollout


def solve(model_label: str, params, pbp_fn, T: int, N_e: int):
    """Run backward induction for the model; returns (pi, actions, e_grid, lam_grid)."""
    if model_label == BASELINE_MODEL:
        tm_fn, n_chi = (lambda t: _baseline_tm(t, params)), 2
    else:
        tm_fn, n_chi = (lambda t: _negbin_tm(t, params)), params.k + 1
    _, pi, actions, e_grid, lam_grid = _backward_induction(
        params, transition_matrix_fn=tm_fn, price_bin_probs_fn=pbp_fn,
        n_chi=n_chi, T=T, N_e=N_e,
    )
    return pi, actions, e_grid, lam_grid


def bi_expected_cost(result: dict, beta: float = 1.0) -> float:
    """Exact expected cost of the optimal (Backward Induction) policy for a config.

    Evaluates the solved policy `pi` (from a solve()/sweep-step/baseline result dict)
    analytically and averages over the rollout initial-state distribution:
    χ₀=0, e₀ ~ Uniform[e_min,e_max] (battery grid), λ̂₀ ~ price marginal at t=0.
    With beta=1 (default) this is the expected *undiscounted* total cost, directly
    comparable to the Monte-Carlo "Mean cost".
    """
    model_label = result["model"]
    params, pbp_fn = result["params"], result["pbp_fn"]
    pi, actions, e_grid, T = result["pi"], result["actions"], result["e_grid"], result["T"]
    if model_label == BASELINE_MODEL:
        tm_fn, n_chi = (lambda t: _baseline_tm(t, params)), 2
    else:
        tm_fn, n_chi = (lambda t: _negbin_tm(t, params)), params.k + 1
    Jpi = _evaluate_policy(
        params, transition_matrix_fn=tm_fn, price_bin_probs_fn=pbp_fn, n_chi=n_chi,
        action_fn=lambda t, chi: actions[pi[t, chi]], T=T, N_e=len(e_grid), beta=beta,
    )
    return float(np.mean(Jpi[0, 0] @ np.asarray(pbp_fn(0))))


def make_scenario(params, seed: int, horizon: int,
                  sampler=None, season: str = "winter", is_weekend: bool = False) -> dict:
    """Generate one rollout scenario with separate sub-seeds for mobility and prices."""
    rng_mob = np.random.default_rng([seed, 0])
    rng_lam = np.random.default_rng([seed, 1])
    rng_e0  = np.random.default_rng([seed, 2])
    mobility_draws = rng_mob.random(horizon)
    phase_draws    = rng_mob.random(horizon)
    e0 = float(rng_e0.uniform(params.e_min, params.e_max))
    if sampler is None:
        lam_path = np.array([
            max(0.0, float(rng_lam.normal(mean_price(t, params), params.sigma_lambda)))
            for t in range(horizon)
        ])
    else:
        dow = 5 if is_weekend else 0
        lam_path = np.array([
            max(0.0, sampler.sample(dow, (t // 60) % 24, season, rng=rng_lam))
            for t in range(horizon)
        ])
    return {"lam_path": lam_path, "mobility_draws": mobility_draws,
            "phase_draws": phase_draws, "e0": e0}


def run_rollouts(pi, actions, e_grid, params, scenarios: list, _rollout_fn, pbp_fn,
                 desc: str | None = None) -> dict:
    """Run all policies on each scenario. Returns {policy_name: [metrics_dict, ...]}.

    If ``desc`` is given, shows a per-rollout progress bar (x/N) under the sweep bars.
    """
    raw = run_rollouts_full(pi, actions, e_grid, params, scenarios, _rollout_fn, pbp_fn, desc=desc)
    return {name: [rollout_metrics(r, params) for r in rolls] for name, rolls in raw.items()}


def run_rollouts_full(pi, actions, e_grid, params, scenarios, _rollout_fn, pbp_fn,
                      desc: str | None = None) -> dict:
    """Like run_rollouts but keeps each raw rollout dict (u_traj/chi_traj/cost_traj)."""
    registry = policy_registry(params, pbp_fn, pi=pi, actions=actions, e_grid=e_grid)
    e0s = [float(sc["e0"]) for sc in scenarios]
    bar = progress = None
    if desc is not None:
        from tqdm import tqdm
        bar = tqdm(total=len(scenarios), desc=desc, unit="rollout", position=2, leave=False)
        progress = lambda: bar.update(1)
    out = run_policies(registry, scenarios, e0s, 0, params, _rollout_fn, progress=progress)
    if bar is not None:
        bar.close()
    return out


def rollout_stats_table(rollouts_by_policy: dict, params):
    """Per-policy summary DataFrame (indexed by policy) from raw rollout dicts.

    The Policy-Rollout page's stats table; columns match build_summary_df's metrics.
    """
    import pandas as pd
    from ev_mdt.plots.sensitivity import _cost_summary
    rows = {policy: _cost_summary([rollout_metrics(r, params) for r in rolls])
            for policy, rolls in rollouts_by_policy.items()}
    return pd.DataFrame(rows).T


def _run_sweep_step(model_label: str, label: str, params, pbp_fn,
                    T: int, N_e: int, N_rollouts: int, seed: int,
                    sampler=None, season: str = "winter", is_weekend: bool = False,
                    _log: Callable | None = None) -> dict:
    """Solve + run rollouts for one sweep configuration."""
    if _log: _log(f"  [{label}] solving (T={T//60}h, N_e={N_e})…")
    pi, actions, e_grid, lam_grid = solve(model_label, params, pbp_fn, T, N_e)
    if _log: _log(f"  [{label}] running {N_rollouts} rollouts…")
    scenarios = [
        make_scenario(params, seed + i, T, sampler=sampler, season=season, is_weekend=is_weekend)
        for i in range(N_rollouts)
    ]
    _rf = rollout_fn(model_label)
    rollouts = run_rollouts(pi, actions, e_grid, params, scenarios, _rf, pbp_fn,
                            desc=f"    [{label}] rollouts")
    sample_rollout = _rf(
        backward_induction_policy, scenarios[0], float(scenarios[0]["e0"]), 0, params,
        pi=pi, actions=actions, e_grid=e_grid,
    )
    return {
        "model":         model_label,
        "label":         label,
        "params":        params,
        "pbp_fn":        pbp_fn,
        "pi":            pi,
        "actions":       actions,
        "e_grid":        e_grid,
        "lam_grid":      lam_grid,
        "T":             T,
        "rollouts":      rollouts,
        "sample_rollout": sample_rollout,
    }


def _gbins_step(label: str, sampler, season: str, is_weekend: bool,
                N_rollouts: int, N_e: int, seed: int, _log=None) -> dict:
    params = build_params(BASELINE_MODEL)
    pbp_fn = _make_gbins_pbp(sampler, params, season, is_weekend)
    return _run_sweep_step(
        BASELINE_MODEL, label, params, pbp_fn, T=24 * 60, N_e=N_e,
        N_rollouts=N_rollouts, seed=seed,
        sampler=sampler, season=season, is_weekend=is_weekend, _log=_log,
    )


def _make_gbins_pbp(sampler, params, season, is_weekend):
    from ev_mdt.pricing.samplers import make_price_bin_probs_fn
    return make_price_bin_probs_fn(sampler, params, season, is_weekend)


# ── Public sweep functions ─────────────────────────────────────────────────────

def sweep_pricing_season(sampler, N_rollouts: int, N_e: int, seed: int,
                          progress_cb: Callable | None = None,
                          _log: Callable | None = None) -> list[dict]:
    """Gaussian Bins (crisis-excluded): vary season, held at weekday."""
    seasons = ["winter", "spring", "summer", "autumn"]
    results = []
    for i, s in enumerate(seasons):
        if progress_cb: progress_cb(i / len(seasons), f"Solving {s}…")
        results.append(_gbins_step(s.capitalize(), sampler, s, False, N_rollouts, N_e, seed, _log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_daytype(sampler, N_rollouts: int, N_e: int, seed: int,
                           progress_cb: Callable | None = None,
                           _log: Callable | None = None) -> list[dict]:
    """Gaussian Bins (crisis-excluded): vary weekday/weekend, held at spring."""
    combos = [("Weekday", False), ("Weekend", True)]
    results = []
    for i, (label, we) in enumerate(combos):
        if progress_cb: progress_cb(i / len(combos), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", we, N_rollouts, N_e, seed, _log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_crisis(sampler_excl, sampler_incl, sampler_crisis,
                          N_rollouts: int, N_e: int, seed: int,
                          progress_cb: Callable | None = None,
                          _log: Callable | None = None) -> list[dict]:
    """Gaussian Bins: vary crisis inclusion, held at spring + weekday."""
    items = [("Excluding crisis", sampler_excl),
             ("Including crisis", sampler_incl),
             ("Crisis only",      sampler_crisis)]
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb: progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_rollouts, N_e, seed, _log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_model(samplers: dict, N_rollouts: int, N_e: int, seed: int,
                         progress_cb: Callable | None = None,
                         _log: Callable | None = None) -> list[dict]:
    """Vary the price model (Gaussian Bins / GMM / MDN), held at spring · weekday · crisis-excluded."""
    items = list(samplers.items())
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb: progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_rollouts, N_e, seed, _log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_penalty(model_label: str, N_rollouts: int, N_e: int, seed: int,
                   progress_cb: Callable | None = None,
                   _log: Callable | None = None) -> list[dict]:
    """Sweep φ ∈ PHI_VALUES. Uses Gaussian parametric pricing."""
    results = []
    for i, phi in enumerate(PHI_VALUES):
        if progress_cb: progress_cb(i / len(PHI_VALUES), f"Solving φ = {phi} €/h…")
        params = build_params(model_label, phi=float(phi))
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, f"{phi} €/h", params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_beta(model_label: str, N_rollouts: int, N_e: int, seed: int,
               progress_cb: Callable | None = None,
               _log: Callable | None = None) -> list[dict]:
    """Sweep the discount factor β ∈ BETA_VALUES over a 24 h horizon."""
    results = []
    for i, beta in enumerate(BETA_VALUES):
        if progress_cb: progress_cb(i / len(BETA_VALUES), f"Solving β = {beta:g}…")
        params = build_params(model_label, beta=float(beta))
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, f"β={beta:g}", params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_horizon(model_label: str, N_rollouts: int, N_e: int, seed: int,
                   progress_cb: Callable | None = None,
                   _log: Callable | None = None) -> list[dict]:
    """Compare T ∈ {24h, 48h, 168h}. Uses Gaussian parametric pricing."""
    results = []
    for i, T_h in enumerate(HORIZON_HOURS):
        if progress_cb: progress_cb(i / len(HORIZON_HOURS), f"Solving T = {T_h} h…")
        params = build_params(model_label)
        T      = T_h * 60
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, f"{T_h} h", params, pbp_fn, T=T, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_departure_profiles(model_label: str, N_rollouts: int, N_e: int, seed: int,
                               progress_cb: Callable | None = None,
                               _log: Callable | None = None) -> list[dict]:
    """Compare departure profiles (p_PD_* overrides) over a 24 h horizon."""
    results = []
    profiles = list(DEPARTURE_PROFILES.items())
    for i, (label, overrides) in enumerate(profiles):
        if progress_cb: progress_cb(i / len(profiles), f"Solving {label}…")
        params = build_params(model_label, **overrides)
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, label, params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_mobility_models(N_rollouts: int, N_e: int, seed: int,
                           progress_cb: Callable | None = None,
                           _log: Callable | None = None) -> list[dict]:
    """Compare NegBin mobility models: {fixed-k, Poisson-k} × {k=5, k=10} (4 configs)."""
    configs = [
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=5",    NegBinParams(k=5)),
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=10",   NegBinParams(k=10)),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=5",  NegBinParams(lambda_k=5.0,  k=NegBinParams.k_max_for_lambda(5.0))),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=10", NegBinParams(lambda_k=10.0, k=NegBinParams.k_max_for_lambda(10.0))),
    ]
    results = []
    for i, (model, label, params) in enumerate(configs):
        if progress_cb: progress_cb(i / len(configs), f"Solving {label}…")
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model, label, params, pbp_fn, T=24 * 60, N_e=N_e,
            N_rollouts=N_rollouts, seed=seed, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def baseline_optimal_result(model_label: str = BASELINE_MODEL, N_e: int = 500) -> dict:
    """Solve one canonical baseline model (Gaussian parametric pricing) for export/display."""
    T = 24 * 60
    params = build_params(model_label)
    pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
    pi, actions, e_grid, lam_grid = solve(model_label, params, pbp_fn, T, N_e)
    return {"model": model_label, "label": model_label, "params": params, "pbp_fn": pbp_fn,
            "pi": pi, "actions": actions, "e_grid": e_grid, "lam_grid": lam_grid, "T": T}


def baseline_model_figures(result: dict, N_rollouts: int, seed: int) -> dict:
    """The three per-model figures: cost bar, optimal-policy heatmap, mean trajectory."""
    from ev_mdt.plots.sensitivity import fig_heatmap_grid, fig_baseline_cost, fig_baseline_trajectories
    model, params, T, pbp_fn = result["model"], result["params"], result["T"], result["pbp_fn"]
    scenarios = [make_scenario(params, seed + i, T) for i in range(N_rollouts)]
    _rf = rollout_fn(model)
    full = run_rollouts_full(result["pi"], result["actions"], result["e_grid"],
                              params, scenarios, _rf, pbp_fn)
    return {
        "baseline_cost":           fig_baseline_cost(full),
        "baseline_optimal_policy": fig_heatmap_grid([result], show_titles=False),
        "baseline_trajectories":   fig_baseline_trajectories(full, scenarios, T, params),
    }


# ── End-to-end orchestration (for CLI / scripts) ───────────────────────────────

def load_fitted_samplers(exclude_crisis: bool = True) -> dict:
    """Load ENTSO-E data and fit Gaussian-bins, GMM, and MDN samplers.

    Returns a dict mapping sampler name → fitted AbstractSampler.
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.pricing.samplers import GaussianBinnedSampler, GMMSampler, MDNSampler

    df = load_prices()
    if exclude_crisis:
        df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]

    samplers = {}
    samplers["Gaussian Bins"] = GaussianBinnedSampler().fit(df)
    samplers["GMM"]           = GMMSampler().fit(df)
    samplers["MDN"]           = MDNSampler().fit(df)
    return samplers


def run_all_sweeps(
    N_rollouts: int = 500,
    N_e: int = 500,
    seed: int = 42,
    sweeps: list[str] | None = None,
    progress_cb: Callable | None = None,
    _log: Callable | None = None,
    _wandb_run=None,
) -> dict[str, list[dict]]:
    """Run selected (or all) sweeps and return a mapping sweep_name -> results list.

    Parameters
    ----------
    sweeps : list of sweep names to run (see ALL_SWEEP_NAMES); None runs all.
    progress_cb : optional callable(fraction: float, message: str) for top-level progress.

    Returns
    -------
    dict mapping each requested sweep name to its result list.
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.pricing.samplers import GaussianBinnedSampler

    if sweeps is None:
        sweeps = list(ALL_SWEEP_NAMES)

    needs_pricing = any(s.startswith("pricing") for s in sweeps)

    def _fit_progress(label: str) -> Callable | None:
        """Route a sampler's _progress(fraction, msg) into the live heartbeat line.

        Updates in place (no new line per epoch/bin); only active when a
        progress_cb is present, since the coarse _log markers already cover the
        no-callback case.
        """
        if progress_cb is None:
            return None
        return lambda _frac, msg: progress_cb(0.0, f"{label}: {msg}")

    sampler_excl = sampler_incl = sampler_crisis = None
    df = df_excl = None
    if needs_pricing:
        if _log: _log("Loading price data…")
        df = load_prices(_log=_log)
        df_excl = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
        if _log: _log("Fitting Gaussian Bins sampler (crisis-excluded)…")
        sampler_excl = GaussianBinnedSampler().fit(df_excl, _progress=_fit_progress("Gaussian Bins (excl. crisis)"))
        if _log: _log("Fitting Gaussian Bins sampler (crisis-included)…")
        sampler_incl = GaussianBinnedSampler().fit(df, _progress=_fit_progress("Gaussian Bins (incl. crisis)"))
        if _log: _log("Fitting Gaussian Bins sampler (crisis-only)…")
        sampler_crisis = GaussianBinnedSampler().fit(
            df[df["timestamp"].dt.year.isin(CRISIS_YEARS)],
            _progress=_fit_progress("Gaussian Bins (crisis only)"))

    all_results: dict[str, list[dict]] = {}
    n = len(sweeps)

    def _cb(sweep_idx: int, name: str) -> Callable | None:
        if progress_cb is None:
            return None
        def inner(f, m):
            progress_cb((sweep_idx + f) / n, f"{name}: {m}")
        return inner

    for i, sweep in enumerate(sweeps):
        cb = _cb(i, sweep)
        if sweep == "pricing_model":
            from ev_mdt.pricing.samplers import GMMSampler, MDNSampler
            # Reuse the crisis-excluded data + Gaussian Bins fit from setup.
            if _log: _log("Fitting GMM sampler…")
            sampler_gmm = GMMSampler().fit(df_excl, _progress=_fit_progress("GMM"))
            if _log: _log("Fitting MDN sampler (neural net — this can take a while)…")
            sampler_mdn = MDNSampler().fit(df_excl, _progress=_fit_progress("MDN"),
                                           _wandb_run=_wandb_run)
            samplers = {
                "Gaussian Bins": sampler_excl,
                "GMM":           sampler_gmm,
                "MDN":           sampler_mdn,
            }
            all_results[sweep] = sweep_pricing_model(samplers, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "pricing_season":
            all_results[sweep] = sweep_pricing_season(sampler_excl, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "pricing_daytype":
            all_results[sweep] = sweep_pricing_daytype(sampler_excl, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "pricing_crisis":
            all_results[sweep] = sweep_pricing_crisis(sampler_excl, sampler_incl, sampler_crisis, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "penalty":
            all_results[sweep] = sweep_penalty(BASELINE_MODEL, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "beta":
            all_results[sweep] = sweep_beta(BASELINE_MODEL, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "horizon":
            all_results[sweep] = sweep_horizon(BASELINE_MODEL, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "departure_profile":
            all_results[sweep] = sweep_departure_profiles(BASELINE_MODEL, N_rollouts, N_e, seed, cb, _log)
        elif sweep == "mobility_model":
            all_results[sweep] = sweep_mobility_models(N_rollouts, N_e, seed, cb, _log)
        else:
            raise ValueError(f"Unknown sweep: {sweep!r}. Valid: {ALL_SWEEP_NAMES}")

    if progress_cb:
        progress_cb(1.0, "Done.")
    return all_results


_SWEEP_FOLDER = {
    "pricing_model":     "pricing_model",
    "pricing_season":    "pricing_season",
    "pricing_daytype":   "pricing_daytype",
    "pricing_crisis":    "pricing_crisis",
    "penalty":           "penalty",
    "beta":              "beta",
    "horizon":           "horizon",
    "departure_profile": "departure_profile",
    "mobility_model":    "mobility_model",
}

# Heatmap-grid columns per sweep (single source of truth; app + CLI + figure export).
HEATMAP_NCOLS = {
    "penalty": 3,
    "beta": 3,
    "pricing_season": 2,
    "mobility_model": 2,
    "pricing_crisis": 3,
}


def save_figures(
    all_results: dict[str, list[dict]],
    out_dir: str | Path = "figures/",
    N_rollouts: int = 200,
    seed: int = 42,
    N_e: int = 500,
    include_baseline: bool = True,
) -> list[Path]:
    """Render all sensitivity figures to PNG and save them under out_dir.

    Also saves the three canonical baseline-model figures if include_baseline=True.

    Returns a list of paths of all saved files.
    """
    from ev_mdt.plots.sensitivity import (
        fig_heatmap_grid, fig_charge_boundary_grid, fig_cost_distribution, figure_to_png,
        fig_baseline_cost, fig_baseline_trajectories, fig_rollout_trajectories,
    )
    from ev_mdt.plots.trip_duration import compute_trip_durations, trip_duration_figure

    out_dir = Path(out_dir)
    saved: list[Path] = []

    for sweep_name, results in all_results.items():
        folder = _SWEEP_FOLDER.get(sweep_name, sweep_name)
        dest = out_dir / "sensitivity_figures" / folder
        dest.mkdir(parents=True, exist_ok=True)

        ncols = HEATMAP_NCOLS.get(sweep_name, 1)
        figs = {
            "policy_heatmaps": fig_heatmap_grid(results, ncols=ncols),
            "charge_border":   fig_charge_boundary_grid(results),
            "cost":            fig_cost_distribution(results),
        }
        for name, fig in figs.items():
            p = dest / f"{name}.png"
            p.write_bytes(figure_to_png(fig))
            saved.append(p)

    if include_baseline:
        bm_dir = out_dir / "baseline_models"
        bm_dir.mkdir(parents=True, exist_ok=True)
        prefixes = {
            BASELINE_MODEL:       "baseline",
            NEGBIN_FIXED_MODEL:   "negbin",
            NEGBIN_SAMPLED_MODEL: "negbin_poisson",
        }
        # NegBin variants share one combined trajectory figure (mobility overlaid).
        negbin_display = {
            NEGBIN_FIXED_MODEL:   ("Negative Binomial (fixed k)",   "#EE6677"),
            NEGBIN_SAMPLED_MODEL: ("Negative Binomial (Poisson k)", "#228833"),
        }
        negbin_bands: list = []
        negbin_scenarios = negbin_T = None

        for model_label in [BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL]:
            prefix = prefixes[model_label]
            result = baseline_optimal_result(model_label, N_e)
            params, T, pbp_fn = result["params"], result["T"], result["pbp_fn"]
            scenarios = [make_scenario(params, seed + i, T) for i in range(N_rollouts)]
            full = run_rollouts_full(result["pi"], result["actions"], result["e_grid"],
                                     params, scenarios, rollout_fn(model_label), pbp_fn)

            for name, fig in (("cost", fig_baseline_cost(full)),
                              ("optimal_policy", fig_heatmap_grid([result], show_titles=False))):
                p = bm_dir / f"{prefix}_{name}.png"
                p.write_bytes(figure_to_png(fig))
                saved.append(p)

            if model_label == BASELINE_MODEL:
                p = bm_dir / "baseline_trajectories.png"
                p.write_bytes(figure_to_png(fig_baseline_trajectories(full, scenarios, T, params)))
                saved.append(p)
            else:
                label, color = negbin_display[model_label]
                chi_list = [r["chi_traj"] for r in full["Backward Induction"]]
                negbin_bands.append((label, color, chi_list, True))
                negbin_scenarios, negbin_T = scenarios, T

        # Single combined NegBin trajectory figure (both variants on one plot).
        if negbin_bands:
            fig = fig_rollout_trajectories(negbin_scenarios, negbin_T, negbin_bands)
            # Export-only: full-width horizontal legend on top (frees the side
            # space the app's side-legend takes) with a slightly larger font.
            # Extra top margin so it clears the first subplot title.
            fig.update_layout(
                margin=dict(l=40, r=30, t=90, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.10,
                            xanchor="left", x=0, font=dict(size=18)),
            )
            p = bm_dir / "negbin_trajectories.png"
            p.write_bytes(figure_to_png(fig))
            saved.append(p)

        # Trip duration figure
        durs = compute_trip_durations()
        p = bm_dir / "trip_duration_by_model.png"
        p.write_bytes(figure_to_png(trip_duration_figure(durs)))
        saved.append(p)

    return saved


# Display names for the baseline/NegBin models in the combined summary table.
_BASELINE_TABLE_LABELS = {
    BASELINE_MODEL:       "Baseline",
    NEGBIN_FIXED_MODEL:   "NegBin (fixed k)",
    NEGBIN_SAMPLED_MODEL: "NegBin (sampled k)",
}


def save_tables(
    all_results: dict[str, list[dict]],
    out_dir: str | Path = "export/tables",
    N_rollouts: int = 500,
    seed: int = 42,
    N_e: int = 500,
    include_baseline: bool = True,
    _log: Callable | None = None,
) -> list[Path]:
    """Write the per-sweep and (optionally) baseline-model summary tables as CSV.

    Mirrors save_figures' directory layout under out_dir:
        sensitivity_figures/<sweep>/summary.csv   — one row per (swept value, policy)
        baseline_models/summary.csv               — combined Baseline + NegBin models

    Each table carries the shared 6-metric set (see build_summary_df). Returns saved paths.
    """
    from ev_mdt.plots.sensitivity import build_summary_df

    out_dir = Path(out_dir)
    saved: list[Path] = []

    for sweep_name, results in all_results.items():
        folder = _SWEEP_FOLDER.get(sweep_name, sweep_name)
        dest = out_dir / "sensitivity_figures" / folder
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / "summary.csv"
        build_summary_df(results).to_csv(p, index=False)
        saved.append(p)
        if _log: _log(f"  Saved table: {p}")

    if include_baseline:
        bm_dir = out_dir / "baseline_models"
        bm_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for model_label in [BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL]:
            label = _BASELINE_TABLE_LABELS[model_label]
            if _log: _log(f"  [{label}] solving + running {N_rollouts} rollouts…")
            result = baseline_optimal_result(model_label, N_e)
            params, T, pbp_fn = result["params"], result["T"], result["pbp_fn"]
            scenarios = [make_scenario(params, seed + i, T) for i in range(N_rollouts)]
            rollouts = run_rollouts(
                result["pi"], result["actions"], result["e_grid"], params,
                scenarios, rollout_fn(model_label), pbp_fn, desc=f"    [{label}] rollouts")
            model_results.append({"rollouts": rollouts, "label": label})
        import pandas as pd
        df = pd.concat([build_summary_df([r]) for r in model_results], ignore_index=True)
        p = bm_dir / "summary.csv"
        df.to_csv(p, index=False)
        saved.append(p)
        if _log: _log(f"  Saved table: {p}")

    return saved


def exact_bi_cost_table(N_e: int = 500, seed: int = 42, _log: Callable | None = None):
    """Analytical (exact) expected cost of the BI policy for every scenario.

    One row for the Baseline config plus one per sensitivity-sweep configuration
    (all 34). No Monte-Carlo: each config is solved and its optimal policy is
    evaluated exactly (β=1, undiscounted). Returns a DataFrame
    [Sweep, Configuration, Exact BI cost (€)].

    Note: this still fits the price samplers (incl. MDN) for the pricing configs,
    but runs only a single throwaway rollout per config to reuse the sweep
    pipeline — the reported cost is purely analytical (from the value function).
    """
    import pandas as pd

    rows = [{
        "Sweep":             "baseline",
        "Configuration":     "Baseline",
        "Exact BI cost (€)": bi_expected_cost(baseline_optimal_result(BASELINE_MODEL, N_e)),
    }]
    if _log: _log("Baseline: exact BI cost computed.")

    results = run_all_sweeps(N_rollouts=1, N_e=N_e, seed=seed, _log=_log)
    for sweep_name, steps in results.items():
        for step in steps:
            rows.append({
                "Sweep":             sweep_name,
                "Configuration":     step["label"],
                "Exact BI cost (€)": bi_expected_cost(step),
            })

    return pd.DataFrame(rows)
