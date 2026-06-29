"""Sensitivity analysis and figure orchestration for the EV charging MDP.

All computation lives here; no Streamlit dependency. The CLI and the app page both
call these functions and pass the returned figures to the renderer / disk.

Costs are computed **analytically (exact)** by evaluating the solved policy with
``ev_mdt.models.common.backward_induction.evaluate_policy`` — there is no
Monte-Carlo cost estimation. Monte-Carlo *rollouts* are still used, but only to
visualise mean price/mobility trajectories (``model_trajectory_figure``) and trip
durations.

Sweep functions
---------------
Each ``sweep_*`` function returns a list of "panel" dicts, one per swept value.
A panel is a solved-config dict containing:

    model, label, params, pbp_fn, pi, actions, e_grid, lam_grid, T
    exact_breakdown  — {policy: {total, charging, penalty, energy_kwh, penalty_min}}
                       (only when ``compute_exact=True``)
"""
from typing import Callable

import numpy as np

from ev_mdt.params import (
    BaselineParams, NegBinParams,
    BASELINE_MODEL, NEGBIN_FIXED_MODEL, NEGBIN_SAMPLED_MODEL,
)
from ev_mdt.models.common.backward_induction import (
    backward_induction as _backward_induction, evaluate_policy as _evaluate_policy,
    scalar_policy_to_action_fn as _scalar_policy_to_action_fn,
)
from ev_mdt.models.baseline.model import transition_matrix as _baseline_tm
from ev_mdt.models.baseline.rollout import simulate_policy_rollout as _baseline_rollout
from ev_mdt.models.negbin.model import transition_matrix as _negbin_tm
from ev_mdt.models.negbin.rollout import simulate_policy_rollout as _negbin_rollout
from ev_mdt.models.common.model_utils import (
    price_bin_probs as _gaussian_pbp, mean_price,
    expected_trip_minutes as _expected_trip_minutes,
    minutes_to_departure as _minutes_to_departure,
)
from ev_mdt.models.common.policies import (
    policy_registry, du_gamma_for_params,
    _du_e_daily, _e_daily_ref, E_CEIL_BASE,
)


# ── Sweep constants (single source of truth) ───────────────────────────────────

PHI_VALUES       = [0, 0.05, 1, 50, 500, 5000]
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
    "horizon",
    "departure_profile",
    "mobility_model",
]

# Mobility-model label → trajectory display name (matches plots.viz.MODEL_COLORS).
_MODEL_DISPLAY = {
    BASELINE_MODEL:       "Baseline",
    NEGBIN_FIXED_MODEL:   "Negative Binomial (fixed k)",
    NEGBIN_SAMPLED_MODEL: "Negative Binomial (Poisson k)",
}


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
    """Return the model-specific simulate_policy_rollout (used for trajectory figures)."""
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


def _mean_initial(J_0, pbp_fn) -> float:
    """Average J at t=0, χ=0 over the rollout initial-state distribution
    (e₀ ~ Uniform on the battery grid, λ̂₀ ~ price marginal at t=0).
    J_0 is the (n_chi, N_e, K) slice returned by evaluate_policy."""
    return float(np.mean(J_0[0] @ np.asarray(pbp_fn(0))))


# ── Exact (analytical) expected cost ───────────────────────────────────────────

def bi_expected_cost(result: dict, beta: float = 1.0, breakdown: bool = False):
    """Exact expected cost of the optimal (Backward Induction) policy for a config.

    Evaluates the solved policy `pi` (from a solve()/sweep-step/baseline result dict)
    analytically and averages over the rollout initial-state distribution:
    χ₀=0, e₀ ~ Uniform[e_min,e_max] (battery grid), λ̂₀ ~ price marginal at t=0.
    With beta=1 (default) this is the expected *undiscounted* total cost.

    With ``breakdown=True`` returns ``{"total", "charging", "penalty"}`` instead of
    just the scalar total.
    """
    model_label = result["model"]
    params, pbp_fn = result["params"], result["pbp_fn"]
    pi, actions, e_grid, T = result["pi"], result["actions"], result["e_grid"], result["T"]
    if model_label == BASELINE_MODEL:
        tm_fn, n_chi = (lambda t: _baseline_tm(t, params)), 2
    else:
        tm_fn, n_chi = (lambda t: _negbin_tm(t, params)), params.k + 1
    out = _evaluate_policy(
        params, transition_matrix_fn=tm_fn, price_bin_probs_fn=pbp_fn, n_chi=n_chi,
        action_fn=lambda t, chi: actions[pi[t, chi]], T=T, N_e=len(e_grid), beta=beta,
        cost_components=breakdown,
    )
    if breakdown:
        Jpi, Jc, Jp, Je, Jm = out
        return {"total": _mean_initial(Jpi, pbp_fn),
                "charging": _mean_initial(Jc, pbp_fn),
                "penalty": _mean_initial(Jp, pbp_fn),
                "energy_kwh": _mean_initial(Je, pbp_fn),
                "penalty_min": _mean_initial(Jm, pbp_fn)}
    return _mean_initial(out, pbp_fn)


def _du_e_ceil(params, gamma: float | None = None) -> float:
    """Demand-scaled DU ceiling (matches policies.next_trip_policy)."""
    if gamma is None:
        gamma = du_gamma_for_params(params)
    ref   = _e_daily_ref()
    ratio = _du_e_daily(params) / ref if ref > 0 else 1.0
    return min(params.e_max, E_CEIL_BASE * ratio ** gamma)


def _vectorized_action_grid(name: str, kwargs: dict, e_grid, lam_grid, params,
                            T: int, probs, cumsum):
    """(T, N_e, K) desired chi=0 charge rates for a benchmark policy, fully
    vectorised — the λ-resolved counterpart of the per-cell scalar policy.

    Returns ``None`` for policies without a vectorised form (the caller then
    falls back to the scalar wrapper). The driving / battery-floor gate is
    applied inside ``evaluate_policy``, exactly as for the scalar action_fn.
    """
    N_e, K = len(e_grid), len(lam_grid)
    u_max, e_max = params.u_max, params.e_max

    if name == "Always-Maximum":
        return np.full((T, N_e, K), u_max)
    if name == "Always-Minimum":
        return np.full((T, N_e, K), params.u_min)
    if name == "Night Charging":
        rate_t = np.where(np.arange(T) % 1440 < 360, u_max, 0.0)
        return np.broadcast_to(rate_t[:, None, None], (T, N_e, K)).copy()
    if name == "Minimum Battery Level":
        rate_e = np.where(e_grid < kwargs["soc_threshold"], u_max, 0.0)
        return np.broadcast_to(rate_e[None, :, None], (T, N_e, K)).copy()
    if name == "Price-Oriented":
        low, high = kwargs["low_threshold"], kwargs["high_threshold"]
        rate_k = np.where(lam_grid <= low, u_max,
                          np.where(lam_grid <= high, u_max / 2, 0.0))
        return np.broadcast_to(rate_k[None, None, :], (T, N_e, K)).copy()

    if name == "Battery Level Urgency":
        thresh = np.clip(1.0 - e_grid / e_max, 0.0, 1.0)              # (N_e,)
        mask = cumsum[:, None, :] <= thresh[None, :, None]           # (T, N_e, K)
        rate = np.where(mask, u_max, 0.0)
        return np.where(e_grid[None, :, None] >= e_max, 0.0, rate)

    if name == "Departure Urgency":
        gamma        = kwargs.get("gamma", None)
        use_reserve  = kwargs.get("use_reserve", True)
        ceil_override = kwargs.get("ceil_override", None)
        e_trip = _expected_trip_minutes(params) * params.mu * params.v * params.omega
        e_ceil = float(ceil_override) if ceil_override is not None else _du_e_ceil(params, gamma)

        slots       = np.array([_minutes_to_departure(t, params) for t in range(T)])  # (T,)
        deliverable = u_max * params.eta_c * params.omega * slots                     # (T,)
        e_diff      = np.maximum(0.0, e_ceil - e_grid[None, :])                        # (1, N_e)
        safe_del    = np.where(deliverable > 0, deliverable, 1.0)
        rho  = np.where(deliverable[:, None] > 0, e_diff / safe_del[:, None], np.inf) # (T, N_e)
        band = np.where(slots > 0, 1.0 / slots, 0.0)                                  # (T,)

        F   = cumsum[:, None, :]                                      # (T, 1, K)
        p1  = F <= rho[:, :, None]                                    # (T, N_e, K)
        p12 = F <= (rho + band[:, None])[:, :, None]
        rate = np.where(p1, u_max, np.where(p12, u_max / 2, 0.0))
        if use_reserve:
            rate = np.where(e_grid[None, :, None] < e_trip, u_max, rate)
        return np.where(e_grid[None, :, None] >= e_max, 0.0, rate)

    return None


def policy_expected_cost(result: dict, policy_fn, beta: float = 1.0,
                         progress_desc: str | None = None,
                         name: str | None = None, breakdown: bool = False,
                         **kwargs):
    """Exact expected cost of a scalar benchmark policy for a given configuration.

    Equivalent to bi_expected_cost but works for any scalar policy from the registry.
    Uses scalar_policy_to_action_fn to vectorize the policy into the (N_e, K) form
    that evaluate_policy expects.

    With ``breakdown=True`` returns ``{"total", "charging", "penalty"}`` instead of
    just the scalar total.
    """
    model_label = result["model"]
    params, pbp_fn = result["params"], result["pbp_fn"]
    e_grid, lam_grid, T = result["e_grid"], result["lam_grid"], result["T"]
    if model_label == BASELINE_MODEL:
        tm_fn, n_chi = (lambda t: _baseline_tm(t, params)), 2
    else:
        tm_fn, n_chi = (lambda t: _negbin_tm(t, params)), params.k + 1
    grid = None
    if name is not None:
        probs  = np.array([pbp_fn(t) for t in range(T)])
        cumsum = probs.cumsum(axis=1)
        grid = _vectorized_action_grid(name, kwargs, e_grid, lam_grid, params, T, probs, cumsum)
    if grid is not None:
        action_fn = lambda t, chi, _g=grid: _g[t]
    else:
        action_fn = _scalar_policy_to_action_fn(policy_fn, e_grid, lam_grid, params, **kwargs)
    out = _evaluate_policy(
        params, transition_matrix_fn=tm_fn, price_bin_probs_fn=pbp_fn, n_chi=n_chi,
        action_fn=action_fn, T=T, N_e=len(e_grid), beta=beta,
        progress_desc=progress_desc, cost_components=breakdown,
    )
    if breakdown:
        Jpi, Jc, Jp, Je, Jm = out
        return {"total": _mean_initial(Jpi, pbp_fn),
                "charging": _mean_initial(Jc, pbp_fn),
                "penalty": _mean_initial(Jp, pbp_fn),
                "energy_kwh": _mean_initial(Je, pbp_fn),
                "penalty_min": _mean_initial(Jm, pbp_fn)}
    return _mean_initial(out, pbp_fn)


def compute_all_exact_costs(result: dict, beta: float = 1.0,
                            desc: str | None = None) -> dict[str, float]:
    """Exact expected cost for every policy in the registry for a solved configuration.

    Returns {policy_name: cost} for all policies in the registry.
    The BI policy uses the fast vectorised index lookup; all others use
    scalar_policy_to_action_fn (one backward pass per policy).

    This is slow: each non-BI policy runs a full backward-pass policy evaluation
    that rebuilds the (N_e × K) action grid via a pure-Python scalar-policy call
    at every (t, χ) cell. Pass ``desc`` to show a per-policy progress bar.
    """
    params, pbp_fn = result["params"], result["pbp_fn"]
    pi, actions, e_grid = result["pi"], result["actions"], result["e_grid"]
    registry = policy_registry(params, pbp_fn, pi=pi, actions=actions, e_grid=e_grid)
    bar = None
    if desc is not None:
        from tqdm import tqdm
        bar = tqdm(total=len(registry), desc=desc, unit="policy", position=2, leave=False)
    costs = {}
    for name, fn, kwargs in registry:
        if bar is not None:
            bar.set_postfix_str(name)
        if name == "Backward Induction":
            costs[name] = bi_expected_cost(result, beta=beta)
        else:
            step_desc = f"      {name}" if desc is not None else None
            costs[name] = policy_expected_cost(result, fn, beta=beta,
                                               progress_desc=step_desc, name=name, **kwargs)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return costs


def compute_all_exact_costs_breakdown(result: dict, beta: float = 1.0,
                                      desc: str | None = None) -> dict[str, dict]:
    """Like compute_all_exact_costs but returns the charging/penalty split.

    Returns ``{policy_name: {"total", "charging", "penalty", ...}}`` for every policy
    in the registry. Same backward-pass cost as compute_all_exact_costs (one extra
    pair of accumulator arrays carried along the recursion).
    """
    params, pbp_fn = result["params"], result["pbp_fn"]
    pi, actions, e_grid = result["pi"], result["actions"], result["e_grid"]
    registry = policy_registry(params, pbp_fn, pi=pi, actions=actions, e_grid=e_grid)
    bar = None
    if desc is not None:
        from tqdm import tqdm
        bar = tqdm(total=len(registry), desc=desc, unit="policy", position=2, leave=False)
    out = {}
    for name, fn, kwargs in registry:
        if bar is not None:
            bar.set_postfix_str(name)
        if name == "Backward Induction":
            out[name] = bi_expected_cost(result, beta=beta, breakdown=True)
        else:
            step_desc = f"      {name}" if desc is not None else None
            out[name] = policy_expected_cost(result, fn, beta=beta,
                                             progress_desc=step_desc, name=name,
                                             breakdown=True, **kwargs)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return out


# ── Rollout scenarios (for trajectory / trip-duration figures only) ─────────────

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


# ── Sweep step (solve + exact cost) ────────────────────────────────────────────

def _run_sweep_step(model_label: str, label: str, params, pbp_fn,
                    T: int, N_e: int, *, compute_exact: bool = True,
                    _log: Callable | None = None) -> dict:
    """Solve one sweep configuration; optionally compute the exact per-policy cost."""
    if _log: _log(f"  [{label}] solving (T={T // 60}h, N_e={N_e})…")
    pi, actions, e_grid, lam_grid = solve(model_label, params, pbp_fn, T, N_e)
    result = {
        "model":    model_label, "label": label, "params": params, "pbp_fn": pbp_fn,
        "pi":       pi, "actions": actions, "e_grid": e_grid, "lam_grid": lam_grid, "T": T,
    }
    if compute_exact:
        if _log: _log(f"  [{label}] computing exact costs…")
        result["exact_breakdown"] = compute_all_exact_costs_breakdown(
            result, desc=f"    [{label}] exact costs")
    return result


def _make_gbins_pbp(sampler, params, season, is_weekend):
    from ev_mdt.pricing.samplers import make_price_bin_probs_fn
    return make_price_bin_probs_fn(sampler, params, season, is_weekend)


def _gbins_step(label: str, sampler, season: str, is_weekend: bool, N_e: int, *,
                compute_exact: bool = True, _log=None) -> dict:
    params = build_params(BASELINE_MODEL)
    pbp_fn = _make_gbins_pbp(sampler, params, season, is_weekend)
    return _run_sweep_step(BASELINE_MODEL, label, params, pbp_fn, T=24 * 60, N_e=N_e,
                           compute_exact=compute_exact, _log=_log)


# ── Public sweep functions ─────────────────────────────────────────────────────

def sweep_pricing_season(sampler, N_e: int, progress_cb: Callable | None = None,
                          _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Gaussian Bins (crisis-excluded): vary season, held at weekday."""
    seasons = ["winter", "spring", "summer", "autumn"]
    results = []
    for i, s in enumerate(seasons):
        if progress_cb: progress_cb(i / len(seasons), f"Solving {s}…")
        results.append(_gbins_step(s.capitalize(), sampler, s, False, N_e,
                                   compute_exact=compute_exact, _log=_log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_daytype(sampler, N_e: int, progress_cb: Callable | None = None,
                           _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Gaussian Bins (crisis-excluded): vary weekday/weekend, held at spring."""
    combos = [("Weekday", False), ("Weekend", True)]
    results = []
    for i, (label, we) in enumerate(combos):
        if progress_cb: progress_cb(i / len(combos), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", we, N_e,
                                   compute_exact=compute_exact, _log=_log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_crisis(sampler_excl, sampler_incl, sampler_crisis, N_e: int,
                          progress_cb: Callable | None = None,
                          _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Gaussian Bins: vary crisis inclusion, held at spring + weekday."""
    items = [("Excluding crisis", sampler_excl),
             ("Including crisis", sampler_incl),
             ("Crisis only",      sampler_crisis)]
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb: progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_e,
                                   compute_exact=compute_exact, _log=_log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_pricing_model(samplers: dict, N_e: int, progress_cb: Callable | None = None,
                         _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Vary the price model (Gaussian Bins / GMM / MDN), held at spring · weekday · crisis-excluded."""
    items = list(samplers.items())
    results = []
    for i, (label, sampler) in enumerate(items):
        if progress_cb: progress_cb(i / len(items), f"Solving {label}…")
        results.append(_gbins_step(label, sampler, "spring", False, N_e,
                                   compute_exact=compute_exact, _log=_log))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_penalty(model_label: str, N_e: int, progress_cb: Callable | None = None,
                   _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Sweep φ ∈ PHI_VALUES. Uses Gaussian parametric pricing."""
    results = []
    for i, phi in enumerate(PHI_VALUES):
        if progress_cb: progress_cb(i / len(PHI_VALUES), f"Solving φ = {phi} €/h…")
        params = build_params(model_label, phi=float(phi))
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, f"{phi} €/h", params, pbp_fn, T=24 * 60, N_e=N_e,
            compute_exact=compute_exact, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_horizon(model_label: str, N_e: int, progress_cb: Callable | None = None,
                   _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Compare T ∈ {24h, 48h, 168h}. Uses Gaussian parametric pricing."""
    results = []
    for i, T_h in enumerate(HORIZON_HOURS):
        if progress_cb: progress_cb(i / len(HORIZON_HOURS), f"Solving T = {T_h} h…")
        params = build_params(model_label)
        T      = T_h * 60
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, f"{T_h} h", params, pbp_fn, T=T, N_e=N_e,
            compute_exact=compute_exact, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_departure_profiles(model_label: str, N_e: int, progress_cb: Callable | None = None,
                               _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
    """Compare departure profiles (p_PD_* overrides) over a 24 h horizon."""
    results = []
    profiles = list(DEPARTURE_PROFILES.items())
    for i, (label, overrides) in enumerate(profiles):
        if progress_cb: progress_cb(i / len(profiles), f"Solving {label}…")
        params = build_params(model_label, **overrides)
        pbp_fn = lambda t, p=params: _gaussian_pbp(t, p)
        results.append(_run_sweep_step(
            model_label, label, params, pbp_fn, T=24 * 60, N_e=N_e,
            compute_exact=compute_exact, _log=_log,
        ))
    if progress_cb: progress_cb(1.0, "Done.")
    return results


def sweep_mobility_models(N_e: int, progress_cb: Callable | None = None,
                           _log: Callable | None = None, *, compute_exact: bool = True) -> list[dict]:
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
            compute_exact=compute_exact, _log=_log,
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


# ── Departure-Urgency calibration sweeps (exact) ───────────────────────────────

def sweep_target_ceiling_exact(
    model_label: str = BASELINE_MODEL,
    step_kwh: float = 5.0,
    N_e: int = 500,
    use_reserve: bool = True,
    progress_cb: Callable | None = None,
    _log: Callable | None = None,
) -> list[dict]:
    """Exact sweep of the Departure Urgency target ceiling (e_base) from e_min+step to e_max.

    One backward-pass evaluation per ceiling value. Returns a list of dicts with keys:
        target_kwh, target_frac, mean_cost, mean_charge_cost, mean_penalty_cost,
        mean_penalty_min, mean_charged
    """
    from tqdm import tqdm
    from ev_mdt.models.common.policies import next_trip_policy

    params = build_params(model_label)
    if _log: _log(f"Solving {model_label} (N_e={N_e})…")
    result = baseline_optimal_result(model_label, N_e)

    ceil_values = np.arange(params.e_min + step_kwh, params.e_max + 1e-6, step_kwh)
    ceil_values = ceil_values[ceil_values <= params.e_max]

    rows = []
    bar = tqdm(list(ceil_values), desc="Target ceiling (exact)", unit="step", position=0)
    for i, ceil_kwh in enumerate(bar):
        bar.set_postfix(ceiling=f"{ceil_kwh:.0f} kWh")
        bd = policy_expected_cost(
            result, next_trip_policy, name="Departure Urgency",
            ceil_override=float(ceil_kwh), use_reserve=use_reserve, breakdown=True,
        )
        rows.append({
            "target_kwh":        float(ceil_kwh),
            "target_frac":       float(ceil_kwh) / params.e_max,
            "mean_cost":         bd["total"],
            "mean_charge_cost":  bd["charging"],
            "mean_penalty_cost": bd["penalty"],
            "mean_penalty_min":  bd["penalty_min"],
            "mean_charged":      bd["energy_kwh"],
        })
        if progress_cb: progress_cb((i + 1) / len(ceil_values), f"ceiling {ceil_kwh:.0f} kWh")
    bar.close()
    return rows


def sweep_gamma_exact(
    use_reserve: bool = True,
    gamma_values: list | None = None,
    N_e: int = 500,
    progress_cb: Callable | None = None,
    _log: Callable | None = None,
) -> dict[str, list[dict]]:
    """Exact sweep of the ceiling scaling exponent γ across three mobility models.

    Returns {model_name: [row_dict, ...]} with exact expected values (no std_cost).
    """
    from tqdm import tqdm
    from ev_mdt.models.common.policies import next_trip_policy

    if gamma_values is None:
        gamma_values = [round(g, 2) for g in np.arange(0.1, 1.01, 0.1)]

    e_daily_ref = _e_daily_ref()
    model_configs = [
        (BASELINE_MODEL,       "Baseline",           build_params(BASELINE_MODEL)),
        (NEGBIN_FIXED_MODEL,   "NegBin fixed k=5",   NegBinParams(k=5)),
        (NEGBIN_SAMPLED_MODEL, "NegBin Poisson k=5", NegBinParams(lambda_k=5.0,
                                                      k=NegBinParams.k_max_for_lambda(5.0))),
    ]

    results: dict[str, list[dict]] = {}
    total_steps = len(model_configs) * len(gamma_values)
    step_i = 0

    outer = tqdm(model_configs, desc="Models (exact)", unit="model", position=0)
    for model_label, model_name, params in outer:
        outer.set_postfix(model=model_name)
        if _log: _log(f"  [{model_name}] solving (N_e={N_e})…")
        result = baseline_optimal_result(model_label, N_e)

        e_daily = _du_e_daily(params)
        ratio   = e_daily / e_daily_ref if e_daily_ref > 0 else 1.0

        rows: list[dict] = []
        inner = tqdm(gamma_values, desc=f"  γ sweep ({model_name})", unit="γ",
                     position=1, leave=False)
        for gamma in inner:
            target_kwh = min(params.e_max, E_CEIL_BASE * ratio ** gamma)
            bd = policy_expected_cost(
                result, next_trip_policy, name="Departure Urgency",
                gamma=gamma, use_reserve=use_reserve, breakdown=True,
            )
            rows.append({
                "gamma":             gamma,
                "target_kwh":        target_kwh,
                "mean_cost":         bd["total"],
                "mean_charge_cost":  bd["charging"],
                "mean_penalty_cost": bd["penalty"],
                "mean_penalty_min":  bd["penalty_min"],
                "mean_charged":      bd["energy_kwh"],
            })
            step_i += 1
            if progress_cb: progress_cb(step_i / total_steps, f"{model_name} γ={gamma:.1f}")
        inner.close()
        results[model_name] = rows
    outer.close()
    return results


# ── Price-sampler fitting ──────────────────────────────────────────────────────

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


# ── Multi-sweep orchestration ──────────────────────────────────────────────────

def run_all_sweeps(
    N_e: int = 500,
    sweeps: list[str] | None = None,
    progress_cb: Callable | None = None,
    _log: Callable | None = None,
    *,
    compute_exact: bool = True,
) -> dict[str, list[dict]]:
    """Run selected (or all) sweeps and return a mapping sweep_name -> panel list.

    Parameters
    ----------
    sweeps        : list of sweep names to run (see ALL_SWEEP_NAMES); None runs all.
    progress_cb   : optional callable(fraction: float, message: str) for top-level progress.
    compute_exact : compute the exact per-policy cost breakdown for each panel
                    (needed for the cost figures; closed-form heatmaps/borders do not need it).
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.pricing.samplers import GaussianBinnedSampler

    if sweeps is None:
        sweeps = list(ALL_SWEEP_NAMES)

    needs_pricing = any(s.startswith("pricing") for s in sweeps)

    def _fit_progress(label: str) -> Callable | None:
        """Route a sampler's _progress(fraction, msg) into the live heartbeat line."""
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
            sampler_mdn = MDNSampler().fit(df_excl, _progress=_fit_progress("MDN"))
            samplers = {
                "Gaussian Bins": sampler_excl,
                "GMM":           sampler_gmm,
                "MDN":           sampler_mdn,
            }
            all_results[sweep] = sweep_pricing_model(samplers, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "pricing_season":
            all_results[sweep] = sweep_pricing_season(sampler_excl, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "pricing_daytype":
            all_results[sweep] = sweep_pricing_daytype(sampler_excl, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "pricing_crisis":
            all_results[sweep] = sweep_pricing_crisis(sampler_excl, sampler_incl, sampler_crisis, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "penalty":
            all_results[sweep] = sweep_penalty(BASELINE_MODEL, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "horizon":
            all_results[sweep] = sweep_horizon(BASELINE_MODEL, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "departure_profile":
            all_results[sweep] = sweep_departure_profiles(BASELINE_MODEL, N_e, cb, _log, compute_exact=compute_exact)
        elif sweep == "mobility_model":
            all_results[sweep] = sweep_mobility_models(N_e, cb, _log, compute_exact=compute_exact)
        else:
            raise ValueError(f"Unknown sweep: {sweep!r}. Valid: {ALL_SWEEP_NAMES}")

    if progress_cb:
        progress_cb(1.0, "Done.")
    return all_results


# Output sub-folder per sweep (single source of truth; CLI + app figure export).
_SWEEP_FOLDER = {name: name for name in ALL_SWEEP_NAMES}


# ── Figure orchestrators (consumed by the CLI and the app) ─────────────────────

def baseline_figures(N_e: int = 500, *, compute_cost: bool = True,
                     model_label: str = BASELINE_MODEL, _log: Callable | None = None):
    """All-policy baseline figures for one model: (result, {name: Figure}).

    Figures: ``policy_heatmaps`` and ``charge_borders`` (all policies in one figure
    each), plus ``cost`` (exact expected-cost bar) when ``compute_cost``.
    """
    from ev_mdt.plots.sensitivity import (
        fig_all_policy_heatmaps, fig_all_policy_charge_borders, fig_baseline_cost,
    )
    if _log: _log(f"Solving {model_label} (N_e={N_e})…")
    result = baseline_optimal_result(model_label, N_e)
    figs = {
        "policy_heatmaps": fig_all_policy_heatmaps(result),
        "charge_borders":  fig_all_policy_charge_borders(result),
    }
    if compute_cost:
        if _log: _log("Computing exact costs for all policies…")
        result["exact_breakdown"] = compute_all_exact_costs_breakdown(result, desc="  exact costs")
        figs["cost"] = fig_baseline_cost(result)
    return result, figs


def sensitivity_figures(dims: list[str] | None = None, N_e: int = 500,
                        progress_cb: Callable | None = None, _log: Callable | None = None,
                        *, compute_cost: bool = True) -> dict[str, dict]:
    """Per-sweep paper figures.

    Returns ``{dim: {"panels": [...], "figures": {name: Figure}}}``. For each sweep
    dimension the figures are per-policy heatmaps and charge borders (one figure each
    for Backward Induction / Departure Urgency / Battery Level Urgency, with one
    subplot panel per swept value) plus an exact expected-cost bar (``compute_cost``).
    """
    from ev_mdt.plots.sensitivity import (
        fig_policy_heatmap_grid, fig_policy_charge_border_grid, fig_cost_distribution,
        PAPER_POLICIES, BI_POLICY, DU_POLICY, BLU_POLICY,
    )
    if dims is None:
        dims = list(ALL_SWEEP_NAMES)
    results = run_all_sweeps(N_e=N_e, sweeps=dims, progress_cb=progress_cb, _log=_log,
                             compute_exact=compute_cost)
    stem = {BI_POLICY: "bi", DU_POLICY: "du", BLU_POLICY: "blu"}
    out: dict[str, dict] = {}
    for dim, panels in results.items():
        figs = {}
        for policy in PAPER_POLICIES:
            figs[f"{stem[policy]}_heatmap"]       = fig_policy_heatmap_grid(panels, policy)
            figs[f"{stem[policy]}_charge_border"] = fig_policy_charge_border_grid(panels, policy)
        if compute_cost:
            figs["cost"] = fig_cost_distribution(panels)
        out[dim] = {"panels": panels, "figures": figs}
    return out


def calibrate_du_figures(N_e: int = 500, gamma_step: float = 0.1, step_kwh: float = 5.0,
                         use_reserve: bool = True, _log: Callable | None = None) -> dict:
    """Departure-Urgency calibration: the e_base (target-ceiling) and γ sweeps (exact).

    Returns ``{"target": (rows, Figure), "gamma": (results, Figure)}``.
    """
    from ev_mdt.plots.calibration import fig_target_sweep, fig_gamma_sweep

    target_rows = sweep_target_ceiling_exact(step_kwh=step_kwh, N_e=N_e,
                                             use_reserve=use_reserve, _log=_log)
    gamma_values = [round(g, 6) for g in np.arange(gamma_step, 1.0 + 1e-9, gamma_step)]
    gamma_results = sweep_gamma_exact(use_reserve=use_reserve, gamma_values=gamma_values,
                                      N_e=N_e, _log=_log)
    return {
        "target": (target_rows, fig_target_sweep(target_rows)),
        "gamma":  (gamma_results, fig_gamma_sweep(gamma_results)),
    }


def model_trajectory_figure(mobility_model: str = BASELINE_MODEL,
                            price_model: str = "Gaussian (parametric)",
                            n: int = 1000, seed: int = 42, T: int = 24 * 60,
                            season: str = "winter", is_weekend: bool = False,
                            _log: Callable | None = None):
    """Mean sampled price + mean fraction-driving over the horizon (two subplots).

    Defaults to Gaussian-parametric pricing and Baseline mobility; pass another
    ``mobility_model`` (MODEL_LABELS) or ``price_model`` (a fitted-sampler name from
    ``load_fitted_samplers``) to swap either.
    """
    from ev_mdt.models.common.policies import always_minimum_policy
    from ev_mdt.plots.sensitivity import fig_rollout_trajectories
    from ev_mdt.plots.viz import MODEL_COLORS

    params  = build_params(mobility_model)
    sampler = None
    if price_model and price_model != "Gaussian (parametric)":
        samplers = load_fitted_samplers()
        if price_model not in samplers:
            raise ValueError(f"Unknown price model {price_model!r}; choose from "
                             f"{['Gaussian (parametric)'] + list(samplers)}")
        sampler = samplers[price_model]

    if _log: _log(f"Generating {n} scenarios (price={price_model}, mobility={mobility_model})…")
    scenarios = [make_scenario(params, seed + i, T, sampler=sampler,
                               season=season, is_weekend=is_weekend) for i in range(n)]
    _rf = rollout_fn(mobility_model)
    if _log: _log("Rolling out mobility…")
    chi_list = [_rf(always_minimum_policy, sc, float(sc["e0"]), 0, params)["chi_traj"]
                for sc in scenarios]
    label = _MODEL_DISPLAY.get(mobility_model, mobility_model)
    color = MODEL_COLORS.get(label, "orange")
    return fig_rollout_trajectories(scenarios, T, [(label, color, chi_list, True)])


def price_model_figures(n_days: int = 1000, season: str | None = None, daytype: str = "all",
                        seed: int = 42, exclude_crisis: bool = True,
                        progress_cb: Callable | None = None, _log: Callable | None = None):
    """Fit the price samplers, simulate price rollouts, and build the comparison figures.

    Returns ``(paths, {"mean_profile": Figure, "std_profile": Figure})``.
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.analysis.prices import fit_samplers, simulate_price_paths, price_figures

    df = load_prices(_log=_log)
    if exclude_crisis:
        df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
    samplers = fit_samplers(df, progress_cb=progress_cb)
    paths = simulate_price_paths(samplers, n_days=n_days, season=season,
                                 daytype=daytype, seed=seed)
    fig_mean, fig_std = price_figures(paths)
    return paths, {"mean_profile": fig_mean, "std_profile": fig_std}


def fit_mdn(n_components: int = 3, hidden_dims: list[int] | None = None,
            epochs: int = 200, batch_size: int = 1024, lr: float = 1e-3,
            exclude_crisis: bool = True, _log: Callable | None = None,
            _progress: Callable | None = None):
    """Fit the MDN price sampler and build the two training-curve figures.

    Returns ``(sampler, history, {"mdn_nll": Figure, "mdn_components": Figure})``.
    """
    from ev_mdt.pricing.entsoe import load_prices
    from ev_mdt.pricing.samplers import MDNSampler
    from ev_mdt.plots.mdn import fig_mdn_nll, fig_mdn_components

    df = load_prices(_log=_log)
    if exclude_crisis:
        df = df[~df["timestamp"].dt.year.isin(CRISIS_YEARS)]
    sampler = MDNSampler(n_components=n_components, hidden_dims=hidden_dims,
                         epochs=epochs, batch_size=batch_size, lr=lr)
    history: list[dict] = []
    sampler.fit(df, _progress=_progress, _history=history)
    figs = {
        "mdn_nll":        fig_mdn_nll(history),
        "mdn_components": fig_mdn_components(history, n_components),
    }
    return sampler, history, figs
