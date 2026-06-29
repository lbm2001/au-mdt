"""ev_mdt — EV Charging MDP package.

High-level API
--------------
    from ev_mdt import solve, sweep, baseline_figures, sensitivity_figures

    result = solve(model="Baseline")          # solve with default params
    sweep_results = sweep(sweeps=["penalty"]) # run one sensitivity sweep (exact)
    _, figs = baseline_figures()              # baseline paper figures
"""
from ev_mdt.params import (
    BaselineParams,
    NegBinParams,
    SolverConfig,
    BASELINE_MODEL,
    NEGBIN_FIXED_MODEL,
    NEGBIN_SAMPLED_MODEL,
    MODEL_LABELS,
    N_e,
    T_hours,
)
from ev_mdt.analysis.sensitivity import (
    solve as _solve_fn,
    build_params,
    run_all_sweeps as sweep,
    ALL_SWEEP_NAMES,
    baseline_figures,
    sensitivity_figures,
    calibrate_du_figures,
    model_trajectory_figure,
    price_model_figures,
    fit_mdn,
)
from ev_mdt.models.common.model_utils import price_bin_probs


def solve(
    model: str = BASELINE_MODEL,
    N_e_override: int | None = None,
    T_hours_override: int | None = None,
    **param_overrides,
):
    """Solve the MDP for the given model and return a result dict.

    Returns the same dict structure as the internal sweep step results, so it
    can be passed directly to any ev_mdt.plots function.

    Parameters
    ----------
    model          : mobility model label (BASELINE_MODEL / NEGBIN_FIXED_MODEL / …)
    N_e_override   : override discretisation grid size (default N_e=500)
    T_hours_override: override horizon in hours (default T_hours=24)
    **param_overrides: passed verbatim to build_params (e.g. phi=10.0, beta=0.98)
    """
    _N_e = N_e_override if N_e_override is not None else N_e
    _T   = (T_hours_override if T_hours_override is not None else T_hours) * 60
    params = build_params(model, **param_overrides)
    pbp_fn = lambda t, p=params: price_bin_probs(t, p)
    pi, actions, e_grid, lam_grid = _solve_fn(model, params, pbp_fn, _T, _N_e)
    return {
        "model":     model,
        "label":     model,
        "params":    params,
        "pbp_fn":    pbp_fn,
        "pi":        pi,
        "actions":   actions,
        "e_grid":    e_grid,
        "lam_grid":  lam_grid,
        "T":         _T,
    }
