import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from dataclasses import dataclass
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    return dataclass, mcolors, mpatches, mticker, np, plt


@app.cell
def _(dataclass):
    @dataclass
    class MDPParams:
        u_max: float = 11.0         # max charge rate (kW)
        u_min: float = 1.4          # min charge rate (kW)
        e_max: float = 40.0         # max battery level (kWh)
        e_min: float = 0.0          # min battery level (kWh)
        phi: float = 1000.0         # unserved-driving penalty (€/h)
        eta_c: float = 0.95         # charging efficiency
        kappa: float = 40.0         # battery capacity (kWh)
        omega: float = 1 / 60       # min → hours conversion (h/min)
        beta: float = 0.999         # discount factor
        sigma_lambda: float = 20.0  # price std dev (€/MWh)
        v: float = 50.0             # average driving speed (km/h)
        mu: float = 0.2             # drive efficiency (kWh/km)

    params = MDPParams()
    params
    return (params,)


@app.cell
def _():
    PARKED  = 0
    DRIVING = 1

    def actual_driving_state(chi: int, e: float, e_min: float) -> int:
        """χ_t^a: same as χ_t unless driving with empty battery."""
        if chi == DRIVING and e <= e_min:
            return DRIVING
        return chi

    return (DRIVING,)


@app.cell
def _(DRIVING, np, params):
    actions = np.array([0.0, params.u_min, params.u_max / 2, params.u_max])

    def actual_action(u: float, chi: int, e: float) -> float:
        """u_t^a: zero when driving with battery above minimum."""
        if chi == DRIVING and e > params.e_min:
            return 0.0
        return u

    return (actual_action,)


@app.cell
def _(np, params):
    def mean_price(t: int) -> float:
        """Time-dependent mean electricity price λ̄_t (€/MWh); t is minute of day [0, 1440)."""
        h = t / 60
        if h < 6:
            return 70.0
        elif h < 9:
            return 150.0
        elif h < 16:
            return 110.0
        elif h < 21:
            return 170.0
        else:
            return 100.0

    def sample_price(t: int, rng) -> float:
        """Sample λ_t from truncated N(λ̄_t, σ²), clipped below at 0."""
        raw = rng.normal(mean_price(t), params.sigma_lambda)
        return float(np.maximum(0.0, raw))

    return (mean_price,)


@app.cell
def _(np):
    def transition_probs(t: int) -> tuple[float, float]:
        """
        Returns (p_PD, p_DP) at minute t of day.
          p_PD: parked  -> driving probability
          p_DP: driving -> parked  probability
        """
        h = t / 60

        if 7.0 <= h < 9.0:
            p_PD = 0.08
        elif 12.0 <= h < 14.0:
            p_PD = 0.03
        elif 16.0 <= h < 18.0:
            p_PD = 0.07
        else:
            p_PD = 0.005

        if 7.5 <= h < 9.5:
            p_DP = 0.15
        elif 12.25 <= h < 14.25:
            p_DP = 0.20
        elif 16.5 <= h < 18.5:
            p_DP = 0.15
        else:
            p_DP = 0.25

        return p_PD, p_DP

    def transition_matrix(t: int):
        """2×2 transition matrix P_t at minute t (row = from-state, col = to-state)."""
        p_PD, p_DP = transition_probs(t)
        return np.array([
            [1 - p_PD, p_PD   ],
            [p_DP,     1 - p_DP],
        ])

    return (transition_probs,)


@app.cell
def _(DRIVING, actual_action, params):
    def reward(chi: int, e: float, u: float, lam: float) -> float:
        """
        One-step reward R_t(S_t, u_t).

        chi : driving state (PARKED=0, DRIVING=1)
        e   : battery energy (kWh)
        u   : desired charge rate (kW)
        lam : electricity price at t (€/MWh)
        """
        u_a = actual_action(u, chi, e)
        charging_cost = lam * params.omega * u_a
        penalty = int(chi == DRIVING and e <= params.e_min) * params.omega * params.phi
        return -(charging_cost + penalty)

    return


@app.cell
def _(DRIVING, params):
    def consumption_fn(chi: int) -> float:
        """Energy consumed per minute in driving state chi (kWh/min)."""
        return params.mu * params.v * params.omega if chi == DRIVING else 0.0

    return (consumption_fn,)


@app.cell
def _(consumption_fn, mean_price, params, transition_probs):
    from utils.backward_induction import backward_induction

    V, pi, action_set, e_grid = backward_induction(
        params,
        mean_price_fn=mean_price,
        transition_probs_fn=transition_probs,
        consumption_fn=consumption_fn,
        T=1440,
        N_e=500,
    )
    return action_set, e_grid, pi


@app.cell
def _(mean_price, mticker, np, plt):

    def plot_price():
        minutes = np.arange(1440)
        prices  = np.array([mean_price(t) for t in minutes])

        fig_price, ax = plt.subplots(figsize=(10, 3))
        ax.step(minutes / 60, prices, where="post", color="steelblue", linewidth=2)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("€ / MWh")
        ax.set_title(r"Mean electricity price $\bar{\lambda}_t$")
        ax.set_xlim(0, 24)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
        ax.grid(True, alpha=0.3)
        fig_price.tight_layout()
        plt.show()
    plot_price()
    return


@app.cell
def _(mticker, np, plt, transition_probs):
    def plot_transitions():
        minutes = np.arange(1440)
        probs   = np.array([transition_probs(t) for t in minutes])
        hours   = minutes / 60

        fig_trans, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        axes[0].step(hours, probs[:, 0], where="post", color="tab:orange", linewidth=2)
        axes[0].set_ylabel("Prob. / min")
        axes[0].set_title(r"Parked $\to$ Driving  ($p_t^{P \to D}$)")
        axes[0].grid(True, alpha=0.3)

        axes[1].step(hours, probs[:, 1], where="post", color="tab:green", linewidth=2)
        axes[1].set_ylabel("Prob. / min")
        axes[1].set_title(r"Driving $\to$ Parked  ($p_t^{D \to P}$)")
        axes[1].set_xlabel("Hour of day")
        axes[1].set_xlim(0, 24)
        axes[1].xaxis.set_major_locator(mticker.MultipleLocator(3))
        axes[1].grid(True, alpha=0.3)

        fig_trans.tight_layout()
        plt.show()
    plot_transitions()
    return


@app.cell
def _(action_set, e_grid, mcolors, mpatches, np, pi, plt):
    def plot_policy():
        hours = np.arange(1440) / 60

        # Discrete colormap: one colour per action level
        palette = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]
        cmap    = mcolors.ListedColormap(palette)
        bounds  = np.arange(len(action_set) + 1) - 0.5
        norm    = mcolors.BoundaryNorm(bounds, cmap.N)

        fig_policy, ax = plt.subplots(figsize=(10, 4))

        ax.pcolormesh(hours, e_grid, pi[:, 0, :].T, cmap=cmap, norm=norm)
        ax.set_title("Optimal policy — Parked")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Battery energy (kWh)")
        ax.set_xticks(range(0, 25, 3))

        legend_handles = [
            mpatches.Patch(color=palette[i], label=f"{action_set[i]:.1f} kW")
            for i in range(len(action_set))
        ]
        fig_policy.legend(
            handles=legend_handles,
            title="Charge rate",
            loc="lower center",
            ncol=len(action_set),
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
        )
        fig_policy.tight_layout()
        plt.show()
    plot_policy()
    return


@app.cell
def _(action_set, mean_price, mticker, np, params, pi, plt):
    def plot_charge_vs_price():
        minutes   = np.arange(1440)
        hours_min = minutes / 60

        # ── price: piecewise mean + ±1σ band ────────────────────────────────
        prices = np.array([mean_price(t) for t in minutes])
        sigma  = params.sigma_lambda

        # ── mean charge rate: average over energy grid, then hourly mean ────
        charge_rates    = action_set[pi[:, 0, :]]                  # (1440, N_e)
        mean_per_minute = charge_rates.mean(axis=1)                # (1440,)
        mean_per_hour   = mean_per_minute.reshape(24, 60).mean(axis=1)  # (24,)

        fig, ax1 = plt.subplots(figsize=(12, 4))

        # Left axis — price
        c_price = "steelblue"
        ax1.fill_between(
            hours_min, prices - sigma, prices + sigma,
            step="post", alpha=0.2, color=c_price, label=r"$\pm\sigma_\lambda$",
        )
        ax1.step(hours_min, prices, where="post", color=c_price, linewidth=2,
                 label=r"$\bar{\lambda}_t$")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("€ / MWh", color=c_price)
        ax1.tick_params(axis="y", labelcolor=c_price)
        ax1.set_xlim(0, 24)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(3))
        ax1.grid(True, alpha=0.2)

        # Right axis — mean charge rate
        c_rate = "tab:red"
        ax2 = ax1.twinx()
        ax2.step(np.arange(24), mean_per_hour, where="post",
                 color=c_rate, linewidth=2, label="Mean charge rate")
        ax2.set_ylabel("Mean charge rate (kW)", color=c_rate)
        ax2.tick_params(axis="y", labelcolor=c_rate)
        ax2.set_ylim(bottom=0)

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

        ax1.set_title(
            r"Mean optimal charge rate vs. electricity price $\bar{\lambda}_t$"
            " (parked, averaged over energy grid)"
        )
        fig.tight_layout()
        plt.show()
    plot_charge_vs_price()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
