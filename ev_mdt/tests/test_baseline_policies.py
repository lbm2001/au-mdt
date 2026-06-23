import unittest

import numpy as np

from ev_mdt.params import BaselineParams
from ev_mdt.models.common.policies import (
    actual_charge_rate,
    backward_induction_policy,
    maximal_charging_policy,
    price_oriented_policy,
)
from ev_mdt.models.baseline.rollout import simulate_policy_rollout
from ev_mdt.models.common.rollout_utils import rollout_metrics


class BaselinePolicyTests(unittest.TestCase):
    def test_maximal_policy_requests_max_rate(self):
        params = BaselineParams(u_max=11.0)

        self.assertEqual(
            maximal_charging_policy(t=0, chi=0, e=20.0, lam=100.0, params=params),
            11.0,
        )

    def test_price_oriented_policy_uses_thresholds(self):
        params = BaselineParams(u_max=12.0)

        self.assertEqual(
            price_oriented_policy(
                t=0, chi=0, e=20.0, lam=50.0, params=params,
                low_threshold=70.0, high_threshold=150.0,
            ),
            12.0,
        )
        self.assertEqual(
            price_oriented_policy(
                t=0, chi=0, e=20.0, lam=100.0, params=params,
                low_threshold=70.0, high_threshold=150.0,
            ),
            6.0,
        )
        self.assertEqual(
            price_oriented_policy(
                t=0, chi=0, e=20.0, lam=180.0, params=params,
                low_threshold=70.0, high_threshold=150.0,
            ),
            0.0,
        )

    def test_actual_charge_rate_blocks_driving_charge_above_minimum(self):
        params = BaselineParams(e_min=0.0, u_max=11.0)

        self.assertEqual(actual_charge_rate(chi=1, e=5.0, desired_u=11.0, params=params), 0.0)
        self.assertEqual(actual_charge_rate(chi=1, e=0.0, desired_u=11.0, params=params), 11.0)

    def test_backward_induction_policy_uses_price_bin_and_nearest_energy(self):
        params = BaselineParams(K=2, lambda_max=200.0)
        actions = np.array([0.0, 1.0, 5.0, 10.0])
        e_grid = np.array([0.0, 10.0])
        pi = np.zeros((1, 2, 2, 2), dtype=int)
        pi[0, 0, 1, 1] = 3

        self.assertEqual(
            backward_induction_policy(
                t=0, chi=0, e=9.0, lam=150.0, params=params,
                pi=pi, actions=actions, e_grid=e_grid,
            ),
            10.0,
        )

    def test_rollout_metrics_are_numeric_for_shared_scenario(self):
        params = BaselineParams(
            u_max=12.0,
            e_max=20.0,
            p_pd_default=0.0,
            p_dp_default=1.0,
            sigma_lambda=0.0,
        )
        scenario = {
            "lam_path": np.array([50.0, 100.0, 200.0]),
            "mobility_draws": np.array([1.0, 1.0, 1.0]),
        }
        rollout = simulate_policy_rollout(
            price_oriented_policy,
            scenario,
            e0=10.0,
            chi0=0,
            params=params,
            low_threshold=70.0,
            high_threshold=150.0,
        )
        metrics = rollout_metrics(rollout, params)

        self.assertEqual(set(metrics), {
            "Total cost (€)",
            "Energy charged (kWh)",
            "Penalty minutes",
            "Final battery (kWh)",
            "Mean charge rate while parked (kW)",
        })
        self.assertEqual(metrics["Penalty minutes"], 0)
        self.assertGreater(metrics["Energy charged (kWh)"], 0.0)


if __name__ == "__main__":
    unittest.main()
