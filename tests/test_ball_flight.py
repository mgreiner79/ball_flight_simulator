import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ball_flight import BallPathSimulator, PitchParameters


class BallFlightTests(unittest.TestCase):
    def _default_params(self, **overrides):
        params = dict(
            v0=42,
            theta=0,
            phi=2,
            spin_rate=1200,
            spin_elevation=180,
            spin_azimuth=0,
            x0=-0.5,
            y0=1.6,
            z0=0,
        )
        params.update(overrides)
        return PitchParameters(**params)

    def test_calculate_returns_time_aligned_trajectory(self):
        params = self._default_params()

        trajectory = BallPathSimulator().calculate(params, t_max=1, dt=0.01)

        self.assertEqual(trajectory.positions.shape[1], 3)
        self.assertEqual(trajectory.velocities.shape, trajectory.positions.shape)
        self.assertEqual(trajectory.time.shape[0], trajectory.positions.shape[0])
        self.assertEqual(trajectory.lift_coefficient.shape[0], trajectory.time.shape[0])
        self.assertEqual(trajectory.spin_parameter.shape[0], trajectory.time.shape[0])
        np.testing.assert_allclose(trajectory.positions[0], [-0.5, 1.6, 0])

    def test_non_integer_time_step_keeps_arrays_aligned(self):
        trajectory = BallPathSimulator().calculate(self._default_params(), t_max=1, dt=0.3)

        self.assertEqual(trajectory.time.shape[0], trajectory.positions.shape[0])
        self.assertEqual(trajectory.to_array().shape, (trajectory.time.shape[0], 4))

    def test_zero_spin_keeps_zero_lift_time_series(self):
        trajectory = BallPathSimulator().calculate(self._default_params(spin_rate=0), t_max=1, dt=0.01)

        self.assertEqual(trajectory.lift_coefficient.shape[0], trajectory.time.shape[0])
        np.testing.assert_allclose(trajectory.lift_coefficient, 0)
        np.testing.assert_allclose(trajectory.lift_force, 0)

    def test_gyrospin_has_no_release_lift(self):
        trajectory = BallPathSimulator().calculate(
            self._default_params(spin_elevation=0, spin_azimuth=90),
            t_max=0.2,
            dt=0.005,
        )

        np.testing.assert_allclose(trajectory.spin_parameter[0], 0, atol=1e-12)
        np.testing.assert_allclose(trajectory.lift_force[0], [0, 0, 0], atol=1e-12)

    def test_ground_impact_is_interpolated_to_y_zero(self):
        trajectory = BallPathSimulator().calculate(self._default_params(), t_max=5, dt=0.005)

        self.assertAlmostEqual(trajectory.y[-1], 0.0)
        self.assertTrue(np.all(trajectory.y >= 0))

    def test_invalid_simulation_settings_raise_value_error(self):
        params = self._default_params()

        with self.assertRaisesRegex(ValueError, "dt must be"):
            BallPathSimulator().calculate(params, t_max=1, dt=0)

        with self.assertRaisesRegex(ValueError, "t_max must be"):
            BallPathSimulator().calculate(params, t_max=0, dt=0.01)

        with self.assertRaisesRegex(ValueError, "mass must be"):
            BallPathSimulator(m=0).calculate(params, t_max=1, dt=0.01)

    def test_calculate_path_requires_initialization(self):
        with self.assertRaisesRegex(ValueError, "initialize_simulation"):
            BallPathSimulator().calculate_path(t_max=1, dt=0.01)

    def test_legacy_state_api_still_works(self):
        simulator = BallPathSimulator("legacy")
        simulator.initialize_simulation(
            v0=22,
            theta=0,
            phi=2,
            spin_rate=0,
            spin_elevation=0,
            spin_azimuth=0,
            x0=-0.5,
            y0=1.6,
            z0=0,
        )
        simulator.calculate_path(t_max=1, dt=0.01)

        self.assertEqual(simulator.get_path_array().shape[1], 4)
        self.assertGreater(simulator.z[-1], simulator.z[0])

    def test_core_import_does_not_load_scene_dependencies(self):
        self.assertNotIn("plotly", sys.modules)
        self.assertNotIn("pyvista", sys.modules)


if __name__ == "__main__":
    unittest.main()
