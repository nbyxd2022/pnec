"""Check the distance rule, equal-budget control and native solver inputs."""

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_secondary_peak_taper as audit
import run_weak_scale_audit as weak
from test_fixed_input_audit import fixtures


class TaperTests(unittest.TestCase):
    def test_distances_are_measured_from_endpoint(self):
        rows = fixtures()[0]
        rows["delta_trace"] = [1, 2, 3]
        rows["best_dx"] = [2, -2, 2.01]
        rows["mode_dx"] = [14, 14, 14]
        targeted, matched = audit.arm_traces(rows)
        # Second peaks at 16, 12 and 16.01 px. Only the first point qualifies.
        np.testing.assert_allclose(targeted, [(12 / 16)**2, 2, 3])
        np.testing.assert_allclose(matched / rows["delta_trace"], np.full(3, targeted.sum() / 6))
        self.assertAlmostEqual(targeted.sum(), matched.sum())

    def test_two_dimensional_distance_missing_minimum_and_zero_increments(self):
        rows = fixtures()[0]
        rows["delta_trace"] = [4, 2, 0]
        rows["mode_dx"], rows["mode_dy"] = 12, 9
        rows["second_cost"][1] = np.inf
        tapered, matched = audit.arm_traces(rows)
        np.testing.assert_allclose(tapered, [4 * (12 / 15)**2, 2, 0])
        self.assertEqual(matched[2], 0)
        rows["delta_trace"] = 0
        for arm in audit.arm_traces(rows):
            np.testing.assert_array_equal(arm, [0, 0, 0])

    def test_native_wire_preserves_original_modes_and_applies_weak_scale_once(self):
        inputs = fixtures()
        inputs[0]["mode_dx"] = [0, 9, 16]
        tapered, matched = audit.arm_traces(inputs[0])
        initial = (np.eye(3), np.array([1., 0., 0.]))
        for extra, modes in ((None, weak.ALL_MODES), ((tapered, matched), weak.TAPER_MODES)):
            output = "".join(name + " 1 0 0 0 1 0 0 0 1\n" for name in modes)
            with patch("run_weak_scale_audit.subprocess.run", return_value=SimpleNamespace(stdout=output)) as call:
                rotations = weak.solve_pair(Path("pnec_fixed_refine"), inputs, initial, taper_and_matched=extra)
            self.assertEqual(tuple(rotations), modes)
            args, kwargs = call.call_args
            self.assertEqual(args[0][1:], [] if extra is None else ["--secondary-peak"])
            lines = kwargs["input"].splitlines()
            self.assertEqual(float(lines[0].split()[2]), 1 / 32)
            matrix = np.loadtxt(StringIO("\n".join(lines[5:])), ndmin=2)
            self.assertEqual(matrix.shape, (3, 25 if extra is None else 27))
            np.testing.assert_array_equal(matrix[:, 24], inputs[0]["delta_trace"])
            if extra is not None:
                np.testing.assert_array_equal(matrix[:, 25], tapered)
                np.testing.assert_array_equal(matrix[:, 26], matched)

    def test_window_checks_coverage_controls_and_computes_signed_differences(self):
        inputs = fixtures()
        inputs[0]["mode_dx"] = [0, 9, 16]
        eye = np.eye(3)
        angle = .01
        step = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        rotations = dict.fromkeys(weak.TAPER_MODES, eye)
        rotations["matched_budget"] = step
        captured = {"initial": (eye, np.array([1., 0., 0.])), "symmetric_analytic": (eye, None)}
        prior = {"frames": [0, 2], "pairs": 2, "observations": 6,
                 "fixed_input_rotation_metrics": {k: {"rpe_1_deg": 0., "rpe_n_deg": 0.}
                                                  for k in ["symmetric_analytic", "weak_per_match"]}}
        truth = np.stack([eye, eye, eye])
        with patch.object(audit, "pair_inputs", return_value=(inputs, captured)), \
                patch.object(audit, "solve_pair", return_value=rotations):
            summary = audit.evaluate(None, {}, prior, truth, None)
            self.assertEqual(summary["observations_attenuated"], 2)
            self.assertEqual(summary["pairs_with_attenuation"], 2)
            self.assertAlmostEqual(summary["far_peak_taper_minus_control_deg"]["matched_budget"]["rpe_1_deg"],
                                   -np.rad2deg(angle))
            prior["fixed_input_rotation_metrics"]["weak_per_match"]["rpe_n_deg"] = 2.
            with self.assertRaisesRegex(ValueError, "archived weak_per_match"):
                audit.evaluate(None, {}, prior, truth, None)


if __name__ == "__main__":
    unittest.main()
