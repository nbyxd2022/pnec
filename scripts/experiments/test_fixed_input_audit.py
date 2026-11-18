"""Check diagnostic geometry, pairing and report validation without native replay."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import run_fixed_input_audit as audit


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def fixtures():
    names = ["solver_index", "feature_id", "ratio", "delta_trace", "mean_trace",
             "best_cost", "second_cost", "endpoint_cost", "best_dx", "best_dy",
             "mode_dx", "mode_dy", "regularization"]
    names += [f"{prefix}{k}" for prefix in ("host_b", "target_b") for k in range(3)]
    names += [f"{prefix}{r}{c}" for prefix in ("host_cov", "target_cov")
              for r in range(3) for c in range(3)]
    rows = np.zeros(3, dtype=[(name, float) for name in names])
    rng = np.random.RandomState(17)
    left, right = rng.normal(size=(2, 3, 3))
    left /= np.linalg.norm(left, axis=1)[:, None]
    right /= np.linalg.norm(right, axis=1)[:, None]
    for prefix, bearings in (("host_b", left), ("target_b", right)):
        for k in range(3):
            rows[f"{prefix}{k}"] = bearings[:, k]
    host = np.stack([2e-6 * (np.eye(3) - b[:, None] @ b[None]) for b in left])
    target = np.stack([3e-6 * (np.eye(3) - b[:, None] @ b[None]) for b in right])
    for prefix, covariance in (("host_cov", host), ("target_cov", target)):
        for r in range(3):
            for c in range(3):
                rows[f"{prefix}{r}{c}"] = covariance[:, r, c]
    rows["solver_index"] = [4, 1, 7]
    rows["feature_id"] = [41, 11, 71]
    rows["delta_trace"] = [0, 1e-6, 4e-6]
    rows["mean_trace"] = rows["delta_trace"].mean()
    rows["regularization"] = 1e-13
    rows["ratio"] = [.1, .4, .7]
    return rows, left, right, host, target


def write_points(folder, host, target, inputs):
    rows = inputs[0]
    prefix = folder / f"fixed_{host}_{target}"
    np.savetxt(str(prefix) + "_points.csv", np.column_stack([rows[n] for n in rows.dtype.names]),
               delimiter=",", header=",".join(rows.dtype.names), comments="", fmt="%.17g")
    with (folder / f"matches_{host}_{target}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_index", "feature_id", "ransac_inlier"])
        # Match diagnostics use a different order from the solver's inlier vector.
        for i in [1, 0, 2]:
            writer.writerow([int(rows["solver_index"][i]), int(rows["feature_id"][i]), 1])
        writer.writerow([2, 21, 0])
    return Path(str(prefix) + "_points.csv"), folder / f"matches_{host}_{target}.csv"


class AuditTests(unittest.TestCase):
    def test_residual_variance_matches_direct_symmetric_formula(self):
        inputs = fixtures()
        rows, left, right, host, target = inputs
        angle = .13
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        translation = np.array([.2, -.3, .9])
        result = audit.variance_effect(inputs, rotation, translation)
        t = translation / np.linalg.norm(translation)
        expected = []
        for i, (a, b) in enumerate(zip(left, right)):
            cross = skew(a)
            g = np.cross(rotation @ b, t)
            nominal = (t @ cross @ rotation @ target[i] @ rotation.T @ cross.T @ t
                       + g @ host[i] @ g + rows["regularization"][i])
            projection = np.eye(3) - b[:, None] @ b[None]
            per_cov = rows["delta_trace"][i] / 2 * projection
            flat_cov = rows["mean_trace"][i] / 2 * projection
            extra = t @ cross @ rotation @ per_cov @ rotation.T @ cross.T @ t
            flat = t @ cross @ rotation @ flat_cov @ rotation.T @ cross.T @ t
            residual = t @ np.cross(a, rotation @ b)
            expected.append([extra / nominal, flat / nominal, residual**2 / nominal])
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result, audit.variance_effect(inputs, rotation, -3 * translation))

    def test_score_and_tie_ranks(self):
        np.testing.assert_array_equal(audit.ranks(np.array([4., 1., 4., 2.])), [2.5, 0, 2.5, 1])
        summary = audit.effect_summary(np.array([[0., 0., .1], [0., 0., 20.]]))
        self.assertIsNone(summary["rank_correlation_increment_vs_nominal_standardized_residual2"])
        for mode in audit.MODES[1:]:
            self.assertEqual(summary[mode]["mean_gaussian_nll_delta"], 0)
        high = audit.effect_summary(np.array([[3., 3., 20.], [1., 1., 10.]]))
        self.assertLess(high[audit.MODES[2]]["mean_gaussian_nll_delta"], 0)

    def test_point_identity_and_covariance_validation(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            inputs = fixtures()
            points, matches = write_points(folder, 0, 1, inputs)
            loaded = audit.read_points(points, matches)
            np.testing.assert_array_equal(loaded[0]["solver_index"], [4, 1, 7])
            matches.write_text(matches.read_text().replace("41", "42"))
            with self.assertRaisesRegex(ValueError, "identities differ"):
                audit.read_points(points, matches)
            points, matches = write_points(folder, 0, 1, inputs)
            inputs[0]["target_cov00"][0] = -1
            write_points(folder, 0, 1, inputs)
            with self.assertRaisesRegex(ValueError, "projected covariance"):
                audit.read_points(points, matches)

    def test_analyze_capture_and_reject_changed_baseline(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            folder = out / "capture/ablation"
            folder.mkdir(parents=True)
            dataset = out / "dataset"
            (dataset / "poses").mkdir(parents=True)
            truth = np.zeros((3, 3, 4))
            truth[:, :, :3] = np.eye(3)
            truth[:, 2, 3] = np.arange(3)
            np.savetxt(dataset / "poses/08.txt", truth.reshape(3, 12))
            previous = np.zeros((3, 8))
            previous[:, 0] = np.arange(3) * .1
            previous[:, 7] = 1
            np.savetxt(folder / "PNEC.txt", previous)
            expected = {"selected_target_frames": [1, 2], "windows": [[0, 2]]}
            (out / "manifest.json").write_text(json.dumps({"action": "smoke", "inputs": expected}))
            pose_fields = ["mode"] + [f"R{r}{c}" for r in range(3) for c in range(3)] + ["t0", "t1", "t2"]
            for target in [1, 2]:
                write_points(folder, target - 1, target, fixtures())
                with (folder / f"fixed_{target - 1}_{target}_poses.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(pose_fields)
                    for mode in ["initial", *audit.MODES]:
                        writer.writerow([mode, *np.eye(3).ravel(), 0, 0, 1])
            with patch.object(audit, "baseline_inputs", return_value=(expected, previous[:, 0], previous.copy(), [])):
                report = audit.analyze(out, dataset)
                self.assertTrue(report["baseline_prefix_exact_match"])
                self.assertEqual(report["windows"][0]["pairs"], 2)
                self.assertEqual(report["windows"][0]["fixed_input_rotation_metrics"][audit.MODES[2]]["rpe_n_deg"], 0)
                self.assertTrue((out / "report.json").is_file())
                previous[2, 3] = 1
                np.savetxt(folder / "PNEC.txt", previous)
                with self.assertRaisesRegex(ValueError, "Baseline prefix changed"):
                    audit.analyze(out, dataset)


if __name__ == "__main__":
    unittest.main()
