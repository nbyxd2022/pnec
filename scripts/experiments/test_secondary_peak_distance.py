"""Regression checks for secondary-peak coordinates, epipolar geometry and matching."""

import unittest

import numpy as np

import analyze_secondary_peak_distance as audit


class DistanceTests(unittest.TestCase):
    def test_target_plane_with_rotation_and_observed_host(self):
        a = .47
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        t = np.array([.3, -.7, .1])
        p = np.array([.4, -.2, 5.])
        left = p / np.linalg.norm(p)
        target = rotation.T @ (p - t)
        target /= np.linalg.norm(target)
        normal = rotation.T @ np.cross(t, left)
        normal /= np.linalg.norm(normal)
        perturbed = np.cos(.002) * target + np.sin(.002) * normal
        rays = np.array([target, perturbed])
        hosts = np.tile(left, (2, 1))
        for baseline in [t, -3 * t]:
            np.testing.assert_allclose(audit.epipolar_angles(hosts, rays, rotation, baseline),
                                       [0., 2.], atol=1e-10)
        self.assertTrue(np.isnan(audit.epipolar_angles(hosts, rays, rotation, np.zeros(3))).all())
        self.assertTrue(np.isnan(audit.epipolar_angles(hosts, rays, rotation, left)).all())

    def test_endpoint_offsets_cutoffs_and_unmatched_bins(self):
        names = ["best_cost", "second_cost", "best_dx", "best_dy", "mode_dx", "mode_dy",
                 "delta_trace", "ratio"]
        rows = np.zeros(9, dtype=[(k, float) for k in names])
        rows["second_cost"] = 1
        rows["ratio"] = [.11, .13, .42, .42, .42, .7, .12, .12, .12]
        rows["best_cost"] = rows["ratio"]
        rows["best_dx"] = [1, -1, 0, 0, 0, 0, 3, 0, 0]
        rows["mode_dx"] = [7, 13, 6, 6, 13, 5, 12, 10, 6]
        rows["delta_trace"] = [1, 3, 2, 4, 9, 100, 100, 101, 102]
        # Bin 0.10-0.15 compares distances 8 and 12. Bin 0.40-0.45 has two
        # near and one far, hence the same overlap weight (one) as the first.
        counts, means = audit.compare_frame(rows, np.arange(9., dtype=float))
        self.assertEqual((counts["near_compared"], counts["far_compared"], counts["near_available"]),
                         (4, 2, 5))
        # Row 8 is an additional near point in the first bin. Mark it unobservable.
        angles = np.arange(9., dtype=float)
        angles[8] = np.nan
        counts, means = audit.compare_frame(rows, angles)
        self.assertEqual([counts[k] for k in ("near_compared", "far_compared", "overlap_weight",
                                              "best_outside", "middle", "invalid_plane")],
                         [3, 2, 2, 1, 1, 1])
        trace = audit.FIELDS.index("added_trace_rad2")
        np.testing.assert_allclose(means[:, trace], [2, 6])
        np.testing.assert_allclose(means[:, audit.FIELDS.index("second_to_endpoint_px")], [7, 12.5])
        np.testing.assert_allclose(means[:, -1], [3.25, 8.5])
        rows["second_cost"][:] = np.inf
        counts, empty = audit.compare_frame(rows, angles)
        self.assertIsNone(empty)
        self.assertEqual(counts["missing_two_minima"], 9)
        self.assertEqual(audit.window_summary([10, 11], counts, [])["comparable_frame_pairs"], 0)

    def test_equal_frame_weights_and_squared_error(self):
        a = np.ones((2, len(audit.FIELDS)))
        b = a.copy()
        b[1, -1] = 9
        summary = audit.window_summary([20, 23], {}, [a, b])
        self.assertEqual(summary["comparable_frame_pairs"], 2)
        self.assertEqual(summary["far"]["rms_epipolar_mrad"], np.sqrt(5.))
        self.assertEqual(summary["far_over_near"]["squared_epipolar_mrad2"], 5.)


if __name__ == "__main__":
    unittest.main()
