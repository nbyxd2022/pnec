"""Exercise cross-sequence frame indexing, baseline replay and capture reuse on a tiny clip."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import run_taper_transfer as runner
from run_fixed_input_audit import MODES, record_pairs
from run_weak_scale_audit import TAPER_MODES
from test_fixed_input_audit import fixtures, write_points


def z_rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])


class TransferTests(unittest.TestCase):
    def test_native_command_uses_local_ids_and_acquires_correct_image_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for i in (450, 451, 452):
                (source / f"{i:06d}.png").touch()
            with patch.object(Path, "symlink_to", autospec=True) as link, \
                    patch("run_fixed_input_audit.subprocess.run") as launch:
                record_pairs(root, root / "pnec_vo", [], source, 450,
                             np.array([0., .1, .2]), range(2, 3), root)
            self.assertEqual([c.args[1].name for c in link.call_args_list],
                             ["000450.png", "000451.png", "000452.png"])
            self.assertEqual(launch.call_args.kwargs["env"]["PNEC_MATCH_DIAGNOSTICS_FRAMES"], "2")
            self.assertEqual(launch.call_args.kwargs["env"]["PNEC_PHOTOMETRIC_INCREMENT"], "symmetric_analytic")

    def test_global_ground_truth_matches_local_capture_and_reuse_detects_tampering(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            build = root / "build"
            dataset = root / "dataset"
            out = root / "result"
            build.mkdir()
            repo.mkdir()
            (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}\n")
            for name in ["pnec_vo", "pnec_fixed_refine"]:
                (build / name).write_bytes(b"fixture")
            config = repo / "config.yaml"
            config.write_bytes(b"fixture-config")
            images = dataset / "sequences/09/image_0"
            images.mkdir(parents=True)
            (dataset / "poses").mkdir()
            for frame in range(1, 5):
                (images / f"{frame:06d}.png").write_bytes(b"image")
            np.savetxt(images.parent / "times.txt", np.arange(5.) / 10.)
            angles = np.array([0., .02, .05, .10, .21])
            gt = np.array([np.column_stack((z_rotation(a), np.zeros(3))) for a in angles])
            np.savetxt(dataset / "poses/09.txt", gt.reshape(5, 12))

            def fake_capture(repo, exe, config, source, start, times, targets, out):
                self.assertEqual((start, list(targets)), (1, [2, 3]))
                folder = out / "capture/ablation"
                folder.mkdir(parents=True)
                steps = np.r_[0., np.diff(angles[start:])]
                rows = np.zeros((4, 8))
                rows[:, 0], rows[:, 6], rows[:, 7] = times, np.sin(steps / 2), np.cos(steps / 2)
                np.savetxt(folder / "PNEC.txt", rows)
                for target in targets:
                    write_points(folder, target - 1, target, fixtures())
                    with (folder / f"fixed_{target - 1}_{target}_poses.csv").open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["mode", *[f"R{r}{c}" for r in range(3) for c in range(3)], "t0", "t1", "t2"])
                        for mode in ("initial", *MODES):
                            writer.writerow([mode, *z_rotation(steps[target]).ravel(), 1, 0, 0])

            def fake_solve(executable, inputs, initial, *, taper_and_matched):
                return dict.fromkeys(TAPER_MODES, initial[0])

            with patch.object(runner, "HOLDOUT", (1, 2, 5)), \
                    patch.object(runner, "replay_config", return_value=[config]), \
                    patch.object(runner.subprocess, "check_output", return_value="fixture\n"), \
                    patch.object(runner, "record_pairs", side_effect=fake_capture) as acquire, \
                    patch("run_secondary_peak_taper.solve_pair", side_effect=fake_solve):
                runner.run(repo, build, dataset, "09", out)
                runner.run(repo, build, dataset, "09", out)
                acquire.assert_called_once()
                w = json.loads((out / "report.json").read_bytes())["window"]
                self.assertEqual((w["dataset_frames"], w["local_frames"], w["pairs"]), ([2, 4], [1, 3], 2))
                self.assertLess(w["max_capture_vs_online_rotation_difference_deg"], 1e-7)
                self.assertLess(w["fixed_input_rotation_metrics"]["symmetric_analytic"]["rpe_n_deg"], 1e-5)
                (out / "capture/ablation/fixed_1_2_points.csv").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runner.run(repo, build, dataset, "09", out)


if __name__ == "__main__":
    unittest.main()
