"""Compare weak uniform and per-match refinement on selected KITTI 09 frame pairs."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np

from analyze_photometric_drift import gap_errors, project_rotations
from analyze_weak_allocation_full import read_json
from run_fixed_input_audit import atomic_json, record_pairs
from run_photometric_increment import WEAK_ALLOCATION_MODES, digest, pose_rows, replay_config, restore_rotations
from run_taper_transfer import capture_index
from run_weak_scale_audit import CONTROL_TOLERANCE_DEG, WEAK_SCALE, checked_file, control_difference_deg, pair_inputs, solve_pair


SEQUENCE = "09"
SELECTED = (1058, 1104, 1105, 1106, 1119)
TARGETS = sorted({frame + offset for frame in SELECTED for offset in (-1, 0, 1)})
BASELINE, UNIFORM, PER_MATCH = WEAK_ALLOCATION_MODES


def evaluate(out, solve_exe, reference, online_errors):
    hashes = capture_index(out, TARGETS)
    frames = []
    for target in TARGETS:
        inputs, poses = pair_inputs(out, hashes, target)
        solutions = solve_pair(solve_exe, inputs, poses["initial"])
        difference = control_difference_deg(solutions[BASELINE], poses[BASELINE][0])
        if difference > CONTROL_TOLERANCE_DEG:
            raise ValueError(f"Frame {target}: native audit baseline differs by {difference:.9g} degrees")
        truth_step = reference[target - 1].T @ reference[target]
        fixed = {mode: control_difference_deg(solutions[mode], truth_step) for mode in WEAK_ALLOCATION_MODES}
        online = {mode: float(values[target - 1]) for mode, values in online_errors.items()}
        frames.append({"target_frame": target, "selected": target in SELECTED, "inliers": len(inputs[0]),
                       "added_target_trace_rad2": float(WEAK_SCALE * inputs[0]["delta_trace"].sum()),
                       "online_error_deg": online, "fixed_input_error_deg": fixed,
                       "online_per_match_minus_uniform_deg": online[PER_MATCH] - online[UNIFORM],
                       "fixed_per_match_minus_uniform_deg": fixed[PER_MATCH] - fixed[UNIFORM],
                       "replayed_analytic_difference_deg": difference})
    return frames, hashes


def run(repo, build, dataset, output_root):
    archive = repo / "docs/experiments/results"
    drift = read_json(archive / "pnec_kitti_09_weak_allocation_rotation_drift_741b480_user_report.json")
    original = Path(drift["source_run"])
    saved = drift["input_sha256"]
    online_report_path = checked_file(original / "report.json", saved["report"])
    original_manifest = read_json(checked_file(original / "manifest.json", saved["manifest"]))
    if original_manifest != read_json(online_report_path)["manifest"]:
        raise ValueError("Original replay manifest differs from its report")
    dataset_sequence = dataset / "sequences/09"
    times_file = checked_file(dataset_sequence / "times.txt", saved["times"])
    gt = checked_file(dataset / "poses/09.txt", saved["ground_truth"])
    times = np.loadtxt(times_file, ndmin=1)
    times -= times[0]
    np.testing.assert_allclose(np.loadtxt(original / "09/times.txt", ndmin=1), times, rtol=0, atol=6e-13)
    reference = project_rotations(np.loadtxt(gt, ndmin=2).reshape(-1, 3, 4)[:, :, :3])
    if len(reference) != len(times):
        raise ValueError("Ground truth and timestamp coverage differ")

    source = dataset_sequence / "image_0"
    end = max(TARGETS) + 1
    for frame in range(end):
        name = f"{frame:06d}.png"
        checked_file(source / name, original_manifest["image_sha256"][name])
    config = replay_config(repo, SEQUENCE)
    if {p.name: digest(p) for p in config} != read_json(original / "09/config_sha256.json"):
        raise ValueError("Current KITTI config differs from original replay")
    poses = {mode: checked_file(original / "09" / mode / "ablation/PNEC.txt", sha)
             for mode, sha in saved["relative_poses"].items()}
    online_errors = {mode: gap_errors(reference, restore_rotations(path, times, 0), 1)
                     for mode, path in poses.items()}

    cache = (build / "CMakeCache.txt").read_text().splitlines()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache:
        raise ValueError("Build directory belongs to another repository")
    executable, solve_exe = build / "pnec_vo", build / "pnec_fixed_refine"
    out = Path(mkdtemp(prefix="pnec_kitti09_weak_outliers_", dir=output_root))
    record_pairs(repo, executable, config, source, 0, times[:end], TARGETS, out, rotation_feedback=True)
    prefix = pose_rows(out / "capture/ablation/PNEC.txt", times[:end])
    if not np.array_equal(prefix, pose_rows(poses[BASELINE], times)[:end]):
        raise ValueError(f"Analytic baseline prefix differs from the original full replay: {out}")

    frames, capture_hashes = evaluate(out, solve_exe, reference, online_errors)
    report = {"sequence": SEQUENCE, "source_run": str(original), "source_report_sha256": saved["report"],
              "tool_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "scope": "Original analytic baseline history through the last target frame",
              "shared_inputs": ["RANSAC inliers", "bearings", "analytic covariance", "Ceres initialization"],
              "baseline_prefix_exact_match": True, "replay_end_exclusive": end,
              "target_increment_scale": WEAK_SCALE, "selected_target_frames": list(SELECTED),
              "binary_sha256": {path.name: digest(path) for path in (executable, solve_exe)},
              "capture_sha256": capture_hashes, "frames": frames}
    atomic_json(out / "report.json", report)
    print("target selected inliers online_delta_deg fixed_delta_deg", flush=True)
    for row in frames:
        print(f"{row['target_frame']} {int(row['selected'])} {row['inliers']} "
              f"{row['online_per_match_minus_uniform_deg']:+.6f} "
              f"{row['fixed_per_match_minus_uniform_deg']:+.6f}", flush=True)
    print(f"Report: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.build.resolve(strict=True),
        args.dataset.resolve(strict=True), args.output_root.resolve(strict=True))
