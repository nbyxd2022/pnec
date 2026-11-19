"""Compare fixed PoseLib and SIMFIT++ thresholds on saved KITTI 09 KLT matches."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from subprocess import check_output
from time import perf_counter

import numpy as np
import poselib
from simfitpp import SIMFITPP, PoseLibEssential
from simfitpp.utils import calib_matrix_to_camera_dict

from analyze_photometric_drift import project_rotations
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest
from run_weak_scale_audit import control_difference_deg


START_THRESHOLD = 1.0
SEEDS = (0, 1, 2)
RANSAC = {"min_iterations": 500, "max_iterations": 500, "success_prob": 0.9999}


def read_matches(path, target):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 10 or any(
        (int(row["host_frame"]), int(row["target_frame"]), int(row["solver_index"]))
        != (target - 1, target, index)
        or row["ransac_enabled"] != "1"
        or row["ransac_inlier"] not in ("0", "1")
        for index, row in enumerate(rows)
    ):
        raise ValueError(f"Unexpected match records for frame {target}: {path}")
    points = [np.array([[float(row[f"{side}_{axis}_px"]) for axis in ("x", "y")]
                        for row in rows]) for side in ("host", "target")]
    if not all(np.isfinite(array).all() for array in points):
        raise ValueError(f"Nonfinite match coordinates: {path}")
    return points[0], points[1], sum(row["ransac_inlier"] == "1" for row in rows)


def solve(host, target, camera, reference, threshold, seed):
    start = perf_counter()
    pose, info = poselib.estimate_relative_pose(
        host, target, camera, camera,
        {**RANSAC, "max_epipolar_error": float(threshold), "seed": int(seed)}, {},
    )
    elapsed = (perf_counter() - start) * 1000
    inliers = np.asarray(info["inliers"], dtype=bool)
    if inliers.shape != (len(host),):
        raise ValueError("PoseLib returned an invalid inlier mask")
    count = int(inliers.sum())
    error = None
    if count >= 5 and np.isfinite(pose.R).all():
        error = control_difference_deg(pose.R, reference)
    return {"inliers": count, "rotation_error_deg": error, "solve_ms": elapsed}, inliers


def evaluate_pair(file, target, gt, intrinsics, camera, tight_threshold):
    host, next_frame, native_inliers = read_matches(file, target)
    # PoseLib maps host rays into the target camera. KITTI rotations map cameras into the first frame.
    reference = gt[target].T @ gt[target - 1]
    trials = []
    for seed in SEEDS:
        estimator = PoseLibEssential(**RANSAC)
        estimator.opt["seed"] = seed
        np.random.seed(seed)
        fixed, fixed_mask = solve(host, next_frame, camera, reference, START_THRESHOLD, seed)
        start = perf_counter()
        threshold, success = SIMFITPP(estimator).estimate_threshold(
            host, next_frame, START_THRESHOLD, K_A=intrinsics, K_B=intrinsics,
        )
        threshold_ms = (perf_counter() - start) * 1000
        adaptive, adaptive_mask = solve(host, next_frame, camera, reference, threshold, seed)
        tight_fixed, _ = solve(host, next_frame, camera, reference, tight_threshold, seed)
        trials.append({
            "seed": seed,
            "adapted_threshold_px": float(threshold),
            "threshold_estimation_succeeded": bool(success),
            "threshold_estimation_ms": threshold_ms,
            "fixed": fixed,
            "adaptive": adaptive,
            "tight_fixed": tight_fixed,
            "mask_diff_count": int(np.count_nonzero(fixed_mask != adaptive_mask)),
        })
    return {"target_frame": target, "matches": len(host), "native_pnec_inliers": native_inliers,
            "matches_sha256": digest(file), "trials": trials}


def main(args):
    repo = Path(__file__).resolve().parents[2] if args.repo is None else args.repo.resolve(strict=True)
    archives = repo / "docs/experiments/results"
    capture = archives / "pnec_kitti_09_feedback_inputs_49cec8d_user_report.json"
    source = json.loads(capture.read_bytes())
    baseline = archives / "pnec_kitti_09_original_feedback_a89d06a_user_report.json"
    manifest = json.loads(baseline.read_bytes())["manifest"]
    gt_bytes = args.ground_truth.read_bytes().replace(b"\r\n", b"\n")
    gt_hash = sha256(gt_bytes).hexdigest()
    if gt_hash != manifest["ground_truth_sha256"]:
        raise ValueError("Ground truth differs from the archived KITTI 09 baseline")
    gt = project_rotations(np.loadtxt(gt_bytes.splitlines()).reshape(-1, 3, 4)[:, :, :3])
    config = repo / "data/config_kitti04-10.yaml"
    params = dict(line.split(":", 1) for line in config.read_text().splitlines() if line.startswith("Camera."))
    if any(float(params["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI input")
    fx, fy, cx, cy = (float(params["Camera." + name]) for name in ("fx", "fy", "cx", "cy"))
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera = calib_matrix_to_camera_dict(intrinsics)
    rows = {}
    for mode, directory in (("weak_uniform", args.uniform), ("weak_per_match", args.per_match)):
        rows[mode] = []
        for frame in source["frames"]:
            target = frame["target_frame"]
            name = f"matches_{target - 1}_{target}.csv"
            path = directory / name
            if digest(path) != source["capture_sha256"][mode]["capture/ablation/" + name]:
                raise ValueError(f"Capture differs from the archived {mode} run: {path}")
            rows[mode].append(evaluate_pair(path, target, gt, intrinsics, camera, args.tight_fixed_threshold))
    report = {
        "sequence": "09",
        "scope": "Selected paired KLT captures and offline PoseLib essential matrix estimation",
        "simfitpp_commit": check_output(["git", "-c", f"safe.directory={args.simfitpp_repo.resolve().as_posix()}",
                                        "-C", str(args.simfitpp_repo), "rev-parse", "HEAD"], text=True).strip(),
        "dependencies": {package: version(package) for package in ("simfitpp", "poselib", "numpy", "scipy")},
        "capture_archive_sha256": digest(capture), "ground_truth_sha256_lf": gt_hash,
        "intrinsics": intrinsics.tolist(), "intrinsics_source_sha256": digest(config),
        "runner_sha256": digest(Path(__file__)), "fixed_threshold_px": START_THRESHOLD,
        "post_hoc_fixed_threshold_px": args.tight_fixed_threshold,
        "ransac_options": RANSAC, "seeds": SEEDS, "frames_by_feedback_capture": rows,
    }
    atomic_json(args.output, report)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ground-truth", "uniform", "per-match", "simfitpp-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--tight-fixed-threshold", type=float, required=True,
                        help="Additional fixed threshold in pixels for a post-hoc sensitivity comparison")
    args = parser.parse_args()
    if not np.isfinite(args.tight_fixed_threshold) or args.tight_fixed_threshold <= 0:
        parser.error("--tight-fixed-threshold must be a positive finite number")
    main(args)
