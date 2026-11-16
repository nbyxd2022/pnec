"""Compare native Ceres refinements on a fixed set of KITTI 01 frame pairs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np

from analyze_photometric_drift import project_rotations, rotation_vectors_deg
from run_photometric_increment import CAMERA, HOLDOUT, MODES, digest, pose_rows


CASES = (517, 519, 521, 522, 524, 672, 681, 696, 697, 699)
CONTROLS = (516, 525, 671, 700)
TARGETS = sorted((*CASES, *CONTROLS))
START, ANCHOR, END = HOLDOUT
SMOKE_TARGETS = (START + 5, START + 7)
QUATERNION = ("qx", "qy", "qz", "qw")
POSE = ("tx", "ty", "tz", *QUATERNION)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def quaternion_matrices(values):
    q = np.asarray(values)
    norms = np.linalg.norm(q, axis=-1)
    if not np.allclose(norms, 1, rtol=0, atol=1e-6):
        raise ValueError("Invalid refinement quaternion norm")
    normalized = q / norms[..., None]
    vector = normalized[..., :3]
    w = normalized[..., 3, None, None]
    cross = np.cross(vector[..., None, :], np.eye(3)).swapaxes(-1, -2)
    return 2 * (vector[..., :, None] * vector[..., None, :] + w * cross) + (2 * w**2 - 1) * np.eye(3)


def pose_delta_deg(left, right):
    return np.linalg.norm(rotation_vectors_deg(left @ right.swapaxes(-1, -2)), axis=-1)


def read_phase(phase, targets, holdout_times, original):
    frame_count = max(targets) - START + 1
    times = np.loadtxt(phase / "times.txt", ndmin=1)
    if not np.array_equal(times, holdout_times[:frame_count]):
        raise ValueError(f"{phase}: replay times differ from the holdout")
    poses = pose_rows(phase / "results/ablation/PNEC.txt", times)
    pose_rows(phase / "results/rot_avg/poses.txt", times)
    expected = original[:frame_count]
    logged_rotations = quaternion_matrices(poses[:, 4:])
    prefix_delta = pose_delta_deg(logged_rotations, quaternion_matrices(expected[:, 4:]))
    bridge = {"exact_pose_prefix": bool(np.array_equal(poses, expected)), "frames": frame_count,
              "max_rotation_difference_deg": float(prefix_delta.max()),
              "max_translation_difference": float(np.max(np.linalg.norm(poses[:, 1:4] - expected[:, 1:4], axis=1)))}
    print(f"{phase.name} original prefix {json.dumps(bridge)}", flush=True)
    if bridge["max_rotation_difference_deg"] > 1e-5:
        raise ValueError(f"{phase}: original replay rotations differ from the holdout")

    results = {}
    ablation = phase / "results/ablation"
    for target in targets:
        frame_id = target - START
        comparisons = read_csv(ablation / f"refinement_{frame_id}.csv")
        if [row["mode"] for row in comparisons] != list(MODES):
            raise ValueError(f"{target}: missing or reordered comparison modes")
        common = [(int(row["host_frame"]), int(row["target_frame"]), int(row["inliers"]),
                   *(float(row["init_" + key]) for key in POSE)) for row in comparisons]
        if any(values != common[0] for values in common) or common[0][:2] != (frame_id - 1, frame_id):
            raise ValueError(f"{target}: inconsistent frame IDs, inliers or Ceres initialization")
        init = common[0][3:]
        if not np.isfinite(init).all():
            raise ValueError(f"{target}: nonfinite initializer")
        points = read_csv(ablation / f"refinement_increments_{frame_id}.csv")
        match_rows = read_csv(ablation / f"matches_{frame_id - 1}_{frame_id}.csv")
        indices = {int(row["solver_index"]) for row in points}
        inlier_indices = {int(row["solver_index"]) for row in match_rows if row["ransac_inlier"] == "1"}
        if indices != inlier_indices or len(indices) != len(points) or len(points) != common[0][2]:
            raise ValueError(f"{target}: refinement input count does not match captured RANSAC inliers")
        traces = np.array([float(row["per_match_increment_trace"]) for row in points])
        base = np.array([float(row["original_covariance_trace"]) for row in points])
        if not np.isfinite(traces).all() or not np.isfinite(base).all() or np.any(traces < 0) or np.any(base <= 0):
            raise ValueError(f"{target}: invalid captured covariance trace")
        trace_sum = float(traces.sum())
        raw_poses = {}
        stats = {}
        for row in comparisons:
            mode = row["mode"]
            trace = float(row["increment_trace"])
            count = int(row["nonzero_increments"])
            expected_trace = 0 if mode == "original" else trace_sum
            expected_count = {"original": 0, "uniform": len(points) if trace_sum > 0 else 0,
                              "per_match": int(np.count_nonzero(traces))}[mode]
            if not math.isclose(trace, expected_trace, rel_tol=1e-12, abs_tol=0) or count != expected_count:
                raise ValueError(f"{target}: inconsistent increment total or activation count")
            pair_times = [float(row["host_time_s"]), float(row["target_time_s"])]
            if not np.allclose(pair_times, times[frame_id - 1:frame_id + 1], atol=1e-9, rtol=0):
                raise ValueError(f"{target}: comparison timestamps differ from the replay")
            raw_poses[mode] = [float(row[key]) for key in POSE]
            stats[mode] = {"usable": row["solution_usable"] == "1", "termination": int(row["termination"]),
                           "initial_cost": float(row["initial_cost"]), "final_cost": float(row["final_cost"]),
                           "nonzero_increments": count}
            if row["solution_usable"] not in ("0", "1") or not np.isfinite(raw_poses[mode]).all() or not np.isfinite(
                    [stats[mode]["initial_cost"], stats[mode]["final_cost"]]).all():
                raise ValueError(f"{target}: invalid native refinement output")
        rotations = {mode: quaternion_matrices(row[3:]) for mode, row in raw_poses.items()}
        baseline_delta = pose_delta_deg(rotations["original"][None], logged_rotations[frame_id][None])[0]
        if baseline_delta > 1e-5:
            raise ValueError(f"{target}: original refinement row differs from the replay pose")
        results[target] = {"inliers": len(points), "total_covariance_increment_trace": trace_sum,
                           "median_increment_over_original_trace": float(np.median(traces / base)) if len(points) else None,
                           "initial_pose": init, "refined_poses": raw_poses, "native_status": stats}
    return bridge, results


def load_holdout(holdout, repo):
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    previous_report = json.loads((holdout / "report.json").read_text(encoding="utf-8"))
    if (manifest != previous_report["manifest"] or manifest["action"] != "holdout"
            or (manifest["replay_start"], manifest["evaluation_start"], manifest["end_exclusive"]) != HOLDOUT
            or "01" not in manifest["sequences"] or manifest["modes"] != list(MODES)):
        raise ValueError("Expected the completed photometric holdout run")
    slot = holdout / "01"
    data = repo / "data"
    config = (data / CAMERA["01"], data / "test_config.yaml",
              data / "tracking/KITTI/01.json", data / "tracking/KITTI/kitti_calib.json")
    if json.loads((slot / "config_sha256.json").read_text(encoding="utf-8")) != {p.name: digest(p) for p in config}:
        raise ValueError("KITTI 01 configurations differ from the holdout")
    times = np.loadtxt(slot / "times.txt", ndmin=1)
    if times.shape != (END - START,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid KITTI 01 holdout timestamps")
    online = {mode: pose_rows(slot / mode / "ablation/PNEC.txt", times) for mode in MODES}
    return times, online, config


def launch_phase(root, targets, repo, executable, holdout, times, config):
    root.mkdir()
    images = root / "images"
    images.mkdir()
    for frame in range(START, max(targets) + 1):
        filename = f"{frame:06d}.png"
        (images / filename).symlink_to((holdout / "01/images" / filename).resolve(strict=True))
    time_path = root / "times.txt"
    np.savetxt(time_path, times[:max(targets) - START + 1], fmt="%.12f")
    ids = ",".join(str(target - START) for target in targets)
    env = dict(os.environ, PNEC_PHOTOMETRIC_INCREMENT="original", OPENBLAS_NUM_THREADS="1",
               PNEC_MATCH_DIAGNOSTICS_FRAMES=ids, PNEC_REFINEMENT_DIAGNOSTICS_FRAMES=ids)
    print(f"{root.name}: running KITTI 01 from {START} to {max(targets)}, log {root / 'run.log'}", flush=True)
    output = root / "results"
    with (root / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run([executable, *config, images, time_path, str(output) + "/", "", "true"],
                       cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def score(ground_truth, online, comparisons):
    raw = np.loadtxt(ground_truth, ndmin=2)[ANCHOR:max(TARGETS) + 1]
    if raw.shape != (max(TARGETS) - ANCHOR + 1, 12) or not np.isfinite(raw).all():
        raise ValueError("Expected KITTI 01 ground truth for all selected frames")
    gt = project_rotations(raw.reshape(-1, 3, 4)[:, :, :3])
    anchored = gt[0].T @ gt
    output = []
    for frame in TARGETS:
        item = comparisons[frame]
        g0, g1 = anchored[frame - ANCHOR - 1:frame - ANCHOR + 1]
        init = quaternion_matrices(item["initial_pose"][3:])
        metrics = {}
        for mode in MODES:
            fixed = quaternion_matrices(item["refined_poses"][mode][3:])
            live = quaternion_matrices(online[mode][frame - START, 4:])
            error, live_error = rotation_vectors_deg(np.stack((g0 @ fixed @ g1.T, g0 @ live @ g1.T)))
            metrics[mode] = {
                "fixed_input_error_deg": float(np.linalg.norm(error)),
                "fixed_input_signed_error_xyz_deg": error.tolist(),
                "online_error_deg": float(np.linalg.norm(live_error)),
                "online_signed_error_xyz_deg": live_error.tolist(),
                "fixed_input_change_from_init_deg": float(pose_delta_deg(fixed[None], init[None])[0]),
                "fixed_minus_online_rotation_deg": float(pose_delta_deg(fixed[None], live[None])[0]),
            }
        output.append({"target_frame": frame, "selection": "case" if frame in CASES else "neighbor",
                       **item, "rotation_metrics": metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analyze-only", type=Path, help="Read an existing comparison run directory")
    args = parser.parse_args()
    if not args.analyze_only and (not args.build or not args.output_root):
        parser.error("Recording needs --build and --output-root")
    repo = Path(__file__).resolve().parents[2]
    holdout = args.holdout.resolve(strict=True)
    times, online, config = load_holdout(holdout, repo)
    if args.analyze_only:
        run = args.analyze_only.resolve(strict=True)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["holdout_sha256"] != digest(holdout / "report.json") or manifest["target_frames"] != TARGETS:
            raise ValueError("Existing comparison refers to a different holdout or target schedule")
        smoke_bridge, _ = read_phase(run / "smoke", SMOKE_TARGETS, times, online["original"])
    else:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=repo, check=True)
        build = args.build.resolve(strict=True)
        cache = (build / "CMakeCache.txt").read_text()
        if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache.splitlines():
            raise ValueError("Build directory was configured for a different repository")
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build)], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "pnec_refinement_compare", "--", "-j2"], check=True)
        binary = (build / "pnec_refinement_compare").resolve(strict=True)
        run = Path(mkdtemp(prefix="pnec_refinement_compare_01_", dir=args.output_root.resolve(strict=True)))
        manifest = {"sequence": "01", "git_commit": revision, "binary_sha256": digest(binary),
                    "holdout_sha256": digest(holdout / "report.json"), "replay_start": START,
                    "case_frames": list(CASES), "neighbor_frames": list(CONTROLS), "target_frames": TARGETS}
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Run directory: {run}", flush=True)
        launch_phase(run / "smoke", SMOKE_TARGETS, repo, binary, holdout, times, config)
        smoke_bridge, smoke = read_phase(run / "smoke", SMOKE_TARGETS, times, online["original"])
        if not all(row["usable"] for item in smoke.values() for row in item["native_status"].values()):
            raise ValueError("A native smoke refinement was unusable")
        if not any(item["total_covariance_increment_trace"] > 0 for item in smoke.values()):
            raise ValueError("Photometric increments were inactive in the smoke run")
        launch_phase(run / "capture", TARGETS, repo, binary, holdout, times, config)

    bridge, results = read_phase(run / "capture", TARGETS, times, online["original"])
    report = {"manifest": manifest, "smoke_bridge": smoke_bridge, "original_prefix_bridge": bridge,
              "ground_truth_sha256": digest(args.gt), "pairs": score(args.gt, online, results)}
    path = run / "refinement_comparison.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("target group inliers fixed_angle_deg(original/uniform/per_match) online_per_match_deg usable")
    for item in report["pairs"]:
        rows = item["rotation_metrics"]
        fixed = "/".join(f"{rows[mode]['fixed_input_error_deg']:.4f}" for mode in MODES)
        usable = "/".join(str(int(item["native_status"][mode]["usable"])) for mode in MODES)
        print(f"{item['target_frame']} {item['selection']} {item['inliers']} {fixed} "
              f"{rows['per_match']['online_error_deg']:.4f} {usable}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
