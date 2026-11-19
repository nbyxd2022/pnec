"""Compare final symmetric refinements on identical inputs from one baseline pass."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp, NamedTemporaryFile

import numpy as np

from analyze_photometric_drift import compose_steps, project_rotations
from analyze_symmetric_photometric_drift import signed_summary
from run_photometric_increment import (
    SYMMETRIC_MODES, digest, pose_rows, replay_config, restore_rotations,
    rotation_metrics,
)


ARCHIVE = "docs/experiments/results/pnec_kitti_08_symmetric_increments_full_dbee80e_user_report.json"
WINDOWS = ((2600, 2800), (3400, 3600))
MODES = SYMMETRIC_MODES


def atomic_json(path, data):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix="fixed-audit-", suffix=".tmp", delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def protocol(action):
    windows = ((0, 7),) if action == "smoke" else WINDOWS
    selected = [target for lo, hi in windows for target in range(lo + 1, hi + 1)]
    return windows, selected, max(selected) + 1


def baseline_inputs(repo, dataset, action):
    archive = repo / ARCHIVE
    recorded = json.loads(archive.read_bytes())
    saved = recorded["reference_symmetric_full_run"]
    baseline = Path(saved["run"]).resolve(strict=True)
    if digest(baseline / "report.json") != saved["report_sha256"]:
        raise ValueError("Archived baseline report changed")
    original = json.loads((baseline / "report.json").read_bytes())
    if (original["results"]["08"] != saved["scores"]
            or original["manifest"]["git_commit"] != saved["git_commit"]):
        raise ValueError("Archived baseline provenance or metrics changed")
    config = replay_config(repo, "08")
    hashes = {p.name: digest(p) for p in config}
    if hashes != json.loads((baseline / "08/config_sha256.json").read_bytes()):
        raise ValueError("Baseline and current configurations differ")
    sequence = dataset / "sequences/08"
    times = np.loadtxt(sequence / "times.txt", ndmin=1)
    if (times.shape != (4071,) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
            or digest(sequence / "times.txt") != recorded["manifest"]["times_sha256"]
            or digest(dataset / "poses/08.txt") != recorded["manifest"]["ground_truth_sha256"]):
        raise ValueError("KITTI 08 timestamps or GT differ from the archived run")
    times -= times[0]
    windows, selected, end = protocol(action)
    relative_file = baseline / "08/symmetric_analytic/ablation/PNEC.txt"
    if digest(relative_file) != saved["pose_sha256"]["ablation/PNEC.txt"]:
        raise ValueError("Archived baseline rotations changed")
    previous = pose_rows(relative_file, times)
    return {
        "baseline_run": str(baseline), "baseline_report_sha256": saved["report_sha256"],
        "baseline_relative_pose_sha256": digest(relative_file),
        "dataset": str(dataset), "timestamps_sha256": digest(sequence / "times.txt"),
        "ground_truth_sha256": digest(dataset / "poses/08.txt"),
        "config_sha256": hashes, "windows": [list(w) for w in windows],
        "selected_target_frames": selected, "replay_start": 0, "end_exclusive": end,
    }, times[:end], previous[:end], config


def matrix_columns(rows, prefix, size=3):
    return np.stack([rows[f"{prefix}{r}{c}"] for r in range(size)
                     for c in range(size)], axis=-1).reshape(-1, size, size)


def vector_columns(rows, prefix):
    return np.stack([rows[f"{prefix}{k}"] for k in range(3)], axis=-1)


def read_poses(path):
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if [row["mode"] for row in records] != ["initial", *MODES]:
        raise ValueError(f"Incomplete or unordered refinement outputs: {path}")
    result = {}
    for row in records:
        rotation = np.array([float(row[f"R{r}{c}"]) for r in range(3)
                             for c in range(3)]).reshape(3, 3)
        translation = np.array([float(row[f"t{k}"]) for k in range(3)])
        if (not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
                or abs(np.linalg.det(rotation) - 1) > 1e-8
                or np.linalg.norm(translation) <= 1e-12):
            raise ValueError(f"Invalid fixed-input pose: {path}/{row['mode']}")
        result[row["mode"]] = (rotation, translation)
    return result


def read_points(path, match_path):
    rows = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    matches = np.genfromtxt(match_path, delimiter=",", names=True, ndmin=1)
    accepted = matches[matches["ransac_inlier"] == 1]
    if not len(rows) or len(rows) != len(accepted):
        raise ValueError(f"Incomplete fixed-input inliers: {path}")
    identities = dict(zip(accepted["solver_index"], accepted["feature_id"]))
    if (len(identities) != len(rows) or len(set(rows["solver_index"])) != len(rows)
            or any(identities.get(index) != feature
                   for index, feature in zip(rows["solver_index"], rows["feature_id"]))):
        raise ValueError(f"Recorded inputs and RANSAC identities differ: {path}")
    left, right = vector_columns(rows, "host_b"), vector_columns(rows, "target_b")
    host_cov, target_cov = matrix_columns(rows, "host_cov"), matrix_columns(rows, "target_cov")
    core = np.column_stack((left, right, host_cov.reshape(-1, 9), target_cov.reshape(-1, 9),
                            rows["delta_trace"], rows["ratio"], rows["regularization"]))
    if (not np.isfinite(core).all() or np.any(rows["delta_trace"] < 0)
            or np.any(rows["ratio"] < 0) or np.any(rows["ratio"] > 1)
            or np.any(rows["regularization"] <= 0)
            or not np.allclose(np.linalg.norm(left, axis=1), 1, rtol=0, atol=1e-8)
            or not np.allclose(np.linalg.norm(right, axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError(f"Invalid bearing, covariance or increment: {path}")
    for covariance in (host_cov, target_cov):
        if (not np.allclose(covariance, covariance.swapaxes(1, 2), rtol=1e-7, atol=1e-14)
                or np.min(np.linalg.eigvalsh(covariance)) < -1e-12):
            raise ValueError(f"Invalid projected covariance: {path}")
    if not np.allclose(rows["mean_trace"], rows["delta_trace"].mean(), rtol=1e-12, atol=1e-20):
        raise ValueError(f"Uniform trace differs from the per-match mean: {path}")
    return rows, left, right, host_cov, target_cov


def variance_effect(inputs, rotation, translation):
    rows, left, right, host_cov, target_cov = inputs
    t = translation / np.linalg.norm(translation)
    rotated_right = right @ rotation.T
    host_gradient = np.cross(rotated_right, t)
    target_gradient = np.cross(t, left) @ rotation
    variance = (np.einsum("ni,nij,nj->n", host_gradient, host_cov, host_gradient)
                + np.einsum("ni,nij,nj->n", target_gradient, target_cov, target_gradient)
                + rows["regularization"])
    tangent_norm2 = np.maximum(0, np.sum(target_gradient**2, axis=1)
                               - np.sum(target_gradient * right, axis=1)**2)
    residual = np.sum(np.cross(t, left) * rotated_right, axis=1)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Nonpositive nominal symmetric residual variance")
    relative = .5 * rows["delta_trace"] * tangent_norm2 / variance
    uniform_relative = .5 * rows["mean_trace"] * tangent_norm2 / variance
    nominal_q = residual**2 / variance
    return np.column_stack((relative, uniform_relative, nominal_q))


def ranks(values):
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=float)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / 2
    return result


def effect_summary(values):
    relative, uniform, q = values.T
    a, b = ranks(relative), ranks(q)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    result = {"points": len(values),
              "rank_correlation_increment_vs_nominal_standardized_residual2":
                  float(np.dot(a, b) / denominator) if denominator > 1e-12 else None}
    increment_ranks = ranks(relative)
    result["increment_rank_quartiles"] = {}
    for name, mask in (("low", increment_ranks <= .25 * (len(values) - 1)),
                       ("high", increment_ranks >= .75 * (len(values) - 1))):
        result["increment_rank_quartiles"][name] = {
            "points": int(mask.sum()),
            "nominal_standardized_residual2_median": float(np.median(q[mask])) if mask.any() else None,
        }
    for mode, extra in ((MODES[1], uniform), (MODES[2], relative)):
        weight_ratio = 1 / (1 + extra)
        nll_delta = .5 * (np.log1p(extra) - q * extra / (1 + extra))
        result[mode] = {
            "median_inverse_variance_ratio_to_baseline": float(np.median(weight_ratio)),
            "fraction_inverse_variance_ratio_below_half": float(np.mean(weight_ratio < .5)),
            "mean_gaussian_nll_delta": float(np.mean(nll_delta)),
        }
    return result


def analyze(out, dataset):
    manifest = json.loads((out / "manifest.json").read_bytes())
    repo = Path(__file__).resolve().parents[2]
    expected, times, previous, _ = baseline_inputs(repo, dataset, manifest["action"])
    if manifest["inputs"] != expected:
        raise ValueError("Current audit inputs differ from the recording manifest")
    actual = pose_rows(out / "capture/ablation/PNEC.txt", times)
    if not np.array_equal(actual, previous):
        raise ValueError("Baseline prefix changed. Preserve capture and inspect before interpreting the shadow solves")
    baseline_rotations = restore_rotations(out / "capture/ablation/PNEC.txt", times, 0)
    baseline_steps = baseline_rotations[:-1].swapaxes(1, 2) @ baseline_rotations[1:]
    truth = np.loadtxt(dataset / "poses/08.txt", ndmin=2).reshape(-1, 3, 4)
    reference = project_rotations(truth[:, :, :3])
    captures, point_effects, pair_rows, hashes = {}, {}, [], {}
    for target in expected["selected_target_frames"]:
        host = target - 1
        prefix = out / f"capture/ablation/fixed_{host}_{target}"
        points_path = Path(str(prefix) + "_points.csv")
        poses_path = Path(str(prefix) + "_poses.csv")
        matches_path = out / f"capture/ablation/matches_{host}_{target}.csv"
        inputs = read_points(points_path, matches_path)
        poses = read_poses(poses_path)
        if not np.allclose(poses[MODES[0]][0], baseline_steps[host], rtol=0, atol=2e-7):
            raise ValueError(f"Frame {target}: diagnostic baseline differs from recorded baseline step")
        captures[target] = poses
        init_effect = variance_effect(inputs, *poses["initial"])
        true_rotation = reference[host].T @ reference[target]
        true_translation = reference[host].T @ (truth[target, :, 3] - truth[host, :, 3])
        gt_effect = (variance_effect(inputs, true_rotation, true_translation)
                     if np.linalg.norm(true_translation) > 1e-9 else None)
        point_effects[target] = (init_effect, gt_effect)
        errors = {}
        for mode, (rotation, _) in poses.items():
            cosine = (np.sum(true_rotation * rotation) - 1) / 2
            errors[mode] = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
        pair_rows.append({"target_frame": target, "inliers": len(inputs[0]),
                          "rotation_error_deg": errors,
                          "at_common_initial_pose": effect_summary(init_effect),
                          "at_gt_motion": effect_summary(gt_effect) if gt_effect is not None else None})
        for path in (points_path, poses_path, matches_path):
            hashes[path.relative_to(out).as_posix()] = digest(path)
    windows = []
    for lo, hi in expected["windows"]:
        estimates = {mode: compose_steps(np.stack([captures[k][mode][0]
                                                   for k in range(lo + 1, hi + 1)]))
                     for mode in MODES}
        signed = {}
        for mode in MODES:
            errors = np.stack([reference[k - 1] @ captures[k][mode][0] @ reference[k].T
                               for k in range(lo + 1, hi + 1)])
            signed[mode] = signed_summary(errors)
        initial = np.concatenate([point_effects[k][0] for k in range(lo + 1, hi + 1)])
        gt = [point_effects[k][1] for k in range(lo + 1, hi + 1)
              if point_effects[k][1] is not None]
        windows.append({"frames": [lo, hi], "pairs": hi - lo,
                        "fixed_input_rotation_metrics": {
                            mode: rotation_metrics(reference[lo:hi + 1], rotations)
                            for mode, rotations in estimates.items()},
                        "signed_error_summary": signed,
                        "point_effect_at_common_initial_pose": effect_summary(initial),
                        "point_effect_at_gt_motion": effect_summary(np.concatenate(gt)) if gt else None,
                        "gt_motion_unavailable_pairs": hi - lo - len(gt)})
    hashes["capture/ablation/PNEC.txt"] = digest(out / "capture/ablation/PNEC.txt")
    hashes["manifest.json"] = digest(out / "manifest.json")
    report = {
        "schema": 1, "manifest": manifest, "baseline_prefix_exact_match": True,
        "analysis_source_sha256": {name: digest(Path(__file__).resolve().parent / name) for name in (
            "run_fixed_input_audit.py", "run_photometric_increment.py",
            "analyze_photometric_drift.py", "analyze_symmetric_photometric_drift.py")},
        "interpretation": {
            "inputs": "Each pair shares matches, RANSAC inliers, source/target KLT covariances and one Ceres initial pose",
            "feedback": "Only the symmetric analytic baseline is fed back; other refinements are recorded only",
            "scope": "Development windows selected from previous trajectory analysis; not held-out validation",
            "gt_scores": "Epipolar geometry diagnostics assume static correct correspondences. They are not pixel endpoint truth or proof of covariance calibration",
            "nll": "Gaussian residual-score change relative to nominal variance at the same specified pose; negative is better under that diagnostic model",
            "inverse_variance": "Ratio evaluates the residual denominator at a common pose; it is not the full nonlinear optimizer influence or an information-gain measure",
            "timing": "Diagnostic solves add overhead; capture timing is not a method speed benchmark",
        },
        "input_sha256": hashes, "windows": windows, "pairs": pair_rows,
    }
    atomic_json(out / "report.json", report)
    for window in windows:
        print("Window", window["frames"], "pairs", window["pairs"], flush=True)
        for mode, metrics in window["fixed_input_rotation_metrics"].items():
            print(mode, json.dumps(metrics), flush=True)
        print("Point effects at GT motion:", json.dumps(window["point_effect_at_gt_motion"]), flush=True)
    print("Report:", out / "report.json", flush=True)
    return report


def record_pairs(repo, executable, config, source, start_frame, times, targets, out, *,
                 rotation_feedback=False, mode=MODES[0], shared_weak=False):
    """Record selected frame pairs from the requested continuous replay."""
    images = out / "images"
    images.mkdir()
    for frame in range(start_frame, start_frame + len(times)):
        name = f"{frame:06d}.png"
        (images / name).symlink_to((source / name).resolve(strict=True))
    np.savetxt(out / "times.txt", times, fmt="%.12f")
    target = out / "capture"
    target.mkdir()
    environment = dict(os.environ, PNEC_PHOTOMETRIC_INCREMENT=mode, OPENBLAS_NUM_THREADS="1")
    for flag in ("PNEC_FIXED_INPUT_AUDIT", "PNEC_MATCH_DIAGNOSTICS_FRAMES", "PNEC_SHARED_WEAK_OUTPUTS"):
        environment.pop(flag, None)
    if shared_weak:
        environment["PNEC_SHARED_WEAK_OUTPUTS"] = "1"
    else:
        environment["PNEC_FIXED_INPUT_AUDIT"] = "1"
        environment["PNEC_MATCH_DIAGNOSTICS_FRAMES"] = ",".join(map(str, targets))
    environment.pop("PNEC_PREVIOUS_ROTATION_BY_VIEW", None)
    if rotation_feedback:
        environment["PNEC_PREVIOUS_ROTATION_BY_VIEW"] = "1"
    command = [str(executable), *map(str, config), str(images), str(out / "times.txt"),
               str(target) + "/", "", "true"]
    print("Run directory:", out, flush=True)
    print("Baseline frames:", len(times), "; fixed-input diagnostic pairs:",
          len(targets), "; native log:", target / "run.log", flush=True)
    with (target / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args):
    repo = Path(__file__).resolve().parents[2]
    dataset = args.dataset.resolve(strict=True)
    if args.analyze_only:
        return analyze(args.analyze_only.resolve(strict=True), dataset)
    if args.action is None or args.build is None or args.output_root is None:
        raise ValueError("Recording needs action, --build and --output-root")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "."], cwd=repo, check=True,
                   stdout=subprocess.DEVNULL)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                         "scripts/experiments", "include", "src"], cwd=repo)
    if untracked.strip():
        raise ValueError("Commit the diagnostic source files before recording")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    build = args.build.resolve(strict=True)
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build directory belongs to a different repository")
    inputs, times, _, config = baseline_inputs(repo, dataset, args.action)
    source = dataset / "sequences/08/image_0"
    for frame in range(len(times)):
        if not (source / f"{frame:06d}.png").is_file():
            raise FileNotFoundError(f"Missing KITTI 08 image {frame}")
    subprocess.run(["cmake", "--build", str(build), "--target", "pnec_vo", "--", "-j2"], check=True)
    executable = build / "pnec_vo"
    out = Path(mkdtemp(prefix=f"pnec_fixed_input_{args.action}_", dir=args.output_root.resolve(strict=True)))
    manifest = {"git_commit": revision, "binary_sha256": digest(executable), "action": args.action,
                "inputs": inputs, "online_mode": MODES[0], "diagnostic_modes": list(MODES),
                "source_sha256": {str(p.relative_to(repo).as_posix()): digest(p) for p in (
                    Path(__file__).resolve(), repo / "src/rel_pose_estimation/frame2frame.cc",
                    repo / "src/rel_pose_estimation/pnec.cc",
                    repo / "include/rel_pose_estimation/photometric_increment.h")}}
    atomic_json(out / "manifest.json", manifest)
    record_pairs(repo, executable, config, source, 0, times, inputs["selected_target_frames"], out)
    return analyze(out, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "windows"), nargs="?")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--analyze-only", type=Path)
    run(parser.parse_args())
