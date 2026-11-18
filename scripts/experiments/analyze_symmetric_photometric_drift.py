"""Analyze saved symmetric KITTI 08 rotations and their signed error accumulation."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np

from analyze_photometric_drift import (
    compose_steps, full_scope, project_rotations, relative_steps,
    rotation_vectors_deg, tail_replacements,
)
from run_photometric_increment import (
    SYMMETRIC_MODES, digest, pose_rows, restore_rotations,
)


BASELINE, UNIFORM, CANDIDATE = SYMMETRIC_MODES
ARCHIVE = "docs/experiments/results/pnec_kitti_08_symmetric_increments_full_dbee80e_user_report.json"


def find_run(root, expected):
    for path in sorted(root.glob("pnec_increment_full_*/report.json")):
        try:
            recorded = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if recorded == expected:
            return path.parent.resolve(strict=True)
    raise ValueError("The archived symmetric KITTI 08 full run was not found")


def load_inputs(run, dataset, expected):
    report_path = run / "report.json"
    recorded = json.loads(report_path.read_bytes())
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if recorded != expected or manifest != recorded["manifest"]:
        raise ValueError("The report or manifest differs from the archived run")
    sequence, = manifest["sequences"]
    count = manifest["end_exclusive"]
    tail = manifest["tail_evaluation_start"]
    source_times = dataset / "sequences" / sequence / "times.txt"
    truth_path = dataset / "poses" / f"{sequence}.txt"
    if (digest(source_times) != manifest["times_sha256"]
            or digest(truth_path) != manifest["ground_truth_sha256"]):
        raise ValueError("KITTI timestamps or ground truth differ from the report")
    times = np.loadtxt(source_times, ndmin=1)
    if (times.shape != (count,) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Incomplete KITTI timestamp coverage")
    times -= times[0]
    truth = np.loadtxt(truth_path, ndmin=2)
    if truth.shape != (count, 12) or not np.isfinite(truth).all():
        raise ValueError("Incomplete KITTI ground truth coverage")

    baseline = recorded["reference_symmetric_full_run"]
    baseline_run = Path(baseline["run"]).resolve(strict=True)
    baseline_report_path = baseline_run / "report.json"
    if (digest(baseline_report_path) != baseline["report_sha256"]
            or baseline["report_sha256"] != manifest["reference_symmetric_full_report_sha256"]):
        raise ValueError("Archived symmetric baseline report changed")
    baseline_report = json.loads(baseline_report_path.read_bytes())
    if (baseline_report["results"][sequence] != baseline["scores"]
            or baseline_report["manifest"]["git_commit"] != baseline["git_commit"]
            or baseline_report["manifest"]["binary_sha256"] != baseline["binary_sha256"]):
        raise ValueError("Symmetric baseline scores or provenance differ")
    directories = {
        BASELINE: baseline_run / sequence / BASELINE,
        UNIFORM: run / sequence / UNIFORM,
        CANDIDATE: run / sequence / CANDIDATE,
    }
    for time_path in (run / sequence / "times.txt", baseline_run / sequence / "times.txt"):
        saved_times = np.loadtxt(time_path, ndmin=1)
        if saved_times.shape != times.shape or not np.allclose(saved_times, times, rtol=0, atol=6e-13):
            raise ValueError(f"Saved timestamps differ: {time_path}")
    estimates, pose_hashes = {}, {}
    for mode, folder in directories.items():
        paths = {name: folder / name for name in ("ablation/PNEC.txt", "rot_avg/poses.txt")}
        hashes = {name: digest(path) for name, path in paths.items()}
        if mode == BASELINE and hashes != baseline["pose_sha256"]:
            raise ValueError("Saved symmetric baseline poses changed")
        pose_rows(paths["rot_avg/poses.txt"], times)
        estimates[mode] = restore_rotations(paths["ablation/PNEC.txt"], times, 0)
        pose_hashes[mode] = hashes
    reference = project_rotations(truth.reshape(-1, 3, 4)[:, :, :3])
    scores = {**baseline["scores"], **recorded["results"][sequence]}
    inputs = {
        "report": digest(report_path), "manifest": digest(manifest_path),
        "baseline_report": digest(baseline_report_path),
        "ground_truth": digest(truth_path), "timestamps": digest(source_times),
        "pose_files": pose_hashes,
    }
    return reference, estimates, scores, tail, inputs


def signed_summary(errors):
    vectors = rotation_vectors_deg(errors)
    signed_sum = vectors.sum(axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    total_angle = float(lengths.sum())
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    lag_one = []
    for axis in range(3):
        left, right = centered[:-1, axis], centered[1:, axis]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        lag_one.append(float(np.dot(left, right) / denominator) if denominator > 1e-12 else None)
    net = compose_steps(errors)[-1]
    return {
        "steps": len(errors),
        "mean_signed_error_xyz_deg": mean.tolist(),
        "signed_error_sum_xyz_deg": signed_sum.tolist(),
        "step_rmse_deg": float(np.sqrt(np.mean(lengths**2))),
        "directional_coherence": float(np.linalg.norm(signed_sum) / total_angle) if total_angle > 1e-12 else None,
        "centered_lag_one_correlation_xyz": lag_one,
        "net_window_error_xyz_deg": rotation_vectors_deg(net[None])[0].tolist(),
    }


def analyze(repo, run, dataset, expected):
    reference, estimates, scores, tail, inputs = load_inputs(run, dataset, expected)
    print("Saved input hashes and baseline coverage verified.", flush=True)
    roles = {"baseline": BASELINE, "uniform": UNIFORM, "candidate": CANDIDATE}
    full = full_scope(reference, estimates, 0, scores, **roles)
    print("Full-sequence rotation metrics reproduced.", flush=True)
    tail_estimates = {mode: rotations[tail:] for mode, rotations in estimates.items()}
    tail_result = full_scope(reference[tail:], tail_estimates, tail,
                             {mode: values["tail"] for mode, values in scores.items()}, **roles)
    print("Tail rotation metrics reproduced.", flush=True)

    # G_i R_est,i G_(i+1)^T expresses each step error in the fixed GT reference axes.
    # The product telescopes to G_lo R_est,lo:hi G_hi^T for each window.
    transported = {
        mode: reference[:-1] @ relative_steps(rotations) @ reference[1:].swapaxes(1, 2)
        for mode, rotations in estimates.items()
    }
    for result in (full, tail_result):
        for window in result["windows"]:
            lo, hi = window["frames"]
            window["signed_error_summary"] = {
                mode: signed_summary(errors[lo:hi]) for mode, errors in transported.items()
            }
    substitutions = tail_replacements(reference[tail:], tail_estimates, tail_result,
                                       baseline=BASELINE, candidate=CANDIDATE)
    print("Window directions and saved-step substitutions computed.", flush=True)
    helpers = ("analyze_symmetric_photometric_drift.py", "analyze_photometric_drift.py",
               "run_photometric_increment.py")
    return {
        "sequence": "08",
        "estimator_git_commit": expected["manifest"]["git_commit"],
        "tool_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "input_sha256": inputs,
        "analysis_source_sha256": {name: digest(repo / "scripts/experiments" / name) for name in helpers},
        "method_roles": roles,
        "signed_error_coordinates": "Fixed reference axes of the ground-truth camera trajectory",
        "interpretation": "Window means and correlations describe this saved trajectory. "
                          "Step substitutions recompose saved rotations and keep the other online outputs fixed.",
        "full": full, "tail": tail_result, "tail_step_substitutions": substitutions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--run", type=Path)
    location.add_argument("--results-root", type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="KITTI dataset root; only times and GT are read")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    expected = json.loads((repo / ARCHIVE).read_bytes())
    run = args.run.resolve(strict=True) if args.run else find_run(args.results_root.resolve(strict=True), expected)
    output = analyze(repo, run, args.dataset.resolve(strict=True), expected)
    path = run / "symmetric_rotation_drift.json"
    with NamedTemporaryFile(mode="w", dir=run, prefix="symmetric-drift-", suffix=".tmp",
                            encoding="utf-8", delete=False) as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)

    delta_key = f"{CANDIDATE}_minus_{BASELINE}_rpe_n_deg"
    for scope in ("full", "tail"):
        result = output[scope]
        print(f"{scope} frames {result['frames']}: RPE metrics verified")
        for band in result["gap_bands"]:
            delta = band[f"{CANDIDATE}_minus_{BASELINE}_deg"]
            print(f"  gap {band['gap_frames']}: candidate-minus-baseline contribution {delta:+.6f} deg")
        worst = max(result["windows"], key=lambda row: row[delta_key])
        best = min(result["windows"], key=lambda row: row[delta_key])
        for label, window in (("largest excess", worst), ("largest improvement", best)):
            print(f"  {label} window {window['frames']}: {window[delta_key]:+.6f} deg")
            for mode, summary in window["signed_error_summary"].items():
                print(f"    {mode}: mean_xyz={summary['mean_signed_error_xyz_deg']} "
                      f"coherence={summary['directional_coherence']}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
