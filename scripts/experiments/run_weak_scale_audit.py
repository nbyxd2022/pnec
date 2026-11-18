"""Evaluate a fixed 1/32 target covariance increment on the saved KITTI 08 windows."""

import argparse
from io import StringIO
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_fixed_input_assignment import ARCHIVE
from analyze_photometric_drift import compose_steps, project_rotations
from run_fixed_input_audit import MODES, atomic_json, read_points, read_poses
from run_photometric_increment import digest, rotation_metrics


WEAK_SCALE = 1 / 32
ALL_MODES = (*MODES, "weak_uniform", "weak_per_match")
# Compare the reconstructed rotations with the three actual native solves of
# each frame. This tolerance is in degrees, well below the reported RPE gaps.
CONTROL_TOLERANCE_DEG = 1e-4


def checked_file(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Capture hash mismatch: {path}")
    return path


def solve_pair(executable, inputs, initial):
    rows, left, right, host_cov, target_cov = inputs
    rotation, translation = initial
    regularizer = rows["regularization"][0]
    if not np.all(rows["regularization"] == regularizer):
        raise ValueError("Inconsistent regularization within the pair")
    wire = StringIO()
    wire.write(f"{len(rows)} {regularizer:.17g} {WEAK_SCALE:.17g}\n")
    np.savetxt(wire, rotation, fmt="%.17g")
    np.savetxt(wire, translation[None, :], fmt="%.17g")
    records = np.column_stack((left, right, host_cov.reshape(-1, 9),
                               target_cov.reshape(-1, 9), rows["delta_trace"]))
    np.savetxt(wire, records, fmt="%.17g")
    raw = subprocess.run([str(executable)], input=wire.getvalue(), text=True,
                         stdout=subprocess.PIPE, check=True).stdout.splitlines()
    solutions = {}
    for line in raw:
        label, *values = line.split()
        if label in solutions or len(values) != 9:
            raise ValueError(f"Invalid native output: {line}")
        matrix = np.array(values, dtype=float).reshape(3, 3)
        if (not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0)
                or abs(np.linalg.det(matrix) - 1) > 1e-8):
            raise ValueError(f"Invalid native rotation for {label}")
        solutions[label] = matrix
    if set(solutions) != set(ALL_MODES):
        raise ValueError(f"Incomplete native modes: {list(solutions)}")
    return solutions


def control_difference_deg(replayed, saved):
    # For rotations the Frobenius chord length equals 2 sqrt(2) sin(angle/2).
    chord = np.linalg.norm(replayed - saved) / (2 * np.sqrt(2))
    return float(np.rad2deg(2 * np.arcsin(np.clip(chord, 0, 1))))


def evaluate_window(capture, hashes, window, reference, executable):
    lo, hi = window["frames"]
    rotations = {name: [] for name in ALL_MODES}
    max_difference = {name: 0.0 for name in MODES}
    observations = 0
    def captured(name):
        relative = "capture/ablation/" + name
        return checked_file(capture / relative, hashes[relative])

    for target in range(lo + 1, hi + 1):
        pair = f"{target - 1}_{target}"
        inputs = read_points(captured(f"fixed_{pair}_points.csv"),
                             captured(f"matches_{pair}.csv"))
        poses = read_poses(captured(f"fixed_{pair}_poses.csv"))
        try:
            solutions = solve_pair(executable, inputs, poses["initial"])
        except (ValueError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Native refinement failed at {pair}: {exc}") from exc
        for name in MODES:
            diff = control_difference_deg(solutions[name], poses[name][0])
            if diff > CONTROL_TOLERANCE_DEG:
                raise ValueError(f"Frame {target}, {name}: replay differs by {diff:.9g} deg")
            max_difference[name] = max(max_difference[name], diff)
        for name in ALL_MODES:
            rotations[name].append(solutions[name])
        observations += len(inputs[0])
        if (target - lo) % 50 == 0:
            print(f"Window {lo}-{hi}: {target - lo}/{hi - lo} pairs", flush=True)
    scores = {name: rotation_metrics(reference[lo:hi + 1], compose_steps(steps))
              for name, steps in rotations.items()}
    for name in MODES:
        for metric, value in scores[name].items():
            if abs(value - window["fixed_input_rotation_metrics"][name][metric]) > CONTROL_TOLERANCE_DEG:
                raise ValueError(f"Window {lo}-{hi}: archived {name} {metric} differs")
    return {"frames": [lo, hi], "pairs": hi - lo, "observations": observations,
            "max_replayed_control_rotation_difference_deg": max_difference,
            "fixed_input_rotation_metrics": scores}


def run(capture, executable):
    repo = Path(__file__).resolve().parents[2]
    report = json.loads((capture / "report.json").read_bytes())
    archive = repo / "docs/experiments/results" / ARCHIVE
    if report != json.loads(archive.read_bytes()) or not report["baseline_prefix_exact_match"]:
        raise ValueError("Expected the archived two-window KITTI 08 capture")
    hashes = report["input_sha256"]
    checked_file(capture / "manifest.json", hashes["manifest.json"])
    frozen = report["manifest"]["inputs"]
    gt_path = checked_file(Path(frozen["dataset"]) / "poses/08.txt", frozen["ground_truth_sha256"])
    truth = np.loadtxt(gt_path, ndmin=2).reshape(-1, 3, 4)
    reference = project_rotations(truth[:, :, :3])
    windows = [evaluate_window(capture, hashes, w, reference, executable) for w in report["windows"]]
    result = {
        "sequence": "08", "scope": "Fixed-input development windows, baseline feedback",
        "target_increment_scale": WEAK_SCALE, "control_tolerance_deg": CONTROL_TOLERANCE_DEG,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_capture": str(capture), "source_report_sha256": digest(capture / "report.json"),
        "source_manifest_sha256": hashes["manifest.json"], "ground_truth_sha256": frozen["ground_truth_sha256"],
        "runner_sha256": digest(Path(__file__).resolve()), "executable_sha256": digest(executable),
        "windows": windows,
    }
    out = capture / "weak_scale_rotation.json"
    atomic_json(out, result)
    print(json.dumps(windows, indent=2), flush=True)
    print(f"Report: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    args = parser.parse_args()
    run(args.capture.resolve(strict=True), args.executable.resolve(strict=True))
