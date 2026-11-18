"""Compare targeted far-peak attenuation with equal per-pair trace reduction on KITTI 08."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_photometric_drift import compose_steps
from analyze_secondary_peak_distance import BEST_MAX, FAR_MIN
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, rotation_metrics
from run_weak_scale_audit import (CONTROL_TOLERANCE_DEG, TAPER_MODES, WEAK_SCALE,
                                  control_difference_deg, load_capture, pair_inputs, solve_pair)


WEAK_ARCHIVE = "pnec_kitti_08_weak_scale_0f1d5c8_user_report.json"


def arm_traces(rows):
    """Return unscaled increments in solver order. The native solver applies WEAK_SCALE once."""
    raw = rows["delta_trace"]
    best = np.hypot(rows["best_dx"], rows["best_dy"])
    second = np.hypot(rows["best_dx"] + rows["mode_dx"], rows["best_dy"] + rows["mode_dy"])
    two = np.isfinite(rows["best_cost"]) & np.isfinite(rows["second_cost"])
    if not np.all(np.isfinite(best[two]) & np.isfinite(second[two])):
        raise ValueError("Two recorded minima have nonfinite offsets")
    far = two & (best <= BEST_MAX) & (second > FAR_MIN)
    tapered = raw.copy()
    tapered[far] *= (FAR_MIN / second[far]) ** 2
    total = float(raw.sum())
    retention = float(tapered.sum()) / total if total else 1.0
    matched = retention * raw
    if not np.isclose(matched.sum(), tapered.sum(), rtol=1e-12, atol=0):
        raise ValueError("Different total target increments in the two arms")
    return tapered, matched


def evaluate(capture, hashes, prior_window, reference, executable):
    lo, hi = prior_window["frames"]
    rotations = {name: [] for name in TAPER_MODES}
    totals = np.zeros(3)  # weak, targeted, matched target-side traces
    changed_points = changed_pairs = observations = 0
    largest_gap = largest_control_error = 0.0
    for target in range(lo + 1, hi + 1):
        inputs, poses = pair_inputs(capture, hashes, target)
        rows = inputs[0]
        tapered, matched = arm_traces(rows)
        try:
            solutions = solve_pair(executable, inputs, poses["initial"], taper_and_matched=(tapered, matched))
        except (ValueError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Native refinement failed on {target - 1}->{target}") from exc
        difference = control_difference_deg(solutions["symmetric_analytic"], poses["symmetric_analytic"][0])
        if difference > CONTROL_TOLERANCE_DEG:
            raise ValueError(f"Frame {target}: analytic control differs by {difference:.9g} deg")
        largest_control_error = max(largest_control_error, difference)
        for name in TAPER_MODES:
            rotations[name].append(solutions[name])
        raw = rows["delta_trace"]
        changed = int(np.count_nonzero(tapered < raw))
        changed_points += changed
        changed_pairs += int(changed > 0)
        observations += len(rows)
        sums = WEAK_SCALE * np.array([raw.sum(), tapered.sum(), matched.sum()])
        largest_gap = max(largest_gap, abs(sums[1] - sums[2]))
        totals += sums
        if (target - lo) % 50 == 0:
            print(f"Window {lo}-{hi}: {target - lo}/{hi - lo} pairs", flush=True)

    if observations != prior_window["observations"] or hi - lo != prior_window["pairs"]:
        raise ValueError(f"Window {lo}-{hi}: coverage differs from weak-scale capture")
    scores = {name: rotation_metrics(reference[lo:hi + 1], compose_steps(steps))
              for name, steps in rotations.items()}
    for name in ("symmetric_analytic", "weak_per_match"):
        for metric, value in scores[name].items():
            archived = prior_window["fixed_input_rotation_metrics"][name][metric]
            if not np.isclose(archived, value, rtol=0, atol=CONTROL_TOLERANCE_DEG):
                raise ValueError(f"Window {lo}-{hi}: archived {name} {metric} differs")
    deltas = {name: {metric: value - scores[name][metric]
                     for metric, value in scores["far_peak_taper"].items()}
              for name in TAPER_MODES if name != "far_peak_taper"}
    return {"frames": [lo, hi], "pairs": hi - lo, "observations": observations,
            "observations_attenuated": changed_points, "pairs_with_attenuation": changed_pairs,
            "sum_added_target_trace_rad2": dict(zip(("weak_per_match", "far_peak_taper", "matched_budget"),
                                                    totals.tolist())),
            "max_per_pair_budget_difference_rad2": float(largest_gap),
            "max_replayed_analytic_rotation_difference_deg": largest_control_error,
            "fixed_input_rotation_metrics": scores, "far_peak_taper_minus_control_deg": deltas}


def run(capture, executable):
    repo = Path(__file__).resolve().parents[2]
    report, reference = load_capture(capture, repo)
    hashes = report["input_sha256"]
    prior_path = repo / "docs/experiments/results" / WEAK_ARCHIVE
    prior = json.loads(prior_path.read_bytes())
    if (prior["source_report_sha256"] != digest(capture / "report.json")
            or prior["source_manifest_sha256"] != hashes["manifest.json"]
            or prior["ground_truth_sha256"] != report["manifest"]["inputs"]["ground_truth_sha256"]
            or prior["target_increment_scale"] != WEAK_SCALE
            or [w["frames"] for w in prior["windows"]] != [w["frames"] for w in report["windows"]]):
        raise ValueError("Capture and archived weak-scale controls differ")
    windows = [evaluate(capture, hashes, w, reference, executable) for w in prior["windows"]]
    result = {"sequence": "08", "scope": "Fixed-input development windows with baseline feedback",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "source_capture": str(capture), "source_report_sha256": prior["source_report_sha256"],
              "source_manifest_sha256": hashes["manifest.json"], "ground_truth_sha256": prior["ground_truth_sha256"],
              "weak_control_report_sha256": digest(prior_path), "runner_sha256": digest(Path(__file__).resolve()),
              "executable_sha256": digest(executable), "control_tolerance_deg": CONTROL_TOLERANCE_DEG,
              "target_increment_scale": WEAK_SCALE, "best_within_px": BEST_MAX, "second_beyond_px": FAR_MIN,
              "taper": "(second_beyond_px / second_to_endpoint_px)^2 for eligible points",
              "matched_budget": "one factor per frame pair times every original per-match increment",
              "windows": windows}
    out = capture / "secondary_peak_taper_rotation.json"
    atomic_json(out, result)
    print(json.dumps(windows, indent=2), flush=True)
    print(f"Report: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture.resolve(strict=True), args.executable.resolve(strict=True))
