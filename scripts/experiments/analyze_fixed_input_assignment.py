"""Compare measured and shuffled photometric increments on saved KITTI 08 inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_photometric_drift import project_rotations
from run_fixed_input_audit import atomic_json, ranks, read_points, variance_effect
from run_photometric_increment import digest


ARCHIVE = "pnec_kitti_08_fixed_input_windows_f3dbadb_user_report.json"
PERMUTATIONS = 32
SEED = 17


def nll_delta(extra, nominal_q):
    return .5 * (np.log1p(extra) - nominal_q * extra / (1 + extra))


def rank_correlation(a, b):
    a, b = ranks(a), ranks(b)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def compare_assignment(values, trace, rng):
    per_point, uniform, nominal_q = values.T
    mean_trace = trace.mean()
    # Keep each point's projection and nominal denominator in every permutation.
    # Reassign only the measured traces, preserving their distribution per frame.
    normalized_trace = trace / mean_trace if mean_trace > 0 else trace
    shuffled = np.stack([uniform * rng.permutation(normalized_trace)
                         for _ in range(PERMUTATIONS)])
    correlations = [rank_correlation(extra, nominal_q) for extra in shuffled]
    return (float(nll_delta(per_point, nominal_q).mean()),
            float(nll_delta(uniform, nominal_q).mean()),
            nll_delta(shuffled, nominal_q).mean(axis=1),
            rank_correlation(per_point, nominal_q), correlations)


def paired_summary(info):
    actual, uniform, shuffled, rho, shuffled_rho = zip(*info)
    actual = np.array(actual)
    shuffled = np.stack(shuffled)
    pair_null_medians = np.median(shuffled, axis=1)
    valid_rhos = [r for r in rho if r is not None]
    null_rhos = [r for per_frame in shuffled_rho for r in per_frame if r is not None]
    return {
        "frame_pairs": len(info),
        "mean_nll_delta_vs_nominal_equal_pair_weights": {
            "measured_assignment": float(actual.mean()),
            "uniform_assignment": float(np.mean(uniform)),
            "shuffled_assignment_median": float(np.median(shuffled.mean(axis=0))),
            "shuffled_assignment_min_max": [float(shuffled.mean(axis=0).min()),
                                            float(shuffled.mean(axis=0).max())],
        },
        "measured_beats_its_shuffled_median_pairs": int(np.sum(actual < pair_null_medians)),
        "measured_beats_uniform_pairs": int(np.sum(actual < uniform)),
        "median_pair_rank_correlation": {
            "measured_assignment": float(np.median(valid_rhos)) if valid_rhos else None,
            "shuffled_assignment": float(np.median(null_rhos)) if null_rhos else None,
            "measured_defined_pairs": len(valid_rhos),
            "shuffled_defined_comparisons": len(null_rhos),
        },
    }


def measure(run, report):
    manifest = report["manifest"]
    evidence = report["input_sha256"]
    manifest_path = run / "manifest.json"
    if digest(manifest_path) != evidence["manifest.json"] or json.loads(manifest_path.read_bytes()) != manifest:
        raise ValueError("Capture manifest differs from the validated report")
    if not report["baseline_prefix_exact_match"] or manifest["action"] != "windows":
        raise ValueError("Expected the fixed-input window capture with an exact baseline prefix")
    frozen = manifest["inputs"]
    gt_path = Path(frozen["dataset"]) / "poses/08.txt"
    if digest(gt_path) != frozen["ground_truth_sha256"]:
        raise ValueError("KITTI ground truth differs from the capture")
    truth = np.loadtxt(gt_path, ndmin=2).reshape(-1, 3, 4)
    reference = project_rotations(truth[:, :, :3])
    original = {p["target_frame"]: p for p in report["pairs"]}
    rng = np.random.RandomState(SEED)
    windows = []
    for lo, hi in frozen["windows"]:
        point_count = 0
        comparisons = []
        for target in range(lo + 1, hi + 1):
            host = target - 1
            point_file = f"capture/ablation/fixed_{host}_{target}_points.csv"
            matches_file = f"capture/ablation/matches_{host}_{target}.csv"
            for filename in (point_file, matches_file):
                if digest(run / filename) != evidence[filename]:
                    raise ValueError(f"Capture file differs from the validated report: {filename}")
            inputs = read_points(run / point_file, run / matches_file)
            true_rotation = reference[host].T @ reference[target]
            true_translation = reference[host].T @ (truth[target, :, 3] - truth[host, :, 3])
            if np.linalg.norm(true_translation) <= 1e-9:
                raise ValueError(f"Missing true translation direction at frame {target}")
            values = variance_effect(inputs, true_rotation, true_translation)
            trace = inputs[0]["delta_trace"]
            measured, uniform, null, rho, null_rhos = compare_assignment(values, trace, rng)
            expected = original[target]["at_gt_motion"]
            if (len(trace) != expected["points"]
                    or not np.isclose(measured, expected["symmetric_per_match"]["mean_gaussian_nll_delta"],
                                     rtol=1e-7, atol=1e-8)
                    or not np.isclose(uniform, expected["symmetric_uniform"]["mean_gaussian_nll_delta"],
                                     rtol=1e-7, atol=1e-8)):
                raise ValueError(f"Could not reproduce the saved residual scores at frame {target}")
            point_count += len(trace)
            comparisons.append((measured, uniform, null, rho, null_rhos))
        windows.append({"frames": [lo, hi], "observations": point_count,
                        **paired_summary(comparisons)})
    return windows


def main(results_root):
    repo = Path(__file__).resolve().parents[2]
    archive = repo / "docs/experiments/results" / ARCHIVE
    expected = json.loads(archive.read_bytes())
    candidates = []
    for path in sorted(results_root.glob("pnec_fixed_input_windows_*/report.json")):
        if json.loads(path.read_bytes()) == expected:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one archived window report under {results_root}; found {len(candidates)}")
    source, = candidates
    windows = measure(source.parent, expected)
    result = {
        "source_report_sha256": digest(source),
        "source_manifest_sha256": expected["input_sha256"]["manifest.json"],
        "script_sha256": digest(Path(__file__).resolve()),
        "permutations_per_pair": PERMUTATIONS,
        "seed": SEED,
        "scope": "Geometric residual scores at ground-truth motion on two chosen development windows",
        "interpretation": "Record-level trace shuffle within each pair; descriptive comparison, no p-value or covariance calibration claim",
        "windows": windows,
    }
    output = source.with_name("photometric_assignment_audit.json")
    atomic_json(output, result)
    print(json.dumps(windows, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    main(parser.parse_args().results_root.resolve(strict=True))
