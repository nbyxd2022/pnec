"""Compare weak allocation with both outputs following one PNEC feedback trajectory."""

import argparse
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np

from analyze_photometric_drift import gap_errors
from analyze_weak_allocation_full import read_json
from audit_kitti09_weak_outliers import BASELINE, PER_MATCH, SEQUENCE, TARGETS, UNIFORM, replay_inputs
from run_fixed_input_audit import atomic_json, record_pairs
from run_photometric_increment import digest, pose_rows, restore_rotations, rotation_metrics
from run_weak_scale_audit import WEAK_SCALE


def score(capture, original_report, baseline_path, times, reference, *, baseline=BASELINE):
    folder = capture / "capture/ablation"
    paths = {baseline: folder / "PNEC.txt", UNIFORM: folder / "weak_uniform.txt",
             PER_MATCH: folder / "weak_per_match.txt"}
    if not np.array_equal(pose_rows(paths[baseline], times), pose_rows(baseline_path, times)):
        raise ValueError(f"Feedback trajectory differs from the archived full run: {capture}")
    tail = original_report["manifest"]["tail_evaluation_start"]
    sequence, = original_report["manifest"]["sequences"]
    targets = TARGETS if sequence == SEQUENCE else ()
    estimates = {mode: restore_rotations(path, times, 0) for mode, path in paths.items()}
    step_errors = {mode: gap_errors(reference, rotation, 1) for mode, rotation in estimates.items()}
    return {
        "baseline_exact_pose_rows_match": True,
        "frames": len(times), "tail_evaluation_start": tail,
        "shared_feedback_metrics": {
            mode: {**rotation_metrics(reference, rotation),
                   "tail": rotation_metrics(reference[tail:], rotation[tail:])}
            for mode, rotation in estimates.items()
        },
        "independent_feedback_metrics": {
            mode: {**{key: metrics[key] for key in ("rpe_1_deg", "rpe_n_deg")},
                   "tail": {key: metrics["tail"][key] for key in ("rpe_1_deg", "rpe_n_deg")}}
            for mode, metrics in original_report["results"][sequence].items()
        },
        "target_frame_errors_deg": [
            {"target_frame": target, **{mode: float(errors[target - 1]) for mode, errors in step_errors.items()}}
            for target in targets
        ],
        "relative_poses_sha256": {mode: digest(path) for mode, path in paths.items()},
    }


def find_original_run(repo, output_root, sequence):
    archive = repo / "docs/experiments/results" / f"pnec_kitti_{sequence}_original_feedback_a89d06a_user_report.json"
    expected = read_json(archive)
    pattern = f"pnec_original_feedback_09_10.*/{sequence}/pnec_increment_full_*/report.json"
    for path in sorted(output_root.glob(pattern)):
        if read_json(path) == expected:
            return path.parent, expected
    raise FileNotFoundError(f"KITTI {sequence}: no replay matches {archive.name}")


def run(repo, build, dataset, output_root, *, sequence=SEQUENCE, feedback_source=BASELINE):
    original, saved, times, reference, source, config, poses = replay_inputs(repo, dataset, full=True, sequence=sequence)
    original_report = read_json(original / "report.json")
    baseline_path, source_report_sha256 = poses[BASELINE], saved["report"]
    prefix = f"pnec_kitti{sequence}_shared_feedback_"
    if feedback_source == "original":
        reference_manifest = original_report["manifest"]
        original, original_report = find_original_run(repo, output_root, sequence)
        manifest = original_report["manifest"]
        if (manifest != read_json(original / "manifest.json")
                or any(manifest[key] != reference_manifest[key] for key in (
                    "times_sha256", "ground_truth_sha256", "image_sha256", "tail_evaluation_start",
                    "ransac_sampling", "replay_start", "evaluation_start", "end_exclusive"))):
            raise ValueError("Original-weight replay inputs differ from the analytic run")
        if {p.name: digest(p) for p in config} != read_json(original / sequence / "config_sha256.json"):
            raise ValueError("Original-weight configuration differs from the current replay")
        baseline_path = original / sequence / "original/ablation/PNEC.txt"
        source_report_sha256 = digest(original / "report.json")
        prefix = f"pnec_kitti{sequence}_original_shared_feedback_"
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache:
        raise ValueError("Build directory belongs to another repository")
    executable = build / "pnec_vo"
    out = Path(mkdtemp(prefix=prefix, dir=output_root))
    record_pairs(repo, executable, config, source, 0, times, (), out,
                 rotation_feedback=True, mode=feedback_source, shared_weak=True)
    report = {
        "sequence": sequence, "source_run": str(original), "source_report_sha256": source_report_sha256,
        "tool_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "binary_sha256": digest(executable), "target_increment_scale": WEAK_SCALE,
        "feedback_source": feedback_source,
        "feedback": "The feedback solution drives the next frame; weak estimates share inliers and Ceres initialization",
        **score(out, original_report, baseline_path, times, reference, baseline=feedback_source),
    }
    atomic_json(out / "report.json", report)
    for scope in ("shared_feedback_metrics", "independent_feedback_metrics"):
        print(scope, flush=True)
        for mode, values in report[scope].items():
            print(f"  {mode}: RPE1={values['rpe_1_deg']:.6f} "
                  f"RPEn={values['rpe_n_deg']:.6f} tail_RPEn={values['tail']['rpe_n_deg']:.6f}", flush=True)
    print(f"Report: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("09", "10"), default=SEQUENCE)
    parser.add_argument("--feedback-source", choices=(BASELINE, "original"), default=BASELINE)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.build.resolve(strict=True),
        args.dataset.resolve(strict=True), args.output_root.resolve(strict=True),
        sequence=args.sequence, feedback_source=args.feedback_source)
