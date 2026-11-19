"""Cross KITTI 09 inliers and Ceres initializations from the two saved weak modes."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np

from analyze_photometric_drift import relative_steps
from audit_kitti09_weak_outliers import PER_MATCH, SEQUENCE, TARGETS, UNIFORM, replay_inputs
from run_fixed_input_audit import atomic_json, record_pairs
from run_photometric_increment import digest, pose_rows, restore_rotations
from run_taper_transfer import capture_index
from run_weak_scale_audit import CONTROL_TOLERANCE_DEG, WEAK_SCALE, control_difference_deg, pair_inputs, solve_pair


MODES = (UNIFORM, PER_MATCH)


def compare_pair(executable, target, pairs, reference, saved_steps):
    initial = {mode: pairs[mode][1]["initial"] for mode in MODES}
    ids = {mode: set(pairs[mode][0][0]["feature_id"]) for mode in MODES}
    init_translation = [initial[m][1] / np.linalg.norm(initial[m][1]) for m in MODES]
    axis_cosine = abs(np.dot(*init_translation))
    result = {
        "target_frame": target,
        "inliers": {mode: len(values) for mode, values in ids.items()},
        "shared_inlier_ids": len(ids[UNIFORM] & ids[PER_MATCH]),
        "initial_rotation_difference_deg": control_difference_deg(initial[UNIFORM][0], initial[PER_MATCH][0]),
        "initial_translation_axis_difference_deg": float(np.rad2deg(np.arccos(np.clip(axis_cosine, 0, 1)))),
        "native_replay_difference_deg": {},
        "errors_deg_by_inlier_source_then_init_source": {},
    }
    truth = reference[target - 1].T @ reference[target]
    for inlier_source, (inputs, _) in pairs.items():
        errors = {}
        for init_source, init in initial.items():
            rotations = solve_pair(executable, inputs, init)
            errors[init_source] = {mode: control_difference_deg(rotations[mode], truth) for mode in MODES}
            if init_source == inlier_source:
                difference = control_difference_deg(rotations[inlier_source], saved_steps[inlier_source][target - 1])
                if difference > CONTROL_TOLERANCE_DEG:
                    raise ValueError(f"Frame {target}, {inlier_source}: native rotation differs by {difference:.9g} degrees")
                result["native_replay_difference_deg"][inlier_source] = difference
        result["errors_deg_by_inlier_source_then_init_source"][inlier_source] = errors
    return result


def run(repo, build, dataset, output_root):
    original, saved, times, reference, source, config, poses = replay_inputs(repo, dataset)
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache:
        raise ValueError("Build directory belongs to another repository")
    end = max(TARGETS) + 1
    executable, solve_exe = build / "pnec_vo", build / "pnec_fixed_refine"
    out = Path(mkdtemp(prefix="pnec_kitti09_feedback_inputs_", dir=output_root))
    captures = {mode: out / mode for mode in MODES}
    hashes, saved_steps = {}, {}
    for mode, capture in captures.items():
        capture.mkdir()
        record_pairs(repo, executable, config, source, 0, times[:end], TARGETS, capture,
                     rotation_feedback=True, mode=mode)
        prefix = pose_rows(capture / "capture/ablation/PNEC.txt", times[:end])
        if not np.array_equal(prefix, pose_rows(poses[mode], times)[:end]):
            raise ValueError(f"{mode}: prefix differs from the original full replay: {capture}")
        saved_steps[mode] = relative_steps(restore_rotations(poses[mode], times, 0))
        hashes[mode] = capture_index(capture, TARGETS)
    rows = []
    for target in TARGETS:
        pairs = {mode: pair_inputs(capture, hashes[mode], target) for mode, capture in captures.items()}
        rows.append(compare_pair(solve_exe, target, pairs, reference, saved_steps))

    report = {"sequence": SEQUENCE, "source_run": str(original), "source_report_sha256": saved["report"],
              "tool_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "scope": "Ceres re-solves on archived online inliers and initializations for each target frame",
              "replay_end_exclusive": end, "target_increment_scale": WEAK_SCALE,
              "exact_pose_prefix_match": {mode: True for mode in MODES},
              "binary_sha256": {path.name: digest(path) for path in (executable, solve_exe)},
              "capture_sha256": hashes, "frames": rows}
    atomic_json(out / "report.json", report)
    for row in rows:
        print(json.dumps(row), flush=True)
    print(f"Report: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.build.resolve(strict=True),
        args.dataset.resolve(strict=True), args.output_root.resolve(strict=True))
