"""Analyze saved full-sequence weak allocation rotations."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_photometric_drift import full_scope, project_rotations, tail_replacements
from run_fixed_input_audit import atomic_json
from run_photometric_increment import HOLDOUT, WEAK_ALLOCATION_MODES, digest, restore_rotations, rotation_metrics


BASELINE, UNIFORM, PER_MATCH = WEAK_ALLOCATION_MODES


def read_json(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def find_run(root, sequence, expected):
    pattern = f"pnec_weak_allocation_full_09_10.*/{sequence}/pnec_increment_full_*/report.json"
    for path in sorted(root.glob(pattern)):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if report == expected:
            return path.parent
    raise FileNotFoundError(f"KITTI {sequence}: the archived full weak-allocation run was not found under {root}")


def analyze(run, dataset, sequence, expected, clip_report):
    manifest = read_json(run / "manifest.json")
    if manifest != expected["manifest"] or manifest["modes"] != list(WEAK_ALLOCATION_MODES):
        raise ValueError(f"Unexpected manifest for {run}")
    count = manifest["end_exclusive"]
    tail_start = manifest["tail_evaluation_start"]
    _, clip_first, clip_end = HOLDOUT
    times_path = dataset / "sequences" / sequence / "times.txt"
    gt_path = dataset / "poses" / f"{sequence}.txt"
    if digest(times_path) != manifest["times_sha256"] or digest(gt_path) != manifest["ground_truth_sha256"]:
        raise ValueError(f"KITTI {sequence}: GT or timestamps differ from replay")
    times = np.loadtxt(times_path, ndmin=1)
    truth = np.loadtxt(gt_path, ndmin=2)
    if (times.shape != (count,) or truth.shape != (count, 12)
            or not np.isfinite(times).all() or not np.isfinite(truth).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError(f"KITTI {sequence}: invalid timestamp or GT coverage")
    times -= times[0]
    saved_times = np.loadtxt(run / sequence / "times.txt", ndmin=1)
    if saved_times.shape != times.shape or not np.allclose(saved_times, times, rtol=0, atol=6e-13):
        raise ValueError(f"KITTI {sequence}: saved timestamps differ from the dataset")

    reference = project_rotations(truth.reshape(-1, 3, 4)[:, :, :3])
    pose_paths = {mode: run / sequence / mode / "ablation/PNEC.txt" for mode in WEAK_ALLOCATION_MODES}
    estimates = {mode: restore_rotations(path, times, 0) for mode, path in pose_paths.items()}
    scores = expected["results"][sequence]
    roles = {"baseline": BASELINE, "uniform": UNIFORM, "candidate": PER_MATCH}
    full = full_scope(reference, estimates, 0, scores, **roles)
    tail_estimates = {mode: rotations[tail_start:] for mode, rotations in estimates.items()}
    tail = full_scope(reference[tail_start:], tail_estimates, tail_start,
                      {mode: values["tail"] for mode, values in scores.items()}, **roles)
    for scope in (full, tail):
        del scope["per_gap_rmse_deg"]
        for window in scope["windows"]:
            metrics = window["metrics"]
            window["weak_per_match_minus_weak_uniform_rpe_n_deg"] = (
                metrics[PER_MATCH]["rpe_n_deg"] - metrics[UNIFORM]["rpe_n_deg"])

    return {
        "sequence": sequence, "source_run": str(run), "estimator_git_commit": manifest["git_commit"],
        "input_sha256": {"report": digest(run / "report.json"), "manifest": digest(run / "manifest.json"),
                         "ground_truth": manifest["ground_truth_sha256"], "times": manifest["times_sha256"],
                         "relative_poses": {mode: digest(path) for mode, path in pose_paths.items()}},
        "full": full, "tail": tail,
        "same_evaluation_clip": {
            "frames": [clip_first, clip_end - 1],
            "restarted_at_450": {mode: {k: v[k] for k in ("rpe_1_deg", "rpe_n_deg")}
                                 for mode, v in clip_report["results"][sequence].items()},
            "from_full_run": {mode: rotation_metrics(reference[clip_first:clip_end], rot[clip_first:clip_end])
                              for mode, rot in estimates.items()},
        },
        "tail_substitutions": tail_replacements(reference[tail_start:], tail_estimates, tail,
                                                baseline=UNIFORM, candidate=PER_MATCH),
        "scope": "Step substitutions recompose trajectories from archived rotations.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("09", "10"), required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    archives = Path(__file__).resolve().parents[2] / "docs/experiments/results"
    seq = args.sequence
    expected = read_json(archives / f"pnec_kitti_{seq}_weak_allocation_full_cadacca_user_report.json")
    clip = read_json(archives / f"pnec_kitti_{seq}_weak_allocation_3a58335_user_report.json")
    run = find_run(args.results_root, seq, expected)
    report = analyze(run, args.dataset, seq, expected, clip)
    report["analysis_source_sha256"] = digest(Path(__file__).resolve())
    out = run / "weak_allocation_rotation_drift.json"
    atomic_json(out, report)
    print(json.dumps({"sequence": seq, "same_evaluation_clip": report["same_evaluation_clip"],
                      "full_gap_bands": report["full"]["gap_bands"],
                      "tail_windows": report["tail"]["windows"],
                      "tail_substitutions": report["tail_substitutions"]}, indent=2))
    print(f"Report: {out}")
