"""Localize rotation drift from the saved KITTI photometric holdout poses."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_photometric_increment import HOLDOUT, MODES, digest, restore_rotations, rotation_metrics


GAP_BANDS = ((1, 1), (2, 10), (11, 30), (31, 60), (61, 120), (121, 200), (201, 299))
DETAIL_GAPS = (1, 10, 30, 60, 120, 200, 250, 299)
WINDOW_FRAMES = 50


def gap_errors(reference, estimate, gap):
    expected = reference[:-gap].swapaxes(1, 2) @ reference[gap:]
    actual = estimate[:-gap].swapaxes(1, 2) @ estimate[gap:]
    cosine = (np.sum(expected * actual, axis=(1, 2)) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def rmse(values):
    return float(np.sqrt(np.mean(values**2)))


def project_rotations(matrices):
    u, _, vt = np.linalg.svd(matrices)
    u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1, 1)[:, None]
    return u @ vt


def replace_window(per_match, original, targets, lo, hi):
    relative = per_match[:-1].swapaxes(1, 2) @ per_match[1:]
    original_relative = original[:-1].swapaxes(1, 2) @ original[1:]
    inside = (targets >= lo) & (targets < hi)
    relative[inside] = original_relative[inside]
    # Remove roundoff from recovering increments out of accumulated rotations.
    relative = project_rotations(relative)
    absolute = np.empty_like(per_match)
    absolute[0] = np.eye(3)
    for index, rotation in enumerate(relative, 1):
        absolute[index] = absolute[index - 1] @ rotation
    return absolute


def analyze(run, ground_truth, sequence):
    report_path = run / "report.json"
    manifest_path = run / "manifest.json"
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start, first, end = HOLDOUT
    if (recorded["manifest"] != manifest or manifest["action"] != "holdout"
            or (manifest["replay_start"], manifest["evaluation_start"], manifest["end_exclusive"]) != HOLDOUT
            or sequence not in manifest["sequences"] or manifest["modes"] != list(MODES)):
        raise ValueError("The report and manifest must agree with the frozen holdout")
    n = end - first
    slot = run / sequence
    times_path = slot / "times.txt"
    times = np.loadtxt(times_path, ndmin=1)
    if (times.shape != (end - start,) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError(f"{times_path}: incomplete or invalid times")
    raw_gt = np.loadtxt(ground_truth, ndmin=2)[first:end]
    if raw_gt.shape != (n, 12) or not np.isfinite(raw_gt).all():
        raise ValueError(f"{ground_truth}: incomplete KITTI truth")
    reference = project_rotations(raw_gt.reshape(-1, 3, 4)[:, :, :3])
    pose_paths = {mode: slot / mode / "ablation/PNEC.txt" for mode in MODES}
    estimates = {mode: restore_rotations(path, times, first - start) for mode, path in pose_paths.items()}
    errors = {mode: [gap_errors(reference, estimate, gap) for gap in range(1, n)]
              for mode, estimate in estimates.items()}
    curves = {mode: [rmse(values) for values in steps] for mode, steps in errors.items()}
    metrics = {mode: {"rpe_1_deg": curve[0], "rpe_n_deg": sum(curve) / n}
               for mode, curve in curves.items()}
    for mode, values in metrics.items():
        for key, actual in values.items():
            if not math.isclose(actual, recorded["results"][sequence][mode][key], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{sequence}/{mode} {key} does not reproduce the saved report")

    # These signed contributions sum to the total RPE_n change across all gaps.
    bands = []
    for lo, hi in GAP_BANDS:
        contribution = {mode: sum(curve[lo - 1:hi]) / n for mode, curve in curves.items()}
        bands.append({"gap_frames": [lo, hi], "rpe_n_contribution_deg": contribution,
                      "per_match_minus_original_deg": contribution["per_match"] - contribution["original"],
                      "per_match_minus_uniform_deg": contribution["per_match"] - contribution["uniform"]})

    windows = []
    for lo in range(first, end, WINDOW_FRAMES):
        hi = min(lo + WINDOW_FRAMES, end)
        by_gap = {}
        for gap in DETAIL_GAPS:
            target_frames = np.arange(first + gap, end)
            inside = (target_frames >= lo) & (target_frames < hi)
            if inside.any():
                by_gap[str(gap)] = {
                    "pairs": int(inside.sum()),
                    "rmse_deg": {mode: rmse(steps[gap - 1][inside]) for mode, steps in errors.items()},
                }
        # Replay the saved rotations mathematically, replacing only transitions
        # whose target is in this window. Other online outputs remain fixed.
        replacement = replace_window(estimates["per_match"], estimates["original"],
                                     np.arange(first + 1, end), lo, hi)
        replay_metrics = rotation_metrics(reference, replacement)
        anchor_gap = hi - 1 - first
        windows.append({
            "target_frames": [lo, hi - 1],
            "by_gap": by_gap,
            "end_error_from_clip_anchor_deg": {mode: float(steps[anchor_gap - 1][0])
                                               for mode, steps in errors.items()},
            "replace_per_match_with_original": replay_metrics,
            "replacement_change_in_per_match_rpe_n_deg":
                replay_metrics["rpe_n_deg"] - metrics["per_match"]["rpe_n_deg"],
        })
    return {
        "sequence": sequence,
        "evaluation_frames": [first, end - 1],
        "input_sha256": {
            "reported_metrics": digest(report_path), "manifest": digest(manifest_path),
            "times": digest(times_path), "ground_truth": digest(ground_truth),
            "relative_poses": {mode: digest(path) for mode, path in pose_paths.items()},
        },
        "recomputed_metrics": metrics,
        "per_gap_rmse_deg": curves,
        "gap_bands": bands,
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--sequence", choices=("01", "08"), default="01")
    parser.add_argument("--gt", type=Path, required=True, help="KITTI poses for the full sequence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = analyze(args.run, args.gt, args.sequence)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"KITTI {args.sequence}: all three RPE_1 and RPE_n values verified")
    print("gap_frames per_match_minus_original_contribution_deg per_match_minus_uniform_contribution_deg")
    for band in output["gap_bands"]:
        lo, hi = band["gap_frames"]
        print(f"{lo:3d}-{hi:<3d} {band['per_match_minus_original_deg']:+.6f} "
              f"{band['per_match_minus_uniform_deg']:+.6f}")
    print("target_frames rpe_1_deg(original/uniform/per_match) anchored_end_error_deg(original/uniform/per_match) "
          "replacement_change_in_per_match_rpe_n_deg")
    for window in output["windows"]:
        lo, hi = window["target_frames"]
        adjacent = window["by_gap"]["1"]["rmse_deg"]
        anchored = window["end_error_from_clip_anchor_deg"]
        print(f"{lo}-{hi} {'/'.join(f'{adjacent[m]:.4f}' for m in MODES)} "
              f"{'/'.join(f'{anchored[m]:.4f}' for m in MODES)} "
              f"{window['replacement_change_in_per_match_rpe_n_deg']:+.6f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
