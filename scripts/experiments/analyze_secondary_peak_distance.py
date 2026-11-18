"""Test distant secondary photometric minima on the frozen KITTI 08 capture."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analyze_fixed_input_assignment import ARCHIVE
from analyze_photometric_drift import project_rotations
from run_fixed_input_audit import atomic_json, read_points
from run_photometric_increment import digest
from run_weak_scale_audit import checked_file


# Analysis cuts in pixels on the saved +/-16 px candidate grid.
BEST_MAX = 2
NEAR_MAX = 8
FAR_MIN = 12
RATIO_BINS = 20
FIELDS = ("cost_ratio", "best_to_endpoint_px", "second_to_endpoint_px", "best_to_second_px",
          "added_trace_rad2", "abs_epipolar_mrad", "squared_epipolar_mrad2")


def epipolar_angles(left, right, rotation, translation):
    """Target bearing to the GT epipolar plane, conditional on the observed host ray."""
    angles = np.full(len(left), np.nan)
    length = np.linalg.norm(translation)
    if length <= 1e-9:
        return angles
    # GT maps target vectors into the host frame. Row normals here are mapped back to target.
    normals = np.cross(translation / length, left) @ rotation
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-9
    sine = np.sum(right[valid] * normals[valid], axis=1) / norms[valid]
    angles[valid] = 1000 * np.arcsin(np.clip(np.abs(sine), 0, 1))
    return angles


def compare_frame(rows, angles):
    best = np.hypot(rows["best_dx"], rows["best_dy"])
    separation = np.hypot(rows["mode_dx"], rows["mode_dy"])
    second = np.hypot(rows["best_dx"] + rows["mode_dx"], rows["best_dy"] + rows["mode_dy"])
    has_two = np.isfinite(rows["best_cost"]) & np.isfinite(rows["second_cost"])
    at_endpoint = has_two & (best <= BEST_MAX)
    eligible = at_endpoint & np.isfinite(angles)
    near = eligible & (second <= NEAR_MAX)
    far = eligible & (second >= FAR_MIN)
    counts = Counter(total=len(rows), missing_two_minima=int((~has_two).sum()),
                     best_outside=int((has_two & ~at_endpoint).sum()),
                     invalid_plane=int((at_endpoint & ~eligible).sum()),
                     middle=int((eligible & ~near & ~far).sum()),
                     near_available=int(near.sum()), far_available=int(far.sum()),
                     near_compared=0, far_compared=0, overlap_weight=0)
    values = np.column_stack((rows["ratio"], best, second, separation,
                              rows["delta_trace"], angles, angles**2))
    bins = np.minimum((rows["ratio"] * RATIO_BINS).astype(int), RATIO_BINS - 1)
    summed = np.zeros((2, len(FIELDS)))
    for index in np.intersect1d(bins[near], bins[far]):
        a = values[near & (bins == index)]
        b = values[far & (bins == index)]
        # Give the two groups identical stratum weights. No point is duplicated.
        weight = min(len(a), len(b))
        summed += weight * np.stack((a.mean(axis=0), b.mean(axis=0)))
        counts.update(near_compared=len(a), far_compared=len(b), overlap_weight=weight)
    if not counts["overlap_weight"]:
        return counts, None
    return counts, summed / counts["overlap_weight"]


def window_summary(frames, counts, frame_means):
    summary = {"frames": frames, "frame_pairs": frames[1] - frames[0],
               "comparable_frame_pairs": len(frame_means), "counts": dict(counts)}
    if not frame_means:
        summary["status"] = "no comparable near/far groups"
        return summary
    groups = np.stack(frame_means)
    near, far = groups.mean(axis=0)  # equal weight for every comparable frame
    for key, values in (("near", near), ("far", far)):
        summary[key] = dict(zip(FIELDS, values.tolist()))
        summary[key]["rms_epipolar_mrad"] = float(np.sqrt(values[-1]))
    summary["far_over_near"] = {
        key: float(far[i] / near[i]) if near[i] > 0 else None
        for i, key in enumerate(FIELDS) if key in ("added_trace_rad2", "squared_epipolar_mrad2")}
    summary["frame_counts_far_larger"] = {
        key: int((groups[:, 1, i] > groups[:, 0, i]).sum())
        for i, key in enumerate(FIELDS) if key in ("added_trace_rad2", "squared_epipolar_mrad2")}
    return summary


def analyze(capture):
    repo = Path(__file__).resolve().parents[2]
    archive = repo / "docs/experiments/results" / ARCHIVE
    report = json.loads((capture / "report.json").read_bytes())
    if report != json.loads(archive.read_bytes()) or not report["baseline_prefix_exact_match"]:
        raise ValueError("Expected the archived KITTI 08 fixed-input window report")
    hashes = report["input_sha256"]
    checked_file(capture / "manifest.json", hashes["manifest.json"])
    frozen = report["manifest"]["inputs"]
    gt = checked_file(Path(frozen["dataset"]) / "poses/08.txt", frozen["ground_truth_sha256"])
    truth = np.loadtxt(gt, ndmin=2).reshape(-1, 3, 4)
    reference = project_rotations(truth[:, :, :3])
    windows = []
    for window in report["windows"]:
        lo, hi = window["frames"]
        counts = Counter()
        frame_means = []
        for target in range(lo + 1, hi + 1):
            host = target - 1
            prefix = f"capture/ablation/fixed_{host}_{target}"
            points = prefix + "_points.csv"
            matches = f"capture/ablation/matches_{host}_{target}.csv"
            rows, left, right, _, _ = read_points(checked_file(capture / points, hashes[points]),
                                                  checked_file(capture / matches, hashes[matches]))
            rotation = reference[host].T @ reference[target]
            translation = reference[host].T @ (truth[target, :, 3] - truth[host, :, 3])
            frame_counts, means = compare_frame(rows, epipolar_angles(left, right, rotation, translation))
            counts.update(frame_counts)
            if means is not None:
                frame_means.append(means)
        if counts["total"] != sum(p["inliers"] for p in report["pairs"] if lo < p["target_frame"] <= hi):
            raise ValueError(f"Wrong observation count in window {lo}-{hi}")
        summary = window_summary([lo, hi], counts, frame_means)
        windows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    result = {"sequence": "08", "source_capture": str(capture),
              "source_report_sha256": digest(capture / "report.json"),
              "source_manifest_sha256": hashes["manifest.json"], "ground_truth_sha256": digest(gt),
              "script_sha256": digest(Path(__file__).resolve()),
              "comparison": {"best_within_px": BEST_MAX, "near_second_at_most_px": NEAR_MAX,
                             "far_second_at_least_px": FAR_MIN, "cost_ratio_bin_width": 1 / RATIO_BINS,
                             "stratum_weight": "smaller group count in shared ratio bin of same frame",
                             "frame_weight": "equal for each comparable frame",
                             "geometry": "absolute target bearing angle to GT epipolar plane in mrad"},
              "windows": windows}
    out = capture / "secondary_peak_distance.json"
    atomic_json(out, result)
    print(f"Report: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    analyze(parser.parse_args().capture.resolve(strict=True))
