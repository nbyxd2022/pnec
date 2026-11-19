"""Check if the target KLT covariance shape predicts residual tails at matched variance."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_pnec_bearing_covariance import (
    Q_THRESHOLD, REPO, bearing, covariance, digest_lf, directional_variance, load_truth,
    matches_09, matches_10, project_covariance,
)
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config


MAX_VARIANCE_RATIO = 1.10
QUARTILES = (.25, .75)
FIELDS = ("condition_number", "predicted_angular_variance_mrad2", "epipolar_angle_mrad2", "q_above_9")


def measurements(raw, target, rotation, translation, focal, center, regularizer, *, feature_field=None):
    rows = np.genfromtxt(BytesIO(raw), delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if (not len(rows) or not np.all(rows["host_frame"] == target - 1)
            or not np.all(rows["target_frame"] == target) or not np.all(rows["noise_frame"] == "target")):
        raise ValueError(f"Unexpected match records for target frame {target}")
    rows = rows[rows["ransac_inlier"] == 1]
    host_cov, target_cov = covariance(rows, "host_cov", 2), covariance(rows, "target_cov", 2)
    host_eigenvalues = np.linalg.eigvalsh(host_cov)
    target_eigenvalues = np.linalg.eigvalsh(target_cov)
    positive = (host_eigenvalues[:, 0] > 0) & (target_eigenvalues[:, 0] > 0)
    rows = rows[positive]
    host_px = np.column_stack((rows["host_x_px"], rows["host_y_px"]))
    target_px = np.column_stack((rows["target_x_px"], rows["target_y_px"]))
    host_bearing, target_bearing = bearing(host_px, focal, center), bearing(target_px, focal, center)
    spread = np.sqrt(3)
    host_projected = project_covariance(host_px, host_cov[positive], focal, center, spread)
    target_projected = project_covariance(target_px, target_cov[positive], focal, center, spread)
    length = np.linalg.norm(translation)
    if length <= 1e-9:
        raise ValueError(f"Undefined translation direction in frame {target}")
    direction = translation / length
    target_gradient = np.cross(direction, host_bearing) @ rotation
    host_gradient = np.cross(target_bearing @ rotation.T, direction)
    normal = np.linalg.norm(target_gradient, axis=1)
    valid = normal > 1e-9
    variance = (directional_variance(host_gradient, host_projected)
                + directional_variance(target_gradient, target_projected) + regularizer)[valid]
    if np.any(variance <= 0):
        raise ValueError(f"Nonpositive dual residual variance in frame {target}")
    residual = np.sum(target_gradient * target_bearing, axis=1)[valid]
    sine = residual / normal[valid]
    angles = 1000 * np.arcsin(np.clip(sine, -1, 1))
    eigenvalues = target_eigenvalues[positive][valid]
    condition = eigenvalues[:, 1] / eigenvalues[:, 0]
    feature = condition if feature_field is None else rows[feature_field][valid]
    values = np.column_stack((feature, 1e6 * variance / normal[valid] ** 2,
                              angles ** 2, residual ** 2 / variance > Q_THRESHOLD))
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite statistics in frame {target}")
    host_eigenvalues = host_eigenvalues[positive][valid]
    host_condition = host_eigenvalues[:, 1] / host_eigenvalues[:, 0]
    return values, np.abs(condition / host_condition - 1)


def matched_groups(values):
    if len(values) < 4:
        return 0, 0, None
    low_cut, high_cut = np.quantile(values[:, 0], QUARTILES)
    if low_cut == high_cut:
        return 0, 0, None
    lowest, highest = values[:, 0] <= low_cut, values[:, 0] >= high_cut
    low, high = values[lowest], values[highest]
    low = low[np.argsort(low[:, 1], kind="stable")]
    high = high[np.argsort(high[:, 1], kind="stable")]
    low_matched, high_matched = [], []
    low_index, high_index = 0, 0
    while low_index < len(low) and high_index < len(high):
        low_variance, high_variance = low[low_index, 1], high[high_index, 1]
        if low_variance * MAX_VARIANCE_RATIO < high_variance:
            low_index += 1
        elif high_variance * MAX_VARIANCE_RATIO < low_variance:
            high_index += 1
        else:
            low_matched.append(low[low_index])
            high_matched.append(high[high_index])
            low_index += 1
            high_index += 1
    if not low_matched:
        return len(low), len(high), None
    grouped = np.stack((np.mean(low_matched, axis=0), np.mean(high_matched, axis=0)))
    ratios = np.asarray(high_matched)[:, 1] / np.asarray(low_matched)[:, 1]
    return len(low), len(high), (len(low_matched), grouped, ratios)


def describe(means):
    stats = dict(zip(FIELDS, means.tolist()))
    stats["epipolar_rms_mrad"] = float(np.sqrt(stats["epipolar_angle_mrad2"]))
    return stats


def summarize(sequence, captures, args, focal, center, regularizer):
    rotation, translation, gt_hash = load_truth(args.ground_truth_dir, sequence)
    frames, means, shape_changes = [], [], []
    for target, raw, checksum in captures:
        if sha256(raw).hexdigest() != checksum:
            raise ValueError(f"Unexpected KITTI {sequence} capture hash in frame {target}")
        host = target - 1
        rot = rotation[host].T @ rotation[target]
        t = rotation[host].T @ (translation[target] - translation[host])
        values, shape_change = measurements(raw, target, rot, t, focal, center, regularizer)
        shape_changes.append(shape_change)
        n_low, n_high, compared = matched_groups(values)
        row = {"target_frame": target, "matches_sha256": checksum, "eligible": len(values),
               "low_available": n_low, "high_available": n_high}
        if compared is not None:
            count, paired, ratios = compared
            row.update(matched_points_per_group=count, low=describe(paired[0]), high=describe(paired[1]),
                       matched_predicted_variance_ratio_min_max=[float(ratios.min()), float(ratios.max())])
            means.append(paired)
        frames.append(row)

    if not means:
        raise ValueError(f"No comparable pairs in KITTI {sequence}")
    frame_means = np.stack(means)
    low, high = map(describe, frame_means.mean(axis=0))
    changes = np.concatenate(shape_changes)
    return {
        "sequence": sequence, "ground_truth_sha256_lf": gt_hash,
        "frame_pairs": len(frames), "comparable_frame_pairs": len(means),
        "matched_points_per_group": sum(row.get("matched_points_per_group", 0) for row in frames),
        "relative_change_target_vs_host_condition_number": {
            "points": len(changes), "p99": float(np.percentile(changes, 99)),
        },
        "frames_with_higher_q_tail_in_high_group": int(np.sum(frame_means[:, 1, 3] > frame_means[:, 0, 3])),
        "frames_with_higher_squared_error_in_high_group": int(np.sum(frame_means[:, 1, 2] > frame_means[:, 0, 2])),
        "low": low, "high": high, "frames": frames,
    }


def main(args):
    config = replay_config(REPO, "10")[0]
    options = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                   if line.startswith(("Camera.", "PNEC.")))
    if any(float(options["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI input")
    focal = np.array([float(options["Camera." + name]) for name in ("fx", "fy")])
    center = np.array([float(options["Camera." + name]) for name in ("cx", "cy")])
    regularizer = float(options["PNEC.regularization"])
    sequences = [summarize(seq, captures, args, focal, center, regularizer) for seq, captures in (
        ("09", matches_09(args.matches_09)), ("10", matches_10(args.bundle_10, config)),
    )]
    report = {
        "scope": "Descriptive target covariance shape diagnostic at fixed KITTI reference motion",
        "feature": "Target pixel covariance lambda_max/lambda_min",
        "quartiles_within_frame": QUARTILES, "max_predicted_angular_variance_ratio_for_pairing": MAX_VARIANCE_RATIO,
        "matching": "Sort both groups by predicted dual angular variance and pair within each frame",
        "frame_weight": "Equal among comparable frames",
        "q_threshold": Q_THRESHOLD, "regularizer": regularizer, "config_sha256_lf": digest_lf(config),
        "kitti10_capture_bundle_sha256": digest(args.bundle_10), "runner_sha256": digest(Path(__file__)),
        "sequences": sequences,
    }
    atomic_json(args.output, report)
    for result in sequences:
        print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matches-09", "bundle-10", "ground-truth-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
