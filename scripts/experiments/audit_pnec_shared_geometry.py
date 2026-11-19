"""Measure shared geometric residual on spatially held-out KITTI matches."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np

from audit_pnec_bearing_covariance import (
    REPO, bearing, covariance, digest_lf, directional_variance, load_truth, project_covariance, read_archive,
)
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config


TILE_PX = 128
REPORT_NAME = "pnec_kitti_09_10_forward_backward_audit_report.json"
FITS = {"rotation_only": slice(0, 3), "translation_only": slice(3, 5), "joint": slice(0, 5)}


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rotate_left(delta, rotation):
    angle = np.linalg.norm(delta)
    cross = skew(delta)
    sinc = np.sinc(angle / np.pi)
    cosc = 0.5 * np.sinc(angle / (2 * np.pi)) ** 2
    return (np.eye(3) + sinc * cross + cosc * (cross @ cross)) @ rotation


def geometry(left, right, rotation, direction, host_cov, target_cov, regularizer):
    rotated_right = right @ rotation.T
    target_gradient = np.cross(direction, left) @ rotation
    host_gradient = np.cross(rotated_right, direction)
    normals = np.linalg.norm(target_gradient, axis=1)
    residual = np.sum(target_gradient * right, axis=1)
    variance = (directional_variance(host_gradient, host_cov)
                + directional_variance(target_gradient, target_cov) + regularizer)
    return residual, normals, variance, rotated_right


def joint_jacobian(left, rotated_right, direction, tangent):
    inner = np.sum(left * rotated_right, axis=1)
    rotation_jacobian = inner[:, None] * direction - (rotated_right @ direction)[:, None] * left
    translation_jacobian = np.cross(left, rotated_right) @ tangent
    return np.column_stack((rotation_jacobian, translation_jacobian))


def measure_frame(raw, target, reference, position, focal, center, regularizer):
    rows = np.genfromtxt(BytesIO(raw), delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    host = target - 1
    if (not len(rows) or not np.all(rows["target_frame"] == target)
            or not np.all(rows["host_frame"] == host) or not np.all(rows["noise_frame"] == "target")):
        raise ValueError(f"Unexpected match records in frame {target}")
    rows = rows[rows["ransac_inlier"] == 1]
    left_pixel = np.column_stack((rows["host_x_px"], rows["host_y_px"]))
    right_pixel = np.column_stack((rows["target_x_px"], rows["target_y_px"]))
    left_cov = covariance(rows, "host_cov", 2)
    right_cov = covariance(rows, "target_cov", 2)
    keep = (np.linalg.eigvalsh(left_cov)[:, 0] > 0) & (np.linalg.eigvalsh(right_cov)[:, 0] > 0)
    left_pixel = left_pixel[keep]
    right_pixel = right_pixel[keep]
    left = bearing(left_pixel, focal, center)
    right = bearing(right_pixel, focal, center)
    left_cov = project_covariance(left_pixel, left_cov[keep], focal, center, np.sqrt(3))
    right_cov = project_covariance(right_pixel, right_cov[keep], focal, center, np.sqrt(3))
    rotation = reference[host].T @ reference[target]
    direction = reference[host].T @ (position[target] - position[host])
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        return {"target_frame": target, "status": "undefined_reference_translation"}
    direction /= length
    tangent = np.linalg.svd(direction[None, :], full_matrices=True)[2][1:].T
    residual, normals, variance, rotated_right = geometry(
        left, right, rotation, direction, left_cov, right_cov, regularizer,
    )
    valid = (normals > 1e-9) & np.isfinite(variance) & (variance > 0)
    jacobian = joint_jacobian(left, rotated_right, direction, tangent)
    scale = np.sqrt(variance[valid])
    jacobian = jacobian[valid] / scale[:, None]
    standardized = residual[valid] / scale
    tile = np.floor(left_pixel[valid] / TILE_PX).astype(np.int64)
    split = tile.sum(axis=1) % 2
    reference_squared = standardized ** 2
    fits = {}
    for mode, columns in FITS.items():
        heldout_squared = np.full(len(split), np.nan)
        corrections = []
        for group in (0, 1):
            train, test = split == group, split != group
            if train.sum() < 6 or not test.any():
                return {"target_frame": target, "status": "insufficient_spatial_split"}
            block = jacobian[train, columns]
            values, _, rank, _ = np.linalg.lstsq(block, -standardized[train], rcond=None)
            if rank != block.shape[1] or not np.isfinite(values).all():
                return {"target_frame": target, "status": "rank_deficient_geometry", "mode": mode}
            delta = np.zeros(5)
            delta[columns] = values
            rotation_fit = rotate_left(delta[:3], rotation)
            direction_fit = direction + tangent @ delta[3:]
            direction_fit /= np.linalg.norm(direction_fit)
            left_test, right_test = left[valid][test], right[valid][test]
            target_normal = np.cross(direction_fit, left_test) @ rotation_fit
            residual_fit = np.sum(target_normal * right_test, axis=1)
            heldout_squared[test] = (residual_fit / scale[test]) ** 2
            corrections.append({
                "training_points": int(train.sum()), "scored_points": int(test.sum()),
                "rotation_change_deg": float(np.rad2deg(np.linalg.norm(delta[:3]))),
                "translation_direction_change_deg": float(np.rad2deg(np.arctan(np.linalg.norm(delta[3:])))),
            })
        fits[mode] = {"heldout_corrected_mean_q": float(heldout_squared.mean()),
                      "heldout_corrected_fraction_q_above_9": float(np.mean(heldout_squared > 9)),
                      "fit_changes": corrections}
    return {
        "target_frame": target, "status": "ok", "scored_points": len(reference_squared),
        "reference_mean_q": float(reference_squared.mean()),
        "reference_fraction_q_above_9": float(np.mean(reference_squared > 9)),
        "fits": fits,
    }


def evaluate(bundle, sequence, args, focal, center, regularizer):
    ref = next(row for row in read_archive(REPORT_NAME)["sequences"] if row["sequence"] == sequence)
    if digest(bundle) != ref["bundle_sha256"]:
        raise ValueError(f"KITTI {sequence} bundle differs from the scored capture")
    rotations, positions, gt_hash = load_truth(args.ground_truth_dir, sequence)
    frames = []
    with tarfile.open(bundle, "r:gz") as tar:
        for row in ref["frames"]:
            target = row["target_frame"]
            raw = tar.extractfile(f"matches_{target - 1}_{target}.csv").read()
            if sha256(raw).hexdigest() != row["matches_sha256"]:
                raise ValueError(f"Invalid input hash for frame {target}")
            frames.append(measure_frame(raw, target, rotations, positions, focal, center, regularizer))

    valid = [frame for frame in frames if frame["status"] == "ok"]
    summary = {
        "sequence": sequence, "bundle_sha256": ref["bundle_sha256"], "ground_truth_sha256_lf": gt_hash,
        "frame_pairs": len(frames), "comparable_frames": len(valid),
        "reference_mean_q": float(np.mean([f["reference_mean_q"] for f in valid])) if valid else None,
        "reference_fraction_q_above_9": float(np.mean([f["reference_fraction_q_above_9"] for f in valid])) if valid else None,
        "fits": {}, "frames": frames,
    }
    for mode in FITS:
        rows = [f["fits"][mode] for f in valid]
        stats = {
            "frames_with_lower_heldout_mean_q": sum(r["heldout_corrected_mean_q"] < f["reference_mean_q"]
                                                   for f, r in zip(valid, rows)),
            "frames_with_lower_heldout_tail": sum(r["heldout_corrected_fraction_q_above_9"]
                                                 < f["reference_fraction_q_above_9"] for f, r in zip(valid, rows)),
        }
        for key in ("heldout_corrected_mean_q", "heldout_corrected_fraction_q_above_9"):
            stats[key] = float(np.mean([r[key] for r in rows])) if rows else None
        for name in ("rotation_change_deg", "translation_direction_change_deg"):
            changes = [fit[name] for row in rows for fit in row["fit_changes"]]
            stats[name] = {"median": float(np.median(changes)),
                           "p90": float(np.percentile(changes, 90))} if changes else None
        summary["fits"][mode] = stats
    return summary


def main(args):
    config = replay_config(REPO, "10")[0]
    parameters = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                      if line.startswith(("Camera.", "PNEC.")))
    if any(float(parameters["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI images")
    focal = np.array([float(parameters["Camera." + name]) for name in ("fx", "fy")])
    center = np.array([float(parameters["Camera." + name]) for name in ("cx", "cy")])
    regularizer = float(parameters["PNEC.regularization"])
    results = [evaluate(bundle, seq, args, focal, center, regularizer)
               for seq, bundle in (("09", args.bundle_09), ("10", args.bundle_10))]
    report = {
        "scope": "Spatial cross-fit diagnostic on frozen inliers at a KITTI reference-motion anchor",
        "interpretation": "A transferable residual component can reflect common motion or scene/model mismatch",
        "tile_size_px": TILE_PX, "split": "checkerboard parity of host pixel tile; evaluate the opposite parity",
        "fit": "One linearized least-squares step, exact held-out residual, dual variance frozen at reference motion",
        "config_sha256_lf": digest_lf(config), "runner_sha256": digest(Path(__file__)), "sequences": results,
    }
    atomic_json(args.output, report)
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-09", "bundle-10", "ground-truth-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
