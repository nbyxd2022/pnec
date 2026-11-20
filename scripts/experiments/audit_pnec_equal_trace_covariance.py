"""Compare frozen KLT covariances with equal-trace isotropic controls."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np

from audit_pnec_bearing_covariance import REPO, bearing, covariance, load_truth, project_covariance, read_archive
from audit_pnec_shared_geometry import geometry, joint_jacobian, rotate_left, TILE_PX
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config

CAPTURE = "pnec_kitti_09_10_forward_backward_audit_report.json"
GEOMETRY = "pnec_kitti_09_10_shared_geometry_audit_report.json"


def matched_frames(sequence, bundle, truth, capture, prior):
    config = replay_config(REPO, sequence)[0]
    fields = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                  if line.startswith(("Camera.", "PNEC.")))
    if any(float(fields["Camera." + key]) for key in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI images")
    focal = np.array([float(fields["Camera." + key]) for key in ("fx", "fy")])
    center = np.array([float(fields["Camera." + key]) for key in ("cx", "cy")])
    regularizer = float(fields["PNEC.regularization"])
    if digest(bundle) != capture["bundle_sha256"]:
        raise ValueError(f"KITTI {sequence} bundle differs from the frozen capture")
    rotations, positions, truth_sha = load_truth(truth, sequence)
    prior_frames = {row["target_frame"]: row for row in prior["frames"]}
    frames = []
    with tarfile.open(bundle, "r:gz") as tar:
        for known in capture["frames"]:
            target = known["target_frame"]
            name = f"matches_{target - 1}_{target}.csv"
            raw = tar.extractfile(name).read()
            if sha256(raw).hexdigest() != known["matches_sha256"]:
                raise ValueError(f"KITTI {sequence} match hash differs for {target}")
            motion = rotations[target - 1].T @ rotations[target]
            translation = rotations[target - 1].T @ (positions[target] - positions[target - 1])
            frame = compare_frame(raw, motion, translation, focal, center, regularizer, prior_frames[target])
            frames.append(frame)
    return frames, truth_sha


def compare_frame(raw, rotation, translation, focal, center, regularizer, prior):
    rows = np.genfromtxt(BytesIO(raw), delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    rows = rows[rows["ransac_inlier"] == 1]
    host_cov, target_cov = covariance(rows, "host_cov", 2), covariance(rows, "target_cov", 2)
    positive = (np.linalg.eigvalsh(host_cov)[:, 0] > 0) & (np.linalg.eigvalsh(target_cov)[:, 0] > 0)
    host_px = np.column_stack((rows["host_x_px"], rows["host_y_px"]))[positive]
    target_px = np.column_stack((rows["target_x_px"], rows["target_y_px"]))[positive]
    left, right = bearing(host_px, focal, center), bearing(target_px, focal, center)
    host_cov, target_cov = host_cov[positive], target_cov[positive]
    direction = translation / np.linalg.norm(translation)
    tangent = np.linalg.svd(direction[None, :], full_matrices=True)[2][1:].T
    e, norms, full, rotated_right = geometry(
        left, right, rotation, direction,
        project_covariance(host_px, host_cov, focal, center, np.sqrt(3)),
        project_covariance(target_px, target_cov, focal, center, np.sqrt(3)), regularizer,
    )
    iso_host = np.trace(host_cov, axis1=1, axis2=2)[:, None, None] * np.eye(2) / 2
    iso_target = np.trace(target_cov, axis1=1, axis2=2)[:, None, None] * np.eye(2) / 2
    _, _, isotropic, _ = geometry(
        left, right, rotation, direction,
        project_covariance(host_px, iso_host, focal, center, np.sqrt(3)),
        project_covariance(target_px, iso_target, focal, center, np.sqrt(3)), regularizer,
    )
    keep = (norms > 1e-9) & np.isfinite(full) & (full > 0)
    frame = prior["target_frame"]
    if (keep.sum() != prior["scored_points"]
            or not np.isclose(np.mean(e[keep] ** 2 / full[keep]), prior["reference_mean_q"], rtol=0, atol=1e-10)):
        raise ValueError(f"Prior reference score differs for frame {frame}")
    if not np.all(np.isfinite(isotropic[keep]) & (isotropic[keep] > 0)):
        raise ValueError(f"Invalid isotropic variance for frame {frame}")

    # Fit once per split, weighted only by the epipolar-plane normal, then share the held-out errors.
    jacobian = joint_jacobian(left, rotated_right, direction, tangent)[keep]
    split = (np.floor(host_px[keep] / TILE_PX).astype(int).sum(axis=1) % 2).astype(bool)
    left, right, e, norms = left[keep], right[keep], e[keep], norms[keep]
    held_out = np.empty_like(e)
    for train in (split, ~split):
        test = ~train
        if train.sum() < 6 or not test.any():
            raise ValueError(f"Insufficient spatial split in frame {frame}")
        delta, _, rank, _ = np.linalg.lstsq(
            jacobian[train] / norms[train, None], -e[train] / norms[train], rcond=None,
        )
        if rank != 5:
            raise ValueError(f"Rank-deficient spatial fit in frame {frame}")
        corrected_rotation = rotate_left(delta[:3], rotation)
        corrected_direction = direction + tangent @ delta[3:]
        corrected_direction /= np.linalg.norm(corrected_direction)
        gradient = np.cross(corrected_direction, left[test]) @ corrected_rotation
        held_out[test] = np.sum(gradient * right[test], axis=1)

    return np.stack((e ** 2, held_out ** 2)), np.stack((full[keep], isotropic[keep]))


def summary(sequence, frames, calibration_frames, truth_sha, bundle_sha):
    # Rows are reference and spatially held-out errors. Columns are full and isotropic variance.
    scales = np.mean([(error[:, None] / variance).mean(axis=-1) for error, variance in calibration_frames], axis=0)
    methods = {}
    for name, factors in (("raw", np.ones_like(scales)), ("other_sequence_scaled", scales)):
        differences = []
        for error, variance in frames:
            scaled = factors[:, :, None] * variance
            nll = 0.5 * (error[:, None] / scaled + np.log(scaled)).mean(axis=-1)
            differences.append(nll[:, 0] - nll[:, 1])
        delta = np.array(differences)
        methods[name] = {
            "scale_full_and_isotropic": factors.tolist(),
            "mean_full_minus_isotropic_nll": delta.mean(axis=0).tolist(),
            "median_full_minus_isotropic_nll": np.median(delta, axis=0).tolist(),
            "frames_with_lower_full_nll": (delta < 0).sum(axis=0).tolist(),
        }
    return {"sequence": sequence, "frame_pairs": len(frames), "observations": sum(e.shape[1] for e, _ in frames),
            "ground_truth_sha256_lf": truth_sha, "bundle_sha256": bundle_sha, **methods}


def main(args):
    sequences = ("09", "10")
    capture = {item["sequence"]: item for item in read_archive(CAPTURE)["sequences"]}
    prior = {item["sequence"]: item for item in read_archive(GEOMETRY)["sequences"]}
    inputs = {
        seq: matched_frames(seq, bundle, args.ground_truth_dir, capture[seq], prior[seq])
        for seq, bundle in zip(sequences, (args.bundle_09, args.bundle_10))
    }
    results = []
    for seq, other in (sequences, sequences[::-1]):
        frames, digest_truth = inputs[seq]
        results.append(summary(seq, frames, inputs[other][0], digest_truth, capture[seq]["bundle_sha256"]))
    report = {
        "scope": "Descriptive Gaussian score on frozen PNEC inliers, equal pixel covariance trace at each endpoint",
        "error_order": ["KITTI reference", "spatially held out"],
        "scale_order": ["full KLT", "isotropic"],
        "averaging": "equal weight per frame pair",
        "runner_sha256": digest(Path(__file__)),
        "sequences": results,
    }
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-09", "bundle-10", "ground-truth-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
