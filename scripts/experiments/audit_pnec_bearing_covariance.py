"""Audit PNEC bearing covariance against Appendix D on saved KITTI matches."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np

from analyze_photometric_drift import project_rotations
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config


REPO = Path(__file__).resolve().parents[2]
ARCHIVES = REPO / "docs/experiments/results"
Q_THRESHOLD = 9.0
# Appendix D uses five samples with n=2 and kappa=1.
WEIGHTS = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
MODELS = ("native_target", "triple_native_target", "appendix_d_target", "appendix_d_both")


def read_archive(filename):
    return json.loads((ARCHIVES / filename).read_bytes())


def matches_09(directory):
    audit = read_archive("pnec_kitti_09_feedback_inputs_49cec8d_user_report.json")
    hashes = audit["capture_sha256"]["weak_uniform"]
    for row in audit["frames"]:
        target = row["target_frame"]
        name = f"matches_{target - 1}_{target}.csv"
        yield target, (directory / name).read_bytes(), hashes["capture/ablation/" + name]


def matches_10(bundle, config):
    with tarfile.open(bundle, "r:gz") as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
        if (manifest["sequence"] != "10" or not manifest["original_prefix_exact"]
                or manifest["target_frames"] != list(range(300, 400))
                or manifest["config_sha256"][config.name] != digest_lf(config)):
            raise ValueError("Unexpected KITTI 10 capture manifest")
        for target in manifest["target_frames"]:
            name = f"matches_{target - 1}_{target}.csv"
            yield target, tar.extractfile(name).read(), manifest["matches_sha256"][name]


def digest_lf(path):
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_truth(dataset, sequence):
    baseline = read_archive(f"pnec_kitti_{sequence}_original_feedback_a89d06a_user_report.json")
    truth_path = dataset / f"{sequence}.txt"
    checksum = digest_lf(truth_path)
    if checksum != baseline["manifest"]["ground_truth_sha256"]:
        raise ValueError(f"Ground truth differs from the archived KITTI {sequence} baseline")
    truth = np.loadtxt(truth_path).reshape(-1, 3, 4)
    return project_rotations(truth[:, :, :3]), truth[:, :, 3], checksum


def covariance(rows, prefix, size):
    cols = [rows[f"{prefix}{i}{j}"] for i in range(size) for j in range(size)]
    return np.column_stack(cols).reshape(-1, size, size)


def bearing(pixels, focal, center):
    xy = (pixels - center) / focal
    rays = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project_covariance(pixels, pixel_cov, focal, center, spread):
    # The native lower Cholesky factor is read column by column.
    offsets = np.linalg.cholesky(pixel_cov).swapaxes(1, 2) * spread
    origin = pixels[:, None, :]
    sigma = bearing(np.concatenate((origin, origin + offsets, origin - offsets), axis=1), focal, center)
    delta = sigma - np.einsum("s,nsi->ni", WEIGHTS, sigma)[:, None, :]
    return np.einsum("s,nsi,nsj->nij", WEIGHTS, delta, delta)


def directional_variance(gradient, covariances):
    return np.einsum("ni,nij,nj->n", gradient, covariances, gradient)


def analyze_pair(raw, target, rotation, translation, focal, center, regularizer):
    rows = np.genfromtxt(BytesIO(raw), delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    if (not len(rows) or not np.all(rows["target_frame"] == target)
            or not np.all(rows["host_frame"] == target - 1)
            or not np.array_equal(rows["solver_index"], np.arange(len(rows)))
            or not np.all(rows["noise_frame"] == "target")
            or not np.all(rows["ransac_enabled"] == 1)
            or not np.isin(rows["ransac_inlier"], [0, 1]).all()):
        raise ValueError(f"Unexpected native match records in frame {target}")
    cov_host = covariance(rows, "host_cov", 2)
    cov_target = covariance(rows, "target_cov", 2)
    native = covariance(rows, "projected_cov", 3)
    accepted = rows["ransac_inlier"] == 1
    positive = (np.linalg.eigvalsh(cov_host)[:, 0] > 0) & (np.linalg.eigvalsh(cov_target)[:, 0] > 0)
    excluded = rows["feature_id"][~positive & accepted].astype(int).tolist()
    if not (positive & accepted).any():
        raise ValueError(f"No inliers with positive definite pixel covariances in frame {target}")
    positive_rows = rows[positive]
    cov_host, cov_target, native = cov_host[positive], cov_target[positive], native[positive]
    host_px = np.column_stack((positive_rows["host_x_px"], positive_rows["host_y_px"]))
    target_px = np.column_stack((positive_rows["target_x_px"], positive_rows["target_y_px"]))

    reproduced = project_covariance(target_px, cov_target, focal, center, 1.0)
    relative_error = np.linalg.norm(native - reproduced, axis=(1, 2)) / np.linalg.norm(native, axis=(1, 2))
    if not np.isfinite(relative_error).all() or relative_error.max() > 1e-8:
        raise ValueError(f"Native projection reconstruction differs in frame {target}")
    corrected = project_covariance(target_px, cov_target, focal, center, np.sqrt(3))
    host_corrected = project_covariance(host_px, cov_host, focal, center, np.sqrt(3))

    if np.linalg.norm(translation) <= 1e-9:
        raise ValueError(f"Undefined epipolar translation direction in frame {target}")
    direction = translation / np.linalg.norm(translation)
    left, right = bearing(host_px, focal, center), bearing(target_px, focal, center)
    target_gradient = np.cross(direction, left) @ rotation
    host_gradient = np.cross(right @ rotation.T, direction)
    residual2 = np.sum(target_gradient * right, axis=1) ** 2
    native_var = directional_variance(target_gradient, native)
    corrected_var = directional_variance(target_gradient, corrected)
    host_var = directional_variance(host_gradient, host_corrected)
    denominators = np.column_stack((native_var, 3 * native_var, corrected_var,
                                    corrected_var + host_var)) + regularizer
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValueError(f"Invalid residual variance in frame {target}")

    inlier = positive_rows["ransac_inlier"] == 1
    trace_ratio = (np.trace(corrected, axis1=1, axis2=2) / np.trace(native, axis1=1, axis2=2))[inlier]
    relative_to_scalar = np.abs(denominators[:, 2] / denominators[:, 1] - 1)[inlier]
    relative_to_full_scale = np.abs(denominators[:, 2] / (3 * denominators[:, 0]) - 1)[inlier]
    q = (residual2[:, None] / denominators)[inlier]
    counts = {
        "target_frame": target, "matches": len(rows),
        "native_inliers": int(accepted.sum()), "analyzed_inliers": int(inlier.sum()),
        "nonpositive_pixel_covariance_matches": int((~positive).sum()),
        "excluded_inlier_feature_ids": excluded,
        "max_native_projection_relative_error": float(relative_error.max()),
    }
    return counts, q, trace_ratio, relative_to_scalar, relative_to_full_scale


def summarize_model(values):
    return {
        "q_median": float(np.median(values)),
        "q_p90": float(np.percentile(values, 90)),
        "fraction_q_above_9": float(np.mean(values > Q_THRESHOLD)),
    }


def analyze_sequence(sequence, source, dataset, captures, focal, center, regularizer):
    rotations, positions, gt_hash = load_truth(dataset, sequence)
    frames, score_rows, trace_ratios, scalar_errors, full_scale_errors = [], [], [], [], []
    for target, raw, checksum in captures:
        if sha256(raw).hexdigest() != checksum:
            raise ValueError(f"Capture hash differs for KITTI {sequence}, target frame {target}")
        host = target - 1
        rotation = rotations[host].T @ rotations[target]
        translation = rotations[host].T @ (positions[target] - positions[host])
        counts, q, ratio, scalar_error, full_scale_error = analyze_pair(
            raw, target, rotation, translation, focal, center, regularizer,
        )
        frames.append({**counts, "matches_sha256": checksum,
                       "models": {name: summarize_model(q[:, j]) for j, name in enumerate(MODELS)}})
        score_rows.append(q)
        trace_ratios.append(ratio)
        scalar_errors.append(scalar_error)
        full_scale_errors.append(full_scale_error)
    q = np.concatenate(score_rows)
    sample_stats = {name: summarize_model(q[:, j]) for j, name in enumerate(MODELS)}
    ratios = np.concatenate(trace_ratios)
    errors = np.concatenate(scalar_errors)
    return {
        "sequence": sequence, "source": source, "ground_truth_sha256_lf": gt_hash,
        "pairs": len(frames), "analyzed_inliers": len(q),
        "max_native_projection_relative_error": max(row["max_native_projection_relative_error"] for row in frames),
        "appendix_d_target_over_native_bearing_trace": {
            "median": float(np.median(ratios)), "p01": float(np.percentile(ratios, 1)),
            "p99": float(np.percentile(ratios, 99)),
        },
        "relative_residual_variance_difference_appendix_d_vs_triple_native": {
            "p99": float(np.percentile(errors, 99)), "max": float(errors.max()),
        },
        "relative_residual_denominator_difference_vs_triple_native_including_regularizer": {
            "p99": float(np.percentile(np.concatenate(full_scale_errors), 99)),
        },
        "models": sample_stats,
        "frame_median_fraction_q_above_9": {
            name: float(np.median([frame["models"][name]["fraction_q_above_9"] for frame in frames]))
            for name in MODELS
        },
        "frames": frames,
    }


def main(args):
    config = replay_config(REPO, "10")[0]
    parameters = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                      if line.startswith(("Camera.", "PNEC.")))
    if any(float(parameters["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI data")
    focal = np.array([float(parameters["Camera." + name]) for name in ("fx", "fy")])
    center = np.array([float(parameters["Camera." + name]) for name in ("cx", "cy")])
    regularizer = float(parameters["PNEC.regularization"])
    sections = [
        analyze_sequence("09", "Selected weak-uniform feedback pairs from 49cec8d", args.ground_truth_dir,
                         matches_09(args.matches_09), focal, center, regularizer),
        analyze_sequence("10", "Original feedback pairs, target frames 300–399", args.ground_truth_dir,
                         matches_10(args.bundle_10, config), focal, center, regularizer),
    ]
    report = {
        "scope": "Epipolar residuals at KITTI reference motion on captured RANSAC inliers",
        "interpretation": "Residual-scale diagnostic; no individual pixel ground truth or pose re-estimation",
        "q": "Squared epipolar residual divided by projected residual variance plus native regularizer",
        "regularizer": regularizer, "config_sha256_lf": digest_lf(config),
        "focal_px": focal.tolist(), "center_px": center.tolist(),
        "kitti10_capture_bundle_sha256": digest(args.bundle_10),
        "runner_sha256": digest(Path(__file__)), "sequences": sections,
    }
    atomic_json(args.output, report)
    for section in sections:
        print(json.dumps({k: v for k, v in section.items() if k != "frames"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matches-09", "bundle-10", "ground-truth-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
