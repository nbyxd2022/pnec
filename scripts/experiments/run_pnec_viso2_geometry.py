"""Run standalone libviso2 and compare its geometry on frozen PNEC KITTI matches."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import mkdtemp

import numpy as np

from audit_pnec_bearing_covariance import REPO, bearing, covariance, load_truth, project_covariance, read_archive
from audit_pnec_shared_geometry import geometry
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config


VISO2_COMMIT = "6e59c408289cc40cbc6a2205e2309be36dc7a7de"
SEQUENCES = ("09", "10")
START = 299
TARGETS = range(START + 1, 400)
PAIR_COUNT = len(TARGETS)
MOTION_CONVENTION = "p_current = R_current_from_previous @ p_previous + t_current_from_previous"
CROSS_FIT = "pnec_kitti_09_10_shared_geometry_audit_report.json"
FB_CAPTURE = "pnec_kitti_09_10_forward_backward_audit_report.json"


def camera_config(sequence):
    config = replay_config(REPO, sequence)[0]
    settings = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                    if line.startswith(("Camera.", "PNEC.")))
    focal = np.array([float(settings["Camera." + name]) for name in ("fx", "fy")])
    center = np.array([float(settings["Camera." + name]) for name in ("cx", "cy")])
    if any(float(settings["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI images")
    return focal, center, float(settings["PNEC.regularization"])


def calibration(path, focal, center):
    rows = dict(line.split(":", 1) for line in path.read_text().splitlines() if line.startswith(("P0:", "P1:")))
    intrinsics = np.array([[focal[0], 0, center[0]], [0, focal[1], center[1]], [0, 0, 1]])
    offsets = []
    for camera in ("P0", "P1"):
        projections = np.array([float(value) for value in rows[camera].split()]).reshape(3, 4)
        if not np.allclose(projections[:, :3], intrinsics, rtol=0, atol=2e-4):
            raise ValueError(f"KITTI {camera} intrinsics differ from the frozen PNEC calibration: {path}")
        offsets.append(np.linalg.solve(projections[:, :3], projections[:, 3]))
    horizontal, vertical, depth = offsets[0] - offsets[1]
    if horizontal <= 0 or abs(vertical) > 1e-6 or abs(depth) > 1e-6 or abs(focal[0] - focal[1]) > 1e-6:
        raise ValueError(f"VISO2-S requires a rectified horizontal pair with equal focal lengths: {path}")
    return float(horizontal)


def parse_motion(path):
    lines = path.read_text().splitlines()
    if len(lines) != PAIR_COUNT:
        raise ValueError(f"Expected {PAIR_COUNT} adjacent frame results, including failed estimates")
    rows = []
    for target, line in zip(TARGETS, lines):
        index, ok, matches, inliers, *matrix = line.split()
        ok = int(ok)
        if int(index) != target - START or ok not in (0, 1) or len(matrix) != 12 * ok:
            raise ValueError(f"Invalid VISO2-S row for frame {target}")
        row = {"target_frame": target, "matches": int(matches), "inliers": int(inliers),
               "motion_current_from_previous": None}
        if ok:
            motion = np.array([float(value) for value in matrix]).reshape(3, 4)
            invert_motion(motion)
            row["motion_current_from_previous"] = motion.tolist()
        if not 0 <= row["inliers"] <= row["matches"]:
            raise ValueError(f"Invalid VISO2-S counts for frame {target}")
        rows.append(row)
    return rows


def invert_motion(motion):
    rotation = motion[:, :3]
    if (not np.isfinite(motion).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-8):
        raise ValueError("VISO2-S returned an invalid rigid transform")
    return rotation.T, -rotation.T @ motion[:, 3]


def run_sequence(sequence, args, binary, out, cv2):
    left_refs = read_archive(f"pnec_kitti_{sequence}_original_feedback_a89d06a_user_report.json")["manifest"]
    slot = args.dataset / "sequences" / sequence
    calibration_path = slot / "calib.txt"
    focal, center, _ = camera_config(sequence)
    baseline = calibration(calibration_path, focal, center)
    left_path = slot / "image_0" / f"{START:06d}.png"
    first = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Cannot read {left_path}")
    height, width = first.shape
    command = [str(binary), str(width), str(height), str(PAIR_COUNT + 1), str(focal[0]),
               str(center[0]), str(center[1]), str(baseline)]
    raw = out / f"viso2_{sequence}.txt"
    log = out / f"viso2_{sequence}.log"
    image_hashes = {}
    with raw.open("wb") as stdout, log.open("wb") as stderr:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr) as job:
            for frame in range(START, TARGETS.stop):
                name = f"{frame:06d}.png"
                for camera in ("image_0", "image_1"):
                    path = slot / camera / name
                    checksum = digest(path)
                    if camera == "image_0" and checksum != left_refs["image_sha256"][name]:
                        raise ValueError(f"Left image differs from the original PNEC run: {path}")
                    image_hashes[f"{camera}/{name}"] = checksum
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None or image.shape != first.shape:
                        raise ValueError(f"Invalid rectified grayscale image: {path}")
                    job.stdin.write(image.tobytes())
            job.stdin.close()
            if job.wait():
                raise RuntimeError(f"VISO2-S execution failed. See {log}")
    poses = parse_motion(raw)
    print(f"KITTI {sequence}: {sum(row['motion_current_from_previous'] is not None for row in poses)}/{PAIR_COUNT} "
          "estimates", flush=True)
    return {"sequence": sequence, "left_right_baseline_m": baseline, "calibration_sha256": digest(calibration_path),
            "image_sha256": image_hashes, "motion_rows": poses, "raw_motion_sha256": digest(raw)}


def capture(args):
    import cv2

    checkout = args.viso2_repo.resolve(strict=True)
    commit = subprocess.check_output(["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != VISO2_COMMIT:
        raise ValueError("Use the frozen standalone libviso2 commit")
    binary = args.binary.resolve(strict=True)
    out = Path(mkdtemp(prefix="pnec_viso2_stereo_", dir=args.output_root.resolve(strict=True)))
    print(f"Run directory: {out}", flush=True)
    sequences = [run_sequence(seq, args, binary, out, cv2) for seq in SEQUENCES]
    output = out / "pnec_viso2_stereo_09_10.json"
    atomic_json(output, {"method": "standalone libviso2 defaults", "libviso2_git_commit": commit,
                         "motion_convention": MOTION_CONVENTION, "image_start": START,
                         "target_frames": list(TARGETS), "binary_sha256": digest(binary),
                         "wrapper_sha256": digest(Path(__file__).with_name("run_viso2_stream.cc")),
                         "opencv_version": cv2.__version__, "sequences": sequences})
    print(f"Report: {output}")


def residual_stats(raw, rotation, translation, stereo_rotation, stereo_translation, focal, center, regularizer):
    rows = np.genfromtxt(BytesIO(raw), delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1)
    rows = rows[rows["ransac_inlier"] == 1]
    host = covariance(rows, "host_cov", 2)
    target = covariance(rows, "target_cov", 2)
    keep = (np.linalg.eigvalsh(host)[:, 0] > 0) & (np.linalg.eigvalsh(target)[:, 0] > 0)
    rows = rows[keep]
    left_px = np.column_stack((rows["host_x_px"], rows["host_y_px"]))
    right_px = np.column_stack((rows["target_x_px"], rows["target_y_px"]))
    left, right = bearing(left_px, focal, center), bearing(right_px, focal, center)
    host_projected = project_covariance(left_px, host[keep], focal, center, np.sqrt(3))
    target_projected = project_covariance(right_px, target[keep], focal, center, np.sqrt(3))
    direction = translation / np.linalg.norm(translation)
    stereo_direction = stereo_translation / np.linalg.norm(stereo_translation)
    e, normals, variance, _ = geometry(left, right, rotation, direction, host_projected, target_projected, regularizer)
    eligible = (normals > 1e-9) & np.isfinite(variance) & (variance > 0)
    e_stereo = np.sum((np.cross(stereo_direction, left) @ stereo_rotation) * right, axis=1)
    q_ref, q_stereo = e[eligible] ** 2 / variance[eligible], e_stereo[eligible] ** 2 / variance[eligible]
    if not len(q_ref):
        raise ValueError("No eligible measurements")
    relative_cosine = (np.trace(rotation.T @ stereo_rotation) - 1) / 2
    return {
        "scored_points": len(q_ref),
        "reference_mean_q": float(q_ref.mean()), "viso2_mean_q": float(q_stereo.mean()),
        "reference_fraction_q_above_9": float(np.mean(q_ref > 9)),
        "viso2_fraction_q_above_9": float(np.mean(q_stereo > 9)),
        "rotation_difference_deg": float(np.rad2deg(np.arccos(np.clip(relative_cosine, -1, 1)))),
        "translation_direction_difference_deg":
            float(np.rad2deg(np.arccos(np.clip(direction @ stereo_direction, -1, 1)))),
    }


def compare_sequence(stereo, sequence, bundle, gt_dir):
    fb = next(row for row in read_archive(FB_CAPTURE)["sequences"] if row["sequence"] == sequence)
    cross_fit = next(row for row in read_archive(CROSS_FIT)["sequences"] if row["sequence"] == sequence)
    if digest(bundle) != fb["bundle_sha256"]:
        raise ValueError(f"Frozen matching bundle differs for KITTI {sequence}")
    left_refs = read_archive(f"pnec_kitti_{sequence}_original_feedback_a89d06a_user_report.json")["manifest"]
    for frame in range(START, TARGETS.stop):
        name = f"{frame:06d}.png"
        if stereo["image_sha256"].get(f"image_0/{name}") != left_refs["image_sha256"][name]:
            raise ValueError(f"VISO2-S left image hash differs for KITTI {sequence}, frame {frame}")
    if [row["target_frame"] for row in stereo["motion_rows"]] != list(TARGETS):
        raise ValueError("Missing or duplicate VISO2-S pose rows")
    frozen = {row["target_frame"]: row for row in fb["frames"]}
    joint = {row["target_frame"]: row for row in cross_fit["frames"]}
    rotations, positions, gt_hash = load_truth(gt_dir, sequence)
    focal, center, regularizer = camera_config(sequence)
    frames = []
    with tarfile.open(bundle, "r:gz") as tar:
        for row in stereo["motion_rows"]:
            target = row["target_frame"]
            host = target - 1
            frame = {"target_frame": target, "viso2_matches": row["matches"], "viso2_inliers": row["inliers"]}
            frames.append(frame)
            if row["motion_current_from_previous"] is None:
                frame["status"] = "viso2_failed"
                continue
            rstereo, tstereo = invert_motion(np.array(row["motion_current_from_previous"]))
            rref = rotations[host].T @ rotations[target]
            tref = rotations[host].T @ (positions[target] - positions[host])
            if min(np.linalg.norm(tref), np.linalg.norm(tstereo)) <= 1e-9:
                frame["status"] = "undefined_translation_direction"
                continue
            raw = tar.extractfile(f"matches_{host}_{target}.csv").read()
            if sha256(raw).hexdigest() != frozen[target]["matches_sha256"]:
                raise ValueError(f"Match hash differs for {target}")
            previous = joint[target]
            if previous["status"] != "ok":
                frame["status"] = "prior_cross_fit_unavailable"
                continue
            stats = residual_stats(raw, rref, tref, rstereo, tstereo, focal, center, regularizer)
            if (stats["scored_points"] != previous["scored_points"]
                    or stats["reference_fraction_q_above_9"] != previous["reference_fraction_q_above_9"]
                    or not np.isclose(stats["reference_mean_q"], previous["reference_mean_q"], rtol=0, atol=1e-10)):
                raise ValueError(f"Prior residual calculation differs for {target}")
            frame.update(stats, status="ok",
                         cross_fit_mean_q=previous["fits"]["joint"]["heldout_corrected_mean_q"],
                         cross_fit_fraction_q_above_9=previous["fits"]["joint"]["heldout_corrected_fraction_q_above_9"])
    valid = [f for f in frames if f["status"] == "ok"]
    aggregates = {
        key: float(np.mean([f[key] for f in valid])) if valid else None
        for key in ("reference_mean_q", "viso2_mean_q", "cross_fit_mean_q",
                    "reference_fraction_q_above_9", "viso2_fraction_q_above_9", "cross_fit_fraction_q_above_9")
    }
    for key in ("rotation_difference_deg", "translation_direction_difference_deg"):
        aggregates[f"median_{key}"] = float(np.median([f[key] for f in valid])) if valid else None
    return {"sequence": sequence, "ground_truth_sha256_lf": gt_hash, "frame_pairs": len(frames),
            "comparable_frames": len(valid),
            "frames_with_lower_viso2_tail": sum(f["viso2_fraction_q_above_9"] < f["reference_fraction_q_above_9"]
                                                for f in valid),
            "frames_with_lower_viso2_mean_q": sum(f["reference_mean_q"] - f["viso2_mean_q"] > 1e-10
                                                   for f in valid),
            **aggregates, "frames": frames}


def score(args):
    capture_report = json.loads(args.run.read_bytes())
    if (capture_report["libviso2_git_commit"] != VISO2_COMMIT or capture_report["motion_convention"] != MOTION_CONVENTION
            or capture_report["image_start"] != START or capture_report["target_frames"] != list(TARGETS)
            or [row["sequence"] for row in capture_report["sequences"]] != list(SEQUENCES)):
        raise ValueError("VISO2-S run does not match the frozen design")
    entries = zip(capture_report["sequences"], SEQUENCES, (args.bundle_09, args.bundle_10))
    results = [compare_sequence(stereo, seq, bundle, args.ground_truth_dir) for stereo, seq, bundle in entries]
    atomic_json(args.output, {"scope": "Independent stereo geometry on frozen PNEC matches and reference variance",
                              "viso2_run_sha256": digest(args.run), "runner_sha256": digest(Path(__file__)),
                              "sequences": results})
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    collect = actions.add_parser("capture")
    collect.set_defaults(handler=capture)
    for name in ("dataset", "viso2-repo", "binary", "output-root"):
        collect.add_argument("--" + name, type=Path, required=True)
    evaluate = actions.add_parser("score")
    evaluate.set_defaults(handler=score)
    for name in ("run", "bundle-09", "bundle-10", "ground-truth-dir", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.handler(args)
