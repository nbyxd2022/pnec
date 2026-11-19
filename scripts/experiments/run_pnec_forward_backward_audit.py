"""Capture and score the KLT forward–backward error on fixed KITTI windows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from tempfile import mkdtemp

import numpy as np

from audit_pnec_bearing_covariance import REPO, digest_lf, load_truth, read_archive
from audit_pnec_covariance_shape import MAX_VARIANCE_RATIO, matched_groups, measurements
from run_fixed_input_audit import atomic_json, record_pairs
from run_photometric_increment import digest, pose_rows, replay_config


TARGETS = range(300, 400)
FB_FIELD = "target_fb_error_sq_px2"
FILE_BY_TARGET = {target: f"matches_{target - 1}_{target}.csv" for target in TARGETS}


def reference_report(sequence):
    return read_archive(f"pnec_kitti_{sequence}_original_shared_feedback_a52d5e6_user_report.json")


def capture(args):
    sequence = args.sequence
    source = reference_report(sequence)
    reference = Path(source["source_run"])
    report_path = reference / "report.json"
    if digest(report_path) != source["source_report_sha256"]:
        raise ValueError("Archived original replay report differs")
    baseline = json.loads(report_path.read_bytes())["manifest"]
    config = replay_config(REPO, sequence)
    config_hashes = {path.name: digest(path) for path in config}
    if config_hashes != json.loads((reference / sequence / "config_sha256.json").read_bytes()):
        raise ValueError("Current KITTI configuration differs from the archived replay")
    dataset = args.dataset.resolve(strict=True) / "sequences" / sequence
    times_file = dataset / "times.txt"
    if digest(times_file) != baseline["times_sha256"]:
        raise ValueError("KITTI timestamps differ from the archived replay")
    times = np.loadtxt(times_file, ndmin=1)
    times -= times[0]
    images = dataset / "image_0"
    for frame in range(TARGETS.stop):
        name = f"{frame:06d}.png"
        if digest(images / name) != baseline["image_sha256"][name]:
            raise ValueError(f"KITTI image differs from the archived replay: {name}")
    expected = pose_rows(reference / sequence / "original/ablation/PNEC.txt", times)[:TARGETS.stop]
    binary = args.binary.resolve(strict=True)
    out = Path(mkdtemp(prefix=f"pnec_kitti{sequence}_fb_", dir=args.output_root.resolve(strict=True)))
    record_pairs(REPO, binary, config, images, 0, times[:TARGETS.stop], TARGETS, out,
                 rotation_feedback=True, mode="original", forward_backward=True)

    diagnostics = out / "capture/ablation"
    if not np.array_equal(pose_rows(diagnostics / "PNEC.txt", times[:TARGETS.stop]), expected):
        raise ValueError("PNEC pose prefix differs from the original replay")
    files = [diagnostics / name for name in FILE_BY_TARGET.values()]
    hashes = {}
    for file in files:
        with file.open("rb") as stream:
            if stream.readline().rstrip().split(b",")[-1] != FB_FIELD.encode():
                raise ValueError(f"Missing forward–backward diagnostics in {file}")
        hashes[file.name] = digest(file)
    manifest = {
        "sequence": sequence, "feedback": "original", "target_frames": list(TARGETS),
        "original_prefix_exact": True, "binary_sha256": digest(binary),
        "source_report_sha256": source["source_report_sha256"], "config_sha256": config_hashes,
        "matches_sha256": hashes,
    }
    manifest_path = out / "manifest.json"
    atomic_json(manifest_path, manifest)
    bundle = out / f"kitti{sequence}_pnec_fb_matches.tar.gz"
    with tarfile.open(bundle, "x:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for file in files:
            tar.add(file, arcname=file.name)
    print(f"KITTI {sequence}: verified pose prefix and {len(files)} match CSVs")
    print(f"Bundle: {bundle}")


def angle_stats(values):
    return {
        "mean_fb_error_sq_px2": float(values[0]),
        "mean_predicted_angular_variance_mrad2": float(values[1]),
        "epipolar_rms_mrad": float(np.sqrt(values[2])),
        "fraction_q_above_9": float(values[3]),
    }


def legacy_matches_hash(raw):
    lines = [line.rsplit(b",", 1)[0] for line in raw.splitlines()]
    return sha256(b"\n".join(lines) + b"\n").hexdigest()


def score_bundle(bundle, sequence, dataset, focal, center, regularizer):
    config = replay_config(REPO, sequence)
    config_hashes = {path.name: digest_lf(path) for path in config}
    opts = json.loads(config[2].read_bytes())["value0"]
    gate = float(opts["config.optical_flow_max_recovered_dist2"])
    rotations, positions, gt_hash = load_truth(dataset, sequence)
    legacy = {}
    if sequence == "10":
        old = read_archive("pnec_kitti_10_simfitpp_holdout_497bab6_report.json")
        legacy = {row["target_frame"]: row["matches_sha256"] for row in old["pairs"]}

    frames, group_means = [], []
    with tarfile.open(bundle, "r:gz") as tar:
        members = tar.getmembers()
        names = set(FILE_BY_TARGET.values())
        if (len(members) != len(names) + 1 or {member.name for member in members} != names | {"manifest.json"}
                or not all(member.isfile() for member in members)):
            raise ValueError(f"Unexpected KITTI {sequence} archive contents")
        manifest = json.load(tar.extractfile("manifest.json"))
        if (manifest["sequence"] != sequence or manifest["feedback"] != "original"
                or manifest["target_frames"] != list(TARGETS) or not manifest["original_prefix_exact"]
                or manifest["source_report_sha256"] != reference_report(sequence)["source_report_sha256"]
                or manifest["config_sha256"] != config_hashes or set(manifest["matches_sha256"]) != names):
            raise ValueError(f"Unexpected KITTI {sequence} capture manifest")
        for target, name in FILE_BY_TARGET.items():
            raw = tar.extractfile(name).read()
            checksum = sha256(raw).hexdigest()
            if checksum != manifest["matches_sha256"][name]:
                raise ValueError(f"Invalid hash for {name}")
            if target in legacy and legacy_matches_hash(raw) != legacy[target]:
                raise ValueError(f"The original matching columns differ in {name}")
            host = target - 1
            rotation = rotations[host].T @ rotations[target]
            translation = rotations[host].T @ (positions[target] - positions[host])
            values, _ = measurements(raw, target, rotation, translation, focal, center, regularizer,
                                     feature_field=FB_FIELD)
            available = values[:, 0] >= 0
            if np.any(values[available, 0] >= gate):
                raise ValueError(f"Accepted match exceeds the original KLT threshold in {name}")
            n_low, n_high, paired = matched_groups(values[available])
            row = {"target_frame": target, "matches_sha256": checksum, "eligible": int(available.sum()),
                   "unavailable": int((~available).sum()), "low_available": n_low, "high_available": n_high}
            if paired is not None:
                count, means, ratios = paired
                row.update(matched_points_per_group=count, low=angle_stats(means[0]), high=angle_stats(means[1]),
                           matched_variance_ratio_min_max=[float(ratios.min()), float(ratios.max())])
                group_means.append(means)
            frames.append(row)

    if not group_means:
        raise ValueError(f"No comparable frames in KITTI {sequence}")
    means = np.stack(group_means)
    tail_delta = means[:, 1, 3] - means[:, 0, 3]
    return {
        "sequence": sequence, "binary_sha256": manifest["binary_sha256"], "bundle_sha256": digest(bundle),
        "ground_truth_sha256_lf": gt_hash, "legacy_csv_prefixes_verified": len(legacy),
        "klt_fb_gate_sq_px2": gate, "frame_pairs": len(frames), "comparable_frame_pairs": len(means),
        "matched_points_per_group": sum(row.get("matched_points_per_group", 0) for row in frames),
        "frames_with_higher_q_tail_in_high_group": int(np.sum(tail_delta > 0)),
        "median_high_minus_low_q_tail": float(np.median(tail_delta)),
        "low": angle_stats(means[:, 0].mean(axis=0)), "high": angle_stats(means[:, 1].mean(axis=0)),
        "frames": frames,
    }


def score(args):
    config = replay_config(REPO, "10")[0]
    params = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                  if line.startswith(("Camera.", "PNEC.")))
    if any(float(params["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI input")
    focal = np.array([float(params["Camera." + name]) for name in ("fx", "fy")])
    center = np.array([float(params["Camera." + name]) for name in ("cx", "cy")])
    regularizer = float(params["PNEC.regularization"])
    results = [score_bundle(bundle, seq, args.ground_truth_dir, focal, center, regularizer)
               for seq, bundle in (("09", args.bundle_09), ("10", args.bundle_10))]
    report = {
        "scope": "Fixed matches and KITTI reference motion; no pose re-estimation",
        "feature": FB_FIELD, "target_frames": list(TARGETS), "predicted_variance_max_ratio": MAX_VARIANCE_RATIO,
        "comparison": "Within-frame extreme quartiles and paired dual epipolar variance; equal frame weights",
        "runner_sha256": digest(Path(__file__)), "sequences": results,
    }
    atomic_json(args.output, report)
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    collect = actions.add_parser("capture")
    collect.set_defaults(run=capture)
    collect.add_argument("--sequence", choices=("09", "10"), required=True)
    for name in ("dataset", "binary", "output-root"):
        collect.add_argument("--" + name, type=Path, required=True)
    evaluate = actions.add_parser("score")
    evaluate.set_defaults(run=score)
    for name in ("bundle-09", "bundle-10", "ground-truth-dir", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.run(args)
