"""Capture and score the frozen SIMFIT++ comparison on KITTI 10."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tarfile
from tempfile import mkdtemp

import numpy as np

from analyze_photometric_drift import project_rotations
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, pose_rows, replay_config


REPO = Path(__file__).resolve().parents[2]
SEQUENCE = "10"
TARGETS = range(300, 400)
FILES = [f"matches_{target - 1}_{target}.csv" for target in TARGETS]
TIGHT_THRESHOLD = 0.30
SIMFITPP_COMMIT = "497bab640b084eed25e228f17ed5c60065349def"


def archive(name):
    return json.loads((REPO / "docs/experiments/results" / name).read_bytes())


def baseline_manifest():
    return archive("pnec_kitti_10_original_feedback_a89d06a_user_report.json")["manifest"]


def capture(args):
    baseline = baseline_manifest()
    source = archive("pnec_kitti_10_original_shared_feedback_a52d5e6_user_report.json")
    reference = Path(source["source_run"])
    if digest(reference / "report.json") != source["source_report_sha256"]:
        raise ValueError("The archived original-feedback reference report differs")
    baseline_path = reference / SEQUENCE / "original/ablation/PNEC.txt"
    config = replay_config(REPO, SEQUENCE)
    config_hashes = {path.name: digest(path) for path in config}
    if config_hashes != json.loads((reference / SEQUENCE / "config_sha256.json").read_bytes()):
        raise ValueError("Current KITTI config differs from the original-feedback run")
    dataset = args.dataset.resolve(strict=True) / "sequences" / SEQUENCE
    times_path = dataset / "times.txt"
    if digest(times_path) != baseline["times_sha256"]:
        raise ValueError("KITTI timestamps differ from the archived run")
    full_times = np.loadtxt(times_path, ndmin=1)
    full_times -= full_times[0]
    if full_times.shape != (baseline["end_exclusive"],):
        raise ValueError("Invalid KITTI time coverage")
    expected_prefix = pose_rows(baseline_path, full_times)[: TARGETS.stop]
    binary = args.binary.resolve(strict=True)

    out = Path(mkdtemp(prefix="pnec_kitti10_simfitpp_holdout_", dir=args.output_root.resolve(strict=True)))
    print(f"Run directory: {out}", flush=True)
    images = out / "images"
    images.mkdir()
    for frame in range(TARGETS.stop):
        name = f"{frame:06d}.png"
        file = (dataset / "image_0" / name).resolve(strict=True)
        if digest(file) != baseline["image_sha256"][name]:
            raise ValueError(f"KITTI image differs from the archived replay: {name}")
        (images / name).symlink_to(file)
    times_file = out / "times.txt"
    np.savetxt(times_file, full_times[: TARGETS.stop], fmt="%.12f")
    results = out / "capture"
    results.mkdir()
    env = dict(os.environ, PNEC_PHOTOMETRIC_INCREMENT="original", PNEC_PREVIOUS_ROTATION_BY_VIEW="1",
               PNEC_MATCH_DIAGNOSTICS_FRAMES=",".join(map(str, TARGETS)), OPENBLAS_NUM_THREADS="1")
    for name in ("PNEC_FIXED_INPUT_AUDIT", "PNEC_SHARED_WEAK_OUTPUTS"):
        env.pop(name, None)
    command = [str(binary), *map(str, config), str(images), str(times_file), str(results) + "/", "", "true"]
    with (out / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    diagnostics = results / "ablation"
    if not np.array_equal(pose_rows(diagnostics / "PNEC.txt", full_times[: TARGETS.stop]), expected_prefix):
        raise ValueError("Original PNEC poses differ from the archived replay prefix")
    # Missing CSVs count against coverage and remain visible in the score.
    hashes = {name: digest(diagnostics / name) for name in FILES if (diagnostics / name).is_file()}
    if not hashes:
        raise ValueError(f"No consecutive match CSVs found in {diagnostics}")
    manifest = {
        "sequence": SEQUENCE, "feedback": "original", "target_frames": list(TARGETS),
        "original_prefix_exact": True, "binary_sha256": digest(binary),
        "source_report_sha256": source["source_report_sha256"],
        "config_sha256": config_hashes, "matches_sha256": hashes,
    }
    manifest_path = out / "manifest.json"
    atomic_json(manifest_path, manifest)
    bundle = out / "kitti10_simfitpp_matches.tar.gz"
    with tarfile.open(bundle, "x:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for name in hashes:
            tar.add(diagnostics / name, arcname=name)
    print(f"Verified prefix, captured {len(hashes)}/{len(FILES)} consecutive pairs", flush=True)
    print(f"Bundle: {bundle}", flush=True)


def summarize(pairs, seeds):
    summary = {}
    for seed_index, seed in enumerate(seeds):
        trials = [pair["trials"][seed_index] for pair in pairs]
        modes = {}
        for mode in ("fixed", "tight_fixed", "adaptive"):
            errors = [trial[mode]["rotation_error_deg"] for trial in trials]
            valid = [error for error in errors if error is not None]
            modes[mode] = {"valid": len(valid), "rms_deg": None, "p90_deg": None}
            if valid:
                modes[mode].update(rms_deg=float(np.sqrt(np.mean(np.square(valid)))),
                                   p90_deg=float(np.percentile(valid, 90)))
        summary[str(seed)] = {"methods": modes, "adapted_threshold_successes":
                              sum(trial["threshold_estimation_succeeded"] for trial in trials)}
    return summary


def score(args):
    from compare_kitti09_simfitpp import SEEDS, START_THRESHOLD, calib_matrix_to_camera_dict, evaluate_pair

    checkout = args.simfitpp_repo.resolve(strict=True)
    commit = subprocess.check_output(["git", "-c", f"safe.directory={checkout.as_posix()}",
                                      "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != SIMFITPP_COMMIT or version("poselib") != "2.0.4":
        raise ValueError("Use the same SIMFIT++ checkout and PoseLib 2.0.4 as the KITTI 09 pilot")
    baseline = baseline_manifest()
    bundle = args.capture.resolve(strict=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    source = archive("pnec_kitti_10_original_shared_feedback_a52d5e6_user_report.json")
    if (manifest["sequence"] != SEQUENCE or manifest["feedback"] != "original"
            or manifest["target_frames"] != list(TARGETS) or not manifest["original_prefix_exact"]
            or manifest["source_report_sha256"] != source["source_report_sha256"]):
        raise ValueError("Capture does not match the frozen KITTI 10 protocol")
    gt_bytes = args.ground_truth.read_bytes().replace(b"\r\n", b"\n")
    gt_hash = sha256(gt_bytes).hexdigest()
    if gt_hash != baseline["ground_truth_sha256"]:
        raise ValueError("Ground truth differs from the archived KITTI 10 baseline")
    gt = project_rotations(np.loadtxt(gt_bytes.splitlines()).reshape(-1, 3, 4)[:, :, :3])
    config = replay_config(REPO, SEQUENCE)[0]
    config_bytes = config.read_bytes()
    expected_hash = manifest["config_sha256"][config.name]
    canonical_bytes = config_bytes.replace(b"\r\n", b"\n")
    if expected_hash not in (sha256(config_bytes).hexdigest(), sha256(canonical_bytes).hexdigest()):
        raise ValueError("Current camera config differs from the capture")
    params = dict(line.split(":", 1) for line in config_bytes.decode("utf-8").splitlines()
                  if line.startswith("Camera."))
    if any(float(params["Camera." + name]) for name in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified KITTI input")
    fx, fy, cx, cy = (float(params["Camera." + name]) for name in ("fx", "fy", "cx", "cy"))
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera = calib_matrix_to_camera_dict(intrinsics)
    pairs, missing, insufficient = [], [], []
    hashes = manifest["matches_sha256"]
    for target, name in zip(TARGETS, FILES):
        if name not in hashes:
            missing.append(target)
            continue
        file = bundle / name
        if digest(file) != hashes[name]:
            raise ValueError(f"Match file differs from the capture: {file}")
        with file.open(encoding="utf-8") as stream:
            if sum(1 for line in stream if line.strip()) < 11:
                insufficient.append(target)
                continue
        pairs.append(evaluate_pair(file, target, gt, intrinsics, camera, TIGHT_THRESHOLD))
    summary = summarize(pairs, SEEDS)
    report = {
        "sequence": SEQUENCE, "target_frames": list(TARGETS),
        "ground_truth_sha256_lf": gt_hash, "capture_manifest_sha256": digest(manifest_path),
        "simfitpp_commit": commit,
        "dependencies": {name: version(name) for name in ("simfitpp", "poselib", "numpy", "scipy")},
        "runner_sha256": digest(Path(__file__)),
        "fixed_threshold_px": START_THRESHOLD, "tight_fixed_threshold_px": TIGHT_THRESHOLD,
        "missing_target_frames": missing, "insufficient_match_frames": insufficient,
        "summary_by_seed": summary, "pairs": pairs,
    }
    atomic_json(args.output, report)
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    collect = subparsers.add_parser("capture", help="Run frames 0–399 and bundle the 100 target pairs")
    collect.set_defaults(run=capture)
    for name in ("dataset", "binary", "output-root"):
        collect.add_argument("--" + name, type=Path, required=True)
    evaluate = subparsers.add_parser("score", help="Compare three threshold settings on the saved matches")
    evaluate.set_defaults(run=score)
    for name in ("capture", "simfitpp-repo", "ground-truth", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.run(args)
