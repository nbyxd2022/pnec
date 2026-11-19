"""Evaluate the frozen KITTI 08 taper on the existing KITTI 09 or 10 clip protocol."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_photometric_drift import project_rotations, relative_steps
from analyze_secondary_peak_distance import BEST_MAX, FAR_MIN
from run_fixed_input_audit import atomic_json, record_pairs
from run_photometric_increment import HOLDOUT, digest, replay_config, restore_rotations
from run_secondary_peak_taper import evaluate_pairs
from run_weak_scale_audit import CONTROL_TOLERANCE_DEG, TAPER_MODES, WEAK_SCALE, checked_file


def clip_inputs(dataset, sequence):
    start, first, end = HOLDOUT
    source = dataset / "sequences" / sequence
    time_path = source / "times.txt"
    gt_path = dataset / "poses" / (sequence + ".txt")
    times = np.loadtxt(time_path, ndmin=1)
    poses = np.loadtxt(gt_path, ndmin=2)
    if (times.ndim != 1 or len(times) < end or poses.shape != (len(times), 12)
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or not np.isfinite(poses[start:end]).all()):
        raise ValueError(f"Invalid KITTI {sequence} timestamp or ground-truth coverage")
    images = source / "image_0"
    image_hashes = {}
    for frame in range(start, end):
        image = images / f"{frame:06d}.png"
        if not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"Missing or empty KITTI image: {image}")
        image_hashes[image.name] = digest(image)
    clip_times = times[start:end] - times[start]
    reference = project_rotations(poses[start:end].reshape(-1, 3, 4)[:, :, :3])
    return images, clip_times, reference, {
        "dataset": str(dataset), "times_sha256": digest(time_path), "ground_truth_sha256": digest(gt_path),
        "image_sha256": image_hashes, "replay_start": start, "evaluation_start": first, "end_exclusive": end}


def capture_index(out, targets):
    ablation = out / "capture/ablation"
    hashes = {"capture/ablation/PNEC.txt": digest(ablation / "PNEC.txt")}
    for target in targets:
        pair = f"{target - 1}_{target}"
        for name in (f"fixed_{pair}_points.csv", f"fixed_{pair}_poses.csv", f"matches_{pair}.csv"):
            path = ablation / name
            hashes[path.relative_to(out).as_posix()] = digest(path)
    return hashes


def run(repo, build, dataset, sequence, out):
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build directory belongs to another repository")
    capture_exe, solve_exe = build / "pnec_vo", build / "pnec_fixed_refine"
    images, times, reference, inputs = clip_inputs(dataset, sequence)
    config = replay_config(repo, sequence)
    start, first, end = HOLDOUT
    lo, hi = first - start, end - start - 1
    targets = range(lo + 1, hi + 1)
    manifest = {"sequence": sequence, "inputs": inputs,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "binary_sha256": {p.name: digest(p) for p in (capture_exe, solve_exe)},
                "config_sha256": {p.name: digest(p) for p in config}, "modes": list(TAPER_MODES),
                "feedback": "symmetric_analytic", "target_increment_scale": WEAK_SCALE,
                "best_within_px": BEST_MAX, "second_beyond_px": FAR_MIN,
                "control_tolerance_deg": CONTROL_TOLERANCE_DEG}
    index_path = out / "capture_index.json"
    if out.exists():
        if not index_path.is_file():
            raise ValueError(f"Existing capture is incomplete or still running: {out}. Use a fresh output directory to retry.")
        if json.loads((out / "manifest.json").read_bytes()) != manifest:
            raise ValueError(f"Inputs differ from the completed capture: {out}")
        index = json.loads(index_path.read_bytes())
        checked_file(out / "manifest.json", index["manifest_sha256"])
    else:
        out.mkdir(parents=True)
        atomic_json(out / "manifest.json", manifest)
        record_pairs(repo, capture_exe, config, images, start, times, targets, out)
        index = {"manifest_sha256": digest(out / "manifest.json"), "files": capture_index(out, targets)}
        atomic_json(index_path, index)
    hashes = index["files"]
    poses_path = "capture/ablation/PNEC.txt"
    online = restore_rotations(checked_file(out / poses_path, hashes[poses_path]), times, 0)
    metrics = evaluate_pairs(out, hashes, (lo, hi), reference, solve_exe, online_steps=relative_steps(online))
    metrics["local_frames"] = metrics.pop("frames")
    metrics["dataset_frames"] = [first, end - 1]
    report = {"sequence": sequence, "scope": "Cross-sequence fixed-input rotation with analytic baseline feedback",
              "git_commit": manifest["git_commit"], "manifest_sha256": index["manifest_sha256"],
              "capture_index_sha256": digest(index_path), "binary_sha256": manifest["binary_sha256"],
              "target_increment_scale": WEAK_SCALE, "best_within_px": BEST_MAX, "second_beyond_px": FAR_MIN,
              "window": metrics}
    atomic_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("09", "10"), required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory, or a completed capture to re-analyze")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.build.resolve(strict=True), args.dataset.resolve(strict=True),
        args.sequence, args.output.resolve())
