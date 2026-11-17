"""Compare direct two-frame PNEC with saved chains on 16 fixed KITTI 08 pairs."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import numpy as np

from run_photometric_increment import MODES, digest, restore_rotations


FULL_REPORT_SHA = "c5f71fceae7ec829a66fd013ce0854fec966e8fea1b770da5e69ad38188845e4"
CHAIN_SHA = {
    "original": "6f10534efb757a7d70395edc7b546d5ec144539fbef019834f82faed9a531f6a",
    "uniform": "aa4099b86b573be95fc129e6893a644b1e8b7e8b0d20e228d52f1711d3a2ac9b",
    "per_match": "601940bcbd23c4cff08f580ae9f7482aa63477e7479d7cf4ba25afe8b718e100",
}
WINDOWS = ((2600, 2800), (3400, 3600))
GAPS = (5, 10)
START_OFFSETS = (0, 50, 100, 150)
MIN_INLIERS = 10  # The frozen PNEC configuration's RANSAC sample size.


def rotation_error(reference, estimate):
    cosine = (np.sum(reference * estimate) - 1) / 2
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))


def match_counts(path, times):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        if (int(row["solver_index"]) != index or row["host_frame"] != "0" or row["target_frame"] != "1"
                or row["ransac_enabled"] != "1" or row["ransac_inlier"] not in ("0", "1")
                or not np.allclose([float(row["host_time_s"]), float(row["target_time_s"])],
                                   times, atol=1e-8, rtol=0)):
            raise ValueError(f"{path}: unexpected pair or RANSAC diagnostic")
    return len(rows), sum(row["ransac_inlier"] == "1" for row in rows)


def estimate_pair(args, config, image_dir, times, output, host, target):
    slot = output / f"{host:06d}_{target:06d}"
    images = slot / "images"
    images.mkdir(parents=True)
    for index, frame in enumerate((host, target)):
        (images / f"{index:06d}.png").symlink_to((image_dir / f"{frame:06d}.png").resolve(strict=True))
    pair_times = np.array([0.0, times[target] - times[host]])
    times_file = slot / "times.txt"
    np.savetxt(times_file, pair_times, fmt="%.12f")
    env = dict(os.environ, PNEC_PHOTOMETRIC_INCREMENT="original", PNEC_MATCH_DIAGNOSTICS_FRAMES="1",
               OPENBLAS_NUM_THREADS="1")
    with (slot / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([args.binary, *config, images, times_file, str(slot) + "/", "", "true"],
                                   cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        return None, {"status": "estimator_failed", "exit_code": completed.returncode}
    try:
        matches, inliers = match_counts(slot / "ablation/matches_0_1.csv", pair_times)
        counts = {"matches": matches, "ransac_inliers": inliers}
        if inliers < MIN_INLIERS:
            return None, {"status": "insufficient_inliers", **counts}
        matrix = restore_rotations(slot / "ablation/PNEC.txt", pair_times, 0)[1]
    except (OSError, ValueError, KeyError) as error:
        return None, {"status": "invalid_pair_output", "detail": str(error)}
    return matrix, {"status": "ok", **counts}


def prepare_inputs(args):
    report_file = args.run / "report.json"
    if digest(report_file) != FULL_REPORT_SHA:
        raise ValueError("Expected the archived KITTI 08 full-sequence run")
    manifest = json.loads(report_file.read_bytes())["manifest"]
    source = args.dataset / "sequences/08"
    gt = args.dataset / "poses/08.txt"
    if (digest(args.binary) != manifest["binary_sha256"] or digest(gt) != manifest["ground_truth_sha256"]
            or digest(source / "times.txt") != manifest["times_sha256"]):
        raise ValueError("Executable, KITTI timestamps or GT differ from the saved run")
    saved = args.run / "08"
    config = [args.repo / "data" / name for name in (
        "config_kitti04-10.yaml", "test_config.yaml", "tracking/KITTI/08.json", "tracking/KITTI/kitti_calib.json")]
    if {file.name: digest(file) for file in config} != json.loads((saved / "config_sha256.json").read_bytes()):
        raise ValueError("PNEC configuration differs from the full-sequence run")
    times = np.loadtxt(source / "times.txt", ndmin=1)
    times -= times[0]
    saved_times = np.loadtxt(saved / "times.txt", ndmin=1)
    gt_rows = np.loadtxt(gt, ndmin=2)
    count = manifest["end_exclusive"]
    if (times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or saved_times.shape != times.shape or not np.allclose(times, saved_times, rtol=0, atol=6e-13)
            or gt_rows.shape != (count, 12) or not np.isfinite(gt_rows).all()):
        raise ValueError("Expected the complete time and GT coverage of KITTI 08")
    u, _, vt = np.linalg.svd(gt_rows.reshape(-1, 3, 4)[:, :, :3])
    u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1, 1)[:, None]
    poses = {mode: saved / mode / "ablation/PNEC.txt" for mode in MODES}
    if {mode: digest(path) for mode, path in poses.items()} != CHAIN_SHA:
        raise ValueError("Saved adjacent rotations differ from the archived full-sequence drift analysis")
    chain = {mode: restore_rotations(path, saved_times, 0) for mode, path in poses.items()}
    return manifest, config, source / "image_0", times, u @ vt, chain


def run(args):
    manifest, config, images, times, reference, chain = prepare_inputs(args)
    pairs = [(window, window[0] + offset, window[0] + offset + gap)
             for window in WINDOWS for offset in START_OFFSETS for gap in GAPS]
    for frame in {frame for _, host, target in pairs for frame in (host, target)}:
        image = images / f"{frame:06d}.png"
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(image)
    output = Path(mkdtemp(prefix="pnec_kitti08_skip_edges_", dir=args.output_root))
    print(f"Run directory: {output}", flush=True)
    rows = []
    for window, host, target in pairs:
        gt = reference[host].T @ reference[target]
        direct, status = estimate_pair(args, config, images, times, output, host, target)
        chain_errors = {mode: rotation_error(gt, estimate[host].T @ estimate[target])
                        for mode, estimate in chain.items()}
        row = {"window": window, "frames": [host, target], "gap_frames": target - host,
               **status, "adjacent_chain_error_deg": chain_errors,
               "direct_error_deg": rotation_error(gt, direct) if direct is not None else None}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    summary = []
    for window in WINDOWS:
        for gap in GAPS:
            group = [row for row in rows if row["window"] == window and row["gap_frames"] == gap]
            valid = [row for row in group if row["status"] == "ok"]
            summary.append({"window": window, "gap_frames": gap, "attempted": len(group), "valid": len(valid),
                            "direct_lower_error_than_chain": {
                                mode: sum(row["direct_error_deg"] < row["adjacent_chain_error_deg"][mode]
                                          for row in valid) for mode in MODES}})
    report = {"sequence": "08", "inputs": manifest, "reference_full_report_sha256": FULL_REPORT_SHA,
              "adjacent_chains_sha256": CHAIN_SHA,
              "policy": "Fresh original PNEC estimate per left-camera pair, without intervening images. "
                        "GT is used to score both the direct estimate and each saved adjacent chain.",
              "pairs": rows, "summary": summary}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}; SHA256={digest(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("repo", "run", "dataset", "binary", "output-root"):
        parser.add_argument(f"--{option}", type=lambda value: Path(value).resolve(strict=True), required=True)
    run(parser.parse_args())
