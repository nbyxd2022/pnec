"""Replay frozen KITTI clips with original, uniform and per-match PNEC refinement."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import mkdtemp
from time import perf_counter
from uuid import uuid4

import numpy as np


MODES = ("original", "uniform", "per_match")
CAMERA = {
    "01": "config_kitti00-02.yaml",
    "08": "config_kitti04-10.yaml",
    "09": "config_kitti04-10.yaml",
    "10": "config_kitti04-10.yaml",
}
HOLDOUT = (450, 500, 800)  # replay start, evaluation start, exclusive end
SMOKE = (0, 0, 8)
TIMING_COLUMNS = ("ID FrameLoading FeatureCreation NEC-ES IT-ES AVG-IT-ES CERES "
                  "OPTIMIZATION TOTAL COVARIANCE_INCREMENT INCREMENT_INLIERS "
                  "INCREMENT_NONZERO INCREMENT_TRACE")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def pose_rows(path, times):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape != (len(times), 8) or not np.isfinite(rows).all():
        raise ValueError(f"{path}: invalid pose coverage")
    if not np.allclose(rows[:, 0], times, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: output times differ from input")
    return rows


def restore_rotations(path, times, warmup):
    rows = pose_rows(path, times)
    quaternions = rows[1 + warmup:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(norms, 1, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: invalid quaternion norm")
    quaternions = quaternions / norms[:, None]
    vector, w = quaternions[:, :3], quaternions[:, 3, None, None]
    cross = np.cross(vector[:, None, :], np.eye(3)).swapaxes(1, 2)
    relative = 2 * (vector[:, :, None] * vector[:, None, :] + w * cross) + (2 * w**2 - 1) * np.eye(3)
    absolute = np.empty((len(relative) + 1, 3, 3))
    absolute[0] = np.eye(3)
    for index, rotation in enumerate(relative, 1):
        absolute[index] = absolute[index - 1] @ rotation
    return absolute


def timing_summary(path, frames, warmup, mode):
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != TIMING_COLUMNS:
            raise ValueError(f"{path}: missing instrumented timing columns")
        rows = np.loadtxt(stream, ndmin=2)
    if (rows.shape != (frames, 13) or not np.isfinite(rows).all()
            or np.any(rows < 0) or not np.array_equal(rows[:, 0], np.arange(frames))):
        raise ValueError(f"{path}: incomplete or invalid timing rows")
    # Timing corresponds to estimates of frame transitions, whose first target is after the anchor.
    total, added, inliers, nonzero, trace = rows[warmup + 1:, 8:13].T
    if mode == "original":
        if np.any(rows[:, 9:]):
            raise ValueError(f"{path}: original mode applied an increment")
    elif (not np.any(nonzero > 0) or np.any(nonzero > inliers)
          or not np.array_equal(trace > 0, nonzero > 0)
          or (mode == "uniform" and np.any(nonzero[trace > 0] != inliers[trace > 0]))):
        raise ValueError(f"{path}: invalid activity in {mode} mode")
    return {"median_total_ms": float(np.median(total)), "p95_total_ms": float(np.percentile(total, 95)),
            "median_increment_ms": float(np.median(added)), "p95_increment_ms": float(np.percentile(added, 95)),
            "pairs_with_nonzero_increment": int(np.count_nonzero(nonzero)),
            "inliers_logged": int(inliers.sum()), "inliers_with_increment": int(nonzero.sum())}


def rotation_metrics(reference, estimate):
    errors = []
    for gap in range(1, len(reference)):
        expected = reference[:-gap].swapaxes(1, 2) @ reference[gap:]
        actual = estimate[:-gap].swapaxes(1, 2) @ estimate[gap:]
        cosine = (np.sum(expected * actual, axis=(1, 2)) - 1) / 2
        degrees = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        errors.append(float(np.sqrt(np.mean(degrees**2))))
    return {"rpe_1_deg": errors[0], "rpe_n_deg": sum(errors) / len(reference)}


def reuse_slot(target, times, warmup, mode):
    pose_rows(target / "rot_avg/poses.txt", times)
    estimate = restore_rotations(target / "ablation/PNEC.txt", times, warmup)
    summary = timing_summary(target / "timing.txt", len(times), warmup, mode)
    elapsed = None
    checkpoint = target / "wall_seconds.txt"
    if checkpoint.is_file():
        elapsed = float(checkpoint.read_text(encoding="utf-8"))
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"{checkpoint}: invalid elapsed time")
    return estimate, {"wall_seconds_including_baseline_ablation_methods": elapsed,
                      "reused_existing_output": True, **summary}


def run(args):
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=repo, check=True)
    build = args.build.resolve(strict=True)
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache.splitlines():
        raise ValueError("Build directory was configured for a different repository")
    if not args.resume:
        subprocess.run(["cmake", "--build", str(build), "--target", "pnec_vo", "--", "-j2"], check=True)
    executable = (build / "pnec_vo").resolve(strict=True)
    start, first, end = SMOKE if args.action == "smoke" else HOLDOUT
    warmup = first - start
    if args.action == "smoke":
        sequences = ("08",)
    else:
        sequences = (args.sequence,) if args.sequence else ("01", "08")
    manifest = {"git_commit": revision, "binary_sha256": digest(executable), "sequences": list(sequences),
                "action": args.action, "replay_start": start, "evaluation_start": first, "end_exclusive": end,
                "modes": list(MODES), "regularizer_and_init": "Nominal PNEC through weighted eigensolver",
                "ransac_sampling": "Fixed OpenGV seed and srand(1) before each RANSAC"}
    if args.resume:
        out = args.resume.resolve(strict=True)
        saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        original_revision = saved["git_commit"]
        runner = Path(__file__).resolve().relative_to(repo).as_posix()
        subprocess.run(["git", "diff", "--quiet", original_revision, revision, "--",
                        ".", f":(exclude){runner}"], cwd=repo, check=True)
        manifest["git_commit"] = original_revision
        if saved != manifest:
            raise ValueError("Resume requires the original binary and run manifest")
    else:
        out = Path(mkdtemp(prefix=f"pnec_increment_{args.action}_", dir=args.output_root.resolve(strict=True)))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {out}", flush=True)
    data = repo / "data"
    scores = {}
    for sequence in sequences:
        slot = out / sequence
        images = slot / "images"
        images.mkdir(parents=True, exist_ok=bool(args.resume))
        source = args.dataset.resolve(strict=True) / "sequences" / sequence
        times = np.loadtxt(source / "times.txt", ndmin=1)[start:end]
        if len(times) != end - start or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"KITTI {sequence}: clip times missing or invalid")
        times -= times[0]
        time_file = slot / "times.txt"
        if time_file.exists():
            saved_times = np.loadtxt(time_file, ndmin=1)
            if saved_times.shape != times.shape or not np.allclose(saved_times, times, rtol=0, atol=6e-13):
                raise ValueError(f"{time_file}: resume timestamps differ")
        else:
            np.savetxt(time_file, times, fmt="%.12f")
        expected_images = set()
        for index in range(start, end):
            name = f"{index:06d}.png"
            expected_images.add(name)
            destination = images / name
            original = (source / "image_0" / name).resolve(strict=True)
            if destination.is_symlink() or destination.exists():
                if not destination.is_symlink() or destination.resolve() != original:
                    raise ValueError(f"{destination}: resume image target differs")
            else:
                destination.symlink_to(original)
        if {path.name for path in images.glob("*.png")} != expected_images:
            raise ValueError(f"{images}: unexpected images in replay directory")
        config = [data / CAMERA[sequence], data / "test_config.yaml",
                  data / "tracking/KITTI" / f"{sequence}.json", data / "tracking/KITTI/kitti_calib.json"]
        config_hashes = {path.name: digest(path) for path in config}
        config_file = slot / "config_sha256.json"
        if config_file.exists():
            if json.loads(config_file.read_text(encoding="utf-8")) != config_hashes:
                raise ValueError(f"{config_file}: resume configurations differ")
        else:
            config_file.write_text(json.dumps(config_hashes, indent=2) + "\n", encoding="utf-8")
        estimates = {}
        scores[sequence] = {}
        runs = (*MODES, "original_repeat") if args.action == "smoke" else MODES
        for label in runs:
            mode = "original" if label == "original_repeat" else label
            target = slot / label
            if args.resume and target.exists():
                try:
                    estimates[label], scores[sequence][label] = reuse_slot(target, times, warmup, mode)
                except (FileNotFoundError, ValueError) as error:
                    preserved = slot / f"{label}.interrupted-{uuid4().hex[:8]}"
                    target.rename(preserved)
                    print(f"KITTI {sequence} {label}: preserved {preserved.name} ({error})", flush=True)
                else:
                    print(f"KITTI {sequence} {label}: reused {len(times)} validated frames", flush=True)
                    continue
            target.mkdir()
            env = dict(os.environ, PNEC_PHOTOMETRIC_INCREMENT=mode, OPENBLAS_NUM_THREADS="1")
            env.pop("PNEC_MATCH_DIAGNOSTICS_FRAMES", None)
            command = [executable, *config, images, time_file, str(target) + "/", "", "true"]
            print(f"KITTI {sequence} {label} running", flush=True)
            tic = perf_counter()
            with (target / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = perf_counter() - tic
            estimates[label] = restore_rotations(target / "ablation/PNEC.txt", times, warmup)
            pose_rows(target / "rot_avg/poses.txt", times)
            scores[sequence][label] = {
                "wall_seconds_including_baseline_ablation_methods": elapsed,
                **timing_summary(target / "timing.txt", len(times), warmup, mode)}
            checkpoint = target / "wall_seconds.tmp"
            checkpoint.write_text(f"{elapsed}\n", encoding="utf-8")
            checkpoint.replace(target / "wall_seconds.txt")
        if args.action == "smoke":
            original = slot / "original/ablation/PNEC.txt"
            repeat = slot / "original_repeat/ablation/PNEC.txt"
            if original.read_bytes() != repeat.read_bytes():
                raise ValueError(f"KITTI {sequence}: repeated original poses differ")
        else:
            gt_path = args.dataset / "poses" / f"{sequence}.txt"
            gt_rows = np.loadtxt(gt_path, ndmin=2)[first:end]
            if gt_rows.shape != (end - first, 12) or not np.isfinite(gt_rows).all():
                raise ValueError(f"KITTI {sequence}: incomplete GT coverage")
            u, _, vt = np.linalg.svd(gt_rows.reshape(-1, 3, 4)[:, :, :3])
            u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1, 1)[:, None]
            for mode in MODES:
                scores[sequence][mode].update(rotation_metrics(u @ vt, estimates[mode]))

    report = {"manifest": manifest, "runner_git_commit": revision, "results": scores}
    report_tmp = out / "report.json.tmp"
    report_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_tmp.replace(out / "report.json")
    print(json.dumps(scores, indent=2))
    print(f"Report: {out / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "holdout"))
    parser.add_argument("--sequence", choices=tuple(CAMERA),
                        help="Run a single sequence in holdout mode")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output-root", type=Path)
    location.add_argument("--resume", type=Path, help="Reuse complete outputs in an existing run directory")
    args = parser.parse_args()
    if args.sequence and args.action != "holdout":
        parser.error("--sequence requires holdout mode")
    run(args)


if __name__ == "__main__":
    main()
