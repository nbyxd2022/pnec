"""Replay frozen KITTI clips for the photometric increment or symmetric analytic comparison."""

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
ANALYTIC_COMPARISON_MODES = ("original", "per_match", "symmetric_analytic")
SYMMETRIC_MODES = ("symmetric_analytic", "symmetric_uniform", "symmetric_per_match")
TAPER_FEEDBACK_MODES = (SYMMETRIC_MODES[0], "weak_per_match", "matched_budget", "far_peak_taper")
WEAK_ALLOCATION_MODES = (SYMMETRIC_MODES[0], "weak_uniform", "weak_per_match")
ROTATION_FEEDBACK_LOG = "Previous-view PNEC rotation feedback active after graph pruning"
SYMMETRIC_CONTROL = {
    "source": "CVPR 2023, Section 3.2, equations 2-3",
    "scope": "Host and target KLT bearing covariances in final Ceres refinement only",
}
CAMERA = {
    "01": "config_kitti00-02.yaml",
    "08": "config_kitti04-10.yaml",
    "09": "config_kitti04-10.yaml",
    "10": "config_kitti04-10.yaml",
}
HOLDOUT = (450, 500, 800)  # replay start, evaluation start, exclusive end
SMOKE = (0, 0, 8)
FULL_TAIL_START = HOLDOUT[2]
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
    if mode in ("original", "symmetric_analytic"):
        if np.any(rows[:, 9:]):
            raise ValueError(f"{path}: {mode} applied a photometric increment")
    elif (not np.any(nonzero > 0) or np.any(nonzero > inliers)
          or not np.array_equal(trace > 0, nonzero > 0)
          or (mode in ("uniform", "symmetric_uniform", "weak_uniform")
              and np.any(nonzero[trace > 0] != inliers[trace > 0]))):
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


def validate_symmetric_activity(target, mode, *, feedback=False):
    if mode not in (*SYMMETRIC_MODES, *TAPER_FEEDBACK_MODES, *WEAK_ALLOCATION_MODES):
        return
    symmetric_logged = feedback_logged = False
    with (target / "run.log").open(encoding="utf-8", errors="replace") as log:
        for line in log:
            symmetric_logged |= "Symmetric analytic Ceres refinement ran" in line
            feedback_logged |= ROTATION_FEEDBACK_LOG in line
            if symmetric_logged and (not feedback or feedback_logged):
                return
    if not symmetric_logged:
        raise ValueError(f"{target}: symmetric analytic refinement was not logged")
    raise ValueError(f"{target}: no previous rotation feedback logged after pruning")


def reuse_slot(target, times, warmup, mode, *, feedback=False):
    validate_symmetric_activity(target, mode, feedback=feedback)
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


def full_sequence_inputs(dataset, sequence):
    source = dataset / "sequences" / sequence
    times_path = source / "times.txt"
    gt_path = dataset / "poses" / f"{sequence}.txt"
    times = np.loadtxt(times_path, ndmin=1)
    gt = np.loadtxt(gt_path, ndmin=2)
    if (times.ndim != 1 or len(times) < FULL_TAIL_START + 2
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or gt.shape != (len(times), 12) or not np.isfinite(gt).all()):
        raise ValueError(f"KITTI {sequence}: invalid full-sequence timestamps, GT or tail coverage")
    for frame in range(len(times)):
        image = source / "image_0" / f"{frame:06d}.png"
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty KITTI image: {image}")
    return len(times), gt, {"times_sha256": digest(times_path), "ground_truth_sha256": digest(gt_path)}


def replay_config(repo, sequence):
    data = repo / "data"
    return [data / CAMERA[sequence], data / "test_config.yaml",
            data / "tracking/KITTI" / f"{sequence}.json", data / "tracking/KITTI/kitti_calib.json"]


def load_full_reference(path, manifest, config, reference_modes=MODES):
    sequence, = manifest["sequences"]
    run = path.resolve(strict=True)
    raw_report = (run / "report.json").read_bytes()
    report = json.loads(raw_report)
    recorded = report["manifest"]
    expected = dict(manifest, git_commit=recorded["git_commit"],
                    binary_sha256=recorded["binary_sha256"], modes=list(reference_modes))
    if reference_modes == ("symmetric_analytic",):
        expected["symmetric_analytic_control"] = SYMMETRIC_CONTROL
        if "reference_full_report_sha256" in recorded:
            expected["reference_full_report_sha256"] = recorded["reference_full_report_sha256"]
    if recorded != expected or recorded != json.loads((run / "manifest.json").read_bytes()):
        raise ValueError("Archived run does not match the current full-sequence protocol")
    config_hashes = {file.name: digest(file) for file in config}
    if config_hashes != json.loads((run / sequence / "config_sha256.json").read_bytes()):
        raise ValueError("Archived run used different PNEC configuration files")
    scores = report["results"][sequence]
    if set(scores) != set(reference_modes):
        raise ValueError("Archived run must contain the expected reference modes")
    for mode in reference_modes:
        for metrics in (scores[mode], scores[mode]["tail"]):
            values = [metrics[key] for key in ("rpe_1_deg", "rpe_n_deg", "p95_total_ms")]
            if not np.isfinite(values).all() or any(value < 0 for value in values):
                raise ValueError(f"Invalid archived metric for {mode}")
    result = {"run": str(run), "report_sha256": sha256(raw_report).hexdigest(),
              "git_commit": recorded["git_commit"], "binary_sha256": recorded["binary_sha256"],
              "runner_git_commit": report["runner_git_commit"], "scores": scores}
    if reference_modes == ("symmetric_analytic",):
        times = np.loadtxt(run / sequence / "times.txt", ndmin=1)
        if (times.shape != (manifest["end_exclusive"],) or not np.isfinite(times).all()
                or not np.all(np.diff(times) > 0)):
            raise ValueError("Archived symmetric run has incomplete timestamp coverage")
        target = run / sequence / "symmetric_analytic"
        restore_rotations(target / "ablation/PNEC.txt", times, 0)
        pose_rows(target / "rot_avg/poses.txt", times)
        result["pose_sha256"] = {
            name: digest(target / name) for name in ("ablation/PNEC.txt", "rot_avg/poses.txt")}
    return result


def load_speed_reference(path, manifest, config):
    run = path.resolve(strict=True)
    raw = (run / "report.json").read_bytes()
    report = json.loads(raw)
    recorded = report["manifest"]
    fields = ("sequences", "action", "replay_start", "evaluation_start",
              "end_exclusive", "regularizer_and_init", "ransac_sampling")
    if (recorded != json.loads((run / "manifest.json").read_bytes())
            or any(recorded[key] != manifest[key] for key in fields)
            or "per_match" not in recorded["modes"]):
        raise ValueError("Speed reference does not match the replay protocol")
    sequence, = manifest["sequences"]
    if ({file.name: digest(file) for file in config}
            != json.loads((run / sequence / "config_sha256.json").read_bytes())):
        raise ValueError("Speed reference used different PNEC configuration files")
    times = np.loadtxt(run / sequence / "times.txt", ndmin=1)
    reuse_slot(run / sequence / "per_match", times,
               manifest["evaluation_start"] - manifest["replay_start"], "per_match")
    files = ("ablation/PNEC.txt", "rot_avg/poses.txt")
    return {"run": str(run), "report_sha256": sha256(raw).hexdigest(),
            "git_commit": recorded["git_commit"], "binary_sha256": recorded["binary_sha256"],
            "pose_sha256": {file: digest(run / sequence / "per_match" / file) for file in files},
            "scores": report["results"][sequence]["per_match"]}


def run(args):
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=repo, check=True)
    build = args.build.resolve(strict=True)
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={repo.as_posix()}" not in cache.splitlines():
        raise ValueError("Build directory was configured for a different repository")
    start, first, end = SMOKE if args.action == "smoke" else HOLDOUT
    full_gt = None
    if args.action == "smoke":
        sequences = ("08",)
    else:
        sequences = (args.sequence,) if args.sequence else ("01", "08")
    if args.action == "full":
        end, full_gt, input_hashes = full_sequence_inputs(args.dataset.resolve(strict=True), args.sequence)
        start = first = 0
    warmup = first - start
    feedback = args.compare_taper_feedback or args.compare_weak_allocation
    modes = MODES
    if args.compare_taper_feedback:
        modes = TAPER_FEEDBACK_MODES
    elif args.compare_weak_allocation:
        modes = WEAK_ALLOCATION_MODES
    elif args.reference_symmetric_full_run:
        modes = SYMMETRIC_MODES[1:]
    elif args.compare_symmetric_increments:
        modes = SYMMETRIC_MODES
    elif args.reference_speed_run:
        modes = ("per_match",)
    elif args.reference_full_run:
        modes = ("symmetric_analytic",)
    elif args.compare_symmetric_analytic:
        modes = ANALYTIC_COMPARISON_MODES

    if not args.resume:
        subprocess.run(["cmake", "--build", str(build), "--target", "pnec_vo", "--", "-j2"], check=True)
    executable = (build / "pnec_vo").resolve(strict=True)
    manifest = {"git_commit": revision, "binary_sha256": digest(executable), "sequences": list(sequences),
                "action": args.action, "replay_start": start, "evaluation_start": first, "end_exclusive": end,
                "modes": list(modes), "regularizer_and_init": "Nominal PNEC through weighted eigensolver",
                "ransac_sampling": "Fixed OpenGV seed and srand(1) before each RANSAC"}
    if args.action == "full":
        manifest.update(tail_evaluation_start=FULL_TAIL_START, **input_hashes)
    full_reference = None
    if args.reference_full_run:
        full_reference = load_full_reference(args.reference_full_run, manifest,
                                             replay_config(repo, args.sequence))
        manifest["reference_full_report_sha256"] = full_reference["report_sha256"]
    symmetric_full_reference = None
    if args.reference_symmetric_full_run:
        symmetric_full_reference = load_full_reference(
            args.reference_symmetric_full_run, manifest, replay_config(repo, args.sequence),
            reference_modes=("symmetric_analytic",))
        manifest["reference_symmetric_full_report_sha256"] = symmetric_full_reference["report_sha256"]
    speed_reference = None
    if args.reference_speed_run:
        speed_reference = load_speed_reference(args.reference_speed_run, manifest,
                                              replay_config(repo, args.sequence))
        manifest["speed_reference_report_sha256"] = speed_reference["report_sha256"]
    if any(mode in SYMMETRIC_MODES for mode in modes):
        manifest["symmetric_analytic_control"] = SYMMETRIC_CONTROL
    if feedback:
        dataset = args.dataset.resolve(strict=True)
        sequence_dir = dataset / "sequences" / args.sequence
        manifest["times_sha256"] = digest(sequence_dir / "times.txt")
        manifest["ground_truth_sha256"] = digest(dataset / "poses" / f"{args.sequence}.txt")
        manifest["image_sha256"] = {
            path.name: digest(path) for path in
            (sequence_dir / "image_0" / f"{frame:06d}.png" for frame in range(start, end))
        }
        feedback_protocol = {
            "previous_rotation": "predecessor edge in the currently retained view graph",
            "feedback": "each mode feeds its own previous relative rotation into the next solve",
            "target_increment_scale": 1 / 32,
        }
        if args.compare_taper_feedback:
            manifest["taper_feedback"] = {
                **feedback_protocol, "best_within_px": 2, "second_beyond_px": 12,
                "taper": "(12 / second_to_endpoint_px)^2 for eligible matches",
                "matched_budget": "same-pair taper total computed on this mode's own inliers",
            }
        else:
            manifest["weak_allocation_feedback"] = {
                **feedback_protocol,
                "weak_per_match": "scaled photometric trace for each inlier",
                "weak_uniform": "sum of scaled traces divided by this mode's inlier count",
            }
    if args.compare_symmetric_increments or args.reference_symmetric_full_run:
        manifest["symmetric_increment_ablation"] = {
            "base": "Host and target KLT covariances in final Ceres refinement",
            "increment": "Target-side photometric trace, uniform or per-match",
            "search": "15x15 normalized raw SSD, +/-16px, minima separation >=4px",
            "trace": "ratio * squared tangent displacement / (4 * squared ray norm)",
        }
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
        time_reference = speed_reference or symmetric_full_reference
        if time_reference is not None:
            saved_times = np.loadtxt(Path(time_reference["run"]) / sequence / "times.txt", ndmin=1)
            if saved_times.shape != times.shape or not np.allclose(saved_times, times, rtol=0, atol=6e-13):
                raise ValueError("Reference timestamps differ from the dataset")
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
        config = replay_config(repo, sequence)
        config_hashes = {path.name: digest(path) for path in config}
        config_file = slot / "config_sha256.json"
        if config_file.exists():
            if json.loads(config_file.read_text(encoding="utf-8")) != config_hashes:
                raise ValueError(f"{config_file}: resume configurations differ")
        else:
            config_file.write_text(json.dumps(config_hashes, indent=2) + "\n", encoding="utf-8")
        estimates = {}
        scores[sequence] = {}
        repeat = modes[0] + "_repeat"
        runs = (*modes, repeat) if args.action == "smoke" else modes
        for label in runs:
            mode = modes[0] if label == repeat else label
            target = slot / label
            if args.resume and target.exists():
                try:
                    estimates[label], scores[sequence][label] = reuse_slot(
                        target, times, warmup, mode, feedback=feedback)
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
            env.pop("PNEC_FIXED_INPUT_AUDIT", None)
            env.pop("PNEC_PREVIOUS_ROTATION_BY_VIEW", None)
            if feedback:
                env["PNEC_PREVIOUS_ROTATION_BY_VIEW"] = "1"
            command = [executable, *config, images, time_file, str(target) + "/", "", "true"]
            print(f"KITTI {sequence} {label} running", flush=True)
            tic = perf_counter()
            with (target / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = perf_counter() - tic
            validate_symmetric_activity(target, mode, feedback=feedback)
            estimates[label] = restore_rotations(target / "ablation/PNEC.txt", times, warmup)
            pose_rows(target / "rot_avg/poses.txt", times)
            scores[sequence][label] = {
                "wall_seconds_including_baseline_ablation_methods": elapsed,
                **timing_summary(target / "timing.txt", len(times), warmup, mode)}
            checkpoint = target / "wall_seconds.tmp"
            checkpoint.write_text(f"{elapsed}\n", encoding="utf-8")
            checkpoint.replace(target / "wall_seconds.txt")
        if args.action == "smoke":
            original = slot / modes[0] / "ablation/PNEC.txt"
            repeated = slot / repeat / "ablation/PNEC.txt"
            if original.read_bytes() != repeated.read_bytes():
                raise ValueError(f"KITTI {sequence}: repeated baseline poses differ")
        else:
            gt_path = args.dataset / "poses" / f"{sequence}.txt"
            gt_rows = full_gt if full_gt is not None else np.loadtxt(gt_path, ndmin=2)[first:end]
            if gt_rows.shape != (end - first, 12) or not np.isfinite(gt_rows).all():
                raise ValueError(f"KITTI {sequence}: incomplete GT coverage")
            u, _, vt = np.linalg.svd(gt_rows.reshape(-1, 3, 4)[:, :, :3])
            u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1, 1)[:, None]
            reference = u @ vt
            for mode in modes:
                estimate = estimates[mode]
                scores[sequence][mode].update(rotation_metrics(reference, estimate))
                if args.action == "full":
                    # Frame 800 is the tail anchor. Reuse the continuous trajectory and timings.
                    tail = FULL_TAIL_START
                    scores[sequence][mode]["tail"] = {
                        **rotation_metrics(reference[tail:], estimate[tail:]),
                        **timing_summary(slot / mode / "timing.txt", len(times), tail, mode),
                    }

    report = {"manifest": manifest, "runner_git_commit": revision, "results": scores}
    if feedback:
        sequence_scores = scores[args.sequence]
        candidate = "far_peak_taper" if args.compare_taper_feedback else "weak_per_match"
        report[f"{candidate}_minus_control_deg"] = {
            mode: {metric: sequence_scores[candidate][metric] - sequence_scores[mode][metric]
                   for metric in ("rpe_1_deg", "rpe_n_deg")}
            for mode in modes if mode != candidate
        }
    if full_reference is not None:
        report["reference_full_run"] = full_reference
    if symmetric_full_reference is not None:
        report["reference_symmetric_full_run"] = symmetric_full_reference
    if speed_reference is not None:
        sequence = args.sequence
        current = scores[sequence]["per_match"]
        poses = {file: digest(out / sequence / "per_match" / file)
                 for file in speed_reference["pose_sha256"]}
        check = {
            "reference": speed_reference,
            "exact_pose_match": poses == speed_reference["pose_sha256"],
            "current_pose_sha256": poses,
            "accuracy_equal": all(current[key] == speed_reference["scores"][key]
                                  for key in ("rpe_1_deg", "rpe_n_deg")),
            "increment_ms": {
                "reference_median": speed_reference["scores"]["median_increment_ms"],
                "current_median": current["median_increment_ms"],
                "reference_p95": speed_reference["scores"]["p95_increment_ms"],
                "current_p95": current["p95_increment_ms"],
            },
        }
        report["photometric_speed_check"] = check
    report_tmp = out / "report.json.tmp"
    report_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_tmp.replace(out / "report.json")
    print(json.dumps(scores, indent=2))
    print(f"Report: {out / 'report.json'}")
    if speed_reference is not None:
        print(json.dumps(check, indent=2))
        if not check["exact_pose_match"] or not check["accuracy_equal"]:
            raise ValueError("Photometric implementation changed the saved poses; inspect the speed report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "holdout", "full"))
    parser.add_argument("--sequence", choices=tuple(CAMERA),
                        help="Choose one sequence for holdout or full evaluation")
    analytic = parser.add_mutually_exclusive_group()
    analytic.add_argument("--compare-symmetric-analytic", action="store_true",
                          help="Compare original, per-match, and double-sided analytic PNEC on the KITTI 08 holdout")
    analytic.add_argument("--reference-full-run", type=Path,
                          help="Replay only double-sided analytic PNEC and report the archived full runs for the selected KITTI sequence alongside it")
    analytic.add_argument("--reference-speed-run", type=Path,
                          help="Check optimized per-match computation against a recorded KITTI 08 holdout")
    analytic.add_argument("--compare-symmetric-increments", action="store_true",
                          help="Compare double-sided PNEC with uniform and per-match photometric increments")
    analytic.add_argument("--reference-symmetric-full-run", type=Path,
                          help="Replay only symmetric uniform/per-match increments and reuse a recorded KITTI 08 full symmetric baseline")
    analytic.add_argument("--compare-taper-feedback", action="store_true",
                          help="Run the four taper comparison modes with previous rotation feedback on KITTI 09 or 10")
    analytic.add_argument("--compare-weak-allocation", action="store_true",
                          help="Compare weak uniform and per-match increments with analytic PNEC on KITTI 09 or 10")
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output-root", type=Path)
    location.add_argument("--resume", type=Path, help="Reuse complete outputs in an existing run directory")
    args = parser.parse_args()
    if args.sequence and args.action == "smoke":
        parser.error("--sequence is only used by holdout and full")
    if args.action == "full" and not args.sequence:
        parser.error("full requires --sequence")
    if args.compare_symmetric_analytic and (args.action != "holdout" or args.sequence != "08"):
        parser.error("--compare-symmetric-analytic requires holdout --sequence 08")
    if args.reference_full_run and args.action != "full":
        parser.error("--reference-full-run requires full --sequence")
    if args.reference_speed_run and (args.action != "holdout" or args.sequence != "08"):
        parser.error("--reference-speed-run requires holdout --sequence 08")
    if args.compare_symmetric_increments and (
            args.action == "full" or (args.action == "holdout" and args.sequence != "08")):
        parser.error("--compare-symmetric-increments supports smoke or holdout --sequence 08")
    if args.compare_taper_feedback and (args.action != "holdout" or args.sequence not in ("09", "10")):
        parser.error("--compare-taper-feedback requires holdout --sequence 09 or 10")
    if args.compare_weak_allocation and (
            args.action not in ("holdout", "full") or args.sequence not in ("09", "10")):
        parser.error("--compare-weak-allocation requires holdout or full with --sequence 09 or 10")
    if args.reference_symmetric_full_run and (args.action != "full" or args.sequence != "08"):
        parser.error("--reference-symmetric-full-run requires full --sequence 08")
    run(args)


if __name__ == "__main__":
    main()
