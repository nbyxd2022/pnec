"""Compare RoMa v2 covariance direction and per-match scale in PNEC."""

import argparse
from functools import partial
from importlib.metadata import version
import json
from itertools import groupby
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from analyze_photometric_drift import compose_steps, gap_errors, rmse
from audit_pnec_bearing_covariance import REPO, bearing, load_truth, project_covariance, read_archive
from run_fixed_input_audit import atomic_json
from run_photometric_increment import digest, replay_config
from run_weak_scale_audit import control_difference_deg, solve_pair


SEQUENCES = ("09", "10")
PILOT_TARGETS = range(300, 400, 10)
CONTIGUOUS_TARGETS = range(800, 900)
ROTATION_GAPS = (10, 30, 50)
SAMPLE_COUNT = 512
SEED = 17
CHECKPOINT_SHA = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
DINOV3_COMMIT = "adc254450203739c8149213a7a69d8d905b4fcfa"


def image_pairs(dataset, sequence, targets):
    images = dataset / "sequences" / sequence / "image_0"
    expected = read_archive(f"pnec_kitti_{sequence}_original_feedback_a89d06a_user_report.json")
    hashes = expected["manifest"]["image_sha256"]
    for target in targets:
        names = [f"{frame:06d}.png" for frame in (target - 1, target)]
        pair = tuple(images / name for name in names)
        if any(digest(path) != hashes[name] for path, name in zip(pair, names)):
            raise ValueError(f"KITTI {sequence} input differs from archived images at {target}")
        yield target, pair, [hashes[name] for name in names]


def isotropic(covariances):
    """Preserve the total pixel variance independently at every endpoint."""
    return np.trace(covariances, axis1=-2, axis2=-1)[..., None, None] * np.eye(2) / 2


def pixel_covariances(precision, source_hw, image_hw):
    """Convert inverse covariance by a coordinate congruence before isotropizing."""
    ratio = (np.asarray(source_hw) / np.asarray(image_hw))[::-1]
    pixel_precision = precision * ratio[:, None] * ratio[None, :]
    return np.linalg.inv(pixel_precision)


def collect_pair(model, impl, pair, torch, image, sdpa_kernel, math_backend):
    sizes = []
    for path in pair:
        with image.open(path) as photo:
            sizes.append((photo.height, photo.width))
    torch.manual_seed(SEED)
    with torch.inference_mode(), sdpa_kernel(math_backend):
        prediction = model.match(*map(str, pair))
        # The stock sampler's density estimation is kept, with its distance computation in FP32.
        with patch.object(impl, "kde", partial(impl.kde, half=False)):
            matches, overlap, precision_a, precision_b = model.sample(prediction, SAMPLE_COUNT)
        if matches.shape != (SAMPLE_COUNT, 4) or any(
            p is None or p.shape != (SAMPLE_COUNT, 2, 2) or p.dtype != torch.float32
            for p in (precision_a, precision_b)
        ):
            raise ValueError("Unexpected RoMa v2 sample or endpoint matrix format")
        h_a, w_a = sizes[0]
        h_b, w_b = sizes[1]
        host, target = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
        # RoMa places the first pixel center at 0.5, OpenCV/KITTI at 0.
        pixels = np.stack((host.cpu().double().numpy(), target.cpu().double().numpy())) - 0.5
        precisions = np.stack((precision_a.cpu().double().numpy(), precision_b.cpu().double().numpy()))
        weights = overlap.cpu().float().numpy().reshape(-1)

    valid = np.isfinite(pixels).all(axis=(0, 2)) & np.isfinite(precisions).all(axis=(0, 2, 3))
    valid &= np.isfinite(weights) & (weights > 0)
    for side, (height, width) in enumerate(sizes):
        valid &= ((pixels[side] >= 0) & (pixels[side] <= (width - 1, height - 1))).all(axis=1)
    valid[valid] &= (np.linalg.eigvalsh(precisions[:, valid]) > 0).all(axis=(0, 2))
    cov = np.stack([pixel_covariances(matrices[valid], (model.anchor_height, model.anchor_width), size)
                    for matrices, size in zip(precisions, sizes)])
    return pixels[:, valid], cov, weights[valid]


def capture(args):
    from PIL import Image
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from romav2 import romav2 as impl

    if version("romav2") != "2.0.1" or torch.version.cuda != "11.8" or not torch.cuda.is_available():
        raise ValueError("Use the tested CUDA FP32 RoMa v2 2.0.1 environment")
    patched_source = Path(impl.__file__).resolve().parent
    if patched_source.parent != args.fp32_run.resolve(strict=True):
        raise ValueError(f"Use PYTHONPATH to select the verified FP32 copy, got {patched_source}")
    hub = Path(torch.hub.get_dir())
    if digest(hub / "checkpoints/romav2.0.1.pt") != CHECKPOINT_SHA:
        raise ValueError("RoMa checkpoint does not match the official SHA-256")
    if not (hub / f"facebookresearch_dinov3_{DINOV3_COMMIT}/hubconf.py").is_file():
        raise ValueError("Missing pinned DINOv3 cache")

    targets = CONTIGUOUS_TARGETS if args.contiguous else PILOT_TARGETS
    pairs = [(seq, target, paths, hashes) for seq in SEQUENCES
             for target, paths, hashes in image_pairs(args.dataset, seq, targets)]
    args.run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    model = impl.RoMaV2(impl.RoMaV2.Cfg(setting="fast", compile=False))
    model.bidirectional = True
    if (model.H_lr, model.W_lr) != (model.anchor_height, model.anchor_width):
        raise ValueError("The comparison requires the checkpoint's anchor image size")

    frames = []
    for seq, target, paths, image_hashes in pairs:
        pixels, cov, overlap = collect_pair(model, impl, paths, torch, Image, sdpa_kernel, SDPBackend.MATH)
        file = args.run / f"{seq}_{target}.npz"
        np.savez(file, pixels=pixels, covariance_px2=cov, overlap=overlap)
        frames.append({"sequence": seq, "target": target, "image_sha256": image_hashes,
                       "file": file.name, "sha256": digest(file), "valid_matches": len(overlap)})
        print(f"KITTI {seq} {target - 1}->{target} valid matches {len(overlap)}/{SAMPLE_COUNT}", flush=True)
    atomic_json(args.run / "capture.json", {
        "romav2": version("romav2"), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "checkpoint_sha256": CHECKPOINT_SHA, "dinov3_commit": DINOV3_COMMIT,
        "patch_sha256": {name: digest(patched_source / name) for name in
                         ("features.py", "dpt.py", "matcher.py", "refiner.py", "vit/__init__.py", "romav2.py")},
        "input_hw": [model.H_lr, model.W_lr], "settings": "FP32 fast, bidirectional, FP32 sampler density",
        "seed": SEED, "samples_per_pair": SAMPLE_COUNT, "capture_script_sha256": digest(Path(__file__)),
        "frames": frames, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    })


def initialization(cv2, host, target, intrinsics):
    """Return target-to-host pose and a covariances-independent mask."""
    if len(host) < 8:
        return None
    cv2.setRNGSeed(SEED)
    essential, mask = cv2.findEssentialMat(host, target, intrinsics, method=cv2.RANSAC,
                                          prob=0.999, threshold=1.0, maxIters=1000)
    if essential is None or essential.shape != (3, 3) or mask is None or np.count_nonzero(mask) < 8:
        return None
    count, rotation, translation, _, _ = cv2.recoverPose(
        essential, host, target, intrinsics, distanceThresh=1000.0, mask=mask.copy(),
    )
    if count < 5:
        return None
    inverse_rotation = rotation.T
    return (inverse_rotation, -inverse_rotation @ translation[:, 0]), mask[:, 0].astype(bool), count


def camera(sequence):
    config = replay_config(REPO, sequence)[0]
    values = dict(line.split(":", 1) for line in config.read_text(encoding="utf-8").splitlines()
                  if line.startswith(("Camera.", "PNEC.")))
    if any(float(values["Camera." + key]) for key in ("k1", "k2", "p1", "p2")):
        raise ValueError("Expected rectified images")
    fx, fy, cx, cy = (float(values["Camera." + key]) for key in ("fx", "fy", "cx", "cy"))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), float(values["PNEC.regularization"])


def estimate_pair(cv2, file, executable, intrinsics, regularizer, *, scale_control=False):
    with np.load(file, allow_pickle=False) as saved:
        host, target = saved["pixels"]
        cov = saved["covariance_px2"]
    initial = initialization(cv2, host, target, intrinsics)
    if initial is None:
        return {"status": "no_shared_initialization", "valid_matches": len(host)}
    pose, inliers, cheiral = initial
    # The regularizer and the nonlinear refinement are the same for both models.
    focal, center = intrinsics.diagonal()[:2], intrinsics[:2, 2]
    pixels = (host[inliers], target[inliers])
    cov = cov[:, inliers]
    left, right = (bearing(coords, focal, center) for coords in pixels)
    rows = np.zeros(inliers.sum(), dtype=[("regularization", "f8"), ("delta_trace", "f8")])
    rows["regularization"] = regularizer
    stats = {"valid_matches": len(host), "inliers": int(inliers.sum()), "cheirality_for_init": int(cheiral)}
    rotations = {"initial": pose[0].tolist()}
    equal_trace = isotropic(cov)
    models = [("equal_trace", equal_trace), ("full", cov)]
    if scale_control:
        # Preserve each endpoint's sum of pixel variances, while removing their per-match assignment.
        uniform = np.broadcast_to(equal_trace.mean(axis=1, keepdims=True), cov.shape)
        models.append(("frame_uniform", uniform))
    for label, covariances in models:
        host_cov, target_cov = (project_covariance(coords, matrix, focal, center, np.sqrt(3))
                                 for coords, matrix in zip(pixels, covariances))
        try:
            # Zero trace increments in the legacy executable make its auxiliary modes identical.
            result = solve_pair(executable, (rows, left, right, host_cov, target_cov), pose)
        except subprocess.CalledProcessError:
            if scale_control:
                stats["rotations"] = rotations
            return {**stats, "status": "native_refinement_failed", "failed_mode": label}
        rotation = result["symmetric_analytic"]
        if any(control_difference_deg(other, rotation) > 1e-8 for other in result.values()):
            raise ValueError("Zero-increment outputs differ in native refinement")
        rotations[label] = rotation.tolist()
    return {**stats, "status": "ok", "rotations": rotations}


def metrics(frames, *, scale_control=False):
    rows = [row["rotation_error_deg"] for row in frames if row["status"] == "ok"]
    scores = {}
    modes = ("initial", "equal_trace", "full")
    if scale_control:
        modes += ("frame_uniform",)
    for label in modes:
        values = np.array([row[label] for row in rows])
        scores[label] = ({"rms": float(np.sqrt(np.mean(values**2))), "median": float(np.median(values)),
                          "p90": float(np.percentile(values, 90))} if len(values) else None)
    result = {"common_pairs": len(rows), "requested_pairs": len(frames), "rotation_deg": scores,
              "full_wins": sum(row["full"] < row["equal_trace"] - 1e-8 for row in rows)}
    if scale_control:
        result["equal_trace_wins_vs_frame_uniform"] = sum(
            row["equal_trace"] < row["frame_uniform"] - 1e-8 for row in rows)
    return result


ORIGINAL_REPORT = "pnec_kitti_09_10_roma_v2_rotation_ablation_bea6f2c_user_report.json"


def check_replayed_frame(actual, expected):
    for key in ("target", "valid_matches", "inliers", "cheirality_for_init"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Archived {key} differs at frame {expected['target']}")
    if expected["status"] != "ok":
        raise ValueError("Expected a successful archived pair")
    rotations = actual.get("rotations", {})
    for label, frozen in expected["rotations"].items():
        if label not in rotations or control_difference_deg(np.array(rotations[label]), np.array(frozen)) > 1e-8:
            raise ValueError(f"Archived {label} rotation differs at frame {expected['target']}")


def contiguous_metrics(frames, reference):
    """Pool rotation errors only across uninterrupted shared frame pairs."""
    chunks = [list(chunk) for valid, chunk in groupby(frames, key=lambda row: row["status"] == "ok") if valid]
    modes = ("initial", "frame_uniform", "equal_trace", "full")
    series = [
        (reference[chunk[0]["target"] - 1:chunk[-1]["target"] + 1],
         {mode: compose_steps([row["rotations"][mode] for row in chunk]) for mode in modes})
        for chunk in chunks
    ]
    summary = {}
    for gap in ROTATION_GAPS:
        windows = [(truth, estimates) for truth, estimates in series if len(truth) > gap]
        count = sum(len(truth) - gap for truth, _ in windows)
        rms = {mode: rmse(np.concatenate([gap_errors(truth, est[mode], gap) for truth, est in windows]))
               for mode in modes} if windows else None
        summary[str(gap)] = {"common_intervals": count, "requested_intervals": len(frames) + 1 - gap,
                             "rotation_rms_deg": rms}
    return summary


def score(args):
    import cv2

    if args.contiguous and args.scale_control:
        raise ValueError("--scale-control is the frozen pilot comparison; --contiguous includes the same control")
    targets = CONTIGUOUS_TARGETS if args.contiguous else PILOT_TARGETS
    scale_control = args.scale_control or args.contiguous
    path = args.run / "capture.json"
    captured = json.loads(path.read_bytes())
    if (captured["checkpoint_sha256"] != CHECKPOINT_SHA or captured["input_hw"] != [512, 512]
            or captured["samples_per_pair"] != SAMPLE_COUNT or
            [(row["sequence"], row["target"]) for row in captured["frames"]]
            != [(seq, target) for seq in SEQUENCES for target in targets]):
        raise ValueError("Unexpected capture protocol")

    capture_sha = digest(path)
    binary_sha = digest(args.executable)
    reference_archive = read_archive(ORIGINAL_REPORT) if scale_control else None
    if reference_archive is not None:
        checks = {"native_binary_sha256": binary_sha, "opencv": cv2.__version__}
        if args.scale_control:
            checks["capture_sha256"] = capture_sha
        for key, value in checks.items():
            if reference_archive[key] != value:
                raise ValueError(f"Control requires archived {key} {reference_archive[key]}, got {value}")
    results = []
    for sequence in SEQUENCES:
        frozen_frames = {}
        if args.scale_control:
            archived = next(row for row in reference_archive["sequences"] if row["sequence"] == sequence)
            frozen_frames = {row["target"]: row for row in archived["frames"]}
        intrinsics, regularizer = camera(sequence)
        frames = []
        for row in captured["frames"]:
            if row["sequence"] == sequence:
                target = row["target"]
                file = args.run / f"{sequence}_{target}.npz"
                if digest(file) != row["sha256"]:
                    raise ValueError(f"Changed match capture: {file}")
                frame = {"target": target, **estimate_pair(
                    cv2, file, args.executable, intrinsics, regularizer, scale_control=scale_control,
                )}
                if args.scale_control:
                    check_replayed_frame(frame, frozen_frames[target])
                frames.append(frame)
        reference, _, gt_hash = load_truth(args.ground_truth_dir, sequence)
        if args.scale_control and gt_hash != archived["gt_sha256_lf"]:
            raise ValueError(f"KITTI {sequence} ground truth differs from the archived comparison")
        for row in frames:
            if row["status"] == "ok":
                target = row["target"]
                expected = reference[target - 1].T @ reference[target]
                row["rotation_error_deg"] = {
                    name: control_difference_deg(np.array(rotation), expected)
                    for name, rotation in row["rotations"].items()
                }
        result = {"sequence": sequence, "gt_sha256_lf": gt_hash,
                  **metrics(frames, scale_control=scale_control), "frames": frames}
        if args.contiguous:
            result["rotation_by_gap_frames"] = contiguous_metrics(frames, reference)
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2), flush=True)
    output = args.run / "rotation_ablation.json"
    if args.contiguous:
        output = args.run / "rotation_contiguous.json"
    elif args.scale_control:
        output = args.run / "rotation_scale_control.json"
    report = {"capture_sha256": capture_sha, "native_binary_sha256": binary_sha,
              "scorer_sha256": digest(Path(__file__)), "opencv": cv2.__version__,
              "residual": "native PNEC with both endpoints; equal trace in original pixel coordinates",
              "sequences": results}
    if reference_archive is not None:
        report["reference_report_sha256"] = digest(REPO / "docs/experiments/results" / ORIGINAL_REPORT)
        report["frame_uniform"] = "Each endpoint uses its own mean isotropic pixel variance over the shared inliers"
    if args.contiguous:
        report["scope"] = "Independent adjacent-pair refinements composed only within uninterrupted shared intervals"
        report["target_frames_inclusive"] = [targets.start, targets.stop - 1]
    atomic_json(output, report)
    print(f"Report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(required=True)
    for action, method, names in (("capture", capture, ("dataset", "fp32-run", "run")),
                                  ("score", score, ("run", "executable", "ground-truth-dir"))):
        sub = actions.add_parser(action)
        sub.set_defaults(method=method)
        sub.add_argument("--contiguous", action="store_true", help="KITTI 09/10 target frames 800 through 899")
        for name in names:
            sub.add_argument("--" + name, type=Path, required=True)
        if action == "score":
            sub.add_argument("--scale-control", action="store_true")
    args = parser.parse_args()
    args.method(args)
