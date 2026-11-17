"""Localize rotation drift from saved KITTI photometric experiment poses."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_photometric_increment import HOLDOUT, MODES, digest, restore_rotations, rotation_metrics


GAP_BANDS = ((1, 1), (2, 10), (11, 30), (31, 60), (61, 120), (121, 200), (201, 299))
DETAIL_GAPS = (1, 10, 30, 60, 120, 200, 250, 299)
WINDOW_FRAMES = 50
SIGNED_WINDOWS = ((500, 550), (650, 700))  # Half-open ground-truth target-frame ranges.
LARGEST_STEPS = 5
FULL_WINDOW_STEPS = 200
FULL_GAP_BANDS = (*GAP_BANDS, (300, 600), (601, 1000), (1001, 2000), (2001, None))


def gap_errors(reference, estimate, gap):
    expected = reference[:-gap].swapaxes(1, 2) @ reference[gap:]
    actual = estimate[:-gap].swapaxes(1, 2) @ estimate[gap:]
    cosine = (np.sum(expected * actual, axis=(1, 2)) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def rmse(values):
    return float(np.sqrt(np.mean(values**2)))


def project_rotations(matrices):
    u, _, vt = np.linalg.svd(matrices)
    u[:, :, -1] *= np.where(np.linalg.det(u @ vt) < 0, -1, 1)[:, None]
    return u @ vt


def relative_steps(absolute):
    return absolute[:-1].swapaxes(1, 2) @ absolute[1:]


def compose_steps(relative):
    absolute = np.empty((len(relative) + 1, 3, 3))
    absolute[0] = np.eye(3)
    for index, rotation in enumerate(relative, 1):
        absolute[index] = absolute[index - 1] @ rotation
    return absolute


def replace_steps(per_match, original, mask):
    relative = relative_steps(per_match)
    relative[mask] = relative_steps(original)[mask]
    # Remove roundoff from recovering increments out of accumulated rotations.
    return compose_steps(project_rotations(relative))


def replace_window(per_match, original, targets, lo, hi):
    return replace_steps(per_match, original, (targets >= lo) & (targets < hi))


def rotation_vectors_deg(rotations):
    """Signed SO(3) log in the input coordinate system, with magnitude in degrees."""
    r = rotations
    skew = 0.5 * np.column_stack((
        r[:, 2, 1] - r[:, 1, 2],
        r[:, 0, 2] - r[:, 2, 0],
        r[:, 1, 0] - r[:, 0, 1],
    ))
    sine = np.linalg.norm(skew, axis=1)
    cosine = np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1)
    angle = np.arctan2(sine, cosine)
    scale = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 1e-12)
    vectors = skew * scale[:, None]
    for index in np.flatnonzero((cosine < 0) & (sine < 1e-6)):
        _, axes = np.linalg.eigh((r[index] + r[index].T) / 2)
        axis = axes[:, -1]
        if np.dot(axis, skew[index]) < 0:
            axis = -axis
        vectors[index] = angle[index] * axis
    return np.rad2deg(vectors)


def signed_window_analysis(reference, estimates, first, per_match_rpe_n):
    targets = np.arange(first + 1, first + len(reference))
    anchor_reference = reference[0].T @ reference
    # D_t = G_(t-1) R_est,t G_t^T lives in the frame-500 GT camera axes.
    # Its chronological product equals E_t G_t^T for anchor-relative E and G.
    transported = {mode: anchor_reference[:-1] @ relative_steps(estimate) @
                   anchor_reference[1:].swapaxes(1, 2) for mode, estimate in estimates.items()}
    logs = {mode: rotation_vectors_deg(steps) for mode, steps in transported.items()}
    difference = transported["per_match"] @ transported["original"].swapaxes(1, 2)
    changes = np.linalg.norm(rotation_vectors_deg(difference), axis=1)
    output = []
    for lo, hi in SIGNED_WINDOWS:
        inside = (targets >= lo) & (targets < hi)
        members = np.flatnonzero(inside)
        ranked = members[np.argsort(-changes[members], kind="stable")]
        largest = np.zeros(len(targets), dtype=bool)
        largest[ranked[:LARGEST_STEPS]] = True
        remaining = inside & ~largest
        summaries = {}
        for mode, steps in transported.items():
            vectors = logs[mode][inside]
            signed_sum = vectors.sum(axis=0)
            total_angle = float(np.linalg.norm(vectors, axis=1).sum())
            net_error = compose_steps(steps[inside])[-1]
            remaining_vectors = logs[mode][remaining]
            remaining_angle = float(np.linalg.norm(remaining_vectors, axis=1).sum())
            remaining_sum = remaining_vectors.sum(axis=0)
            summaries[mode] = {
                "signed_step_sum_xyz_deg": signed_sum.tolist(),
                "sum_step_angles_deg": total_angle,
                "directional_coherence":
                    float(np.linalg.norm(signed_sum) / total_angle) if total_angle > 1e-9 else None,
                "net_window_error_xyz_deg": rotation_vectors_deg(net_error[None])[0].tolist(),
                "signed_step_sum_without_largest_changes_xyz_deg": remaining_sum.tolist(),
                "directional_coherence_without_largest_changes":
                    float(np.linalg.norm(remaining_sum) / remaining_angle) if remaining_angle > 1e-9 else None,
            }
        replacement = {}
        for name, mask in (("largest_five", largest), ("remaining", remaining), ("entire_window", inside)):
            estimate = replace_steps(estimates["per_match"], estimates["original"], mask)
            metrics = rotation_metrics(reference, estimate)
            replacement[name] = {
                "steps_replaced": int(mask.sum()),
                **metrics,
                "change_in_per_match_rpe_n_deg": metrics["rpe_n_deg"] - per_match_rpe_n,
            }
        output.append({
            "target_frames": [lo, hi - 1], "pairs": int(inside.sum()),
            "largest_change_target_frames": targets[ranked[:LARGEST_STEPS]].tolist(),
            "sign_summaries": summaries, "replace_per_match_with_original": replacement,
            "steps": [{"target_frame": int(targets[index]), "method_change_deg": float(changes[index]),
                       "signed_error_xyz_deg": {mode: vectors[index].tolist() for mode, vectors in logs.items()}}
                      for index in members],
        })
    return output


def full_scope(reference, estimates, first, expected):
    n = len(reference)
    curves = {mode: [rmse(gap_errors(reference, estimate, gap)) for gap in range(1, n)]
              for mode, estimate in estimates.items()}
    metrics = {mode: {"rpe_1_deg": curve[0], "rpe_n_deg": sum(curve) / n}
               for mode, curve in curves.items()}
    for mode, values in metrics.items():
        for key, value in values.items():
            if not math.isclose(value, expected[mode][key], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Frames {first}–{first + n - 1}/{mode}: {key} differs from report")

    bands = []
    for lo, limit in FULL_GAP_BANDS:
        if lo >= n:
            break
        hi = min(limit if limit is not None else n - 1, n - 1)
        contribution = {mode: sum(curve[lo - 1:hi]) / n for mode, curve in curves.items()}
        bands.append({"gap_frames": [lo, hi], "rpe_n_contribution_deg": contribution,
                      "per_match_minus_original_deg": contribution["per_match"] - contribution["original"],
                      "per_match_minus_uniform_deg": contribution["per_match"] - contribution["uniform"]})

    windows = []
    # Adjacent windows share the anchor at the boundary, but each step belongs to one window.
    for lo in range(0, n - 1, FULL_WINDOW_STEPS):
        hi = min(lo + FULL_WINDOW_STEPS, n - 1)
        segment = slice(lo, hi + 1)
        local = {mode: rotation_metrics(reference[segment], estimate[segment])
                 for mode, estimate in estimates.items()}
        windows.append({"frames": [first + lo, first + hi], "metrics": local,
                        "per_match_minus_original_rpe_n_deg":
                            local["per_match"]["rpe_n_deg"] - local["original"]["rpe_n_deg"]})
    return {"frames": [first, first + n - 1], "recomputed_metrics": metrics,
            "per_gap_rmse_deg": curves, "gap_bands": bands, "windows": windows}


def tail_replacements(reference, estimates, tail_result):
    first = tail_result["frames"][0]
    targets = first + np.arange(1, len(reference))
    base = estimates["original"]
    candidate = estimates["per_match"]
    change = gap_errors(base, candidate, 1)
    ranked = np.argsort(-change, kind="stable")
    window = max(tail_result["windows"], key=lambda row: row["per_match_minus_original_rpe_n_deg"])
    lo, hi = window["frames"]
    inside = (targets > lo) & (targets <= hi)
    top20 = np.zeros(len(targets), dtype=bool)
    top20[ranked[:20]] = True
    top5 = np.zeros(len(targets), dtype=bool)
    top5[ranked[inside[ranked]][:LARGEST_STEPS]] = True
    masks = {"largest_20_in_tail": top20}
    if window["per_match_minus_original_rpe_n_deg"] > 0:
        masks.update(largest_5_in_window=top5, remaining_in_window=inside & ~top5, entire_window=inside)
    initial_error = tail_result["recomputed_metrics"]["per_match"]["rpe_n_deg"]
    substitutions = {}
    for name, mask in masks.items():
        replacement = replace_steps(candidate, base, mask)
        metrics = rotation_metrics(reference, replacement)
        substitutions[name] = {
            "steps_replaced": int(mask.sum()), **metrics,
            "change_in_tail_rpe_n_deg": metrics["rpe_n_deg"] - initial_error,
            "target_frames": targets[mask].tolist(),
        }
    return {"window_frames": [lo, hi], "window_local_rpe_n_excess_deg": window["per_match_minus_original_rpe_n_deg"],
            "policy": "Rank individual steps by rotation disagreement between the saved methods. "
                      "Select the window by largest local RPE_n excess. Substitutions only recompose saved rotations.",
            "replace_per_match_steps_with_original": substitutions}


def analyze(run, ground_truth, sequence, signed_windows=False):
    report_path = run / "report.json"
    manifest_path = run / "manifest.json"
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (recorded["manifest"] != manifest or sequence not in manifest["sequences"]
            or manifest["modes"] != list(MODES)):
        raise ValueError("The report and manifest must agree on run, sequence and modes")
    start, first, end = (manifest[key] for key in ("replay_start", "evaluation_start", "end_exclusive"))
    full = manifest["action"] == "full"
    if full:
        tail = manifest["tail_evaluation_start"]
        if (start != 0 or first != 0 or tail != HOLDOUT[2] or tail + 1 >= end
                or manifest["sequences"] != [sequence] or digest(ground_truth) != manifest["ground_truth_sha256"]):
            raise ValueError("Full report needs matching GT and complete coverage of the frozen tail")
        if signed_windows:
            raise ValueError("--signed-windows is defined for the original holdout")
    elif manifest["action"] != "holdout" or (start, first, end) != HOLDOUT:
        raise ValueError("Expected a full sequence or the frozen holdout")
    n = end - first
    slot = run / sequence
    times_path = slot / "times.txt"
    times = np.loadtxt(times_path, ndmin=1)
    if (times.shape != (end - start,) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError(f"{times_path}: incomplete or invalid times")
    raw_gt = np.loadtxt(ground_truth, ndmin=2)[first:end]
    if raw_gt.shape != (n, 12) or not np.isfinite(raw_gt).all():
        raise ValueError(f"{ground_truth}: incomplete KITTI truth")
    reference = project_rotations(raw_gt.reshape(-1, 3, 4)[:, :, :3])
    pose_paths = {mode: slot / mode / "ablation/PNEC.txt" for mode in MODES}
    estimates = {mode: restore_rotations(path, times, first - start) for mode, path in pose_paths.items()}
    inputs = {
        "reported_metrics": digest(report_path), "manifest": digest(manifest_path),
        "times": digest(times_path), "ground_truth": digest(ground_truth),
        "relative_poses": {mode: digest(path) for mode, path in pose_paths.items()},
    }
    if full:
        anchors = recorded["results"][sequence]
        full_result = full_scope(reference, estimates, first, anchors)
        tail_estimates = {mode: estimate[tail:] for mode, estimate in estimates.items()}
        tail_result = full_scope(reference[tail:], tail_estimates, tail,
                                 {mode: record["tail"] for mode, record in anchors.items()})
        return {"sequence": sequence, "input_sha256": inputs, "full": full_result, "tail": tail_result,
                "tail_step_substitutions": tail_replacements(reference[tail:], tail_estimates, tail_result)}
    errors = {mode: [gap_errors(reference, estimate, gap) for gap in range(1, n)]
              for mode, estimate in estimates.items()}
    curves = {mode: [rmse(values) for values in steps] for mode, steps in errors.items()}
    metrics = {mode: {"rpe_1_deg": curve[0], "rpe_n_deg": sum(curve) / n}
               for mode, curve in curves.items()}
    for mode, values in metrics.items():
        for key, actual in values.items():
            if not math.isclose(actual, recorded["results"][sequence][mode][key], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{sequence}/{mode} {key} does not reproduce the saved report")

    # These signed contributions sum to the total RPE_n change across all gaps.
    bands = []
    for lo, hi in GAP_BANDS:
        contribution = {mode: sum(curve[lo - 1:hi]) / n for mode, curve in curves.items()}
        bands.append({"gap_frames": [lo, hi], "rpe_n_contribution_deg": contribution,
                      "per_match_minus_original_deg": contribution["per_match"] - contribution["original"],
                      "per_match_minus_uniform_deg": contribution["per_match"] - contribution["uniform"]})

    windows = []
    for lo in range(first, end, WINDOW_FRAMES):
        hi = min(lo + WINDOW_FRAMES, end)
        by_gap = {}
        for gap in DETAIL_GAPS:
            target_frames = np.arange(first + gap, end)
            inside = (target_frames >= lo) & (target_frames < hi)
            if inside.any():
                by_gap[str(gap)] = {
                    "pairs": int(inside.sum()),
                    "rmse_deg": {mode: rmse(steps[gap - 1][inside]) for mode, steps in errors.items()},
                }
        # Replay the saved rotations mathematically, replacing only transitions
        # whose target is in this window. Other online outputs remain fixed.
        replacement = replace_window(estimates["per_match"], estimates["original"],
                                     np.arange(first + 1, end), lo, hi)
        replay_metrics = rotation_metrics(reference, replacement)
        anchor_gap = hi - 1 - first
        windows.append({
            "target_frames": [lo, hi - 1],
            "by_gap": by_gap,
            "end_error_from_clip_anchor_deg": {mode: float(steps[anchor_gap - 1][0])
                                               for mode, steps in errors.items()},
            "replace_per_match_with_original": replay_metrics,
            "replacement_change_in_per_match_rpe_n_deg":
                replay_metrics["rpe_n_deg"] - metrics["per_match"]["rpe_n_deg"],
        })
    report = {
        "sequence": sequence,
        "evaluation_frames": [first, end - 1],
        "input_sha256": inputs,
        "recomputed_metrics": metrics,
        "per_gap_rmse_deg": curves,
        "gap_bands": bands,
        "windows": windows,
    }
    if signed_windows:
        report["signed_windows"] = signed_window_analysis(reference, estimates, first, metrics["per_match"]["rpe_n_deg"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--sequence", choices=("01", "08"), default="01")
    parser.add_argument("--gt", type=Path, required=True, help="KITTI poses for the full sequence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signed-windows", action="store_true", help="Analyze signed drift in the two identified windows")
    args = parser.parse_args()
    output = analyze(args.run, args.gt, args.sequence, args.signed_windows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")

    if "full" in output:
        for scope in ("full", "tail"):
            result = output[scope]
            print(f"KITTI {args.sequence} {scope} frames {result['frames']}: all RPE values verified")
            for band in result["gap_bands"]:
                print(f"Gap {band['gap_frames']}: per_match minus original RPE_n contribution "
                      f"{band['per_match_minus_original_deg']:+.6f} deg")
            for window in sorted(result["windows"], key=lambda row: -row["per_match_minus_original_rpe_n_deg"])[:5]:
                print(f"Window {window['frames']}: local per_match minus original RPE_n "
                      f"{window['per_match_minus_original_rpe_n_deg']:+.6f} deg")
        replacements = output["tail_step_substitutions"]
        print(f"Tail replacement window {replacements['window_frames']}")
        for name, item in replacements["replace_per_match_steps_with_original"].items():
            print(f"{name}: {item['steps_replaced']} steps, tail RPE_n change {item['change_in_tail_rpe_n_deg']:+.6f} deg")
        print(f"Report: {args.output}")
        return

    print(f"KITTI {args.sequence}: all three RPE_1 and RPE_n values verified")
    print("gap_frames per_match_minus_original_contribution_deg per_match_minus_uniform_contribution_deg")
    for band in output["gap_bands"]:
        lo, hi = band["gap_frames"]
        print(f"{lo:3d}-{hi:<3d} {band['per_match_minus_original_deg']:+.6f} "
              f"{band['per_match_minus_uniform_deg']:+.6f}")
    print("target_frames rpe_1_deg(original/uniform/per_match) anchored_end_error_deg(original/uniform/per_match) "
          "replacement_change_in_per_match_rpe_n_deg")
    for window in output["windows"]:
        lo, hi = window["target_frames"]
        adjacent = window["by_gap"]["1"]["rmse_deg"]
        anchored = window["end_error_from_clip_anchor_deg"]
        print(f"{lo}-{hi} {'/'.join(f'{adjacent[m]:.4f}' for m in MODES)} "
              f"{'/'.join(f'{anchored[m]:.4f}' for m in MODES)} "
              f"{window['replacement_change_in_per_match_rpe_n_deg']:+.6f}")
    if args.signed_windows:
        for window in output["signed_windows"]:
            lo, hi = window["target_frames"]
            print(f"Signed window {lo}-{hi}; largest change targets {window['largest_change_target_frames']}")
            print("mode sum_xyz_deg net_xyz_deg coherence coherence_without_largest_changes")
            for mode, summary in window["sign_summaries"].items():
                values = (summary["signed_step_sum_xyz_deg"], summary["net_window_error_xyz_deg"],
                          summary["directional_coherence"], summary["directional_coherence_without_largest_changes"])
                print(mode, json.dumps(values))
            for subset, result in window["replace_per_match_with_original"].items():
                print(f"replace_{subset}: steps={result['steps_replaced']} RPE_n={result['rpe_n_deg']:.6f} "
                      f"delta={result['change_in_per_match_rpe_n_deg']:+.6f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
