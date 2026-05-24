#!/usr/bin/env python3
"""
Rank generated models by ego-trajectory displacement and travelled distance.

The script compares each model's UniDepth ego trajectory pickle against the GT
ego trajectory in drivinggen_input_dataset/ego_condition/ego_motion/<scene>.npy.

Coordinate convention:
    * GT .npy files are global XY positions. They are translated to frame 0
      and rotated into an ego-local frame with columns:
          [forward, lateral_left]
    * UniDepth SLAM locs are stored as [x_right, z_forward]. They are converted
      to [forward, lateral_left] as:
          [z_forward, -x_right]

By default, no best-fit rotation or scaling is applied. That preserves the
generated trajectory's forward/backward and left/right sign. The default ranking
metric combines total travelled-distance error with endpoint-vector error, so a
trajectory that moves backward by the right distance is still penalized.
Optional rotation alignment is available for diagnostic comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-9
DEFAULT_RELATIVE_FLOOR_M = 1.0


@dataclass(frozen=True)
class TrajectoryJob:
    model_dir: Path
    model_name: str
    scene: str
    pred_traj_path: Path
    gt_traj_path: Path


def strip_model_suffix(name: str) -> str:
    suffix = "-drivinggen-samples-outputs"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def parse_scene_args(scene_args: list[str] | None) -> list[str] | None:
    if not scene_args:
        return None

    scenes: list[str] = []
    for item in scene_args:
        path = Path(item)
        if path.exists() and path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    scenes.append(line)
        else:
            scenes.extend(s for s in item.split(",") if s)
    return scenes


def discover_models(benchmark_root: Path, requested: set[str] | None) -> list[Path]:
    model_dirs: list[Path] = []
    for model_dir in sorted(benchmark_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "benchmarks").is_dir():
            continue
        names = {model_dir.name, strip_model_suffix(model_dir.name)}
        if requested and names.isdisjoint(requested):
            continue
        model_dirs.append(model_dir)
    return model_dirs


def find_pred_traj(model_dir: Path, scene: str) -> Path | None:
    scene_dir = model_dir / "benchmarks" / scene
    if not scene_dir.is_dir():
        return None
    matches = sorted(scene_dir.glob("**/unidepth-estimate_ego_traj.pkl"))
    return matches[0] if matches else None


def discover_scenes(model_dirs: Iterable[Path], requested: list[str] | None) -> list[str]:
    if requested:
        return requested

    scenes: set[str] = set()
    for model_dir in model_dirs:
        benchmarks_dir = model_dir / "benchmarks"
        for pred_path in benchmarks_dir.glob("*/**/unidepth-estimate_ego_traj.pkl"):
            try:
                scene = pred_path.relative_to(benchmarks_dir).parts[0]
            except ValueError:
                continue
            scenes.add(scene)
    return sorted(scenes)


def build_jobs(
    model_dirs: list[Path],
    scenes: list[str],
    input_root: Path,
    common_scenes: bool,
    strict: bool,
) -> tuple[list[TrajectoryJob], list[str]]:
    warnings: list[str] = []
    jobs: list[TrajectoryJob] = []
    available_by_model: dict[str, set[str]] = {}

    for model_dir in model_dirs:
        model_name = strip_model_suffix(model_dir.name)
        available_by_model[model_name] = set()
        for scene in scenes:
            gt_traj_path = input_root / "ego_condition" / "ego_motion" / f"{scene}.npy"
            pred_traj_path = find_pred_traj(model_dir, scene)
            if pred_traj_path is not None and gt_traj_path.is_file():
                available_by_model[model_name].add(scene)

    selected_scenes = set(scenes)
    if common_scenes and available_by_model:
        selected_scenes = set.intersection(*available_by_model.values())
        if not selected_scenes:
            raise RuntimeError("No common scenes with GT and predicted ego trajectories across selected models.")

    for model_dir in model_dirs:
        model_name = strip_model_suffix(model_dir.name)
        for scene in scenes:
            if scene not in selected_scenes:
                continue

            gt_traj_path = input_root / "ego_condition" / "ego_motion" / f"{scene}.npy"
            pred_traj_path = find_pred_traj(model_dir, scene)

            missing = []
            if pred_traj_path is None:
                missing.append(str(model_dir / "benchmarks" / scene / "**/unidepth-estimate_ego_traj.pkl"))
            if not gt_traj_path.is_file():
                missing.append(str(gt_traj_path))

            if missing:
                msg = f"Skipping {model_name}/{scene}: missing " + ", ".join(missing)
                if strict:
                    raise FileNotFoundError(msg)
                warnings.append(msg)
                continue

            jobs.append(
                TrajectoryJob(
                    model_dir=model_dir,
                    model_name=model_name,
                    scene=scene,
                    pred_traj_path=pred_traj_path,
                    gt_traj_path=gt_traj_path,
                )
            )

    return jobs, warnings


def load_gt_trajectory(path: Path) -> np.ndarray:
    gt = np.load(path, allow_pickle=True)
    gt = np.asarray(gt, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise ValueError(f"GT trajectory must have shape (T, >=2): {path} has {gt.shape}")
    if len(gt) == 0:
        raise ValueError(f"GT trajectory is empty: {path}")
    return gt[:, :2]


def load_pred_trajectory(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "locs" not in payload:
        raise ValueError(f"Predicted trajectory pickle must contain dict key 'locs': {path}")
    locs = np.asarray(payload["locs"], dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] < 2:
        raise ValueError(f"Predicted locs must have shape (T, >=2): {path} has {locs.shape}")
    if len(locs) == 0:
        raise ValueError(f"Predicted locs are empty: {path}")
    return locs[:, :2]


def gt_global_xy_to_ego_forward_lateral(
    gt_xy: np.ndarray,
    k_ahead: int = 1,
    min_step: float = 1.0,
) -> np.ndarray:
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    origin = gt_xy[0]
    rel = gt_xy - origin

    k = min(max(k_ahead, 1), len(gt_xy) - 1)
    heading = gt_xy[k] - gt_xy[0]
    if np.linalg.norm(heading) < min_step:
        for j in range(1, len(gt_xy)):
            heading = gt_xy[j] - gt_xy[0]
            if np.linalg.norm(heading) >= min_step:
                break

    theta = 0.0 if np.linalg.norm(heading) < EPS else float(np.arctan2(heading[1], heading[0]))
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rel @ rotation


def slam_xright_zforward_to_forward_lateral(locs: np.ndarray) -> np.ndarray:
    locs = np.asarray(locs, dtype=np.float64)
    out = np.empty((len(locs), 2), dtype=np.float64)
    out[:, 0] = locs[:, 1]
    out[:, 1] = -locs[:, 0]
    return out


def resample_traj(traj: np.ndarray, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if len(traj) == target_len:
        return traj.copy()
    if len(traj) == 1:
        return np.repeat(traj, target_len, axis=0)

    old_t = np.linspace(0.0, 1.0, len(traj))
    new_t = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, traj.shape[1]), dtype=np.float64)
    for dim in range(traj.shape[1]):
        values = traj[:, dim]
        finite = np.isfinite(values)
        if finite.sum() == 0:
            out[:, dim] = 0.0
        elif finite.sum() == 1:
            out[:, dim] = values[finite][0]
        else:
            out[:, dim] = np.interp(new_t, old_t[finite], values[finite])
    return out


def align_pred_to_gt_origin_fixed(
    pred_forward_lateral: np.ndarray,
    gt_forward_lateral: np.ndarray,
    allow_scale: bool,
) -> tuple[np.ndarray, float]:
    pred = np.asarray(pred_forward_lateral, dtype=np.float64)
    gt = np.asarray(gt_forward_lateral, dtype=np.float64)
    n = min(len(pred), len(gt))
    if n < 2:
        return pred - pred[:1], 1.0

    pred_rel = pred[:n] - pred[:1]
    gt_rel = gt[:n]
    finite = np.isfinite(pred_rel).all(axis=1) & np.isfinite(gt_rel).all(axis=1)
    if finite.sum() < 2:
        return pred - pred[:1], 1.0

    p = pred_rel[finite]
    q = gt_rel[finite]
    covariance = p.T @ q / len(p)
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return pred - pred[:1], 1.0

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[1] *= -1
        rotation = vt.T @ u.T

    scale = 1.0
    if allow_scale:
        var_p = float((p ** 2).sum() / len(p))
        if var_p > EPS:
            scale = float((q * (rotation @ p.T).T).sum() / (len(p) * var_p))

    pred_full_rel = pred - pred[:1]
    aligned = (scale * (rotation @ pred_full_rel.T)).T
    return aligned, scale


def path_length(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 0.0
    finite = np.isfinite(traj).all(axis=1)
    traj = traj[finite]
    if len(traj) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def relative_denominator(gt: float, floor_m: float) -> float:
    return max(abs(gt), floor_m, EPS)


def signed_relative_error(pred: float, gt: float, floor_m: float) -> float:
    return float((pred - gt) / relative_denominator(gt, floor_m))


def absolute_relative_error(pred: float, gt: float, floor_m: float) -> float:
    return float(abs(pred - gt) / relative_denominator(gt, floor_m))


def prepare_trajectories(job: TrajectoryJob, alignment_mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    gt_global = load_gt_trajectory(job.gt_traj_path)
    pred_locs = load_pred_trajectory(job.pred_traj_path)

    gt_local = gt_global_xy_to_ego_forward_lateral(gt_global)
    pred_local = slam_xright_zforward_to_forward_lateral(pred_locs)

    target_len = max(len(gt_local), len(pred_local))
    gt_local = resample_traj(gt_local, target_len)
    pred_local = resample_traj(pred_local, target_len)

    if alignment_mode == "none":
        pred_local = pred_local - pred_local[:1]
        scale = 1.0
    elif alignment_mode == "rotate":
        pred_local, scale = align_pred_to_gt_origin_fixed(pred_local, gt_local, allow_scale=False)
    elif alignment_mode == "rotate-scale":
        pred_local, scale = align_pred_to_gt_origin_fixed(pred_local, gt_local, allow_scale=True)
    else:
        raise ValueError(f"Unknown alignment mode: {alignment_mode}")

    return gt_local, pred_local, scale


def compute_scene_metrics(job: TrajectoryJob, alignment_mode: str, relative_floor_m: float) -> dict[str, object]:
    gt, pred, scale = prepare_trajectories(job, alignment_mode)

    gt_endpoint = gt[-1] - gt[0]
    pred_endpoint = pred[-1] - pred[0]
    gt_distance = path_length(gt)
    pred_distance = path_length(pred)
    gt_displacement = float(np.linalg.norm(gt_endpoint))
    pred_displacement = float(np.linalg.norm(pred_endpoint))
    endpoint_error = float(np.linalg.norm(pred_endpoint - gt_endpoint))
    ade = float(np.linalg.norm(pred - gt, axis=1).mean())
    endpoint_rel_error = endpoint_error / relative_denominator(gt_displacement, relative_floor_m)

    forward_gt = float(gt_endpoint[0])
    forward_pred = float(pred_endpoint[0])
    lateral_gt = float(gt_endpoint[1])
    lateral_pred = float(pred_endpoint[1])
    distance_abs_rel_error = absolute_relative_error(pred_distance, gt_distance, relative_floor_m)
    distance_signed_rel_error = signed_relative_error(pred_distance, gt_distance, relative_floor_m)
    displacement_abs_rel_error = absolute_relative_error(pred_displacement, gt_displacement, relative_floor_m)
    displacement_signed_rel_error = signed_relative_error(pred_displacement, gt_displacement, relative_floor_m)

    return {
        "model": job.model_name,
        "scene": job.scene,
        "gt_frames": int(len(gt)),
        "pred_path": str(job.pred_traj_path),
        "alignment_mode": alignment_mode,
        "alignment_scale": scale,
        "gt_distance_m": gt_distance,
        "pred_distance_m": pred_distance,
        "distance_abs_error_m": abs(pred_distance - gt_distance),
        "distance_signed_error_m": pred_distance - gt_distance,
        "distance_abs_rel_error": distance_abs_rel_error,
        "distance_signed_rel_error": distance_signed_rel_error,
        "gt_displacement_m": gt_displacement,
        "pred_displacement_m": pred_displacement,
        "displacement_abs_error_m": abs(pred_displacement - gt_displacement),
        "displacement_signed_error_m": pred_displacement - gt_displacement,
        "displacement_abs_rel_error": displacement_abs_rel_error,
        "displacement_signed_rel_error": displacement_signed_rel_error,
        "endpoint_error_m": endpoint_error,
        "endpoint_rel_error": endpoint_rel_error,
        "ade_m": ade,
        "gt_forward_m": forward_gt,
        "pred_forward_m": forward_pred,
        "forward_error_m": forward_pred - forward_gt,
        "gt_lateral_left_m": lateral_gt,
        "pred_lateral_left_m": lateral_pred,
        "lateral_left_error_m": lateral_pred - lateral_gt,
        "combined_abs_rel_error": 0.5 * (distance_abs_rel_error + displacement_abs_rel_error),
        "combined_endpoint_distance_rel_error": 0.5 * (distance_abs_rel_error + endpoint_rel_error),
        "combined_endpoint_distance_error_m": 0.5 * (abs(pred_distance - gt_distance) + endpoint_error),
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def aggregate_model_metrics(scene_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in scene_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for model, rows in sorted(by_model.items()):
        aggregate_rows.append(
            {
                "model": model,
                "num_scenes": len(rows),
                "mean_gt_distance_m": mean([float(r["gt_distance_m"]) for r in rows]),
                "mean_pred_distance_m": mean([float(r["pred_distance_m"]) for r in rows]),
                "mean_distance_abs_error_m": mean([float(r["distance_abs_error_m"]) for r in rows]),
                "mean_distance_abs_rel_error": mean([float(r["distance_abs_rel_error"]) for r in rows]),
                "mean_distance_signed_rel_error": mean([float(r["distance_signed_rel_error"]) for r in rows]),
                "mean_gt_displacement_m": mean([float(r["gt_displacement_m"]) for r in rows]),
                "mean_pred_displacement_m": mean([float(r["pred_displacement_m"]) for r in rows]),
                "mean_displacement_abs_error_m": mean([float(r["displacement_abs_error_m"]) for r in rows]),
                "mean_displacement_abs_rel_error": mean([float(r["displacement_abs_rel_error"]) for r in rows]),
                "mean_displacement_signed_rel_error": mean([float(r["displacement_signed_rel_error"]) for r in rows]),
                "mean_endpoint_error_m": mean([float(r["endpoint_error_m"]) for r in rows]),
                "mean_endpoint_rel_error": mean([float(r["endpoint_rel_error"]) for r in rows]),
                "mean_ade_m": mean([float(r["ade_m"]) for r in rows]),
                "mean_forward_error_m": mean([float(r["forward_error_m"]) for r in rows]),
                "mean_lateral_left_error_m": mean([float(r["lateral_left_error_m"]) for r in rows]),
                "mean_combined_abs_rel_error": mean([float(r["combined_abs_rel_error"]) for r in rows]),
                "mean_combined_endpoint_distance_rel_error": mean(
                    [float(r["combined_endpoint_distance_rel_error"]) for r in rows]
                ),
                "mean_combined_endpoint_distance_error_m": mean(
                    [float(r["combined_endpoint_distance_error_m"]) for r in rows]
                ),
            }
        )
    return aggregate_rows


def add_ranks(rows: list[dict[str, object]], rank_by: str) -> list[dict[str, object]]:
    rows = [dict(row) for row in rows]
    rows.sort(key=lambda row: (float(row[rank_by]), str(row["model"])))
    last_value = None
    last_rank = 0
    for idx, row in enumerate(rows, start=1):
        value = float(row[rank_by])
        if last_value is None or abs(value - last_value) > 1e-12:
            last_rank = idx
            last_value = value
        row["rank"] = last_rank
    return rows


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            if abs(value) < 1:
                return f"{value:.4f}"
            return f"{value:.2f}"
        return str(value)

    table = [[fmt(row.get(col, "")) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(row[idx]) for row in table)) if table else len(col)
        for idx, col in enumerate(columns)
    ]
    print(" ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))
    print("-" * (sum(widths) + len(widths) - 1))
    for row in table:
        print(" ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models by generated ego-trajectory distance and displacement errors."
    )
    parser.add_argument("--input-root", type=Path, default=Path("drivinggen_input_dataset"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--models", nargs="*", help="Model output dirs or stripped model names to include.")
    parser.add_argument("--scenes", nargs="*", help="Scene IDs, comma lists, or text files with one scene per line.")
    parser.add_argument("--limit-models", type=positive_int, help="Process only the first N discovered models.")
    parser.add_argument("--limit-scenes", type=positive_int, help="Process only the first N discovered scenes.")
    parser.add_argument("--common-scenes", action="store_true",
                        help="Rank only on scenes that have GT and predicted ego trajectories for every selected model.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing files instead of skipping jobs.")
    parser.add_argument("--alignment-mode", choices=("none", "rotate", "rotate-scale"), default="none",
                        help="Default preserves generated forward/back and left/right signs.")
    parser.add_argument("--relative-floor-m", type=float, default=DEFAULT_RELATIVE_FLOOR_M,
                        help="Denominator floor for relative errors. Prevents near-static scenes from exploding.")
    parser.add_argument(
        "--rank-by",
        default="mean_combined_endpoint_distance_rel_error",
        choices=(
            "mean_combined_endpoint_distance_rel_error",
            "mean_combined_endpoint_distance_error_m",
            "mean_combined_abs_rel_error",
            "mean_distance_abs_rel_error",
            "mean_distance_abs_error_m",
            "mean_displacement_abs_rel_error",
            "mean_displacement_abs_error_m",
            "mean_endpoint_error_m",
            "mean_endpoint_rel_error",
            "mean_ade_m",
        ),
        help="Aggregate metric to rank by. Lower is better for all options.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("benchmark_results/ego_trajectory_distance_ranking"))
    parser.add_argument("--no-write", action="store_true", help="Only print; do not write CSV/JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.relative_floor_m < 0:
        raise ValueError("--relative-floor-m must be non-negative")

    requested_models = set(args.models) if args.models else None
    scenes = parse_scene_args(args.scenes)
    model_dirs = discover_models(args.benchmark_root, requested_models)
    if args.limit_models is not None:
        model_dirs = model_dirs[: args.limit_models]
    if not model_dirs:
        raise RuntimeError(f"No model benchmark directories found under {args.benchmark_root}")

    scene_ids = discover_scenes(model_dirs, scenes)
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]
    if not scene_ids:
        raise RuntimeError("No scenes selected.")

    jobs, warnings = build_jobs(
        model_dirs=model_dirs,
        scenes=scene_ids,
        input_root=args.input_root,
        common_scenes=args.common_scenes,
        strict=args.strict,
    )
    for warning in warnings:
        print("WARNING:", warning)
    if not jobs:
        raise RuntimeError("No trajectory jobs to evaluate.")

    scene_rows: list[dict[str, object]] = []
    for job in jobs:
        scene_rows.append(compute_scene_metrics(job, args.alignment_mode, args.relative_floor_m))

    aggregate_rows = aggregate_model_metrics(scene_rows)
    ranked_rows = add_ranks(aggregate_rows, args.rank_by)

    print(f"Evaluated {len(scene_rows)} model-scene trajectories.")
    print(f"Models: {len(aggregate_rows)}")
    print(f"Scene selection: {'common scenes only' if args.common_scenes else 'all available scenes'}")
    print(f"Alignment mode: {args.alignment_mode}")
    print(f"Relative denominator floor: {args.relative_floor_m:g} m")
    print(f"Rank metric: {args.rank_by} (lower is better)")
    print()

    print_table(
        ranked_rows,
        [
            "rank",
            "model",
            "num_scenes",
            "mean_combined_endpoint_distance_rel_error",
            "mean_distance_abs_rel_error",
            "mean_endpoint_rel_error",
            "mean_combined_endpoint_distance_error_m",
            "mean_distance_abs_error_m",
            "mean_endpoint_error_m",
            "mean_forward_error_m",
            "mean_lateral_left_error_m",
        ],
    )

    if not args.no_write:
        args.outdir.mkdir(parents=True, exist_ok=True)
        write_csv(args.outdir / "model_ranking.csv", ranked_rows)
        write_csv(args.outdir / "scene_metrics.csv", scene_rows)
        with (args.outdir / "model_ranking.json").open("w") as f:
            json.dump(ranked_rows, f, indent=2)
        with (args.outdir / "scene_metrics.json").open("w") as f:
            json.dump(scene_rows, f, indent=2)
        print()
        print(f"Wrote: {args.outdir / 'model_ranking.csv'}")
        print(f"Wrote: {args.outdir / 'scene_metrics.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
