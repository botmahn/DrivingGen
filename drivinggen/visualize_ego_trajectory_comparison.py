#!/usr/bin/env python3
"""
Create side-by-side benchmark videos for DrivingGen ego trajectories.

For each selected model and scene, the output video stacks:
    1. GT frames from drivinggen_input_dataset/ego_condition/imgs/<scene>.jpg
       followed by drivinggen_input_dataset/videos-fvd/<scene>/
    2. Generated video from benchmark_results/<model>/generated_videos/<scene>.mp4
    3. An animated BEV trajectory overlay:
       GT in green, predicted ego trajectory in red.

Coordinate convention:
    * GT .npy files are global XY positions. They are translated to frame 0
      and rotated into an ego frame where coordinate 0 is forward and
      coordinate 1 is lateral-left.
    * UniDepth SLAM locs are (x_right, z_forward). They are converted to
      (z_forward, -x_right). By default no best-fit rotation is applied, so
      generated left/right and forward/backward signs are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


GT_COLOR = (40, 190, 40)      # BGR: green
PRED_COLOR = (30, 45, 220)    # BGR: red
TEXT_COLOR = (245, 245, 245)
INK_COLOR = (40, 44, 52)
GRID_COLOR = (225, 229, 235)
AXIS_COLOR = (155, 162, 172)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class RenderJob:
    model_dir: Path
    model_name: str
    scene: str
    generated_video: Path
    gt_frame_dir: Path
    gt_condition_frame: Path | None
    gt_traj_path: Path
    pred_traj_path: Path
    output_path: Path


@dataclass
class TrajectoryViewport:
    width: int
    height: int
    pad: int
    lateral_mid: float
    forward_mid: float
    scale: float

    def to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        forward, lateral = float(point[0]), float(point[1])
        # Image x grows rightward, while positive lateral means vehicle-left.
        x = self.width * 0.5 - (lateral - self.lateral_mid) * self.scale
        y = self.height * 0.5 - (forward - self.forward_mid) * self.scale
        return int(round(x)), int(round(y))


def eprint(*items: object) -> None:
    print(*items, file=sys.stderr)


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


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
        if not (model_dir / "generated_videos").is_dir():
            continue

        names = {model_dir.name, strip_model_suffix(model_dir.name)}
        if requested and names.isdisjoint(requested):
            continue
        model_dirs.append(model_dir)
    return model_dirs


def discover_scenes(model_dirs: Iterable[Path], requested: list[str] | None) -> list[str]:
    if requested:
        return requested

    scenes: set[str] = set()
    for model_dir in model_dirs:
        for video_path in (model_dir / "generated_videos").glob("*.mp4"):
            scenes.add(video_path.stem)
    return sorted(scenes)


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or math.isnan(fps):
        fps = 10.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {path}: {width}x{height}")
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def find_pred_traj(model_dir: Path, scene: str) -> Path | None:
    scene_dir = model_dir / "benchmarks" / scene
    if not scene_dir.is_dir():
        return None
    matches = sorted(scene_dir.glob("**/unidepth-estimate_ego_traj.pkl"))
    return matches[0] if matches else None


def build_jobs(
    model_dirs: list[Path],
    scenes: list[str],
    input_root: Path,
    output_root: Path,
    strict: bool,
) -> tuple[list[RenderJob], list[str]]:
    warnings: list[str] = []
    jobs: list[RenderJob] = []

    for model_dir in model_dirs:
        model_name = strip_model_suffix(model_dir.name)
        model_out = output_root / model_name
        for scene in scenes:
            generated_video = model_dir / "generated_videos" / f"{scene}.mp4"
            gt_frame_dir = input_root / "videos-fvd" / scene
            gt_condition_frame = input_root / "ego_condition" / "imgs" / f"{scene}.jpg"
            gt_traj_path = input_root / "ego_condition" / "ego_motion" / f"{scene}.npy"
            pred_traj_path = find_pred_traj(model_dir, scene)

            missing = []
            if not generated_video.is_file():
                missing.append(str(generated_video))
            if not gt_frame_dir.is_dir():
                missing.append(str(gt_frame_dir))
            if not gt_traj_path.is_file():
                missing.append(str(gt_traj_path))
            if pred_traj_path is None:
                missing.append(str(model_dir / "benchmarks" / scene / "**/unidepth-estimate_ego_traj.pkl"))

            if missing:
                msg = f"Skipping {model_name}/{scene}: missing " + ", ".join(missing)
                if strict:
                    raise FileNotFoundError(msg)
                warnings.append(msg)
                continue

            jobs.append(
                RenderJob(
                    model_dir=model_dir,
                    model_name=model_name,
                    scene=scene,
                    generated_video=generated_video,
                    gt_frame_dir=gt_frame_dir,
                    gt_condition_frame=gt_condition_frame if gt_condition_frame.is_file() else None,
                    gt_traj_path=gt_traj_path,
                    pred_traj_path=pred_traj_path,
                    output_path=model_out / f"{scene}.mp4",
                )
            )

    return jobs, warnings


def sorted_gt_frames(gt_frame_dir: Path, condition_frame: Path | None = None) -> list[Path]:
    frames = [
        path
        for path in gt_frame_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frames = sorted(frames)

    if condition_frame is not None and condition_frame.is_file():
        first_stem = frames[0].stem if frames else ""
        if first_stem not in {"0", "00000", "000000"}:
            frames = [condition_frame] + frames
    return frames


def read_gt_frame(frame_paths: list[Path], index: int, fallback_shape: tuple[int, int]) -> np.ndarray:
    if not frame_paths:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    index = max(0, min(index, len(frame_paths) - 1))
    frame = cv2.imread(str(frame_paths[index]), cv2.IMREAD_COLOR)
    if frame is None:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    return frame


def sample_index(i: int, out_count: int, source_count: int) -> int:
    if source_count <= 1 or out_count <= 1:
        return 0
    return int(round(i * (source_count - 1) / (out_count - 1)))


def resize_exact(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_panel_label(frame: np.ndarray, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, frame.shape[0] / 900.0)
    thickness = max(1, int(round(frame.shape[0] / 420.0)))
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = 14, 16 + th
    cv2.rectangle(
        frame,
        (x - 7, y - th - 8),
        (x + tw + 7, y + baseline + 8),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(frame, label, (x, y), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def load_gt_trajectory(path: Path) -> np.ndarray:
    gt = np.load(path, allow_pickle=True)
    gt = np.asarray(gt, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] < 2:
        raise ValueError(f"GT trajectory must have shape (T, >=2): {path} has {gt.shape}")
    return gt[:, :2]


def load_pred_trajectory(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "locs" not in payload:
        raise ValueError(f"Predicted trajectory pickle must contain dict key 'locs': {path}")
    locs = np.asarray(payload["locs"], dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] < 2:
        raise ValueError(f"Predicted locs must have shape (T, >=2): {path} has {locs.shape}")
    return locs[:, :2]


def gt_global_xy_to_ego_forward_lateral(
    gt_xy: np.ndarray,
    k_ahead: int = 1,
    min_step: float = 1.0,
) -> np.ndarray:
    """Match z-sample_ftd.py gt_2_ego, returning (forward, lateral-left)."""
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

    if np.linalg.norm(heading) < 1e-9:
        theta = 0.0
    else:
        theta = float(np.arctan2(heading[1], heading[0]))

    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rel @ rotation


def slam_xright_zforward_to_forward_lateral(locs: np.ndarray) -> np.ndarray:
    locs = np.asarray(locs, dtype=np.float64)
    out = np.empty((len(locs), 2), dtype=np.float64)
    out[:, 0] = locs[:, 1]   # z_forward
    out[:, 1] = -locs[:, 0]  # lateral-left
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


def smooth_traj_moving_average(traj: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(traj) < 3:
        return traj
    window = min(window, len(traj))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return traj

    radius = window // 2
    padded = np.pad(traj, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(traj)
    for dim in range(traj.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def align_pred_to_gt_origin_fixed(
    pred_forward_lateral: np.ndarray,
    gt_forward_lateral: np.ndarray,
    allow_scale: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Origin-fixed Umeyama alignment, matching z-sample_ftd.py behavior."""
    pred = np.asarray(pred_forward_lateral, dtype=np.float64)
    gt = np.asarray(gt_forward_lateral, dtype=np.float64)
    n = min(len(pred), len(gt))
    if n < 2:
        return pred - pred[:1], 1.0, np.eye(2)

    pred_rel = pred[:n] - pred[:1]
    gt_rel = gt[:n]
    finite = np.isfinite(pred_rel).all(axis=1) & np.isfinite(gt_rel).all(axis=1)
    if finite.sum() < 2:
        return pred - pred[:1], 1.0, np.eye(2)

    p = pred_rel[finite]
    q = gt_rel[finite]
    covariance = p.T @ q / len(p)
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return pred - pred[:1], 1.0, np.eye(2)

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[1] *= -1
        rotation = vt.T @ u.T

    scale = 1.0
    if allow_scale:
        var_p = float((p ** 2).sum() / len(p))
        if var_p > 1e-12:
            scale = float((q * (rotation @ p.T).T).sum() / (len(p) * var_p))

    pred_full_rel = pred - pred[:1]
    aligned = (scale * (rotation @ pred_full_rel.T)).T
    return aligned, scale, rotation


def prepare_trajectories(
    gt_traj_path: Path,
    pred_traj_path: Path,
    frame_count: int,
    alignment_mode: str,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    gt_global = load_gt_trajectory(gt_traj_path)
    pred_locs = load_pred_trajectory(pred_traj_path)

    gt_local = gt_global_xy_to_ego_forward_lateral(gt_global)
    pred_local = slam_xright_zforward_to_forward_lateral(pred_locs)

    gt_local = resample_traj(gt_local, frame_count)
    pred_local = resample_traj(pred_local, frame_count)
    if alignment_mode == "none":
        pred_aligned = pred_local - pred_local[:1]
        scale = 1.0
    elif alignment_mode == "rotate":
        pred_aligned, scale, _ = align_pred_to_gt_origin_fixed(pred_local, gt_local, allow_scale=False)
    elif alignment_mode == "rotate-scale":
        pred_aligned, scale, _ = align_pred_to_gt_origin_fixed(pred_local, gt_local, allow_scale=True)
    else:
        raise ValueError(f"Unknown alignment mode: {alignment_mode}")

    if smooth_window > 1:
        gt_local = smooth_traj_moving_average(gt_local, smooth_window)
        pred_aligned = smooth_traj_moving_average(pred_aligned, smooth_window)

    return gt_local, pred_aligned, scale


def nice_grid_step(world_range: float) -> float:
    if world_range <= 0:
        return 1.0
    raw = world_range / 5.0
    exponent = math.floor(math.log10(raw))
    base = raw / (10 ** exponent)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exponent)


def make_viewport(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    width: int,
    height: int,
    min_plot_range: float,
) -> TrajectoryViewport:
    points = np.vstack([gt_traj[:, :2], pred_traj[:, :2], np.zeros((1, 2))])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        points = np.zeros((1, 2), dtype=np.float64)

    forward_min, lateral_min = points.min(axis=0)
    forward_max, lateral_max = points.max(axis=0)

    forward_range = max(float(forward_max - forward_min), min_plot_range)
    lateral_range = max(float(lateral_max - lateral_min), min_plot_range)
    forward_mid = float((forward_min + forward_max) * 0.5)
    lateral_mid = float((lateral_min + lateral_max) * 0.5)

    pad = max(42, int(round(min(width, height) * 0.09)))
    usable_w = max(1, width - 2 * pad)
    usable_h = max(1, height - 2 * pad)
    scale = min(usable_w / lateral_range, usable_h / forward_range)

    return TrajectoryViewport(
        width=width,
        height=height,
        pad=pad,
        lateral_mid=lateral_mid,
        forward_mid=forward_mid,
        scale=scale,
    )


def draw_grid(canvas: np.ndarray, viewport: TrajectoryViewport) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (210, 214, 220), 1, cv2.LINE_AA)

    lateral_span = (w - 2 * viewport.pad) / viewport.scale
    forward_span = (h - 2 * viewport.pad) / viewport.scale
    lat_min = viewport.lateral_mid - lateral_span * 0.5
    lat_max = viewport.lateral_mid + lateral_span * 0.5
    fwd_min = viewport.forward_mid - forward_span * 0.5
    fwd_max = viewport.forward_mid + forward_span * 0.5
    step = nice_grid_step(max(lateral_span, forward_span))

    lat = math.floor(lat_min / step) * step
    while lat <= lat_max + 1e-9:
        x = viewport.to_pixel(np.array([viewport.forward_mid, lat]))[0]
        cv2.line(canvas, (x, viewport.pad), (x, h - viewport.pad), GRID_COLOR, 1, cv2.LINE_AA)
        lat += step

    fwd = math.floor(fwd_min / step) * step
    while fwd <= fwd_max + 1e-9:
        y = viewport.to_pixel(np.array([fwd, viewport.lateral_mid]))[1]
        cv2.line(canvas, (viewport.pad, y), (w - viewport.pad, y), GRID_COLOR, 1, cv2.LINE_AA)
        fwd += step

    origin = viewport.to_pixel(np.array([0.0, 0.0]))
    cv2.line(canvas, (origin[0], viewport.pad), (origin[0], h - viewport.pad), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (viewport.pad, origin[1]), (w - viewport.pad, origin[1]), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.circle(canvas, origin, 4, INK_COLOR, -1, cv2.LINE_AA)

    arrow_start = (w - viewport.pad - 34, h - viewport.pad)
    arrow_end = (w - viewport.pad - 34, h - viewport.pad - 52)
    cv2.arrowedLine(canvas, arrow_start, arrow_end, INK_COLOR, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "Forward", (arrow_end[0] - 38, arrow_end[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, INK_COLOR, 1, cv2.LINE_AA)


def draw_legend(canvas: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 16, 30
    cv2.rectangle(canvas, (8, 8), (132, 72), (255, 255, 255), -1)
    cv2.rectangle(canvas, (8, 8), (132, 72), (212, 216, 222), 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y), (x + 28, y), GT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (x + 36, y + 5), font, 0.5, INK_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + 28), (x + 28, y + 28), PRED_COLOR, 3, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (x + 36, y + 33), font, 0.5, INK_COLOR, 1, cv2.LINE_AA)


def draw_polyline(
    canvas: np.ndarray,
    traj: np.ndarray,
    frame_index: int,
    viewport: TrajectoryViewport,
    color: tuple[int, int, int],
) -> None:
    upto = max(0, min(frame_index + 1, len(traj)))
    if upto <= 0:
        return

    pts = np.array([viewport.to_pixel(point) for point in traj[:upto]], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[-1]), 6, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[-1]), 7, (255, 255, 255), 1, cv2.LINE_AA)


def render_trajectory_panel(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    frame_index: int,
    viewport: TrajectoryViewport,
) -> np.ndarray:
    canvas = np.full((viewport.height, viewport.width, 3), 248, dtype=np.uint8)
    draw_grid(canvas, viewport)
    draw_polyline(canvas, gt_traj, frame_index, viewport, GT_COLOR)
    draw_polyline(canvas, pred_traj, frame_index, viewport, PRED_COLOR)
    draw_legend(canvas)
    draw_panel_label(canvas, "Ego trajectory")
    return canvas


def open_writer(path: Path, fps: float, width: int, height: int, codec: str) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path} with codec {codec}")
    return writer


def render_job(
    job: RenderJob,
    panel_height: int,
    traj_width: int | None,
    codec: str,
    overwrite: bool,
    alignment_mode: str,
    smooth_window: int,
    min_plot_range: float,
    max_frames: int | None,
) -> dict[str, object]:
    if job.output_path.exists() and not overwrite:
        return {"status": "skipped_existing", "output": str(job.output_path)}

    info = video_info(job.generated_video)
    frame_count = info.frame_count
    if frame_count <= 0:
        raise RuntimeError(f"Could not read frame count for {job.generated_video}")
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    frame_paths = sorted_gt_frames(job.gt_frame_dir, job.gt_condition_frame)
    if not frame_paths:
        raise RuntimeError(f"No GT frames found under {job.gt_frame_dir}")

    first_gt = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_gt is None:
        raise RuntimeError(f"Could not read first GT frame: {frame_paths[0]}")
    gt_h, gt_w = first_gt.shape[:2]

    panel_height = make_even(panel_height)
    gt_panel_w = make_even(max(2, int(round(gt_w * panel_height / gt_h))))
    gen_panel_w = make_even(max(2, int(round(info.width * panel_height / info.height))))
    traj_panel_w = make_even(traj_width if traj_width is not None else panel_height)
    out_w = make_even(gt_panel_w + gen_panel_w + traj_panel_w)
    out_h = panel_height

    gt_traj, pred_traj, scale = prepare_trajectories(
        job.gt_traj_path,
        job.pred_traj_path,
        frame_count,
        alignment_mode=alignment_mode,
        smooth_window=smooth_window,
    )
    viewport = make_viewport(gt_traj, pred_traj, traj_panel_w, panel_height, min_plot_range)

    cap = cv2.VideoCapture(str(job.generated_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {job.generated_video}")
    writer = open_writer(job.output_path, info.fps, out_w, out_h, codec)

    last_gen = np.zeros((info.height, info.width, 3), dtype=np.uint8)
    for i in range(frame_count):
        ok, gen_frame = cap.read()
        if ok and gen_frame is not None:
            last_gen = gen_frame
        else:
            gen_frame = last_gen

        gt_idx = sample_index(i, frame_count, len(frame_paths))
        gt_frame = read_gt_frame(frame_paths, gt_idx, fallback_shape=(gt_h, gt_w))

        gt_panel = resize_exact(gt_frame, gt_panel_w, panel_height)
        gen_panel = resize_exact(gen_frame, gen_panel_w, panel_height)
        traj_panel = render_trajectory_panel(gt_traj, pred_traj, i, viewport)

        draw_panel_label(gt_panel, "GT video")
        draw_panel_label(gen_panel, "Generated video")

        stacked = np.hstack([gt_panel, gen_panel, traj_panel])
        if stacked.shape[1] != out_w:
            pad_w = out_w - stacked.shape[1]
            if pad_w > 0:
                stacked = np.hstack([stacked, np.zeros((out_h, pad_w, 3), dtype=np.uint8)])
            else:
                stacked = stacked[:, :out_w]
        writer.write(stacked)

    writer.release()
    cap.release()

    return {
        "status": "rendered",
        "model": job.model_name,
        "scene": job.scene,
        "output": str(job.output_path),
        "fps": info.fps,
        "frames": frame_count,
        "alignment_scale": scale,
        "alignment_mode": alignment_mode,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT/generated/ego-trajectory comparison videos for DrivingGen benchmark outputs."
    )
    parser.add_argument("--input-root", type=Path, default=Path("drivinggen_input_dataset"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/trajectory_comparison_videos"))
    parser.add_argument("--models", nargs="*", help="Model output dirs or stripped model names to include.")
    parser.add_argument("--scenes", nargs="*", help="Scene IDs, comma lists, or text files with one scene per line.")
    parser.add_argument("--limit-models", type=positive_int, help="Process only the first N discovered models.")
    parser.add_argument("--limit-scenes", type=positive_int, help="Process only the first N discovered scenes.")
    parser.add_argument("--max-frames", type=positive_int, help="Render only the first N frames of each output.")
    parser.add_argument("--panel-height", type=positive_int, default=576)
    parser.add_argument("--traj-width", type=positive_int, help="Trajectory panel width. Defaults to panel height.")
    parser.add_argument("--min-plot-range", type=float, default=1.0,
                        help="Minimum forward/lateral world span in metres for the trajectory panel.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, e.g. mp4v, avc1, H264.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing files instead of skipping jobs.")
    parser.add_argument("--alignment-mode", choices=("none", "rotate", "rotate-scale"),
                        help=("Trajectory alignment mode. Default 'none' preserves generated left/right and "
                              "forward/backward signs. 'rotate' matches the old origin-fixed rotation; "
                              "'rotate-scale' also estimates isotropic scale."))
    parser.add_argument("--allow-scale", action="store_true",
                        help="Deprecated alias for --alignment-mode rotate-scale.")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Optional odd moving-average window over trajectories. Default 1 disables smoothing.")
    parser.add_argument("--manifest", type=Path,
                        help="Optional JSON path for render status records. Defaults to <output-dir>/manifest.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    alignment_mode = args.alignment_mode
    if alignment_mode is None:
        alignment_mode = "rotate-scale" if args.allow_scale else "none"
    elif args.allow_scale and alignment_mode != "rotate-scale":
        eprint("WARNING: --allow-scale is ignored because --alignment-mode was set explicitly.")

    requested_models = set(args.models) if args.models else None
    scenes = parse_scene_args(args.scenes)

    model_dirs = discover_models(args.benchmark_root, requested_models)
    if args.limit_models is not None:
        model_dirs = model_dirs[: args.limit_models]
    if not model_dirs:
        eprint(f"No model output directories found under {args.benchmark_root}")
        return 2

    scene_ids = discover_scenes(model_dirs, scenes)
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]
    if not scene_ids:
        eprint("No scenes selected.")
        return 2

    jobs, warnings = build_jobs(
        model_dirs=model_dirs,
        scenes=scene_ids,
        input_root=args.input_root,
        output_root=args.output_dir,
        strict=args.strict,
    )
    for warning in warnings:
        eprint("WARNING:", warning)
    if not jobs:
        eprint("No render jobs to run.")
        return 2

    results: list[dict[str, object]] = []
    print(f"Rendering {len(jobs)} comparison videos to {args.output_dir}")
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.model_name} :: {job.scene}")
        try:
            result = render_job(
                job=job,
                panel_height=args.panel_height,
                traj_width=args.traj_width,
                codec=args.codec,
                overwrite=args.overwrite,
                alignment_mode=alignment_mode,
                smooth_window=args.smooth_window,
                min_plot_range=args.min_plot_range,
                max_frames=args.max_frames,
            )
        except Exception as exc:
            if args.strict:
                raise
            result = {
                "status": "failed",
                "model": job.model_name,
                "scene": job.scene,
                "error": str(exc),
            }
            eprint("ERROR:", job.model_name, job.scene, exc)
        results.append(result)

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump({"jobs": results, "warnings": warnings}, f, indent=2)

    rendered = sum(1 for item in results if item.get("status") == "rendered")
    skipped = sum(1 for item in results if item.get("status") == "skipped_existing")
    failed = sum(1 for item in results if item.get("status") == "failed")
    print(f"Done. rendered={rendered} skipped_existing={skipped} failed={failed}")
    print(f"Manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
