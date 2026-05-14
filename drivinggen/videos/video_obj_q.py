"""
video_obj_q.py
==============
Objective Quality evaluation for the DrivingGen benchmark.

Implements the IEEE P2020 objective video quality metrics (``p2020_v2`` module)
to assess perceptual quality without a reference video.  The P2020 standard
defines a suite of metrics targeting display and streaming artefacts common in
generated content:

  * **mmp_alias** (Modulation Mitigation Probability for aliasing) — the key
    metric returned by this module.  It measures temporal flickering and
    high-frequency spatial aliasing introduced by generative models.  Lower
    raw ``mmp_alias`` values indicate more flickering; after inversion
    (``1 - mmp_alias``) higher scores indicate better quality.

  * Various per-frame sharpness, noise, colour, and blocking metrics — all
    computed but not returned by the public API.

Processing is parallelised with ``ProcessPoolExecutor`` so that multiple
videos are evaluated concurrently across CPU cores.

Design note:
    The ``_process_one_video`` function is defined at module scope (not as a
    lambda or nested function) because ``ProcessPoolExecutor`` requires
    picklable callables for inter-process serialisation.
"""

# Standard-library imports
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

# Third-party imports
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Local P2020 metric implementations (bundled with the benchmark)
from .p2020_v2 import single_frame_metrics as single_frame_metrics_v2
from .p2020_v2 import video_metrics         as video_metrics_v2


# ---------------------------------------------------------------------------
# Per-video worker function (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _process_one_video(images: list) -> Tuple[dict, dict]:
    """
    Compute IEEE P2020 objective quality metrics for a single video.

    Two categories of metrics are computed:

    * **Video-level metrics** (``adj_metric``): Computed over the entire
      frame sequence (e.g. temporal flicker, inter-frame aliasing).  These
      capture artefacts that span multiple frames and cannot be detected by
      looking at individual frames in isolation.

    * **Frame-level metrics** (``frame_mean``): Computed independently for
      each frame (e.g. sharpness, noise level, colour gamut).  The
      per-frame values are averaged with ``nanmean`` so that frames that
      produce NaN (e.g. blank frames) do not bias the result.

    This function is executed inside a worker process spawned by
    ``ProcessPoolExecutor``, so it must be self-contained (no shared state
    with the parent process).

    Args:
        images (list[str]): Ordered list of frame file paths for one video.

    Returns:
        Tuple[dict, dict]:
            - **adj_metric_dict**: Video-level metric values keyed by metric
              name (str → float).
            - **frame_mean_dict**: Per-frame metric values averaged over all
              frames (str → float).
    """
    # Read all frames from disk; cv2.imread returns BGR uint8 (H, W, 3)
    images_np = [cv2.imread(f) for f in images]
    # images_np: list of T arrays, each shape: (H, W, 3)

    # --- Video-level metrics ---
    # video_metrics_v2 expects a list of BGR frames and returns a flat dict
    adj_metric = video_metrics_v2(images_np)
    # adj_metric: {metric_name: scalar_value}

    # --- Per-frame metrics averaged across frames ---
    this_dict = defaultdict(list)  # accumulate per-frame values per metric
    for image in images_np:
        img_metric = single_frame_metrics_v2(image)
        # img_metric: {metric_name: scalar_value} for this single frame
        for k, v in img_metric.items():
            this_dict[k].append(v)   # collect across all T frames

    # Average each metric across frames; nanmean tolerates invalid frames
    frame_mean = {
        k: float(np.nanmean(np.asarray(v)))
        for k, v in this_dict.items()
    }
    # frame_mean: {metric_name: float}

    return adj_metric, frame_mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_objective_quality_v2(video_list: list) -> float:
    """
    Compute dataset-level objective quality and return the ``mmp_alias`` score.

    Overview:
      1. Dispatch ``_process_one_video`` in parallel for all videos using
         ``ProcessPoolExecutor`` (one worker per CPU core minus one).
      2. Aggregate per-video metric values into dataset-level lists.
      3. Average each metric list with ``nanmean``.
      4. Invert ``fmp_alias``-family metrics (lower raw = more flickering →
         ``1 - raw`` makes higher = better, consistent with other metrics).
      5. Return the ``mmp_alias`` score as the canonical objective quality
         value reported by DrivingGen.

    Why ``mmp_alias``?
    ------------------
    Temporal aliasing / flickering is a common failure mode of video
    generation models: the model may produce frames that are individually
    plausible but incoherent across time, leading to high-frequency
    temporal oscillations visible as "shimmer".  The P2020 MMP-alias metric
    specifically targets this artefact and is therefore the most
    discriminative single metric for comparing video generation quality.

    Args:
        video_list (list[list[str]]): Outer list over videos; each inner
            list contains frame file paths in chronological order.

    Returns:
        float: Dataset-level ``mmp_alias`` score (after inversion).
            Higher values indicate less temporal flickering (better quality).
    """
    # Accumulators for metric values across all videos
    val_dict       = defaultdict(list)  # frame-level metrics: {name: [v1, v2, ...]}
    video_val_dict = defaultdict(list)  # video-level metrics: {name: [v1, v2, ...]}

    print('=========================Start Objective Quality=========================')

    # Use all available CPU cores minus one to avoid starving the main process
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f'Using {max_workers} worker(s)')

    # --- Parallel evaluation ---
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # ex.map preserves submission order; results are yielded as they complete
        for adj_metric, frame_mean in tqdm(
            ex.map(_process_one_video, video_list),
            total=len(video_list)
        ):
            # Accumulate video-level metrics
            for k, v in adj_metric.items():
                video_val_dict[k].append(v)   # one entry per video

            # Accumulate frame-averaged metrics
            for k, v in frame_mean.items():
                val_dict[k].append(v)         # one entry per video

    # Convert defaultdicts to plain dicts for clarity
    val_dict       = dict(val_dict)        # {metric_name: [float, ...]}
    video_val_dict = dict(video_val_dict)  # {metric_name: [float, ...]}

    # --- Aggregate and normalise ---
    normed_score_infer: Dict[str, float] = {}

    # Frame-level metrics
    for k, v in val_dict.items():
        score = np.nanmean(np.array(v))   # dataset mean of per-video means
        if 'fmp_alias' in k:
            # fmp_alias family: lower raw value = more flickering;
            # invert so that higher score = better quality
            score = 1 - score
        normed_score_infer[k] = score
        print(f'{k} score mean: {np.nanmean(score)}')

    # Video-level metrics
    for k, v in video_val_dict.items():
        score = np.nanmean(np.array(v))   # dataset mean
        log   = np.array(v)               # raw values (retained for reference)
        if 'fmp_alias' in k:
            score = 1 - score             # same inversion as above
        normed_score_infer[k] = score
        print(f'video {k} score mean: {np.nanmean(score)}')

    # Compute an overall average across all metrics (informational)
    avg = np.mean(np.array([v for v in normed_score_infer.values()]))
    normed_score_infer['avg'] = avg

    print(normed_score_infer)

    # Return the canonical metric: inverted mmp_alias (temporal flickering)
    return float(normed_score_infer['mmp_alias'])
