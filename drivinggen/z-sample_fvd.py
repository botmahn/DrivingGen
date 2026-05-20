"""
z-sample_fvd.py
===============
Main video evaluation orchestration script for the DrivingGen benchmark.

This script computes one or more quality / consistency metrics for a set of
generated driving-scene videos and prints the results to stdout.  Each metric
is gated by the ``--metric`` command-line flag so that expensive GPU
computations can be run selectively.

Supported metrics
-----------------
fvd         Frechet Video Distance — compares feature distributions of
            generated and reference 100-frame clips (get_fvd).
obj_q       Objective perceptual quality: P3020 MMP metrics — MTF50, edge
            rise-time, chromatic aberration, etc. (get_objective_quality_v2).
sub_q       Subjective perceptual quality: CLIP-IQA+ per-frame score
            (get_subjective_quality).
v_consist   Visual / scene consistency: DINOv2 feature similarity + optical
            flow magnitude across consecutive frames (get_scene_consistency_v3).
a_consist   Agent appearance consistency: DINOv2 crop similarity for each
            tracked agent across frames (get_agent_consistency).
a_missing   Agent missing rate: Cosmos-Reason1 VLM binary check for whether
            each expected agent is visible in the generated video
            (get_agent_missing).
all         Run all of the above in sequence.

Input layout
------------
Generated videos are expected to live at:
    {root_path}/{scene}/{model_name}/{exp_id}/images/

Agent metadata (bounding boxes and class labels) is read from pre-computed
pkl files produced by extract_traj_agent_unidepth.py:
    {outdir}/{scene}/{model_name}/{exp_id}/unidepth-estimate_agents_bbox.pkl
    {outdir}/{scene}/{model_name}/{exp_id}/unidepth-estimate_agents_bbox_label.pkl

Usage
-----
    python z-sample_fvd.py \
        --root_path  /data/generated \
        --gt_path    /data/gt_meta.json \
        --outdir     /data/output \
        --model_name cosmos \
        --exp_id     free \
        --metric     fvd
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import argparse
import glob
import json
import math
import os
import pickle
import random
import sys

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import cv2
import matplotlib
matplotlib.use('Agg')           # use non-interactive backend (must precede pyplot import)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------
from decord import VideoReader   # fast GPU-accelerated video decoder

# ---------------------------------------------------------------------------
# DrivingGen metric modules
# ---------------------------------------------------------------------------
from videos.video_distribution import get_fvd              # Frechet Video Distance
from videos.video_obj_q        import get_objective_quality_v2  # P3020 MMP metrics
from videos.video_sub_q        import get_subjective_quality    # CLIP-IQA+
from videos.video_v_consist    import get_scene_consistency_v3  # DINOv2 + optical flow
from videos.video_a_consist    import get_agent_consistency     # agent appearance consistency
from videos.video_a_missing    import get_agent_missing         # agent missing rate (VLM)


# ---------------------------------------------------------------------------
# Video / image loading utilities
# ---------------------------------------------------------------------------

def get_video(video_path):
    """
    Load an MP4 video file and return all frames as a floating-point tensor.

    The decord ``VideoReader`` is used for efficient sequential decoding;
    each frame is converted from ``uint8`` in [0, 255] to ``float32`` in
    [0, 1] before stacking.

    Parameters
    ----------
    video_path : str
        Absolute path to the MP4 file.

    Returns
    -------
    video_frames : torch.Tensor, shape (T, H, W, 3), dtype float32
        All decoded frames in RGB order, normalised to [0, 1].
        T = total frame count, H = frame height, W = frame width.
    """
    # Decode all frames from the MP4 file; each element is a decord AVFrame
    video_frames = VideoReader(video_path)

    # Convert each frame to a float32 torch tensor and normalise to [0, 1]
    # v.asnumpy() : shape (H, W, 3), uint8
    # torch.from_numpy(…) : shape (H, W, 3), uint8
    # / 255.0             : shape (H, W, 3), float32 in [0, 1]
    video_frames = [
        (torch.from_numpy(v.asnumpy()).to(torch.float32)) / 255.
        for v in video_frames
    ]

    # Stack the list of per-frame tensors into a single tensor
    # video_frames : shape (T, H, W, 3), float32, values in [0, 1]
    video_frames = torch.stack(video_frames)

    return video_frames   # shape: (T, H, W, 3)


def get_imgs(video_path):
    """
    Return a sorted list of absolute image file paths from a directory.

    The sort is lexicographic, which is equivalent to chronological order
    when frames are saved with zero-padded numeric filenames (e.g.
    ``00000.png``, ``00001.png``, …).

    Parameters
    ----------
    video_path : str
        Absolute path to the directory containing the frame images.

    Returns
    -------
    imgs : list of str
        Sorted absolute paths to every file in ``video_path``.
    """
    imgs = os.listdir(video_path)                               # unsorted file names
    imgs = [os.path.join(video_path, f) for f in imgs]         # prepend directory
    imgs = sorted(imgs)                                         # lexicographic sort
    return imgs


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def print_sheet_row(metrics, include_header=False):
    """
    Format a metric dictionary as a Google Sheets ``=SPLIT()`` formula and
    print it to stdout so it can be pasted directly into a spreadsheet cell.

    Each call prints one data row; passing ``include_header=True`` additionally
    prints a header row above the data row so the column names are preserved.

    Parameters
    ----------
    metrics : dict
        Nested metric dictionary produced by the main evaluation loop.
        Expected structure (keys may be absent; ``None`` is handled)::

            {
              'distribution': {'fvd': float},
              'quality': {
                  'objective_quality': {
                      'frame_dynamic_range_proxy': float,
                      'mtf50': float, 'mtf10': float,
                      'contrast_transfer_accuracy': float,
                      'edge_rise_time': float,
                      'total_distortion': float,
                      'flare_attenuation': float,
                      'gradient_entropy': float,
                      'blur_extent': float,
                      'chroma_aberration': float,
                      'sequence_dynamic_range_proxy': float,
                      'fmp_alias': float,
                      'mmp_alias': float,
                  },
                  'smoothness': [mse, ssim, lpips],
                  'magnitude': float,
                  'subjective_quality': float,
                  'scene_consistency': float,
              }
            }

    include_header : bool
        When ``True``, print a header ``=SPLIT()`` row before the data row
        (useful when pasting into a blank sheet for the first time).

    Notes
    -----
    The ``=SPLIT("a,b,c", ",")`` formula splits a comma-delimited string into
    adjacent cells when entered in Google Sheets.  All floating-point values
    are formatted to 4 decimal places.
    """
    # Navigate the nested metric dict safely (fall back to empty dict / list)
    q   = metrics.get('quality', {}) or {}
    obj = q.get('objective_quality', {}) or {}
    sm  = q.get('smoothness', []) or []

    # Unpack the smoothness tuple [mse, ssim, lpips] with safe index access
    mse   = sm[0] if len(sm) > 0 else None   # mean squared error
    ssim  = sm[1] if len(sm) > 1 else None   # structural similarity index
    lpips = sm[2] if len(sm) > 2 else None   # learned perceptual image patch similarity

    # Contrast transfer accuracy may appear under two different key names
    cta = obj.get('contrast_transfer_accuracy',
                  obj.get('contrast_transfer accuracy'))

    # Define the ordered column list: (display_name, value)
    cols = [
        ('FVD',                          metrics.get('distribution', {}).get('fvd')),
        ('mse',                          mse),
        ('ssim',                         ssim),
        ('lpips',                        lpips),
        ('flow',                         q.get('magnitude')),
        ('frame_dynamic_range_proxy',    obj.get('frame_dynamic_range_proxy')),
        ('mtf50',                        obj.get('mtf50')),
        ('mtf10',                        obj.get('mtf10')),
        ('contrast_transfer_accuracy',   cta),
        ('edge_rise_time',               obj.get('edge_rise_time')),
        ('total_distortion',             obj.get('total_distortion')),
        ('flare_attenuation',            obj.get('flare_attenuation')),
        ('gradient_entropy',             obj.get('gradient_entropy')),
        ('blur_extent',                  obj.get('blur_extent')),
        ('chroma_aberration',            obj.get('chroma_aberration')),
        ('sequence_dynamic_range_proxy', obj.get('sequence_dynamic_range_proxy')),
        ('fmp_alias',                    obj.get('fmp_alias')),
        ('mmp_alias',                    obj.get('mmp_alias')),
        ('subjective_quality',           q.get('subjective_quality')),
        ('scene_consistency',            q.get('scene_consistency')),
    ]

    def fmt(v):
        """Format a metric value: 4 decimals for numbers, empty string for None."""
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    # Build comma-delimited strings for headers and values
    headers = ",".join(k for k, _ in cols)
    values  = ",".join(fmt(v) for _, v in cols)

    # Optionally print the header row first
    if include_header:
        print(f'=SPLIT("{headers}", ",")')

    # Print the data row as a Google Sheets SPLIT formula
    print(f'=SPLIT("{values}", ",")')


def print_by_metric(all_results: dict,
                    floatfmt: str = ".4f",
                    models_order: list[str] | None = None):
    """
    Pretty-print a cross-model metric comparison table to stdout.

    The table is grouped by metric category (e.g. ``objective_quality``,
    ``subjective_quality``); within each category metrics are rows and models
    are columns.  Columns are right-aligned for numeric values and padded to
    the widest entry for clean alignment.

    Parameters
    ----------
    all_results : dict
        Nested dictionary with the structure::

            {
              "model_a": {"category_1": {"metric_x": float, ...}, ...},
              "model_b": {"category_1": {"metric_x": float, ...}, ...},
              ...
            }

    floatfmt : str
        Python format string for floating-point values (default ``".4f"``).
    models_order : list of str or None
        Explicit column order for the models.  Models present in
        ``all_results`` but absent from this list are appended at the end.
        When ``None`` the insertion order of ``all_results`` is preserved.

    Notes
    -----
    Categories not present for a given model produce a ``"-"`` placeholder
    in the corresponding cell.
    """
    # ------------------------------------------------------------------
    # Step 1: Determine the column order (models)
    # ------------------------------------------------------------------
    if models_order is None:
        # Preserve the dict's insertion order (Python 3.7+ guarantee)
        models = list(all_results.keys())
    else:
        # Respect the caller-specified ordering for the subset that exists
        models = [m for m in models_order if m in all_results]
        # Append any models not listed in models_order at the end
        models += [m for m in all_results.keys() if m not in models]

    # ------------------------------------------------------------------
    # Step 2: Collect all metric category names across every model
    # ------------------------------------------------------------------
    categories = sorted({
        cat
        for m in models
        for cat in all_results[m].keys()
    })

    # ------------------------------------------------------------------
    # Step 3: Print one table per category
    # ------------------------------------------------------------------
    for cat in categories:
        # Build row_map: {metric_name: {model_name: value}}
        row_map = {}
        for m in models:
            for met, v in all_results[m].get(cat, {}).items():
                row_map.setdefault(met, {})[m] = v

        if not row_map:
            continue   # skip categories with no data for any model

        metrics = sorted(row_map.keys())   # alphabetical metric order

        # ------------------------------------------------------------------
        # Step 4: Pre-format every cell as a string to compute column widths
        # ------------------------------------------------------------------
        table = []   # list of rows; each row is [metric_name, val_0, val_1, …]
        for met in metrics:
            row = [met]
            for m in models:
                v = row_map[met].get(m, None)
                if isinstance(v, (int, float)):
                    s = format(v, floatfmt)   # e.g. "0.9231"
                elif v is None:
                    s = "-"                   # missing entry placeholder
                else:
                    s = str(v)
                row.append(s)
            table.append(row)

        # Column headers: first column is the metric name, rest are model names
        headers = ["metric"] + models

        # ------------------------------------------------------------------
        # Step 5: Compute per-column widths and print
        # ------------------------------------------------------------------
        # Transpose the table (including headers) to get per-column value lists
        cols   = list(zip(*([headers] + table)))
        widths = [max(len(x) for x in col) for col in cols]

        print(f"\n[{cat}]")
        # Header row: metric name left-aligned, model values right-aligned
        print(" ".join(
            h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
            for i, h in enumerate(headers)
        ))
        # Separator line
        print("-" * (sum(widths) + len(widths) - 1))
        # Data rows
        for r in table:
            print(" ".join(
                r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i])
                for i in range(len(headers))
            ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Command-line argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='DrivingGen video evaluation — computes FVD, quality, '
                    'and consistency metrics for generated driving scenes.'
    )
    parser.add_argument('--root_path',  type=str,
                        help='Root directory containing per-scene/model/exp sub-folders '
                             'with generated frame images.')
    parser.add_argument('--gt_path',   type=str,
                        help='Path to the ground-truth meta JSON file listing '
                             'reference scene identifiers.')
    parser.add_argument('--outdir',    type=str, default='./vis_depth',
                        help='Output directory where agent pkl files were written '
                             'by extract_traj_agent_unidepth.py.')
    parser.add_argument('--model_name', type=str, default='gt',
                        help='Name of the generative model being evaluated.')
    parser.add_argument('--exp_id',     type=str, default='free',
                        help='Experiment identifier suffix.')
    parser.add_argument('--debug',      type=int, default=0,
                        help='Enable debug mode (non-zero) for additional logging.')
    parser.add_argument('--metric',     type=str, default='fvd',
                        choices=['fvd', 'obj_q', 'sub_q', 'v_consist',
                                 'a_consist', 'a_missing', 'all'],
                        help='Metric(s) to compute. Use "all" to run every metric.')
    args = parser.parse_args()

    # Top-level result accumulator: {model_name: {metric_name: value}}
    all_metrics = {}

    model_name = args.model_name
    exp_id     = args.exp_id

    print(f'eval: {model_name}')

    # Ground-truth mode does not use an exp_id suffix
    exp_id = exp_id if model_name != 'gt' else ''

    # ------------------------------------------------------------------
    # Step 1: Enumerate scene directories under root_path
    # ------------------------------------------------------------------
    runs = os.listdir(args.root_path)
    # Keep only directories (ignore stray files and FVD symlink staging dirs)
    runs = [
        run for run in runs
        if os.path.isdir(os.path.join(args.root_path, run))
        and not run.startswith('+')
    ]
    debug = args.debug

    print(f'all {len(runs)} runs')

    # Load the ground-truth scene list from the JSON meta file
    gt_json = args.gt_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)   # list of reference scene directory strings

    # ------------------------------------------------------------------
    # Step 2: Initialise per-metric accumulators
    # ------------------------------------------------------------------
    preds     = []   # (unused) placeholder for tensor-stack path
    pred_imgs = []   # list of N_scenes × list of sorted image paths
    gt_imgs   = []   # list of N_scenes × list of sorted GT image paths (unused)

    # FVD: prepare directories for symlinked video frame sequences
    if args.metric == 'fvd' or args.metric == 'all':
        # Reference clip directory (pre-built from GT data)
        gt_video_fvd_base  = "data/videos-fvd"
        # Generated clip directory: one sub-folder per scene
        video_fvd_base = args.root_path + f'+{model_name}_fvd'
        os.makedirs(video_fvd_base, exist_ok=True)

    # Agent consistency / missing: accumulators for bbox and label data
    if args.metric == 'a_consist' or args.metric == 'a_missing' or args.metric == 'all':
        agents_bbox       = []   # list of N_valid_scenes agent bbox lists
        agents_label      = []   # list of N_valid_scenes agent label dicts
        valid_agents_runs = []   # scene names that have valid agent pkl files
        img_dirs          = []   # image directories for valid agent scenes

    # ------------------------------------------------------------------
    # Step 3: Per-scene data collection loop
    # ------------------------------------------------------------------
    for run in runs:
        s_name   = run   # scene identifier string
        # Construct the path to the generated frames for this scene
        log_base = os.path.join(args.root_path, s_name, model_name, exp_id)

        # Collect sorted image paths for this scene
        # video_path : directory containing numbered PNG/JPG frames
        video_path   = os.path.join(log_base, 'images')
        # video_frames : list of sorted absolute image paths
        video_frames = get_imgs(video_path)

        # Skip the conditioning frame (index 0) to evaluate only generated frames
        pred_imgs.append(video_frames[1:])   # list of (T-1) image paths

        # ------------------------------------------------------------------
        # FVD: build a symlinked flat directory of frames for this scene
        # ------------------------------------------------------------------
        if args.metric == 'fvd' or args.metric == 'all':
            # Construct a unique name for the symlink directory of this scene
            name = (log_base.split('/')[-3] + '+' +
                    log_base.split('/')[-2] + '+' +
                    log_base.split('/')[-1])
            fvd_path = os.path.join(video_fvd_base, name)

            # Remove stale symlink directory from a previous run
            if os.path.exists(fvd_path):
                os.system(f'rm -rf {fvd_path}')
            os.makedirs(fvd_path, exist_ok=True)

            for idx, img in enumerate(video_frames):
                # Skip the first (conditioning) frame — same as pred_imgs above
                if idx == 0:
                    continue

                # Destination symlink path preserves the original file extension
                # link_path example: .../00001.png
                link_path = os.path.join(fvd_path, f'{idx:05d}{img[-4:]}')

                # Create a symlink only if it does not already exist
                # (idempotent: safe to rerun without errors)
                if not os.path.exists(link_path):
                    # os.symlink(src, dst): src is the real file, dst is the link
                    os.symlink(os.path.abspath(img), link_path)

        # ------------------------------------------------------------------
        # Agent metrics: load precomputed bbox and label pkl files
        # ------------------------------------------------------------------
        if args.metric == 'a_consist' or args.metric == 'a_missing' or args.metric == 'all':
            # pkl files are written by extract_traj_agent_unidepth.py
            outdir = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')

            try:
                # agents_bbox pkl: list of N_agents × list of (frame_id, [x1,y1,x2,y2])
                with open(outdir + '-estimate_agents_bbox.pkl', 'rb') as f:
                    agent_bbox = pickle.load(f)
                # agents_label pkl: dict {"x1-y1-x2-y2": class_name}
                with open(outdir + '-estimate_agents_bbox_label.pkl', 'rb') as f:
                    agent_label = pickle.load(f)
            except Exception:
                # Missing pkl files means no agents were detected for this scene
                print(f"Skipping agents: {outdir}.")
                agent_bbox = None

            if agent_bbox is not None:
                # ----------------------------------------------------------
                # Trim agent tracks: keep only the first continuous segment.
                # A gap is defined as consecutive frame IDs differing by > 10.
                # This removes tracking drift after occlusions.
                # ----------------------------------------------------------
                trans_agent_bbox = []   # trimmed bbox list for all agents

                for box_id, boxes in enumerate(agent_bbox):
                    # Extract the sorted frame indices for this agent's track
                    ids = [box[0] for box in boxes]

                    # Walk forward until a gap > 10 frames is detected
                    ids_valid   = []    # frame IDs in the first continuous segment
                    boxes_valid = []    # corresponding bbox entries

                    for i, id in enumerate(ids):
                        ids_valid.append(id)
                        boxes_valid.append(boxes[i])

                        if i < len(ids) - 1:
                            if id + 10 < ids[i + 1]:
                                # Gap detected: stop at the current frame
                                print(f'trim {ids} at {id}')
                                break   # discard everything after this gap

                    # Replace the full track with the trimmed first segment
                    boxes = boxes_valid   # list of (frame_id, bbox) up to first gap
                    ids   = ids_valid

                    trans_agent_bbox.append(boxes)

                # Accumulate valid-scene agent data for the metric functions
                agents_bbox.append(trans_agent_bbox)      # trimmed bbox tracks
                agents_label.append(agent_label)           # class label dict
                valid_agents_runs.append(run)              # scene name
                img_dirs.append(os.path.join(log_base, 'images'))  # image dir path

    # ------------------------------------------------------------------
    # Step 4: Compute selected metrics
    # ------------------------------------------------------------------

    # ---- FVD -----------------------------------------------------------------
    fvd = -1   # sentinel value indicating "not computed"
    if args.metric == 'fvd' or args.metric == 'all':
        # Compare feature distribution of generated clips vs. ground-truth clips.
        # Both directories contain one sub-folder per scene with symlinked frames.
        # Returns a scalar FVD score (lower is better).
        fvd = get_fvd(video_fvd_base, gt_video_fvd_base)

    # ---- Objective quality (P3020 MMP) ----------------------------------------
    objective_quality    = -1   # sentinel
    gt_objective_quality = -1   # sentinel (GT evaluation, currently disabled)
    if args.metric == 'obj_q' or args.metric == 'all':
        # pred_imgs : list of N_scenes × list of image paths
        # Returns a dict of MMP metric names → scalar values
        objective_quality = get_objective_quality_v2(pred_imgs)

    # ---- Subjective quality (CLIP-IQA+) ----------------------------------------
    subjective_quality    = -1   # sentinel
    gt_subjective_quality = -1   # sentinel
    if args.metric == 'sub_q' or args.metric == 'all':
        # Runs CLIP-IQA+ on every frame path in pred_imgs.
        # Returns a scalar mean aesthetic quality score.
        subjective_quality = get_subjective_quality(pred_imgs)

    # ---- Visual / scene consistency (DINOv2 + optical flow) --------------------
    scene_consistency    = -1   # sentinel
    gt_scene_consistency = -1   # sentinel
    if args.metric == 'v_consist' or args.metric == 'all':
        # Cache directory keyed by data track name to avoid recomputation
        track     = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/v_consist', track)
        os.makedirs(cache_dir, exist_ok=True)

        # One cache subdirectory per scene for storing DINOv2 features
        v_consist_cache_dir = [f'{cache_dir}/{name}' for name in runs]

        # Returns a scalar consistency score (higher = more temporally coherent)
        scene_consistency = get_scene_consistency_v3(pred_imgs, v_consist_cache_dir)

    # ---- Agent appearance consistency (DINOv2 crop similarity) -----------------
    agent_consistency = -1   # sentinel
    if args.metric == 'a_consist' or args.metric == 'all':
        track     = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/a_consist', track)
        os.makedirs(cache_dir, exist_ok=True)

        # Cache per valid scene (scenes without agent pkl files are excluded)
        a_consist_cache_dir = [f'{cache_dir}/{name}' for name in valid_agents_runs]

        # Returns a scalar mean pairwise DINOv2 cosine similarity for agent crops
        agent_consistency = get_agent_consistency(
            valid_agents_runs,    # list of scene names with valid agent tracks
            agents_bbox,          # list of N_scenes trimmed agent bbox tracks
            agents_label,         # list of N_scenes class label dicts
            a_consist_cache_dir,  # list of per-scene cache directories
            img_dirs,             # list of per-scene image directories
        )

    # ---- Agent missing rate (Cosmos-Reason1 VLM) --------------------------------
    if args.metric == 'a_missing' or args.metric == 'all':
        track     = args.root_path.split('/')[-1]
        cache_dir = os.path.join('./cache/a_missing', track)
        os.makedirs(cache_dir, exist_ok=True)

        a_missing_cache_dir = [f'{cache_dir}/{name}' for name in valid_agents_runs]

        # Returns a scalar fraction of expected agents that are missing in the
        # generated video (lower is better: 0.0 = all agents present).
        agent_missing = get_agent_missing(
            valid_agents_runs,    # list of scene names
            agents_bbox,          # trimmed agent bbox tracks
            agents_label,         # class label dicts
            a_missing_cache_dir,  # per-scene cache directories
            img_dirs,             # per-scene image directories
        )

    # ------------------------------------------------------------------
    # Step 5: Aggregate results and print
    # ------------------------------------------------------------------
    # Collect all computed metrics into a flat dict for this model
    metrics = {
        'fvd':               fvd,
        'objective_quality': objective_quality,
        'subjective_quality': subjective_quality,
        'scene_consistency': scene_consistency,
        'agent_consistency': agent_consistency,
        'agent_missing':     agent_missing,
    }
    all_metrics[model_name] = metrics

    # Print each metric value with consistent formatting
    for sub_key, sub_val in all_metrics.items():
        if isinstance(sub_val, (float, int)):
            # Scalar metrics formatted to 4 decimal places
            print(f"  {sub_key}: {sub_val:.4f}")
        else:
            # Lists, dicts, or other composite values printed as-is
            print(f"  {sub_key}: {sub_val}")

    # Optional: cross-model comparison table (uncomment when evaluating multiple models)
    # print_by_metric(all_metrics)
