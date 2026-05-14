"""
z-sample_ftd.py
===============
Trajectory Metrics Evaluation Driver for the DrivingGen benchmark.

This script is the trajectory counterpart of ``z-sample_fvd.py``.  Where the
FVD script evaluates *video-level* quality (pixel fidelity, perceptual quality,
temporal coherence), this script evaluates *trajectory-level* quality of the
ego-vehicle path extracted by the SLAM pipeline.

Evaluation pipeline
-------------------
1. **Load GT trajectories** (``data/ego_condition/ego_motion/<scene>.npy``)
   and convert from global XY to ego-centric local frame via ``gt_2_ego``.
2. **Load predicted trajectories** from per-scene
   ``<log_base>-estimate_ego_traj.pkl`` files produced by
   ``func/extract_traj_ego_unidepth.py``.
3. **Coordinate conversion**: SLAM outputs ``(x_right, z_forward)`` which is
   column-swapped and sign-flipped to match the GT convention via
   ``ego_y_2_x``.
4. **Alignment** (``ego_condition`` track only): Umeyama similarity transform
   (scale + rotation, origin-fixed) aligns the predicted trajectory to the GT
   local frame via ``slam_align_to_gt_fix_origin``.
5. **Smoothing**: Savitzky-Golay polynomial filter removes high-frequency SLAM
   noise via ``smooth_traj_sg``.
6. **Metrics** (selectable via ``--metric``):
   - ``ftd``         — Fréchet Trajectory Distance (MTR encoder features)
   - ``traj_q``      — Comfort/curvature/speed quality score
   - ``traj_consist``— Velocity/acceleration smoothness consistency
   - ``traj_align``  — ADE and DTW vs GT (ego-condition track only)
   - ``all``         — All of the above

Helper printing utilities (``print_sheet_row``, ``print_by_metric``) are
provided for formatting results into Google Sheets or aligned ASCII tables.
"""

# Standard-library imports
import argparse        # CLI argument parsing
import cv2             # OpenCV (imported but unused in the main block)
import glob            # File-system globbing (imported but unused directly)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend before importing pyplot
                       # (required when running on a headless server)
import matplotlib.pyplot as plt  # Plotting (imported for potential debug plots)
import numpy as np     # Numerical array operations
import os              # OS path utilities
import numpy as np     # Re-imported (harmless duplicate; only one instance is active)
from PIL import Image  # PIL image loader (imported but unused in the main block)
import sys             # System utilities (imported but unused directly)
import math            # Math utilities (imported but unused directly)
import random          # Random sampling (imported but unused directly)
import json            # JSON loading for the GT manifest

# Deep-learning / scientific imports
import torch                         # PyTorch tensors
import pickle                        # PKL serialisation for saved SLAM trajectories
from numpy.typing import ArrayLike   # Type alias for array-like arguments

# DrivingGen trajectory metric modules
from trajs.traj_distribution import get_ftd          # Fréchet Trajectory Distance
from trajs.traj_quality      import get_traj_quality  # Comfort / curvature / speed
from trajs.traj_consistency  import get_traj_consistency  # Velocity smoothness
from trajs.traj_alignment    import get_ade, get_dtw  # ADE and dynamic time warping

# Scipy smoothing (imported here rather than at the top to co-locate with usage)
from scipy.signal import savgol_filter  # Savitzky-Golay polynomial smoother


# ---------------------------------------------------------------------------
# Video / image loading helpers (not used in the default evaluation path)
# ---------------------------------------------------------------------------

def get_video(video_path):
    """
    Load an MP4 video as a float32 tensor using ``decord.VideoReader``.

    Note:
        ``VideoReader`` is imported implicitly; call ``from decord import
        VideoReader`` before using this function.  It is not used in the
        default ``__main__`` evaluation path but retained for convenience.

    Args:
        video_path (str): Path to the ``.mp4`` video file.

    Returns:
        torch.Tensor: Float32 tensor of shape ``(T, H, W, 3)`` with values
            in ``[0, 1]``.
    """
    video_frames = VideoReader(video_path)             # decord VideoReader; lazy-loads the file
    # Convert each decoded frame (numpy uint8 H×W×3) to float32 in [0,1]
    video_frames = [(torch.from_numpy(v.asnumpy()).to(torch.float32)) / 255.
                    for v in video_frames]
    # shape of each element: (H, W, 3)
    video_frames = torch.stack(video_frames)           # shape: (T, H, W, 3)
    return video_frames                                # shape: (T, H, W, 3)


def get_imgs(video_path):
    """
    Return a sorted list of image file paths from a directory.

    Args:
        video_path (str): Directory containing frame images (any extension).

    Returns:
        list[str]: Sorted list of absolute image file paths.
    """
    imgs = os.listdir(video_path)                      # list file names (unsorted)
    imgs = [os.path.join(video_path, f) for f in imgs] # prepend directory prefix
    imgs = sorted(imgs)                                # sort lexicographically (chronological for zero-padded names)
    return imgs                                        # list of sorted absolute paths


# ---------------------------------------------------------------------------
# Result formatting utilities
# ---------------------------------------------------------------------------

def print_sheet_row(metrics, include_header=False):
    """
    Print a Google Sheets ``=SPLIT(...)`` formula for a single model's metrics.

    Formats all video + trajectory metrics as a comma-separated string wrapped
    in an ``=SPLIT("...", ",")`` formula so that pasting it into a Google Sheet
    cell automatically distributes values across columns.

    Args:
        metrics (dict): Nested metrics dictionary with keys ``'distribution'``,
            ``'quality'`` (sub-keys ``'objective_quality'``, ``'smoothness'``,
            ``'subjective_quality'``, ``'scene_consistency'``, ``'magnitude'``).
        include_header (bool): If ``True``, also print a header row before the
            value row.  Default: ``False``.
    """
    # Safely extract nested sub-dicts; fall back to empty dict/list if missing
    q   = metrics.get('quality', {}) or {}             # quality sub-dict
    obj = q.get('objective_quality', {}) or {}         # P2020 objective metrics sub-dict
    sm  = q.get('smoothness', []) or []                # [mse, ssim, lpips] list

    # Unpack the smoothness triple with safe indexing
    mse   = sm[0] if len(sm) > 0 else None  # mean-squared error (reference-based)
    ssim  = sm[1] if len(sm) > 1 else None  # SSIM score
    lpips = sm[2] if len(sm) > 2 else None  # LPIPS perceptual distance

    # contrast_transfer_accuracy may appear under two possible key spellings
    cta   = obj.get('contrast_transfer_accuracy', obj.get('contrast_transfer accuracy'))

    # Define column order: (header_label, value)
    cols = [
        ('FVD',                         metrics.get('distribution', {}).get('fvd')),
        ('mse',                         mse),
        ('ssim',                        ssim),
        ('lpips',                       lpips),
        ('flow',                        q.get('magnitude')),              # optical-flow magnitude
        ('frame_dynamic_range_proxy',   obj.get('frame_dynamic_range_proxy')),
        ('mtf50',                       obj.get('mtf50')),                # sharpness at 50% contrast
        ('mtf10',                       obj.get('mtf10')),                # sharpness at 10% contrast
        ('contrast_transfer_accuracy',  cta),
        ('edge_rise_time',              obj.get('edge_rise_time')),
        ('total_distortion',            obj.get('total_distortion')),
        ('flare_attenuation',           obj.get('flare_attenuation')),
        ('gradient_entropy',            obj.get('gradient_entropy')),
        ('blur_extent',                 obj.get('blur_extent')),
        ('chroma_aberration',           obj.get('chroma_aberration')),
        ('sequence_dynamic_range_proxy',obj.get('sequence_dynamic_range_proxy')),
        ('fmp_alias',                   obj.get('fmp_alias')),            # frame-level flicker
        ('mmp_alias',                   obj.get('mmp_alias')),            # video-level flicker (canonical)
        ('subjective_quality',          q.get('subjective_quality')),     # CLIP-IQA+
        ('scene_consistency',           q.get('scene_consistency')),      # DINOv3 cosine similarity
    ]

    def fmt(v):
        """Format a single value: 4 decimal places for numbers, empty string for None."""
        if isinstance(v, (int, float)):
            return f"{v:.4f}"   # fixed 4 decimal places
        return "" if v is None else str(v)  # None → empty cell; other → str

    # Build comma-separated header and value strings
    headers = ",".join(k for k, _ in cols)          # "FVD,mse,ssim,..."
    values  = ",".join(fmt(v) for _, v in cols)     # "0.1234,0.5678,..."

    if include_header:
        print(f'=SPLIT("{headers}", ",")')   # header row formula
    print(f'=SPLIT("{values}", ",")')        # value row formula


def print_by_metric(all_results: dict,
                    floatfmt: str = ".4f",
                    models_order: list[str] | None = None):
    """
    Print an aligned ASCII table grouping results by metric category.

    Results are organised so that each *category* (e.g. ``'objective_quality'``)
    becomes a table where rows are individual metrics and columns are models.
    This makes cross-model comparison easy to read.

    Args:
        all_results (dict): Nested dict of shape
            ``{model_name: {category: {metric_name: value}}}``.
        floatfmt (str): Python format string for floating-point values.
            Default: ``".4f"``.
        models_order (list[str] | None): If provided, columns appear in this
            order; any model not listed is appended at the end.  If ``None``,
            columns follow the insertion order of ``all_results``.

    Example output::

        [objective_quality]
        metric          wan    cosmos
        ---------------------
        mtf50         0.6234  0.5912
        mmp_alias     0.7821  0.8103
    """
    # Step 1: Determine column order (models)
    if models_order is None:
        models = list(all_results.keys())              # preserve dict insertion order
    else:
        # Start with models in the explicit order that exist in all_results
        models = [m for m in models_order if m in all_results]
        # Append any remaining models not listed in models_order
        models += [m for m in all_results.keys() if m not in models]

    # Step 2: Collect all unique category keys across all models
    categories = sorted({cat for m in models for cat in all_results[m].keys()})
    # categories: sorted list of category name strings

    for cat in categories:
        # Step 3: Build a {metric_name: {model_name: value}} map for this category
        row_map = {}
        for m in models:
            for met, v in all_results[m].get(cat, {}).items():
                row_map.setdefault(met, {})[m] = v  # accumulate per-metric, per-model
        if not row_map:
            continue  # skip categories with no data

        metrics = sorted(row_map.keys())  # sort metric names alphabetically

        # Step 4: Pre-format all values as strings (ensures column alignment)
        table = []  # list of rows; each row: [metric_name, val_model1, val_model2, ...]
        for met in metrics:
            row = [met]                         # first column: metric name
            for m in models:
                v = row_map[met].get(m, None)   # value for this metric × model
                if isinstance(v, (int, float)):
                    s = format(v, floatfmt)     # format float to fixed decimals
                elif v is None:
                    s = "-"                     # missing value placeholder
                else:
                    s = str(v)                  # fallback for non-numeric values
                row.append(s)
            table.append(row)                   # add formatted row to table

        # Step 5: Compute column widths (max width across header + all data rows)
        headers = ["metric"] + models           # header row: ["metric", "wan", "cosmos", ...]
        cols = list(zip(*([headers] + table)))  # transpose to get one tuple per column
        widths = [max(len(x) for x in col) for col in cols]  # max string width per column

        # Step 6: Print aligned table
        print(f"\n[{cat}]")  # category header
        # Print header row: first column left-justified, rest right-justified
        print(" ".join(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
                       for i, h in enumerate(headers)))
        print("-" * (sum(widths) + len(widths) - 1))  # separator line
        for r in table:
            # Print each data row with same justification as header
            print(" ".join(r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i])
                           for i in range(len(headers))))


# ---------------------------------------------------------------------------
# Coordinate system helpers
# ---------------------------------------------------------------------------

def gt_2_ego(gt_xy, heading=None, k_ahead=1, min_step=1):
    """
    Convert a global XY GT trajectory to an ego-centric local frame.

    The local frame is defined by:
    - **Origin**: The first waypoint ``gt_xy[0]`` (translation to origin).
    - **Heading axis (+Z)**: The initial forward direction of the vehicle,
      estimated from the displacement between frame 0 and frame ``k_ahead``.
      If the initial displacement is too small (< ``min_step``), the first
      sufficiently large displacement is used instead (robust to stationary
      starts).
    - **Rotation**: A 2D rotation matrix that aligns the heading vector to the
      +Z axis (i.e. the vehicle always faces "up" in the local frame).

    A column-swap and sign-flip are applied after rotation to convert from
    the ``(X_right, Z_forward)`` OpenCV convention to the ``(Z_forward,
    -X_right)`` convention expected by the MTR trajectory encoder.

    Args:
        gt_xy (np.ndarray): Global GT XY positions, shape ``(T, 2)``.
        heading (torch.Tensor | None): Pre-computed heading angle (radians).
            If ``None``, the heading is inferred from the trajectory.
        k_ahead (int): Number of frames ahead to use for heading estimation.
            Default: ``1`` (use frame 1).
        min_step (int): Minimum displacement (metres) required for the heading
            estimate to be considered reliable.  Default: ``1`` metre.

    Returns:
        tuple:
            - **gt_local_xy** (np.ndarray): Trajectory in the ``(X_right,
              Z_forward)`` local frame, shape ``(T, 2)``.  Used as input to
              the MTR encoder (FTD metric).
            - **gt** (np.ndarray): Same trajectory in ``(Z_forward, -X_right)``
              convention, shape ``(T, 2)``.  Used for visualisation.
            - **theta** (torch.Tensor): Scalar; the estimated heading angle in
              radians (angle between +X axis and the vehicle's forward direction).
    """
    gt = torch.from_numpy(gt_xy)   # shape: (T, 2); convert numpy → torch for matrix ops

    # Step 1: Translate so the first waypoint becomes the origin
    origin = gt[0]                 # shape: (2,); position of frame 0 in global coords
    rel_gt = gt - origin           # shape: (T, 2); all positions relative to frame 0

    # Step 2: Estimate the forward heading direction
    k = min(k_ahead, len(gt) - 1)  # clamp k_ahead to valid range [0, T-1]
    v = gt[k] - gt[0]              # shape: (2,); displacement from frame 0 to frame k

    if torch.linalg.norm(v) < min_step:
        # The k_ahead displacement is too small (e.g. vehicle is stationary at start).
        # Search forward for the first frame with a displacement >= min_step.
        for j in range(1, len(gt)):
            v = gt[j] - gt[0]          # shape: (2,); displacement to frame j
            if torch.linalg.norm(v) >= min_step:
                break                  # found a usable displacement vector

    # Step 3: Build a 2D rotation matrix that aligns the heading to the +X axis
    if heading is None:
        heading = v                            # shape: (2,); use inferred displacement vector
        theta = torch.atan2(heading[1], heading[0])  # scalar; angle of heading w.r.t. +X axis
    else:
        theta = heading                         # scalar; caller-supplied heading angle

    # Counter-clockwise rotation matrix R such that R @ heading → +X direction
    R = torch.tensor([[ torch.cos(theta), -torch.sin(theta)],   # shape: (2, 2)
                      [ torch.sin(theta),  torch.cos(theta)]])   # standard CCW rotation

    # Step 4: Rotate all relative positions by R
    gt_local = torch.matmul(rel_gt, R)         # shape: (T, 2); (X_aligned, Y_aligned)

    # Step 5: Save (X_right, Z_forward) copy for MTR encoder input
    gt_local_xy = gt_local.clone()             # shape: (T, 2); (X_right, Z_forward)

    # Step 6: Column-swap + sign-flip to convert to (Z_forward, -X_right) for visualisation
    gt_local[:, [0, 1]] = gt_local[:, [1, 0]] # swap: col0 ← Z_forward, col1 ← X_right
    gt_local[:, 0] = -gt_local[:, 0]          # negate new col0: col0 ← -Z_forward

    gt = gt_local.numpy()                      # shape: (T, 2); (Z_forward, -X_right) for vis

    # Return: MTR-ready (X,Z), vis-ready (Z,-X), and heading angle
    return gt_local_xy, gt, theta
    # gt_local_xy: shape (T, 2) — (X_right, Z_forward), for MTR / FTD
    # gt:          shape (T, 2) — (-Z_forward, X_right), for visualisation
    # theta:       scalar torch.Tensor — heading angle in radians


def smooth_traj_sg(xy, dt=0.1, win_sec=0.4, poly=3):
    """
    Apply a Savitzky-Golay polynomial filter to a 2D trajectory.

    Savitzky-Golay filtering fits a low-degree polynomial to a sliding window
    of samples and evaluates the polynomial at the centre of the window.  It
    simultaneously smooths noise and preserves the shape of the trajectory
    (peaks, slopes) better than a moving average.

    Edge cases handled:
    - ``win_sec / dt`` is rounded to the nearest odd integer (SG requirement).
    - If the computed window exceeds the trajectory length, it is clamped to
      the nearest odd integer ≤ T.
    - For T == 4 (minimum viable length), the window is forced to 3 and the
      polynomial order to 1.
    - If ``poly >= k - 1`` (polynomial order too high for the window), the
      order is reduced automatically.
    - If the window is < 3 (too short to smooth), the input is returned as-is.

    Args:
        xy (array-like): 2D trajectory, shape ``(T, 2)``, units: metres.
        dt (float): Sampling interval in seconds.  Default: ``0.1`` s (10 Hz).
        win_sec (float): Smoothing window duration in seconds.  Default:
            ``0.4`` s (4 frames at 10 Hz).
        poly (int): Polynomial order for the Savitzky-Golay filter.
            Default: ``3`` (cubic).

    Returns:
        np.ndarray: Smoothed trajectory, shape ``(T, 2)``, same units as
            ``xy``.  Returned unchanged if the trajectory is too short to
            smooth.
    """
    xy = np.asarray(xy, float)   # shape: (T, 2); ensure float64 numpy array
    T  = xy.shape[0]              # number of time steps

    # Compute window length in samples from the requested duration
    k = int(round(win_sec / dt))  # scalar int; number of frames in the smoothing window
    if k % 2 == 0:
        k += 1                    # SG requires an odd window length; round up

    # Clamp: window must not exceed sequence length
    if k > T:
        k = T if T % 2 == 1 else T - 1  # largest odd number ≤ T

    # Special case: T == 4 triggers a stricter SG constraint
    if T == 4:
        k    = 3   # only valid odd window ≤ 4
        poly = 1   # poly must be < k - 1 = 2, so poly ≤ 1

    # General case: reduce polynomial order if it would violate poly < k - 1
    elif poly >= k - 1:
        poly = max(1, k - 2)  # keep at least linear; cap at k-2

    # Too short to smooth: return original trajectory unchanged
    if k < 3:
        return xy   # shape: (T, 2); unchanged

    # Apply SG filter along axis=0 (time axis) for both X and Z columns simultaneously
    xy_s = savgol_filter(xy, window_length=k, polyorder=poly, axis=0, mode="interp")
    # mode="interp" uses polynomial extrapolation at boundaries (avoids edge artefacts)
    # xy_s: shape (T, 2); smoothed trajectory
    return xy_s   # shape: (T, 2)


def ego_y_2_x(ego_xz):
    """
    Convert SLAM output from ``(X_right, Z_forward)`` to ``(Z_forward, X_right)`` convention.

    The VisualOdometry SLAM pipeline stores trajectory in ``locs`` with columns
    ordered as ``(x_right, z_forward)``.  The GT ego-centric frame after
    ``gt_2_ego`` uses ``(Z_forward, -X_right)`` for visualisation, but the
    MTR encoder and alignment functions expect ``(X_right, Z_forward)``.

    This function performs the column swap and sign flip so that the predicted
    trajectory matches the same column ordering as the GT local frame produced
    by the *visualisation* branch of ``gt_2_ego``.

    Args:
        ego_xz (array-like): SLAM trajectory in ``(X_right, Z_forward)``
            convention, shape ``(N, 2)`` (or ``(N, 3)`` — only first 2 cols used).

    Returns:
        np.ndarray: Converted trajectory, shape ``(N, 2)``, with columns
            ``(-Z_forward, X_right)`` matching the GT vis convention.
    """
    gt_local = torch.from_numpy(np.asarray(ego_xz, float))
    # shape: (N, 2); initial SLAM convention: col0=X_right, col1=Z_forward

    # Swap columns: col0 ← Z_forward, col1 ← X_right
    gt_local[:, [0, 1]] = gt_local[:, [1, 0]]  # shape: (N, 2)

    # Negate the new col1 (was X_right, now -X_right)
    gt_local[:, 1] = -gt_local[:, 1]            # shape: (N, 2)

    gt = gt_local.numpy()  # shape: (N, 2); converted to numpy for downstream ops
    return gt              # shape: (N, 2)


def umeyama_2d(X: np.ndarray, Y: np.ndarray, with_scale=True):
    """
    Compute the Umeyama similarity transform aligning source ``X`` to target ``Y``.

    The Umeyama algorithm finds the optimal similarity transform
    ``Y ≈ s · R · X + t`` that minimises the mean squared error between the
    transformed source and the target.  It uses SVD on the cross-covariance
    matrix and is numerically stable.

    A reflection fix is applied: if ``det(R) < 0`` the SVD produces a
    reflection (improper rotation) which is corrected by negating the last row
    of ``Vt`` before recomputing ``R``.

    Args:
        X (np.ndarray): Source trajectory, shape ``(N, 2)``.  Must have N ≥ 2.
        Y (np.ndarray): Target trajectory, shape ``(N, 2)``.  Must be
            point-correspondence-aligned with ``X``.
        with_scale (bool): If ``True``, estimate an isotropic scale factor
            ``s``.  If ``False``, force ``s = 1.0`` (rotation + translation
            only).  Default: ``True``.

    Returns:
        tuple:
            - **s** (float): Isotropic scale factor; ``1.0`` if
              ``with_scale=False``.
            - **R** (np.ndarray): Optimal 2D rotation matrix, shape ``(2, 2)``.
            - **t** (np.ndarray): Optimal translation vector, shape ``(2,)``.

    Note:
        The transform is: ``Y_pred = s * (R @ X.T).T + t``
    """
    muX, muY = X.mean(0), Y.mean(0)   # shape: (2,) each; centroid of X and Y
    Xc, Yc   = X - muX, Y - muY      # shape: (N, 2) each; centred (zero-mean) coordinates

    # Cross-covariance matrix (unnormalised)
    H = Xc.T @ Yc / len(X)            # shape: (2, 2); H = Xc^T Yc / N

    # Singular Value Decomposition of H
    U, _, Vt = np.linalg.svd(H)       # U: (2,2), _: (2,), Vt: (2,2)

    # Optimal rotation: R = V U^T
    R = Vt.T @ U.T                    # shape: (2, 2)

    # Fix reflection: det(R) should be +1 for a proper rotation
    if np.linalg.det(R) < 0:
        Vt[1] *= -1                   # flip sign of last row of Vt
        R = Vt.T @ U.T               # shape: (2, 2); recompute R without reflection

    if with_scale:
        # Variance of source trajectory around its centroid
        varX = (Xc ** 2).sum() / len(X)     # scalar float; isotropic variance of X
        # Optimal scale: s = trace(Yc (R Xc^T)^T) / (N * varX)
        s = (Yc * (R @ Xc.T).T).sum() / (len(X) * varX)  # scalar float
    else:
        s = 1.0   # no scale estimation; pure rotation + translation

    # Optimal translation: t = muY - s * R @ muX
    t = muY - s * R @ muX             # shape: (2,)

    return s, R, t
    # s: scalar float — isotropic scale
    # R: shape (2, 2) — 2D rotation matrix
    # t: shape (2,) — translation vector


def slam_align_to_gt_fix_origin(
    pred_xyz:   ArrayLike,      # SLAM output: (N, 2) or (N, 3)  cols: (x_right, z_forward)
    gt_local:   ArrayLike,      # GT local trajectory from gt_2_ego: (N, 2), frame 0 = (0,0)
    with_scale: bool = True,
) -> np.ndarray:
    """
    Align a predicted SLAM trajectory to the GT local frame while keeping the
    origin fixed at ``(0, 0)``.

    Unlike full Umeyama (which estimates scale + rotation + translation), this
    function enforces that the first point stays at the origin.  This avoids a
    degenerate global translation that could artificially reduce trajectory
    error by sliding the entire prediction onto the GT.

    The alignment is:
        1. Remove the first-frame absolute position from both sequences
           (``P_rel = P - P[0]``, ``Q_rel = Q``, which already has frame 0 at origin).
        2. Find the optimal rotation ``R`` and scale ``s`` via SVD on the
           cross-covariance of the centred sequences.
        3. Apply ``aligned = s · R · P_rel`` (no translation added, so the
           origin stays at ``(0, 0)``).

    Args:
        pred_xyz (ArrayLike): Predicted SLAM trajectory, shape ``(N, 2)``
            (ground-plane XZ components; Y-up is discarded if present).
        gt_local (ArrayLike): GT local trajectory from ``gt_2_ego``, shape
            ``(N, 2)``.  Frame 0 must be ``(0, 0)``.
        with_scale (bool): If ``True``, estimate an isotropic scale ``s``.
            If ``False``, force ``s = 1.0`` (rotation only).  Default: ``True``.

    Returns:
        tuple:
            - **aligned** (np.ndarray): Aligned predicted trajectory, shape
              ``(N, 2)``.  Frame 0 is ``(0, 0)``.
            - **s** (float): Estimated (or fixed) isotropic scale.
            - **R** (np.ndarray): Estimated 2D rotation matrix, shape ``(2, 2)``.
    """
    P = np.asarray(pred_xyz, float)   # shape: (N, 2); SLAM trajectory in (x_right, z_forward)
    Q = np.asarray(gt_local, float)   # shape: (N, 2); GT local trajectory, frame 0 = (0,0)

    # Step 1: Centre both trajectories on the first frame
    P_rel = P - P[0]   # shape: (N, 2); SLAM relative to its own first frame
    Q_rel = Q           # shape: (N, 2); GT already starts at (0, 0) by construction

    # Step 2: SVD of cross-covariance to find optimal rotation
    H = P_rel.T @ Q_rel / len(P_rel)  # shape: (2, 2); cross-covariance matrix
    U, _, Vt = np.linalg.svd(H)       # U: (2,2), _: (2,), Vt: (2,2)
    R = Vt.T @ U.T                    # shape: (2, 2); candidate rotation

    # Fix reflection: ensure det(R) = +1
    if np.linalg.det(R) < 0:
        Vt[1] *= -1                   # negate last row to correct reflection
        R = Vt.T @ U.T               # shape: (2, 2); corrected rotation

    # Step 3: Estimate scale (optional)
    if with_scale:
        varP = (P_rel ** 2).sum() / len(P_rel)   # scalar; isotropic variance of SLAM path
        s = (Q_rel * (R @ P_rel.T).T).sum() / (len(P_rel) * varP)  # scalar; optimal scale
    else:
        s = 1.0   # no scale; pure rotation

    # Step 4: Apply the similarity transform; no translation → origin stays at (0,0)
    aligned = (s * (R @ P_rel.T)).T   # shape: (N, 2); aligned prediction in GT local frame

    return aligned, s, R
    # aligned: shape (N, 2) — predicted trajectory in GT coordinate frame, origin-fixed
    # s:       scalar float — isotropic scale factor
    # R:       shape (2, 2) — 2D rotation matrix


def apply_sr(trajectory_xz, s, R):
    """
    Apply a pre-computed similarity transform ``(s, R)`` to a trajectory.

    Used to transform additional trajectory arrays (e.g. agent trajectories in
    the same SLAM coordinate system) into the GT local frame using the scale
    and rotation estimated from the ego trajectory alignment.

    Note:
        This function does **not** add any translation.  The origin of the
        source trajectory is mapped to the origin of the target frame.  If
        ``trajectory_xz`` does not start at ``(0, 0)``, the first point will
        be transformed accordingly.

    Args:
        trajectory_xz (array-like): Source trajectory in the SLAM coordinate
            system ``(x_right, z_forward)``, shape ``(N, 2)`` or ``(N, 3)``
            (extra columns are preserved but transformed column-wise by ``R``).
        s (float): Isotropic scale factor from ``slam_align_to_gt_fix_origin``.
        R (np.ndarray): 2D rotation matrix, shape ``(2, 2)``, from
            ``slam_align_to_gt_fix_origin``.

    Returns:
        np.ndarray: Transformed trajectory, same shape as ``trajectory_xz``,
            in the GT local frame.
    """
    arr = np.asarray(trajectory_xz, float)   # shape: (N, 2); ensure float64 numpy
    aligned = (s * (R @ arr.T)).T            # shape: (N, 2); s·R·arr for each row
    return aligned                           # shape: (N, 2)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ---- Argument parsing -----------------------------------------------
    parser = argparse.ArgumentParser(description='DrivingGen trajectory metric evaluation')

    parser.add_argument('--root_path',   type=str,
                        help='Root directory; each sub-dir is one evaluation run/scene.')
    parser.add_argument('--gt_path',     type=str,
                        help='Path to the GT JSON manifest listing scene names.')
    parser.add_argument('--outdir',      type=str,  default='./vis_depth',
                        help='Output directory where saved trajectory PKLs live.')
    parser.add_argument('--model_name',  type=str,  default='gt',
                        help='Identifier string for the model being evaluated.')
    parser.add_argument('--exp_id',      type=str,  default='free',
                        help='Experiment ID sub-folder suffix (ignored when model_name==gt).')
    parser.add_argument('--debug',       type=int,  default=0,
                        help='Debug flag (0=off, 1=on); enables extra GT logging.')
    parser.add_argument('--metric',      type=str,  default='fvd',
                        help='Which metric(s) to compute: ftd | traj_q | traj_consist | traj_align | all.')
    parser.add_argument('--track',       type=str,  default='ego_condition',
                        help='Evaluation track: "ego_condition" uses GT alignment; others skip alignment.')
    args = parser.parse_args()

    all_metrics = {}   # top-level accumulator: {model_name: {metric_name: value}}

    model_name = args.model_name   # string identifier for results logging
    exp_id     = args.exp_id       # experiment sub-folder tag

    print(f'eval: {model_name}')

    # When evaluating the GT itself, no experiment sub-folder is needed
    exp_id = exp_id if model_name != 'gt' else ''

    # Enumerate all per-scene sub-directories under root_path
    runs = os.listdir(args.root_path)                              # list all entries
    runs = [run for run in runs
            if os.path.isdir(os.path.join(args.root_path, run))]  # keep directories only

    debug = args.debug   # int; 0 = off, 1 = on

    print(f'all {len(runs)} runs')   # log total number of scenes

    # Load the GT manifest (JSON list or dict of scene names)
    gt_json = args.gt_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)   # dict or list; keys/elements are scene name strings

    # Accumulators (populated in the per-scene loop below)
    preds    = []   # list of predicted trajectory arrays, one per scene
    pred_imgs = []  # (unused in default path) list of predicted frame path lists
    gt_imgs   = []  # (unused in default path) list of GT frame path lists

    # Base directory for GT ego motion numpy files
    gt_traj_path_base = 'data/ego_condition/ego_motion'
    # Each GT file: <gt_traj_path_base>/<scene_name>.npy → shape (T_total, ≥2)

    # ---- Pre-load FTD GT trajectories ------------------------------------
    # The FTD metric requires all GT trajectories to be processed and stacked
    # before calling get_ftd, so we pre-load them here.
    if args.metric == 'ftd' or args.metric == 'all':
        gt_trajs_ftd = []   # list of smoothed GT trajectories for FTD
        for s_name in gt_json:
            gt_path = os.path.join(gt_traj_path_base, s_name + '.npy')
            gt = np.load(gt_path, allow_pickle=True)
            # gt: shape (T_total, ≥2); take first 101 frames (10.1 seconds at 10 Hz)

            # Convert global XY → ego-centric (X_right, Z_forward)
            gt_local_xy, gt_local_yx, theta = gt_2_ego(gt[:101, :2])
            # gt_local_xy: shape (101, 2) — (X_right, Z_forward), for MTR encoder
            # gt_local_yx: shape (101, 2) — (-Z_forward, X_right), for visualisation
            # theta: scalar torch.Tensor — heading angle

            if True:   # always smooth GT trajectories for FTD
                gt_local_xy = smooth_traj_sg(gt_local_xy, dt=0.1, win_sec=0.4, poly=3)
                # gt_local_xy: shape (101, 2); Savitzky-Golay smoothed

            gt_trajs_ftd.append(gt_local_xy)   # accumulate; shape per element: (101, 2)
        # gt_trajs_ftd: list of N_scenes arrays each (101, 2) — collected GT for FTD

    # Allocate alignment GT list for the ego_condition track
    if args.track == 'ego_condition':
        gt_trajs_align = []   # list of smoothed GT trajectories for ADE/DTW alignment

    # ---- Per-scene prediction loop ---------------------------------------
    preds = []   # reset (was defined earlier; this is the active accumulator)
    for run in runs:
        s_name = run   # scene identifier string (matches GT file basename)

        # Path to the SLAM output PKL for this scene
        log_base = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')
        # Full path to PKL: <log_base>-estimate_ego_traj.pkl

        if args.track == 'ego_condition':
            # Load the GT for this scene to compute alignment
            gt_path = os.path.join(gt_traj_path_base, s_name + '.npy')
            gt = np.load(gt_path, allow_pickle=True)
            # gt: shape (T_total, ≥2); global XY + optional extra channels

            # Convert global GT to ego-centric frame
            gt_local_xy, gt_local_yx, theta = gt_2_ego(gt[:101, :2])
            # gt_local_xy: shape (101, 2) — (X_right, Z_forward)

        # ---- Load predicted trajectory PKL ---------------------------------
        with open(log_base + '-estimate_ego_traj.pkl', 'rb') as f:
            data = pickle.load(f)              # dict with keys including 'locs'
            pred = data['locs'].astype(np.float32)
        # pred: shape (N, 2); SLAM output in (x_right, z_forward) convention

        # Convert SLAM (x_right, z_forward) → (-z_forward, x_right) for GT matching
        pred_xy = ego_y_2_x(pred)             # shape: (N, 2)

        if args.track == 'ego_condition':
            # Align predicted trajectory to GT local frame (origin-fixed Umeyama)
            pred_xy, s, r = slam_align_to_gt_fix_origin(
                pred_xy, gt_local_xy, with_scale=False
            )
            # pred_xy: shape (N, 2); aligned to GT frame; s: scale (1.0); r: shape (2,2)

            # Smooth both prediction and GT for fair comparison
            pred_xy    = smooth_traj_sg(pred_xy,    dt=0.1, win_sec=0.4, poly=3)
            gt_local_xy = smooth_traj_sg(gt_local_xy, dt=0.1, win_sec=0.4, poly=3)
            # Both: shape (≤101, 2); Savitzky-Golay smoothed

            gt_trajs_align.append(gt_local_xy)   # accumulate smoothed GT for ADE/DTW
        else:
            # Free-drive track: no GT alignment, just smooth the prediction
            pred_xy = smooth_traj_sg(pred_xy, dt=0.1, win_sec=0.4, poly=3)
            # pred_xy: shape (N, 2); smoothed

        preds.append(pred_xy)   # accumulate per-scene prediction; each (N, 2)
    # preds: list of N_scenes arrays, each shape (N_i, 2)

    # Stack all per-scene predictions into a single array
    preds = np.array(preds)
    # preds: shape (N_scenes, T, 2) — assumes all scenes have the same T after smoothing

    # ---- Metric computation -----------------------------------------------

    # FTD: Fréchet Trajectory Distance (distribution-level metric)
    ftd = -1   # sentinel: -1 = not computed
    if args.metric == 'ftd' or args.metric == 'all':
        gt_trajs_ftd = np.array(gt_trajs_ftd)
        # gt_trajs_ftd: shape (N_scenes, 101, 2) — stacked GT trajectories

        ftd = get_ftd(preds, gt_trajs_ftd, stride=10)
        # stride=10: sample one keyframe per 10 frames for MTR windowing
        # ftd: scalar float — Fréchet distance in MTR feature space (lower = better)

    # Trajectory quality: comfort, curvature, speed score
    traj_quality = -1   # sentinel
    if args.metric == 'traj_q' or args.metric == 'all':
        traj_quality = get_traj_quality(preds)
        # traj_quality: scalar float — composite comfort/curvature/speed quality in [0,1]

    # Trajectory consistency: velocity and acceleration smoothness
    traj_consistency = -1   # sentinel
    if args.metric == 'traj_consist' or args.metric == 'all':
        traj_consistency = get_traj_consistency(preds)
        # traj_consistency: scalar float — smoothness score S = 0.5·exp(-σ_v/μ_v) + 0.5·exp(-σ_a/μ_a)

    # Trajectory alignment: ADE and DTW vs GT (ego_condition track only)
    traj_ade = -1   # sentinel
    traj_dtw = -1   # sentinel
    if args.metric == 'traj_align' or args.metric == 'all':
        traj_ade = get_ade(preds, gt_trajs_align)
        # traj_ade: scalar float — mean Average Displacement Error across all scenes (metres)

        traj_dtw = get_dtw(preds, gt_trajs_align)
        # traj_dtw: scalar float — mean Dynamic Time Warping distance across all scenes

    # ---- Aggregate and log results ----------------------------------------
    # Build the metrics dict for this model
    metrics = {
        'ftd':              ftd,              # scalar; -1 if not computed
        'traj_quality':     traj_quality,     # scalar; -1 if not computed
        'traj_consistency': traj_consistency, # scalar; -1 if not computed
        'traj_ade':         traj_ade,         # scalar; -1 if not computed
        'traj_dtw':         traj_dtw          # scalar; -1 if not computed
    }
    all_metrics[model_name] = metrics   # store under model name for multi-model comparison

    # Print each metric value; format scalars with 4 decimal places
    for sub_key, sub_val in all_metrics.items():
        if isinstance(sub_val, (float, int)):
            print(f"  {sub_key}: {sub_val:.4f}")   # formatted scalar
        else:
            print(f"  {sub_key}: {sub_val}")        # dict / list printed as-is
