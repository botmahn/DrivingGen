"""
extract_traj_agent_unidepth.py
==============================
Agent trajectory extraction pipeline for DrivingGen.

Pipeline overview
-----------------
1. UniDepthV2-ViT-L14  — monocular metric depth + intrinsic estimation
2. YOLOv10x            — object detection on the first frame
3. SAMURAI             — single-object tracking across all frames for each
                         detected agent (bounding box + binary mask per frame)
4. Geometry            — pixel-to-camera-to-world unprojection using the
                         estimated depth and ego camera poses
5. Output              — per-scene pkl files:
     • agents_traj.pkl       list of (frame_ids, world_xy_coords, label)
     • agents_bbox.pkl       SAMURAI-tracked bounding boxes
     • agents_bbox_label.pkl class-label dict keyed by "x1-y1-x2-y2" strings

Distributed execution
----------------------
The workload is sharded across RANK / WORLD_SIZE (environment variables) so
the script can be launched with torchrun or any MPI-style launcher.
Within each node a secondary split over --all_id local GPU slots is applied.

Usage
-----
    python extract_traj_agent_unidepth.py \
        --root_path   /data/scenes \
        --outdir      /data/output \
        --gt_meta_path /data/gt_meta.json \
        --model_name  gt \
        --exp_id      free \
        --local_id    0 \
        --all_id      8
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import argparse
import math
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# Third-party numerical / vision imports
# ---------------------------------------------------------------------------
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize
import json

# ---------------------------------------------------------------------------
# UniDepth — monocular metric depth estimation
# ---------------------------------------------------------------------------
from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize, image_grid

# ---------------------------------------------------------------------------
# Visual-SLAM helpers (feature extraction, matching, trajectory estimation)
# ---------------------------------------------------------------------------
from visual_slam.dataset import *
from visual_slam.vo import *

# ---------------------------------------------------------------------------
# YOLOv10 — object detection
# ---------------------------------------------------------------------------
from ultralytics import YOLOv10

# ---------------------------------------------------------------------------
# SAMURAI — single-object video tracker built on SAM2
# ---------------------------------------------------------------------------
# Add the bundled third-party directory to the module search path so that
# "samurai" can be imported without a global installation.
sys.path.append(os.path.abspath('third_parties'))
from samurai.scripts.demo import samurai_main

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised to avoid GPU allocation at import)
# ---------------------------------------------------------------------------
depth_model = None   # UniDepthV2 instance (set by init_depth_model)
det_model   = None   # YOLOv10   instance (set by init_det_model)


# ---------------------------------------------------------------------------
# Model initialisation helpers
# ---------------------------------------------------------------------------

def init_depth_model():
    """
    Download (or load from Hugging Face cache) UniDepthV2-ViT-L14 and move it
    to the available CUDA device.

    Configuration choices
    ---------------------
    - resolution_level = 0   : highest internal processing resolution
    - interpolation_mode = "bilinear" : smooth upsampling of depth predictions

    Side effects
    ------------
    Sets the module-level ``depth_model`` singleton so all subsequent calls to
    ``depth_model.infer()`` use the same loaded weights.
    """
    global depth_model

    print("Torch version:", torch.__version__)

    # Model identifier on Hugging Face Hub
    name = "unidepth-v2-vitl14"

    # Load pre-trained weights; falls back to local HF cache when offline
    depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # -------------------------------------------------------------------------
    # V2-only settings
    # -------------------------------------------------------------------------
    # resolution_level=0 keeps the maximum internal resolution; increasing this
    # value downsamples the feature map for faster (but less detailed) inference.
    depth_model.resolution_level = 0

    # Bilinear interpolation when up-sampling the depth output to the original
    # image resolution (alternative: "nearest" — faster but blockier).
    depth_model.interpolation_mode = "bilinear"

    # Move model to GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = depth_model.to(device)


def init_det_model():
    """
    Load the YOLOv10x detection model from a local checkpoint and move it to
    the first available CUDA device.

    Side effects
    ------------
    Sets the module-level ``det_model`` singleton.

    Notes
    -----
    The checkpoint path is hard-coded to the shared-disk location used during
    the original DrivingGen experiments; update it if running elsewhere.
    """
    global det_model
    # Load YOLOv10x weights and push to GPU
    det_model = YOLOv10(
        '/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/yolov10x.pt'
    ).cuda()


# ---------------------------------------------------------------------------
# Distributed workload scheduling
# ---------------------------------------------------------------------------

def set_task_list(base_path, local_rank, gt_json=None,
                  model_name='gt', exp_id='free', all_id=8):
    """
    Build the list of scene directories this process should handle.

    The full scene list is first shuffled with a fixed seed (2026) for
    reproducibility and then split uniformly across ``WORLD_SIZE`` global
    ranks. Each global rank's slice is further split across ``all_id`` local
    GPU slots so every GPU in a multi-GPU node handles a distinct subset.

    Parameters
    ----------
    base_path : str
        Root directory that contains per-scene sub-folders.
    local_rank : int
        Index of the current GPU / local process (0 … all_id-1).
    gt_json : str, optional
        Path to a JSON file listing ground-truth scene directories.
        Required when ``model_name == 'gt'``; ignored otherwise.
    model_name : str
        Either ``'gt'`` (ground-truth data) or the name of the generative
        model being evaluated (e.g. ``'cosmos'``, ``'wan'``).
    exp_id : str
        Experiment identifier appended to each scene path when
        ``model_name != 'gt'``.
    all_id : int
        Number of local GPU slots per node (default 8).

    Returns
    -------
    runs : list[str]
        Absolute paths to the frame directories this process must process.
        For GT scenes the path ends in ``…/CAM_F0``; for generated scenes
        it ends in ``…/images``.
    """
    runs = []

    # ------------------------------------------------------------------
    # Step 1: Collect the raw scene list
    # ------------------------------------------------------------------
    if model_name == 'gt':
        # Ground-truth mode: scene list is specified in an external JSON file
        with open(gt_json, 'r') as f:
            gt_json = json.load(f)   # list of absolute scene directory strings
        dirs = gt_json
    else:
        # Generated-video mode: discover scenes by listing the root directory
        dirs = os.listdir(base_path)

    # ------------------------------------------------------------------
    # Step 2: Build per-scene paths
    # ------------------------------------------------------------------
    if model_name == 'gt':
        # Ground-truth scenes are referenced directly (no model/exp suffix)
        scenes = dirs
    else:
        # Generated scenes follow the convention: base/scene/model/exp_id
        scenes = [os.path.join(base_path, f, model_name, exp_id) for f in dirs]

    # ------------------------------------------------------------------
    # Step 3: Read distributed-training environment variables
    # ------------------------------------------------------------------
    # RANK      — global rank of this process across all nodes (default 0)
    # WORLD_SIZE — total number of processes launched   (default 1)
    global_rank = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get('WORLD_SIZE', 1))

    # Shuffle with a fixed seed so every process sees the same ordering
    # before slicing; this ensures reproducible assignment without coordination.
    random.seed(2026)
    random.shuffle(scenes)

    # ------------------------------------------------------------------
    # Step 4: Two-level data partitioning
    # ------------------------------------------------------------------
    # Level 1: split across global ranks (nodes)
    data_chunks      = np.array_split(scenes, world_size)
    local_data_chunk = data_chunks[global_rank]  # slice for this node

    # Level 2: split across local GPU slots within this node
    local_data_chunks = np.array_split(local_data_chunk, all_id)
    scenes = local_data_chunks[local_rank % all_id]  # slice for this GPU

    # ------------------------------------------------------------------
    # Step 5: Build final frame-directory paths
    # ------------------------------------------------------------------
    for scene in scenes:
        if model_name == 'gt':
            # Ground-truth sensor data: front-camera frames live in CAM_F0/
            runs.append(os.path.join(scene, 'CAM_F0'))
        else:
            # Generated frames: all images are stored in images/
            runs.append(os.path.join(scene, 'images'))

    print(f"Total {len(runs)} data to process, local index {local_rank}")

    return runs


# ---------------------------------------------------------------------------
# Geometry: pixel → camera → world
# ---------------------------------------------------------------------------

def reconstruct_global_trajectory(pixel_centers, depth_values, Ks,
                                   camera_poses, delta_d=None):
    """
    Reconstruct the top-down (x, z) world-space trajectory of a tracked agent.

    For each frame the pipeline performs a two-step unprojection:

      1. **Pixel → Camera**  ``cam_coord = K⁻¹ · [u, v, 1]ᵀ · Z``
         Converts the 2-D pixel location together with the metric depth Z into
         a 3-D point in the camera's local coordinate frame.

      2. **Camera → World**  ``world_coord = R · cam_coord + T``
         Applies the (R, T) extrinsic to express the point in the fixed world
         frame.  Only the X and Z components are retained for the top-down
         bird's-eye-view trajectory.

    Parameters
    ----------
    pixel_centers : list of (float, float)
        Per-frame (x, y) pixel coordinates of the agent centre.
        Length: N_frames.
    depth_values : list of float
        Per-frame metric depth in **metres** at the agent's pixel centre.
        Length: N_frames.
    Ks : list of np.ndarray, shape (3, 3)
        Per-frame camera intrinsic matrices.
        Layout: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        Length: N_frames.
    camera_poses : list of tuple (R, T)
        Per-frame world-from-camera extrinsics.
        R : np.ndarray, shape (3, 3) — rotation matrix.
        T : np.ndarray, shape (3,)   — translation vector (metres).
        Length: N_frames.
    delta_d : list or np.ndarray of float, optional
        Per-frame depth correction offsets (e.g. from a learned residual).
        When provided, ``Z_c += delta_d[i]`` before unprojection.
        Length: N_frames (or None to skip correction).

    Returns
    -------
    global_trajectory : list of np.ndarray, shape (2,)
        Per-frame [x, z] world coordinates (top-down 2-D projection).
        Length: N_frames.

    Notes
    -----
    The Y axis is dropped because driving scenes are planar and the top-down
    projection (X-Z plane) is the standard representation for trajectory
    evaluation metrics.
    """
    global_trajectory = []

    for i, ((x_c, y_c), Z_c) in enumerate(zip(pixel_centers, depth_values)):

        # ------------------------------------------------------------------
        # Step 1: Pixel coordinates → Camera coordinates
        # ------------------------------------------------------------------
        # Invert the intrinsic matrix for this specific frame.
        # K_inv : shape (3, 3)
        K_inv = np.linalg.inv(Ks[i])

        # Homogeneous pixel coordinate; shape (3,)
        pixel_coord = np.array([x_c, y_c, 1])

        # Optionally apply a per-frame depth correction offset
        if delta_d is not None:
            Z_c += delta_d[i].item()   # scalar depth correction (metres)

        # Back-project: cam_coord = K⁻¹ · [u,v,1]ᵀ · Z
        # cam_coord : shape (3,)  [X_cam, Y_cam, Z_cam] in metres
        cam_coord = (K_inv @ pixel_coord) * Z_c

        # ------------------------------------------------------------------
        # Step 2: Camera coordinates → World coordinates
        # ------------------------------------------------------------------
        R, T = camera_poses[i]  # R : (3,3), T : (3,)

        # Apply extrinsic: world_coord = R · cam_coord + T
        # world_coord : shape (3,)  [X_world, Y_world, Z_world] in metres
        world_coord = R @ cam_coord + T

        # Keep only the top-down (X, Z) components; Y is vertical height
        # Appended element shape: (2,)
        global_trajectory.append(world_coord[[0, -1]])  # [X_world, Z_world]

    return global_trajectory


# ---------------------------------------------------------------------------
# Detection: YOLOv10 on a single RGB frame
# ---------------------------------------------------------------------------

def det_obj(frame):
    """
    Run YOLOv10 detection on a single RGB image and produce a binary mask that
    marks all **movable** objects (to be excluded from the VisualSLAM feature
    matching step), as well as the pixel centres, bounding boxes, and class
    labels of agents that will be tracked by SAMURAI.

    COCO class taxonomy used
    ------------------------
    Movable classes (zeroed in the VisualSLAM mask) — index set:
        {0,1,2,3,4,5,6,7,8,14,15,16,17,18,19,20,21,22,23}
        → person, bicycle, car, motorcycle, airplane, bus, train, truck,
          boat, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
          giraffe

    Tracked classes (trajectory targets) — index set: {0,1,2,3,5,7}
        → person, bicycle, car, motorcycle, bus, truck
        Minimum detection confidence: 0.3

    Parameters
    ----------
    frame : np.ndarray, shape (H, W, 3), dtype uint8
        RGB image in HWC layout (as returned by ``PIL.Image`` → ``np.array``).

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask where pixels belonging to movable objects are set to 0
        and the rest are 255.  Used to suppress dynamic regions during
        VisualSLAM feature extraction.
    person_car_pixel : list of [int, int]
        Pixel-space (x, y) centres of all tracked-class detections that meet
        the confidence threshold.
    person_car_bbox : list of [int, int, int, int]
        Axis-aligned bounding boxes [x1, y1, x2, y2] for the same detections.
    person_car_label : dict {str: str}
        Maps "x1-y1-x2-y2" box key to class name string (e.g. "car").

    Notes
    -----
    The distinction between "movable" and "tracked" classes is deliberate:
    all movable objects corrupt VisualSLAM (hence the mask), but only
    person / vehicle classes are meaningful trajectory targets.
    """
    global det_model

    # Run YOLOv10 on the full-resolution RGB frame
    # result_list has length 1 (batch size 1)
    result = det_model(frame)
    result = result[0]   # unpack single-image result

    # YOLOv10 result attributes (unused ones kept as documentation)
    boxes     = result.boxes      # Boxes: bounding box detections
    masks     = result.masks      # Masks: segmentation masks (None for detection-only)
    keypoints = result.keypoints  # Keypoints: pose estimation (None here)
    probs     = result.probs      # Probs: classification scores (None here)
    obb       = result.obb        # OBB: oriented bounding boxes (None here)

    # ------------------------------------------------------------------
    # COCO class reference (kept for readability; detection uses indices)
    # ------------------------------------------------------------------
    # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    #  5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    #  10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    #  14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    #  20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', ...}

    # Class indices whose pixel regions will be masked out for VisualSLAM
    # (all dynamic / movable objects in the COCO taxonomy up to index 23)
    movable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # Initialise the VisualSLAM mask to all-valid (255 = keep for feature matching)
    # mask : shape (H, W), dtype uint8
    mask = (np.ones((frame.shape[0], frame.shape[1])) * 255).astype(np.uint8)

    # Containers for the trajectory-relevant detections
    person_car_pixel = []          # list of [cx, cy] pixel centres
    person_car_bbox  = []          # list of [x1, y1, x2, y2] boxes
    person_car_label = {}          # "x1-y1-x2-y2" → class name string

    # Mapping from COCO index to human-readable label for tracked classes only
    lookup_dict = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }

    if boxes:
        num_box = len(boxes)   # total number of detections in this frame

        for id in range(num_box):
            # boxes.cls[id] : scalar tensor — COCO class index
            cls_id = int(boxes.cls[id].item())

            if cls_id in movable:
                # ----------------------------------------------------------
                # Mask out the bounding box region so VisualSLAM ignores it
                # ----------------------------------------------------------
                # boxes.data[id, :4] : tensor, shape (4,) — [x1, y1, x2, y2]
                xy = boxes.data[id, :4].cpu().numpy()  # shape (4,), float32
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])

                # Zero-out the bounding-box region in the VisualSLAM mask
                # mask[y1:y2, x1:x2] : shape (H_box, W_box), set to 0
                mask[y1:y2, x1:x2] = 0

                # ----------------------------------------------------------
                # Record tracked-class detections above confidence threshold
                # ----------------------------------------------------------
                # Tracked classes: person, bicycle, car, motorcycle, bus, truck
                # Confidence gate: >= 0.3
                if cls_id in [0, 1, 2, 3, 5, 7] and boxes.conf[id].item() >= 0.3:
                    # Compute the pixel-space centre of the bounding box
                    mid_x = int((x1 + x2) / 2)   # horizontal centre
                    mid_y = int((y1 + y2) / 2)   # vertical centre

                    person_car_pixel.append([mid_x, mid_y])
                    person_car_bbox.append([x1, y1, x2, y2])

                    # Key format matches the label look-up in the main loop
                    person_car_label[f'{x1}-{y1}-{x2}-{y2}'] = lookup_dict[cls_id]

    return mask, person_car_pixel, person_car_bbox, person_car_label


# ---------------------------------------------------------------------------
# Depth estimation from a segmentation mask
# ---------------------------------------------------------------------------

def estimate_depth_from_mask(depth, mask, method='median_top', use_percentile=False):
    """
    Estimate the representative depth of a masked object in a depth image.

    Naively taking the median over the entire object mask is unreliable because
    background leakage and occlusion cause large variance at the object boundary.
    This function mitigates that by restricting the statistics to the **top half**
    of the bounding row range of the mask — pixels that are closer to the camera
    tend to produce more accurate UniDepth estimates than those near the feet
    (ground contact) or the top of tall vehicles.

    Parameters
    ----------
    depth : np.ndarray, shape (H, W), dtype float32 or float64
        Per-pixel metric depth in **metres** as predicted by UniDepthV2.
    mask : np.ndarray, shape (H, W), dtype bool or uint8
        Binary object mask where non-zero pixels belong to the target object
        (as produced by the SAMURAI tracker).
    method : str, one of {'median_top', 'mean_top', 'full_median', 'full_mean'}
        Aggregation strategy:
        - ``'*_top'``  : restrict to the upper 50 % of the mask's row span
          (rows with index < y_min + 0.5 * height), which are more reliable.
        - ``'mean_*'`` : compute the arithmetic mean of valid depth pixels.
        - ``'median_*'``: compute the median of valid depth pixels (default).
    use_percentile : bool
        When True, clip the valid depth values to the [2nd, 98th] percentile
        range before aggregating, removing extreme outliers caused by sensor
        noise or background leakage.

    Returns
    -------
    depth_est : float or None
        Estimated depth value in **metres**.
        Returns ``None`` if the mask contains no valid pixels after filtering.

    Examples
    --------
    >>> depth_est = estimate_depth_from_mask(
    ...     depth_map,             # shape (H, W)
    ...     samurai_mask,          # shape (H, W) bool
    ...     method='mean',
    ...     use_percentile=True,
    ... )
    """
    # Guard: return None if the mask contains no foreground pixels
    if not np.any(mask):
        return None

    # ------------------------------------------------------------------
    # Step 1: Locate the bounding row range of the mask
    # ------------------------------------------------------------------
    # ys, xs : 1-D arrays of row / column indices of foreground pixels
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min + 1   # total height of the object in pixels

    # ------------------------------------------------------------------
    # Step 2: Optionally restrict to the upper half of the mask
    # ------------------------------------------------------------------
    if 'top' in method:
        # Threshold at 50 % of the bounding box height from the top
        # (pixels with row index >= threshold are suppressed)
        threshold = y_min + h * 0.5

        # Work on a copy so we do not modify the caller's mask
        top_mask = mask.copy()
        for y, x in zip(ys, xs):
            if y >= threshold:
                # Zero out pixels in the lower half of the object region
                top_mask[y, x] = False
        target_mask = top_mask    # shape (H, W), bool
    else:
        # Use the full mask without any row-based filtering
        target_mask = mask        # shape (H, W), bool

    # ------------------------------------------------------------------
    # Step 3: Extract valid depth values inside the target mask
    # ------------------------------------------------------------------
    # valid : 1-D array of metric depth values in the selected region
    valid = depth[target_mask > 0]   # shape (N_valid,), float

    if len(valid) == 0:
        return None   # mask became empty after top-half filtering

    # ------------------------------------------------------------------
    # Step 4: Optional percentile clipping to remove extreme outliers
    # ------------------------------------------------------------------
    if use_percentile:
        # Compute the 2nd and 98th percentile gates
        lower, upper = np.percentile(valid, [2, 98])
        # Keep only depth values within the [2%, 98%] range
        valid = valid[(valid >= lower) & (valid <= upper)]   # shape (N_clipped,)

    # ------------------------------------------------------------------
    # Step 5: Aggregate to a single depth estimate
    # ------------------------------------------------------------------
    if 'mean' in method:
        return float(np.mean(valid))    # arithmetic mean in metres
    else:
        return float(np.median(valid))  # median in metres (more robust)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Command-line argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Agent trajectory extraction using UniDepthV2 + SAMURAI + YOLOv10'
    )
    parser.add_argument('--root_path',     type=str,
                        help='Root directory containing per-scene sub-folders.')
    parser.add_argument('--outdir',        type=str, default='./vis_depth',
                        help='Output directory for depth frames, tracks, and pkl files.')
    parser.add_argument('--gt_meta_path',  type=str, default='./vis_depth',
                        help='Path to the ground-truth meta JSON file (used when '
                             '--model_name gt).')
    parser.add_argument('--model_name',    type=str, default='gt',
                        help='Model name; use "gt" for ground-truth data.')
    parser.add_argument('--exp_id',        type=str, default='free',
                        help='Experiment identifier appended to generated-scene paths.')
    parser.add_argument('--local_id',      type=int, default=0,
                        help='Local GPU index (0 … all_id-1).')
    parser.add_argument('--all_id',        type=int, default=8,
                        help='Total number of local GPU slots on this node.')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Model initialisation (done once before the scene loop)
    # ------------------------------------------------------------------
    init_depth_model()   # loads UniDepthV2-ViT-L14 onto GPU
    init_det_model()     # loads YOLOv10x onto GPU

    # ------------------------------------------------------------------
    # Workload distribution: get the scene paths for this GPU
    # ------------------------------------------------------------------
    runs = set_task_list(
        args.root_path,
        int(args.local_id),
        args.gt_meta_path,
        args.model_name,
        args.exp_id,
        args.all_id,
    )

    model_name = args.model_name
    exp_id     = args.exp_id

    # ==================================================================
    # Main per-scene processing loop
    # ==================================================================
    for run in runs:
        print(f'{args.local_id}: {run}')

        # ------------------------------------------------------------------
        # Stage 0: Collect and sort frame file paths
        # ------------------------------------------------------------------
        filenames = os.listdir(run)
        filenames = [os.path.join(run, f) for f in filenames]
        filenames.sort()   # ensure chronological order (lexicographic on zero-padded names)

        # Per-scene accumulators
        rgbs           = []   # list of np.ndarray, each (H, W, 3) uint8
        depths_raw     = []   # list of np.ndarray, each (H, W) float32 — metric depth in m
        points_3d      = []   # list of np.ndarray, each (H, W, 3) — 3-D point cloud in camera frame
        rgb_intrinsics = []   # list of np.ndarray, each (3, 3) — camera intrinsics per frame
        masks          = []   # list of np.ndarray, each (H, W) uint8 — VisualSLAM exclusion masks
        depths_mov     = []   # (unused accumulator, reserved for future use)
        track_masks    = []   # (unused accumulator, reserved for future use)
        track_bboxs    = []   # (unused accumulator, reserved for future use)

        # ------------------------------------------------------------------
        # Build output directory structure
        # ------------------------------------------------------------------
        if model_name == 'gt':
            # GT path pattern: …/<scene_a>+<scene_b>/<cam>/
            s_name   = run.split('/')[-3] + '+' + run.split('/')[-2]
            log_base = os.path.join(args.outdir, s_name, model_name, 'unidepth')
        else:
            # Generated path pattern: …/<scene>/<model>/<exp_id>/images/
            s_name   = run.split('/')[-4]
            log_base = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')

        # Directory for per-frame colourised depth images
        depth_out_dir = os.path.join(log_base, 'depth_frame')
        os.makedirs(depth_out_dir, exist_ok=True)

        # Directory for per-frame tracking overlay images
        seg_out_dir = os.path.join(log_base, 'track_frame')
        os.makedirs(seg_out_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Stage 1: Per-frame depth estimation and detection
        # ------------------------------------------------------------------
        # person_car_bbox and person_car_label are detected only on frame 0
        # and passed to SAMURAI for tracking; initialise here so they are
        # always in scope even if the loop body exits early.
        person_car        = []
        person_car_bbox   = []
        person_car_label  = {}

        for k, filename in enumerate(filenames):
            # When the experiment uses 20 conditioning frames, skip the first
            # 19 (indices 0-18) so processing starts at the 20th frame (index
            # 19), giving 81 frames in total.
            if 'conds_20' in args.exp_id and k < 19:
                continue

            # Load the frame as an RGB numpy array; shape (H, W, 3), dtype uint8
            rgb = np.array(Image.open(filename))

            # ----------------------------------------------------------
            # Detection: run YOLOv10 only on the first processed frame
            # ----------------------------------------------------------
            if k == 0:
                # Detect agents; retain bbox & label for SAMURAI initialisation
                # mask : (H, W) uint8 — VisualSLAM exclusion mask
                # person_car : list of [cx, cy]
                # person_car_bbox : list of [x1, y1, x2, y2]
                # person_car_label : dict {"x1-y1-x2-y2": class_name}
                mask, person_car, person_car_bbox, person_car_label = det_obj(rgb)
            else:
                # For subsequent frames only the VisualSLAM mask is needed
                mask, _, _, _ = det_obj(rgb)

            masks.append(mask)   # accumulate VisualSLAM masks; list of (H, W)

            # ----------------------------------------------------------
            # Store the raw RGB frame for VisualSLAM and downstream use
            # ----------------------------------------------------------
            rgbs.append(rgb.copy())   # shape (H, W, 3), uint8

            # Convert to CHW torch tensor for UniDepthV2 inference
            # rgb_torch : shape (3, H, W), uint8
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

            # ----------------------------------------------------------
            # Stage 2: UniDepthV2 inference — depth + intrinsics + points
            # ----------------------------------------------------------
            # predictions dict keys: "depth", "intrinsics", "points"
            predictions = depth_model.infer(rgb_torch, None)

            # Metric depth map; squeeze removes the batch dimension
            # depth : shape (H, W), float32, values in metres
            depth = predictions["depth"].squeeze().cpu().numpy()
            depths_raw.append(depth.copy())

            # Predicted camera intrinsics for this frame
            # intrinsics : shape (3, 3), float32
            intrinsics = predictions["intrinsics"].squeeze(0).cpu().numpy()
            rgb_intrinsics.append(intrinsics)

            # 3-D point cloud in camera coordinate frame
            # predictions["points"] : shape (1, 3, H, W)  (batch, XYZ, rows, cols)
            # xyz (flat) : shape (H*W, 3) — used for legacy VisualSLAM path
            xyz = predictions["points"].cpu().squeeze(0).reshape(3, -1).t().numpy()
            # points_3d_i : shape (H, W, 3), stored per frame
            points_3d.append(
                predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy().copy()
            )

            # Colourised depth map for optional debug visualisation
            # depth_pred_col : shape (H, W, 3), uint8, jet colormap
            depth_pred_col = colorize(depth, cmap="jet")
            # cv2.imwrite(os.path.join(depth_out_dir, f'{k:05}.png'), depth_pred_col)

        # ------------------------------------------------------------------
        # Helper: write a list of images to an MP4 video
        # ------------------------------------------------------------------
        from moviepy.editor import ImageSequenceClip

        def images_to_video(image_folder, output_video, fps=30):
            """
            Encode a directory of sorted PNG/JPG images into an MP4 file.

            Parameters
            ----------
            image_folder : str
                Directory containing the image frames (PNG or JPG).
            output_video : str
                Output MP4 file path.
            fps : int
                Frames per second for the output video (default 30).
            """
            # Collect all PNG/JPG images and sort lexicographically
            images = [
                os.path.join(image_folder, img)
                for img in os.listdir(image_folder)
                if (img.endswith(".png") or img.endswith(".jpg"))
            ]
            images.sort()   # ensure chronological / zero-padded order

            clip = ImageSequenceClip(images, fps=fps)
            clip.write_videofile(
                output_video,
                codec="libx264",
                verbose=False,
                logger=None,
            )

        # ------------------------------------------------------------------
        # Stage 3: Ego trajectory estimation (VisualSLAM)
        # ------------------------------------------------------------------
        # If a cached ego trajectory already exists on disk, load it;
        # otherwise run the full VisualSLAM pipeline.
        if not os.path.exists(log_base + '-estimate_ego_traj.pkl'):
            # -----------------------------------------------------------------
            # Full VisualSLAM pipeline (runs only if no cached result exists)
            # -----------------------------------------------------------------
            import pdb
            pdb.set_trace()   # breakpoint for debugging missing ego trajectories

            # Path to pre-recorded ground-truth camera intrinsics
            meta_path = '/mnt/cache/zhouyang/dg-bench/nuplan_1.1/val_sensor_data_10hz_0530'
            s_name = s_name.split('+')
            intrinsics_path = os.path.join(
                meta_path,
                s_name[0] + '+' + s_name[1],
                s_name[2],
                'intrinsic.npy',
            )

            # For 20-conditioning-frame experiments, use frames 19-99 (81 frames)
            if 'conds_20' in args.exp_id:
                rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)[19:100]
            else:
                rgb_intrinsics = np.load(intrinsics_path, allow_pickle=True)

            # Wrap the collected data into the VisualSLAM dataset container
            dataset_handler = DatasetHandler(rgbs, depths_raw, rgb_intrinsics)

            check_data(dataset_handler, args.outdir)

            # Part I: Extract ORB/SIFT features from all frames
            # images : list of processed grayscale frames
            # masks  : list of (H, W) uint8 — regions excluded from feature detection
            images = dataset_handler.images
            kp_list, des_list = extract_features_dataset(images, masks)
            # kp_list  : list of N_frames × N_keypoints keypoints
            # des_list : list of N_frames × (N_keypoints, D) descriptors

            # Part II: Match feature descriptors across consecutive frame pairs
            matches, matches2 = match_features_dataset(des_list)

            # Optionally filter matches by Lowe's ratio test
            is_main_filtered_m = True   # set False to use raw matches
            if is_main_filtered_m:
                dist_threshold  = 0.7   # Lowe ratio threshold (0.7 is standard)
                filtered_matches = filter_matches_dataset(matches, dist_threshold, matches2)
                matches = filtered_matches

            # Part III: Estimate camera trajectory from filtered matches
            depth_maps = dataset_handler.depth_maps
            try:
                # trajectory  : shape (3, N_frames) — world-space XYZ of camera centre
                # poses_3x3   : list of N_frames (R 3×3, T 3) tuples
                trajectory, poses_3x3 = estimate_trajectory(
                    matches, kp_list, dataset_handler.k,
                    depth_maps=depth_maps,
                    dataset_handler=dataset_handler,
                )
            except Exception as e:
                print(e)
                print(f'extract ego fail for {run}')
                continue   # skip this scene on failure

            # Extract the 2-D top-down (X, Z) ego positions
            locs = []
            for i in range(0, trajectory.shape[1]):
                current_pos = trajectory[:, i]   # shape (3,)
                locs.append([current_pos.item(0), current_pos.item(2)])   # [X, Z]

            # locs : shape (N_frames, 2), dtype float32
            locs = np.array(locs, dtype=np.float32)

            # Cache the ego trajectory to disk
            ego_traj = {
                'locs':     locs,        # shape (N_frames, 2)
                'poses_3x3': poses_3x3,  # list of N_frames (R, T) tuples
            }
            import pickle
            with open(log_base + '-estimate_ego_traj.pkl', 'wb') as f:
                pickle.dump(ego_traj, f)

        else:
            # Load previously computed ego trajectory from cache
            print('loading existing ego')
            import pickle
            with open(log_base + '-estimate_ego_traj.pkl', 'rb') as f:
                ego_traj = pickle.load(f)
            locs      = ego_traj['locs']        # shape (N_frames, 2)
            poses_3x3 = ego_traj['poses_3x3']   # list of (R 3×3, T 3) tuples

        # ------------------------------------------------------------------
        # Stage 4: SAMURAI tracking for each detected agent
        # ------------------------------------------------------------------
        # Accumulators for all tracked agents in this scene
        mov_track_bbox  = []   # list of N_agents × list of (frame_id, [x1,y1,x2,y2])
        mov_track_mask  = []   # list of N_agents × dict{frame_id: (H,W) bool mask}
        mov_track_label = []   # list of N_agents class name strings

        if len(person_car_bbox) > 0:
            # Import colour palette helpers for visualisation overlays
            import seaborn as sns
            import matplotlib.colors as mcolors

            def sns_to_cv_colors(n: int, palette="husl"):
                """
                Generate ``n`` visually distinct BGR colours using a seaborn palette.

                Parameters
                ----------
                n : int
                    Number of colours to generate.
                palette : str
                    Seaborn colour palette name (default ``"husl"``).

                Returns
                -------
                list of (int, int, int)
                    BGR tuples with values in [0, 255] suitable for OpenCV drawing.
                """
                rgb = sns.color_palette(palette, n)   # list of n (R, G, B) in [0, 1]
                # Swap R↔B for OpenCV's BGR convention
                return [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in rgb]

            # Generate 120 distinct colours; each agent gets a unique colour index
            colors = sns_to_cv_colors(120, "husl")

            # Sort detected bounding boxes left-to-right, top-to-bottom
            # (centre-x primary, centre-y secondary) for deterministic agent IDs
            person_car_bbox_sorted = sorted(
                person_car_bbox,
                key=lambda b: (
                    (b[0] + b[2]) / 2,   # centre x — left to right
                    (b[1] + b[3]) / 2,   # centre y — top to bottom
                )
            )

            for box_id, bbox in enumerate(person_car_bbox_sorted):
                # Retrieve the class label for this detection
                mov_track_label.append(
                    person_car_label[f'{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}']
                )

                # --------------------------------------------------------
                # Run SAMURAI tracker initialised with the YOLO bounding box
                # --------------------------------------------------------
                # track_mask : dict {frame_id: np.ndarray (H, W) bool}
                # track_bbox : list of (frame_id, [x1, y1, x2, y2])
                track_mask, track_bbox = samurai_main(
                    video_path=filenames,     # list of sorted image paths
                    txt_path=bbox,            # initial [x1, y1, x2, y2] box
                    save_to_video=False,
                    seg_out_dir=seg_out_dir,
                )

                # --------------------------------------------------------
                # Draw tracked bounding boxes onto the frame images
                # --------------------------------------------------------
                # Use the original frame directory for the first agent,
                # the tracking output directory for subsequent agents
                # (avoids overwriting source frames).
                if box_id > 0:
                    img_dir = seg_out_dir
                else:
                    img_dir = run

                # Build a frame_id → list-of-boxes look-up for fast rendering
                boxes_dict = {}
                for f_id, bbox_i in track_bbox:
                    if f_id not in boxes_dict:
                        boxes_dict[f_id] = [bbox_i]
                    else:
                        boxes_dict[f_id].append(bbox_i)

                # Overlay boxes onto each image and save to the tracking dir
                for fn in os.listdir(img_dir):
                    img = cv2.imread(os.path.join(img_dir, fn))
                    # img : shape (H, W, 3), uint8, BGR

                    # Draw all boxes for this frame (may be empty)
                    for (x1, y1, x2, y2) in boxes_dict.get(
                        int(fn.split('.')[0].split('_')[0]), []
                    ):
                        cv2.rectangle(img, (x1, y1), (x2, y2), colors[box_id], 2)

                    # Save the annotated frame to the tracking output directory
                    cv2.imwrite(os.path.join(seg_out_dir, fn), img)

                mov_track_mask.append(track_mask)   # append agent's per-frame masks
                mov_track_bbox.append(track_bbox)   # append agent's per-frame boxes

            # Encode the tracking visualisation into a video for review
            images_to_video(seg_out_dir, os.path.join(log_base, "track.mp4"), fps=10)

        # ------------------------------------------------------------------
        # Stage 5: 3-D trajectory reconstruction for each tracked agent
        # ------------------------------------------------------------------
        if len(mov_track_bbox) > 0:
            # Per-agent accumulators for depth-estimation inputs
            mov_pixel_frames  = []   # list of N_agents × list of (cx, cy)
            mov_depth_centers = []   # list of N_agents × list of float (depth m)
            mov_ids           = []   # list of N_agents × list of frame_ids

            for mov_id, mov_box_i in enumerate(mov_track_bbox):
                this_pixel_frames  = []   # (cx, cy) per tracked frame
                this_depth_centers = []   # depth estimate (m) per tracked frame
                mov_mask_i = mov_track_mask[mov_id]   # dict {frame_id: (H,W) bool}

                # Record the frame indices where this agent was successfully tracked
                mov_ids.append([m[0] for m in mov_box_i])

                for (frame_id, bbox) in mov_box_i:
                    x1, y1, x2, y2 = bbox

                    # Pixel-space centre of the tracked bounding box
                    pc_x = int((x1 + x2) / 2)   # horizontal centre
                    pc_y = int((y1 + y2) / 2)   # vertical centre

                    # Retrieve the SAMURAI binary mask for this frame
                    # mask_i : shape (H, W), bool
                    mask_i = (mov_mask_i[frame_id] > 0)
                    assert mask_i.sum() > 0, \
                        f"Empty SAMURAI mask for agent {mov_id} at frame {frame_id}"

                    # Estimate metric depth using the top-half of the mask
                    # (arithmetic mean after 2-98 percentile clipping)
                    # pc_d : float, depth in metres
                    pc_d = estimate_depth_from_mask(
                        depths_raw[frame_id],      # shape (H, W), depth map
                        mov_mask_i[frame_id],      # shape (H, W), agent mask
                        method='mean',
                        use_percentile=True,
                    )

                    this_pixel_frames.append((pc_x, pc_y))
                    this_depth_centers.append(pc_d)

                mov_pixel_frames.append(this_pixel_frames)
                mov_depth_centers.append(this_depth_centers)

            # Reconstruct the world-space trajectory for every tracked agent
            agents_traj = []
            for id_p, (pixel_values, depth_values) in enumerate(
                zip(mov_pixel_frames, mov_depth_centers)
            ):
                mov_label = mov_track_label[id_p]   # class name string
                ids_p = mov_ids[id_p]               # list of frame indices

                # Select the intrinsics and camera poses for the tracked frames
                this_intrinsics = [rgb_intrinsics[id] for id in ids_p]  # list of (3,3)
                this_poses      = [poses_3x3[id]      for id in ids_p]  # list of (R,T)

                # Project pixel+depth → world XZ coordinates
                # focal_global_coords : list of shape (2,) arrays — [X, Z] in metres
                focal_global_coords = reconstruct_global_trajectory(
                    pixel_values,      # list of (cx, cy)
                    depth_values,      # list of float (metres)
                    this_intrinsics,   # list of (3, 3) intrinsics
                    this_poses,        # list of (R 3×3, T 3) extrinsics
                )

                # Store as (frame_ids, world_xy_list, class_label) tuple
                agents_traj.append((ids_p, focal_global_coords, mov_label))

            # ------------------------------------------------------------------
            # Stage 6: Persist results to disk
            # ------------------------------------------------------------------
            import pickle

            # agents_traj.pkl — list of (frame_ids, [(X,Z), …], label) tuples
            with open(log_base + '-estimate_agents_traj.pkl', 'wb') as f:
                pickle.dump(agents_traj, f)

            # agents_bbox.pkl — list of N_agents × list of (frame_id, [x1,y1,x2,y2])
            with open(log_base + '-estimate_agents_bbox.pkl', 'wb') as f:
                pickle.dump(mov_track_bbox, f)

            # agents_bbox_label.pkl — dict {"x1-y1-x2-y2": class_name}
            with open(log_base + '-estimate_agents_bbox_label.pkl', 'wb') as f:
                pickle.dump(person_car_label, f)

        else:
            # ----------------------------------------------------------------
            # No agents detected: remove any stale output files from a
            # previous run to avoid outdated data confusing downstream steps.
            # ----------------------------------------------------------------
            for suffix in [
                '-estimate_agents_traj.pkl',
                '-estimate_agents_bbox.pkl',
                '-estimate_agents_bbox_label.pkl',
            ]:
                stale = log_base + suffix
                if os.path.exists(stale):
                    os.system(f'rm -rf {stale}')
