"""
extract_traj_ego_unidepth.py
============================
Ego-vehicle trajectory extraction pipeline for the DrivingGen benchmark.

This module implements a monocular visual SLAM pipeline that estimates the
ego-vehicle's 3-D trajectory from a sequence of RGB frames by combining:

    1. Monocular depth estimation — UniDepthV2 (ViT-L14) predicts per-pixel
       metric depth and camera intrinsics for each frame.
    2. Movable-object masking     — YOLOv10x detects people, vehicles, and
       animals; their bounding boxes are zeroed in the feature-matching mask
       so dynamic objects do not corrupt the odometry estimate.
    3. Feature-based visual SLAM  — VisualSLAM library (from visual_slam)
       extracts SIFT/ORB keypoints, matches them across frames, filters
       outliers, and estimates the camera-pose trajectory.

For each processed scene the output is a pickle file containing:
    ``locs``       (N, 2) float32 array of (x, z) world positions.
    ``poses_3x3``  list of N (R, t) tuples (3x3 rotation + 3-element translation).
"""

import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import numpy as np
import torch
from PIL import Image

import sys

# UniDepthV2 monocular depth estimator (supports ViT-L14 backbone).
from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize, image_grid

import re

# VisualSLAM pipeline: dataset management, feature extraction/matching, pose estimation.
from visual_slam.dataset import *
from visual_slam.vo import *

import math

# YOLOv10 real-time object detector for movable-object masking.
from ultralytics import YOLOv10

import random
import numpy as np
from scipy.optimize import minimize
import torch
import json


# ---------------------------------------------------------------------------
# Module-level model singletons (loaded once to avoid repeated GPU allocation)
# ---------------------------------------------------------------------------

depth_model = None  # UniDepthV2 instance; populated by init_depth_model()
det_model   = None  # YOLOv10 instance;   populated by init_det_model()


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_depth_model():
    """Load and configure the UniDepthV2 (ViT-L14) monocular depth estimator.

    Downloads the pre-trained weights from the Hugging Face Hub
    (lpiccinelli/unidepth-v2-vitl14) if not already cached, then moves the
    model to the best available device (CUDA if present, else CPU).

    Configuration applied:
        resolution_level = 0       — use the model's native resolution.
        interpolation_mode = "bilinear" — smooth depth upsampling.

    Modifies:
        depth_model (global): Set to the loaded UniDepthV2 instance on the
            appropriate device.
    """
    global depth_model

    print("Torch version:", torch.__version__)

    # Model identifier on the Hugging Face Hub.
    name = "unidepth-v2-vitl14"

    # Download (or load from cache) UniDepthV2 with ViT-L14 backbone.
    depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # UniDepthV2 version and backbone (kept for potential logging).
    version  = "v2"
    backbone = "vitl14"

    # Use the model's native spatial resolution (level 0 = no downscaling).
    depth_model.resolution_level = 0

    # Bilinear interpolation is the default smooth upsampling mode.
    depth_model.interpolation_mode = "bilinear"

    # Select GPU if available; fall back to CPU for systems without CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = depth_model.to(device)


def init_det_model():
    """Load the YOLOv10x object detector and place it on GPU.

    Loads the YOLOv10x checkpoint from a fixed path on the shared disk.
    The model is used for detecting movable objects (people, vehicles, animals)
    so their regions can be masked out during visual SLAM feature matching.

    Modifies:
        det_model (global): Set to the loaded YOLOv10 instance on CUDA.
    """
    global det_model

    # Load YOLOv10x from the project's checkpoint directory and move to CUDA.
    det_model = YOLOv10(
        '/shared_disk/users/yang.zhou/iclr_open_source/DrivingGen/ckpt/yolov10x.pt'
    ).cuda()


# ---------------------------------------------------------------------------
# Distributed task assignment
# ---------------------------------------------------------------------------

def set_task_list(
    base_path: str,
    local_rank: int,
    gt_json: str = None,
    model_name: str = 'gt',
    exp_id: str = 'free',
    all_id: int = 8
):
    """Distribute scene directories across distributed GPU workers.

    Reads the list of scene identifiers from either a JSON metadata file (when
    model_name == 'gt') or from a directory listing (when evaluating a model).
    Scenes are shuffled with a fixed random seed (2026) for reproducibility,
    then split evenly across all distributed workers using RANK / WORLD_SIZE
    environment variables, and further split across GPUs within each node.

    Args:
        base_path (str): Root directory containing per-scene sub-directories.
        local_rank (int): Index of this GPU within its node (0-indexed).
        gt_json (str | None): Path to the JSON file listing ground-truth scene
            identifiers.  Required when model_name == 'gt'.
        model_name (str): Name of the model being evaluated, or 'gt' for the
            ground-truth pass.  Used to construct scene paths.
        exp_id (str): Experiment identifier sub-directory within each scene's
            model output directory.
        all_id (int): Total number of GPUs per node.  Scenes assigned to this
            node are further split into all_id chunks.  Default: 8.

    Returns:
        runs (list[str]): List of absolute paths to frame directories that
            this GPU worker should process.
    """
    runs = []  # final list of frame-directory paths for this worker

    # --- Determine scene list based on whether this is a ground-truth or model pass. ---
    if model_name == 'gt':
        # Ground-truth: scene IDs come from the JSON metadata file.
        with open(gt_json, 'r') as f:
            gt_json = json.load(f)
        dirs = gt_json  # list of scene ID strings
    else:
        # Model output: enumerate sub-directories in base_path.
        dirs = os.listdir(base_path)

    # Build the full scene paths.
    if model_name == 'gt':
        # For ground truth, the scene paths are the IDs themselves (absolute paths).
        scenes = dirs
    else:
        # For model outputs, construct: <base_path>/<scene_id>/<model_name>/<exp_id>
        exp_id = exp_id
        scenes = [os.path.join(base_path, f, model_name, exp_id) for f in dirs]

    # --- Distributed task splitting. ---

    # Read RANK and WORLD_SIZE from environment (set by torchrun / mpirun).
    # Defaults to single-process mode (rank=0, world_size=1).
    global_rank = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get('WORLD_SIZE', 1))

    # Shuffle with a fixed seed so all workers apply the same shuffle order,
    # ensuring that each scene is assigned to exactly one worker.
    random.seed(2026)
    random.shuffle(scenes)

    # Split scenes evenly across all distributed nodes (world_size processes).
    data_chunks = np.array_split(scenes, world_size)
    # data_chunks: list of world_size subarrays

    # Select the chunk for this node's global rank.
    local_data_chunk = data_chunks[global_rank]
    # local_data_chunk: 1-D array of scene paths for this node

    # Further split this node's chunk across its all_id GPUs.
    local_data_chunks = np.array_split(local_data_chunk, all_id)
    # local_data_chunks: list of all_id subarrays

    # Select the chunk for this GPU's local rank.
    scenes = local_data_chunks[local_rank % all_id]
    # scenes: 1-D array of scene paths for this specific GPU

    # --- Build the frame-directory path for each assigned scene. ---
    for scene in scenes:
        if model_name == 'gt':
            # Ground-truth frames live under: <scene>/CAM_F0/
            runs.append(os.path.join(scene, 'CAM_F0'))
        else:
            # Model-generated frames live under: <scene>/images/
            runs.append(os.path.join(scene, 'images'))

    print(f"Total {len(runs)} data to process, local index {local_rank}")

    return runs  # list of absolute paths to frame directories


# ---------------------------------------------------------------------------
# 3-D trajectory reconstruction from depth + poses
# ---------------------------------------------------------------------------

def reconstruct_global_trajectory(
    pixel_centers,   # list of (x_pixel, y_pixel) tuples, length N
    depth_values,    # list of float depth values (meters), length N
    Ks,              # list of (3, 3) camera intrinsic matrices, length N
    camera_poses,    # list of (R, T) tuples — R: (3, 3), T: (3,), length N
    delta_d=None     # optional list of depth correction scalars, length N
):
    """Lift 2-D pixel detections into 3-D world coordinates using depth and camera poses.

    For each frame i, the pipeline is:
        1. Camera coordinates:
               cam_coord = K_inv @ [u, v, 1]^T * Z
           where [u, v] is the pixel centre and Z is the (optionally corrected)
           depth at that pixel.  K_inv is the inverse of the 3x3 camera intrinsic
           matrix for frame i.

        2. World coordinates:
               world_coord = R @ cam_coord + T
           where (R, T) is the camera-to-world pose for frame i.

    Only the X (index 0) and Z (index -1) world coordinates are returned,
    corresponding to the horizontal and forward axes of the driving coordinate
    frame.

    Args:
        pixel_centers (list[tuple[float, float]]): Per-frame pixel coordinates
            of the detected object / point of interest.
            Each element: (x_pixel, y_pixel).  Length: N.
        depth_values (list[float]): Per-frame metric depth in metres at the
            corresponding pixel centre.  Length: N.
        Ks (list[np.ndarray]): Per-frame 3x3 camera intrinsic matrices.
            Each element shape: (3, 3).  Length: N.
        camera_poses (list[tuple[np.ndarray, np.ndarray]]): Per-frame
            camera-to-world pose (rotation matrix R, translation vector T).
            R shape: (3, 3),  T shape: (3,).  Length: N.
        delta_d (list | None): Optional per-frame depth corrections to add to
            depth_values.  Useful when calibrating against GPS or LiDAR.
            Each element is a scalar tensor or float.  Length: N.  Default: None.

    Returns:
        global_trajectory (list[np.ndarray]): Per-frame (x, z) world
            coordinates.  Each element shape: (2,).  Length: N.
    """
    global_trajectory = []  # accumulate (x, z) pairs

    for i, ((x_c, y_c), Z_c) in enumerate(zip(pixel_centers, depth_values)):
        # --- Step 1: Compute inverse of per-frame intrinsic matrix. ---
        K_inv = np.linalg.inv(Ks[i])   # shape: (3, 3)

        # Homogeneous pixel coordinate: [u, v, 1]^T
        pixel_coord = np.array([x_c, y_c, 1])  # shape: (3,)

        # Apply optional depth correction (e.g. from LiDAR or optimisation).
        if delta_d is not None:
            Z_c += delta_d[i].item()   # scalar addition to depth value

        # Back-project to camera space:
        #   cam_coord = K^{-1} [u, v, 1]^T * Z
        # This gives the 3-D point in the camera's local frame.
        cam_coord = (K_inv @ pixel_coord) * Z_c   # shape: (3,)

        # --- Step 2: Transform from camera frame to world frame. ---
        R, T = camera_poses[i]
        # R shape: (3, 3) — rotation matrix (camera-to-world)
        # T shape: (3,)   — translation vector (camera origin in world frame)

        # world_coord = R @ cam_coord + T
        world_coord = R @ cam_coord + T   # shape: (3,)

        # Keep only the X (index 0) and Z (index -1 = 2) components,
        # which correspond to the horizontal and forward driving axes.
        global_trajectory.append(world_coord[[0, -1]])  # shape: (2,)

    return global_trajectory  # list of N (2,) arrays


# ---------------------------------------------------------------------------
# Object detection and masking
# ---------------------------------------------------------------------------

def det_obj(frame: np.ndarray):
    """Run YOLOv10 on a single RGB frame and build a movable-object mask.

    Detects all COCO objects in the frame.  Objects belonging to the 'movable'
    category (people, vehicles, animals) have their bounding boxes zeroed in the
    returned binary mask so that visual SLAM feature extraction ignores them.
    For a subset of movable classes (person, bicycle, car, motorcycle, bus, train,
    truck) the pixel-centre coordinates and bounding boxes are also returned for
    use in depth-based trajectory reconstruction.

    COCO class IDs for movable objects:
        0: person,  1: bicycle,  2: car,  3: motorcycle,  4: airplane,
        5: bus,     6: train,    7: truck, 8: boat,
        14: bird,   15: cat,     16: dog, 17: horse,
        18: sheep,  19: cow,     20: elephant, 21: bear,
        22: zebra,  23: giraffe.

    Person/vehicle subset (used for depth lifting): [0, 1, 2, 3, 5, 6, 7].

    Args:
        frame (np.ndarray): Input RGB image.
            shape: (H, W, 3) — uint8, values in [0, 255].

    Returns:
        mask (np.ndarray): Binary exclusion mask.
            shape: (H, W) — uint8.  255 = valid (static background),
            0 = masked out (movable object bounding box region).
        person_car_pixel (list[list[int, int]]): Pixel centres [cx, cy] of
            detected person/vehicle objects.
        person_car_bbox (list[list[int, int, int, int]]): Bounding boxes
            [x1, y1, x2, y2] of detected person/vehicle objects.
    """
    global det_model

    # Run the YOLOv10 detector on the single input frame.
    result = det_model(frame)
    result = result[0]  # batch size 1 — take the first (only) result

    # Unpack detection outputs (only bounding boxes are used below).
    boxes     = result.boxes      # Boxes object — bounding box detections
    masks     = result.masks      # Segmentation masks (not used here)
    keypoints = result.keypoints  # Pose keypoints (not used here)
    probs     = result.probs      # Classification probabilities (not used here)
    obb       = result.obb        # Oriented bounding boxes (not used here)

    # COCO class IDs considered 'movable' (dynamic objects that corrupt SLAM).
    movable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # Initialise the mask to all-valid (255 = background pixel).
    mask = (np.ones((frame.shape[0], frame.shape[1])) * 255).astype(np.uint8)
    # mask shape: (H, W)

    person_car_pixel = []    # list of [cx, cy] for person/vehicle detections
    person_car_bbox  = []    # list of [x1, y1, x2, y2] for person/vehicle detections

    if boxes:
        num_box = len(boxes)

        for id in range(num_box):
            # boxes.cls[id] — COCO class ID as a float tensor; convert to int.
            cls_id = int(boxes.cls[id].item())

            if cls_id in movable:
                # Retrieve bounding box in xyxy format on CPU as a numpy array.
                xy = boxes.data[id, :4].cpu().numpy()  # shape: (4,)
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])

                # Zero out (mask) the bounding box region to exclude it from
                # feature extraction in the visual SLAM step.
                mask[y1:y2, x1:x2] = 0

                # For person/vehicle classes, record the pixel centre and bounding box.
                if cls_id in [0, 1, 2, 3, 5, 6, 7]:
                    # Pixel centre of the bounding box.
                    mid_x = int((x1 + x2) / 2)
                    mid_y = int((y1 + y2) / 2)
                    person_car_pixel.append([mid_x, mid_y])
                    person_car_bbox.append([x1, y1, x2, y2])

    return mask, person_car_pixel, person_car_bbox
    # mask shape: (H, W) uint8


def drive_roi_mask(h: int, w: int, keep: float = 0.5, side: float = 0.03):
    """Create a binary region-of-interest (ROI) mask for the lower driving region.

    Pixels inside the ROI are set to 255 (valid); pixels outside are 0 (invalid).
    The ROI covers the lower ``keep`` fraction of the frame height, excluding a
    thin ``side`` fraction strip on each horizontal edge.

    This mask restricts feature extraction to the road region, avoiding
    irrelevant features from the sky and far-away scene elements.

    ROI definition:
        top    = 0          (start from the very top of the lower region)
        bottom = int(h * keep)
        left   = int(w * side)
        right  = int(w * (1 - side))

    Args:
        h (int): Frame height in pixels.
        w (int): Frame width in pixels.
        keep (float): Fraction of the frame height to keep (from the top).
            Default: 0.5 — keep the upper 50% of the image.
        side (float): Fraction of the frame width to crop from each side.
            Default: 0.03 — remove 3% from left and right edges.

    Returns:
        m (np.ndarray): Binary ROI mask.
            shape: (h, w) — uint8.  255 inside ROI, 0 outside.
    """
    # Initialise mask to all-zeros (everything excluded).
    m = np.zeros((h, w), np.uint8)
    # m shape: (h, w)

    # Define ROI row bounds.
    top = 0                # start row  (inclusive)
    bot = int(h * keep)    # end row    (exclusive) — lower keep-fraction

    # Define ROI column bounds.
    l = int(w * side)          # left column  (inclusive)
    r = int(w * (1 - side))    # right column (exclusive)

    # Set the ROI interior to 255 (valid pixels).
    m[top:bot, l:r] = 255

    return m  # shape: (h, w) uint8


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -----------------------------------------------------------------------
    # Argument parsing
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Extract ego-vehicle trajectories using UniDepthV2 + VisualSLAM.'
    )
    parser.add_argument('--root_path',     type=str,
                        help='Root directory containing per-scene frame folders.')
    parser.add_argument('--outdir',        type=str, default='./vis_depth',
                        help='Output directory for depth visualisations and trajectory pickles.')
    parser.add_argument('--gt_meta_path',  type=str, default='./vis_depth',
                        help='Path to the ground-truth JSON metadata file.')
    parser.add_argument('--model_name',    type=str, default='gt',
                        help="Model name to evaluate, or 'gt' for ground-truth frames.")
    parser.add_argument('--exp_id',        type=str, default='free',
                        help='Experiment ID sub-directory within the model output folder.')
    parser.add_argument('--local_id',      type=int, default=0,
                        help='Local GPU rank within the current node (0-indexed).')
    parser.add_argument('--all_id',        type=int, default=8,
                        help='Total number of GPUs per node.')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Model initialisation
    # -----------------------------------------------------------------------
    init_depth_model()   # loads UniDepthV2 onto GPU
    init_det_model()     # loads YOLOv10x onto GPU

    # -----------------------------------------------------------------------
    # Distribute scenes across workers
    # -----------------------------------------------------------------------
    runs = set_task_list(
        args.root_path,
        int(args.local_id),
        args.gt_meta_path,
        args.model_name,
        args.exp_id,
        args.all_id
    )
    # runs: list of absolute paths to frame directories for this GPU

    model_name = args.model_name
    exp_id     = args.exp_id

    # Load the full GT scene list for constructing ground-truth reference paths.
    gt_json = args.gt_meta_path
    with open(gt_json, 'r') as f:
        gt_json = json.load(f)  # list of scene ID strings

    # Build a lookup from scene ID to its base path.
    gt_paths = {}
    for gt_base in gt_json:
        gt_paths[gt_base] = gt_base

    # -----------------------------------------------------------------------
    # Per-scene processing loop
    # -----------------------------------------------------------------------
    for run in runs:

        print(f'{args.local_id}: {run}')

        # --- List and sort frame files in the scene directory. ---
        filenames = os.listdir(run)
        filenames = [os.path.join(run, f) for f in filenames]
        filenames.sort()   # alphabetical / numeric order matches temporal order

        # Per-scene accumulators.
        rgbs          = []  # list of (H, W, 3) uint8 RGB arrays
        depths_raw    = []  # list of (H, W) float depth maps (metres)
        points_3d     = []  # list of (H, W, 3) float point-cloud arrays
        rgb_intrinsics = [] # list of (3, 3) intrinsic matrices predicted by UniDepthV2
        masks         = []  # list of (H, W) uint8 combined SLAM feature masks
        depths_mov    = []  # reserved for future use (depth at movable objects)
        track_masks   = []  # reserved for future use (object tracking masks)
        track_bboxs   = []  # reserved for future use (tracked bounding boxes)

        # -----------------------------------------------------------------------
        # Set up output directory for depth visualisation frames.
        # -----------------------------------------------------------------------
        if model_name == 'gt':
            # Ground-truth scenes: <scene_dataset>+<scene_clip> / gt / unidepth/
            s_name   = run.split('/')[-3] + '+' + run.split('/')[-2]
            log_base = os.path.join(args.outdir, s_name, model_name, 'unidepth')
        else:
            # Model-generated scenes: <scene_clip> / <model_name> / <exp_id> / unidepth/
            s_name   = run.split('/')[-4]
            log_base = os.path.join(args.outdir, s_name, model_name, exp_id, 'unidepth')

        # Create the depth-frame output directory.
        depth_out_dir = os.path.join(log_base, 'depth_frame')
        os.makedirs(depth_out_dir, exist_ok=True)

        # -----------------------------------------------------------------------
        # Per-frame depth estimation and detection loop.
        # -----------------------------------------------------------------------
        for k, filename in enumerate(filenames):

            # Skip the first 19 frames when using 20-frame conditioning
            # (the 20th frame is the actual first frame in that mode).
            if 'conds_20' in args.exp_id and k < 19:
                continue

            # --- Load RGB frame. ---
            rgb = np.array(Image.open(filename))
            # rgb shape: (H, W, 3) — uint8

            # --- Detect movable objects to build the SLAM exclusion mask. ---
            mask, person_car, person_car_bbox = det_obj(rgb)
            # mask shape: (H, W) — uint8; 255=valid, 0=movable object

            # Apply the driving ROI mask: also exclude the non-driving upper region.
            mask_wh = drive_roi_mask(576, 1024)
            # mask_wh shape: (576, 1024) — uint8; 255 inside ROI, 0 outside

            # Combine: pixels outside the driving ROI are also excluded.
            mask[mask_wh == 0] = 0
            masks.append(mask)
            # mask shape: (H, W)

            # Store the RGB frame for the SLAM dataset handler.
            rgbs.append(rgb.copy())
            # rgb shape: (H, W, 3)

            # Convert to torch CHW format for the depth model.
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            # rgb_torch shape: (3, H, W)

            # --- Run UniDepthV2 depth + intrinsics inference. ---
            # Pass None as intrinsics to let the model predict them.
            predictions = depth_model.infer(rgb_torch, None)

            # Extract per-pixel metric depth map.
            depth = predictions["depth"].squeeze().cpu().numpy()
            # depth shape: (H, W) — float, metric depth in metres
            depths_raw.append(depth.copy())

            # Extract predicted 3x3 camera intrinsic matrix.
            intrinsics = predictions["intrinsics"].squeeze(0).cpu().numpy()
            # intrinsics shape: (3, 3)
            rgb_intrinsics.append(intrinsics)

            # Extract the dense 3-D point cloud in camera space.
            # predictions["points"] shape: (1, 3, H, W) — (x, y, z) at each pixel
            xyz = predictions["points"].cpu().squeeze(0).reshape(3, -1).t().numpy()
            # xyz shape: (H*W, 3)

            # Store the (H, W, 3) point map for potential later use.
            points_3d.append(
                predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy().copy()
            )
            # points_3d[-1] shape: (H, W, 3)

            # --- Save a colourised depth visualisation to disk. ---
            depth_pred_col = colorize(depth, cmap="jet")
            # depth_pred_col shape: (H, W, 3) — uint8 false-colour image

            cv2.imwrite(os.path.join(depth_out_dir, f'{k:05}.png'), depth_pred_col)

        # -----------------------------------------------------------------------
        # Assemble depth frames into a video for visual inspection.
        # -----------------------------------------------------------------------
        from moviepy.editor import ImageSequenceClip

        def images_to_video(image_folder: str, output_video: str, fps: int = 30):
            """Write all PNG/JPG images in a directory to an MP4 video.

            Args:
                image_folder (str): Directory containing image files.
                output_video (str): Output MP4 file path.
                fps (int): Frames per second.  Default: 30.
            """
            # Collect and sort all PNG and JPG image paths.
            images = [
                os.path.join(image_folder, img)
                for img in os.listdir(image_folder)
                if img.endswith(".png") or img.endswith(".jpg")
            ]
            images.sort()  # sort alphabetically to preserve temporal order

            clip = ImageSequenceClip(images, fps=fps)
            clip.write_videofile(
                output_video,
                codec="libx264",
                verbose=False,
                logger=None   # suppress moviepy progress output
            )

        # Compile depth frames into a video at 10 fps.
        images_to_video(
            depth_out_dir,
            os.path.join(log_base, "depths.mp4"),
            fps=10
        )

        # -----------------------------------------------------------------------
        # Visual SLAM trajectory estimation.
        # -----------------------------------------------------------------------
        if True:  # always run trajectory estimation (condition kept for easy toggling)

            # --- Build the dataset handler with frames, depth maps, and intrinsics. ---
            dataset_handler = DatasetHandler(
                rgbs,            # list of (H, W, 3) uint8 arrays
                depths_raw,      # list of (H, W) float depth maps
                rgb_intrinsics   # list of (3, 3) intrinsic matrices
            )

            # Sanity check on the dataset (logs statistics, checks alignment).
            check_data(dataset_handler, args.outdir)

            # --- Part I: Feature extraction. ---
            # Extract keypoints and descriptors from all frames using the
            # combined movable-object + ROI mask to restrict features to
            # static background regions.
            images   = dataset_handler.images
            kp_list, des_list = extract_features_dataset(images, masks)
            # kp_list:  list of N keypoint lists (one per frame)
            # des_list: list of N descriptor arrays, each shape: (num_kp, D)

            # --- Part II: Feature matching. ---
            matches, matches2 = match_features_dataset(des_list)
            # matches:  list of N-1 match lists between consecutive frames
            # matches2: supplementary matches (used for filtering)

            # Apply Lowe's ratio test to remove ambiguous matches.
            is_main_filtered_m = True
            if is_main_filtered_m:
                dist_threshold = 0.7   # Lowe ratio threshold (0.7 is standard)
                filtered_matches = filter_matches_dataset(matches, dist_threshold, matches2)
                matches = filtered_matches
                # matches: filtered list of N-1 match lists

            # --- Part III: Trajectory estimation. ---
            depth_maps = dataset_handler.depth_maps
            # depth_maps: list of N (H, W) float depth maps

            try:
                # Estimate camera poses using PnP + RANSAC with depth-backed
                # 3-D point correspondences.
                trajectory, poses_3x3 = estimate_trajectory(
                    matches,
                    kp_list,
                    dataset_handler.k,   # shared intrinsic matrix (3, 3) from handler
                    depth_maps=depth_maps,
                    dataset_handler=dataset_handler
                )
                # trajectory shape: (3, N) — world-frame camera positions (x, y, z)
                # poses_3x3: list of N (R: (3,3), t: (3,)) tuples

            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()  # breakpoint for diagnosing SLAM failures
                print(f'extract ego fail for {run}')
                continue  # skip this scene and proceed to the next

            # -----------------------------------------------------------------------
            # Extract (x, z) positions from the trajectory matrix.
            # -----------------------------------------------------------------------
            locs = []
            for i in range(0, trajectory.shape[1]):
                # trajectory[:, i] shape: (3,) — [x, y, z] world position at frame i
                current_pos = trajectory[:, i]

                # Keep only the X (horizontal) and Z (forward) components,
                # which define the 2-D bird's-eye-view driving trajectory.
                locs.append([current_pos.item(0), current_pos.item(2)])
                # appended: [x, z] for frame i

            locs = np.array(locs, dtype=np.float32)
            # locs shape: (N, 2) — columns are [x_world, z_world]

            print(log_base, locs)

            # -----------------------------------------------------------------------
            # Save trajectory to pickle.
            # -----------------------------------------------------------------------
            ego_traj = {
                'locs':      locs,       # shape: (N, 2) float32 — (x, z) world positions
                'poses_3x3': poses_3x3,  # list of N (R, t) tuples
            }

            import pickle
            output_pkl_path = log_base + '-estimate_ego_traj.pkl'
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(ego_traj, f)
            # Output file: <log_base>-estimate_ego_traj.pkl
