"""
traj_distribution.py
====================
Computes the Fréchet Trajectory Distance (FTD) metric for evaluating
driving trajectory quality in the DrivingGen benchmark.

FTD is analogous to FID (Fréchet Inception Distance) but uses trajectory
feature representations extracted from the Motion Transformer (MTR) polyline
encoder instead of image features.

Reference:
    Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Published at NeurIPS 2022.
    Written by Shaoshuai Shi.
    All Rights Reserved.
"""

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import sys

# Add the MTR third-party library to the Python path so its modules are importable.
sys.path.append('third_parties/MTR')
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.models import model as model_utils
from mtr.utils import common_utils


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_config():
    """Load the MTR YAML configuration file and set global numpy random seed.

    Reads the pre-defined Waymo 100% data config for MTR and populates the
    global ``cfg`` object used throughout the MTR codebase.

    Returns:
        cfg: The populated MTR configuration object.
    """
    # Path to the MTR configuration YAML for Waymo 100% data split.
    cfg_file = 'third_parties/MTR/tools/cfgs/waymo/mtr+100_percent_data.yaml'

    # Path to the pre-trained MTR checkpoint (epoch 28, step 176552).
    cktp = 'ckpt/mtr-epoch=28-step=176552.ckpt'

    # Parse the YAML file and merge settings into the global cfg object.
    cfg_from_yaml_file(cfg_file, cfg)

    # Fix numpy random seed for reproducible behaviour across runs.
    np.random.seed(1024)

    return cfg


# ---------------------------------------------------------------------------
# Coordinate transformation helpers
# ---------------------------------------------------------------------------

def transform_trajs_to_center_coords(
    obj_trajs,          # shape: (num_objects, num_timestamps, num_attrs)
    center_xyz,         # shape: (num_center_objects, 3) or (num_center_objects, 2)
    center_heading,     # shape: (num_center_objects,)
    heading_index,      # scalar int — column index of the heading attribute
    rot_vel_index=None  # optional list of two ints — column indices of [vx, vy]
):
    """Rotate all agent trajectories into each center object's ego-centric frame.

    For each center object c, every agent trajectory is translated so that
    the center object's position becomes the origin and then rotated by
    -center_heading[c] around the Z axis.  This produces one transformed
    copy of all trajectories per center object.

    The rotation is a standard 2-D rotation in the XY plane:
        [x', y'] = R(-theta) @ [x, y]
    where R(-theta) = [[cos(-theta), -sin(-theta)],
                       [sin(-theta),  cos(-theta)]]

    Args:
        obj_trajs (torch.Tensor): Raw agent trajectories.
            shape: (num_objects, num_timestamps, num_attrs).
            The first ``center_xyz.shape[1]`` columns are spatial coordinates
            [x, y] or [x, y, z].
        center_xyz (torch.Tensor): World-space position of each center object.
            shape: (num_center_objects, 3) or (num_center_objects, 2).
        center_heading (torch.Tensor): Heading angle (radians) of each center
            object in world frame.
            shape: (num_center_objects,).
        heading_index (int): Column index in ``num_attrs`` dimension that
            stores the heading attribute; will be adjusted by -center_heading.
        rot_vel_index (list[int] | None): Two column indices [vx_col, vy_col]
            whose values should also be rotated into the ego frame.  If None,
            velocity directions are not rotated.

    Returns:
        obj_trajs (torch.Tensor): Ego-centric trajectories.
            shape: (num_center_objects, num_objects, num_timestamps, num_attrs).
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]

    # Sanity checks: center_xyz and center_heading must share the same leading
    # dimension, and the spatial dimension must be 2 or 3.
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    # Expand obj_trajs from (num_objects, T, D) to
    # (num_center_objects, num_objects, T, D) by repeating along axis 0.
    # shape: (1, num_objects, num_timestamps, num_attrs) -> repeat ->
    # shape: (num_center_objects, num_objects, num_timestamps, num_attrs)
    obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)

    # --- Step 1: Translate so each center object sits at the origin. ---
    # Broadcast center_xyz from (num_center_objects, D_xyz) to
    # (num_center_objects, 1, 1, D_xyz) then subtract.
    # Only the first D_xyz columns ([x, y] or [x, y, z]) are translated.
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    # shape after subtraction: (num_center_objects, num_objects, num_timestamps, num_attrs)

    # --- Step 2: Rotate XY coordinates by -center_heading in the XY plane. ---
    # Flatten the (num_objects, num_timestamps) axes for the rotation utility,
    # then reshape back.
    obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
        # shape passed in:  (num_center_objects, num_objects * num_timestamps, 2)
        angle=-center_heading   # shape: (num_center_objects,) — negative = rotate to align heading to +X
    ).view(num_center_objects, num_objects, num_timestamps, 2)
    # shape restored: (num_center_objects, num_objects, num_timestamps, 2)

    # --- Step 3: Adjust the heading attribute by subtracting the center heading. ---
    # After rotation, every agent's heading is expressed relative to the center
    # object's heading rather than the world frame.
    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]
    # shape: (num_center_objects, num_objects, num_timestamps)

    # --- Step 4 (optional): Rotate velocity direction into ego frame. ---
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2  # must provide exactly [vx_col, vy_col]
        obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
            # shape: (num_center_objects, num_objects * num_timestamps, 2)
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)
        # shape: (num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs
    # shape: (num_center_objects, num_objects, num_timestamps, num_attrs)


# ---------------------------------------------------------------------------
# Feature tensor construction for the MTR polyline encoder
# ---------------------------------------------------------------------------

def generate_centered_trajs_for_agents(
    center_objects,     # shape: (num_center_objects, 10)
    obj_trajs_past,     # shape: (num_objects, num_timestamps, 10)
    obj_types,          # shape: (num_objects,)  — string labels
    center_indices,     # shape: (num_center_objects,) — index into obj_trajs_past
    sdc_index,          # int or None — index of the self-driving car in obj_trajs_past
    timestamps,         # shape: (num_timestamps,) — absolute time values (seconds)
    obj_trajs_future    # shape: (num_objects, num_future_timestamps, 10) or None
):
    """Build the full feature tensor fed to MTR's polyline encoder.

    Transforms all agent trajectories into each center object's ego frame, then
    constructs a rich per-agent-per-timestep feature vector by concatenating:
        1. Ego-frame spatial state     — 6 dims  [x, y, z, dx, dy, dz]
        2. Agent type one-hot          — 5 dims  [vehicle, pedestrian, cyclist,
                                                  is_center, (sdc — disabled)]
        3. Time embedding              — (T+1) dims  one-hot position + scalar
        4. Heading sin/cos             — 2 dims  [sin(h), cos(h)]
        5. Velocity                    — 2 dims  [vx, vy]
        6. Acceleration                — 2 dims  [ax, ay]

    Total feature width = 6 + 5 + (T+1) + 2 + 2 + 2 = 17 + T + 1 = T + 18.
    For T=11 this gives 29 dims.

    Args:
        center_objects (np.ndarray): Last observed state of each center agent.
            shape: (num_center_objects, 10).
            Columns: [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid].
        obj_trajs_past (np.ndarray): Historical trajectory of every agent.
            shape: (num_objects, num_timestamps, 10).
            Columns: [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid].
        obj_types (np.ndarray[str]): String type label for each agent.
            shape: (num_objects,).
        center_indices (np.ndarray[int]): Row index in ``obj_trajs_past`` that
            corresponds to each center object.
            shape: (num_center_objects,).
        sdc_index (int | None): Index of the SDC (self-driving car) in
            ``obj_trajs_past``.  Currently unused (SDC one-hot channel is
            commented out).
        timestamps (np.ndarray): Absolute timestamps for each history step.
            shape: (num_timestamps,).
        obj_trajs_future: Not used in the current implementation (future label
            generation is commented out).  Pass None.

    Returns:
        ret_obj_trajs (torch.Tensor): Feature tensor on CUDA.
            shape: (num_center_objects, num_objects, num_timestamps, T+18).
        ret_obj_valid_mask (torch.Tensor): Boolean validity mask on CUDA.
            shape: (num_center_objects, num_objects, num_timestamps).

    Notes:
        Usage context (last point, 11 timestamps, vehicle type, 1 center object,
        no SDC, 0.1 s interval, no future labels) is described in the original
        developer notes.
    """
    # Validate that both input arrays use the expected 10-attribute format.
    assert obj_trajs_past.shape[-1] == 10
    assert center_objects.shape[-1] == 10

    num_center_objects = center_objects.shape[0]          # n_c
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape  # n, T, 10

    # Convert numpy arrays to float32 CPU tensors for processing.
    center_objects = torch.from_numpy(center_objects).float()   # shape: (num_center_objects, 10)
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()   # shape: (num_objects, num_timestamps, 10)
    timestamps     = torch.from_numpy(timestamps)                # shape: (num_timestamps,)

    # --- Transform all trajectories into each center object's ego frame. ---
    # center_objects[:, 0:3] = [cx, cy, cz], center_objects[:, 6] = heading
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],        # shape: (num_center_objects, 3)
        center_heading=center_objects[:, 6],      # shape: (num_center_objects,)
        heading_index=6,                          # column 6 = heading
        rot_vel_index=[7, 8]                      # columns 7-8 = [vel_x, vel_y]
    )
    # shape: (num_center_objects, num_objects, num_timestamps, 10)

    # -----------------------------------------------------------------------
    # Feature 2: One-hot agent type encoding (5 channels)
    # -----------------------------------------------------------------------
    # Channel 0: vehicle, 1: pedestrian, 2: cyclist, 3: is_center_object,
    # 4: is_SDC (currently disabled).
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    # shape: (num_center_objects, num_objects, num_timestamps, 5)

    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE',    :, 0] = 1  # vehicle flag
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # pedestrian flag (original typo preserved)
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST',    :, 2] = 1  # cyclist flag
    # Mark which agent is the center object (one per center object).
    object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
    # SDC channel (index 4) is intentionally left as zero (feature disabled).

    # -----------------------------------------------------------------------
    # Feature 3: Time embedding  (num_timestamps + 1 channels)
    # -----------------------------------------------------------------------
    # Each timestep t gets a one-hot indicator at position t plus the raw
    # scalar timestamp value in the last channel.
    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    # shape: (num_center_objects, num_objects, num_timestamps, num_timestamps+1)

    # One-hot positional encoding: position t is set to 1 for timestep t.
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1

    # Append the raw timestamp scalar in the last channel for all agents.
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps
    # shape: (num_center_objects, num_objects, num_timestamps, num_timestamps+1)

    # -----------------------------------------------------------------------
    # Feature 4: Heading sin/cos encoding (2 channels)
    # -----------------------------------------------------------------------
    # Encodes heading as a unit vector to avoid discontinuity at ±pi.
    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    # shape: (num_center_objects, num_objects, num_timestamps, 2)

    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])  # sin(heading)
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])  # cos(heading)

    # -----------------------------------------------------------------------
    # Feature 6: Acceleration (2 channels) — finite difference of velocity
    # -----------------------------------------------------------------------
    # vel shape: (num_center_objects, num_objects, num_timestamps, 2)
    vel = obj_trajs[:, :, :, 7:9]   # columns 7-8 = [vel_x, vel_y]
    # shape: (num_center_objects, num_objects, num_timestamps, 2)

    # torch.roll shifts the time axis by 1 so vel_pre[t] = vel[t-1].
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    # shape: (num_center_objects, num_objects, num_timestamps, 2)

    # Finite difference: a = (v[t] - v[t-1]) / dt,  dt = 0.1 s
    acce = (vel - vel_pre) / 0.1
    # shape: (num_center_objects, num_objects, num_timestamps, 2)

    # The first timestep's acceleration is invalid due to roll wrap-around;
    # copy the second timestep's acceleration to fill it.
    acce[:, :, 0, :] = acce[:, :, 1, :]

    # -----------------------------------------------------------------------
    # Concatenate all features along the last dimension.
    # -----------------------------------------------------------------------
    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6],       # Feature 1: [x,y,z,dx,dy,dz] in ego frame  — 6 dims
        object_onehot_mask,            # Feature 2: agent type one-hot              — 5 dims
        object_time_embedding,         # Feature 3: time embedding                  — (T+1) dims
        object_heading_embedding,      # Feature 4: [sin(h), cos(h)]                — 2 dims
        obj_trajs[:, :, :, 7:9],       # Feature 5: velocity [vx, vy]              — 2 dims
        acce,                          # Feature 6: acceleration [ax, ay]           — 2 dims
    ), dim=-1)
    # shape: (num_center_objects, num_objects, num_timestamps, 6+5+(T+1)+2+2+2) = (n_c, n, T, T+18)

    # -----------------------------------------------------------------------
    # Validity mask and zeroing of invalid entries
    # -----------------------------------------------------------------------
    # Column -1 (index 9) of obj_trajs_past contains the validity flag (0 or 1).
    ret_obj_valid_mask = obj_trajs[:, :, :, -1]
    # shape: (num_center_objects, num_objects, num_timestamps)

    # Zero out feature vectors for timesteps where validity == 0.
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    # Move both tensors to GPU for downstream model inference.
    return ret_obj_trajs.cuda(), (ret_obj_valid_mask > 0).cuda()
    # ret_obj_trajs shape:    (num_center_objects, num_objects, num_timestamps, T+18)
    # ret_obj_valid_mask shape: (num_center_objects, num_objects, num_timestamps)


# ---------------------------------------------------------------------------
# Kinematic quantity computation from (x, y) positions
# ---------------------------------------------------------------------------

def compute_heading_xy(xy: np.ndarray) -> np.ndarray:
    """Compute per-frame heading angles from a 2-D (x, y) trajectory.

    Uses central differences for interior frames and one-sided differences for
    the first and last frames to avoid reducing the trajectory length.

    Formulas:
        heading[t]   = arctan2(y[t+1] - y[t-1], x[t+1] - x[t-1])  for 0 < t < N-1
        heading[0]   = arctan2(y[1]   - y[0],   x[1]   - x[0])
        heading[N-1] = arctan2(y[N-1] - y[N-2], x[N-1] - x[N-2])

    Args:
        xy (np.ndarray): 2-D position trajectory.
            shape: (N, 2) — columns are [x, y].

    Returns:
        heading (np.ndarray): Per-frame heading angle in radians.
            shape: (N,).
    """
    x, y = xy[:, 0], xy[:, 1]   # shape: (N,) each
    N = len(x)

    # --- Central differences for interior frames (indices 1 … N-2). ---
    # dx_mid[i] = x[i+2] - x[i]  which corresponds to central diff at i+1
    dx_mid = x[2:] - x[:-2]     # shape: (N-2,)
    dy_mid = y[2:] - y[:-2]     # shape: (N-2,)
    heading_mid = np.arctan2(dy_mid, dx_mid)   # shape: (N-2,)

    # --- One-sided differences at boundary frames. ---
    heading_first = np.arctan2(y[1] - y[0],   x[1] - x[0])    # scalar — forward diff at t=0
    heading_last  = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])  # scalar — backward diff at t=N-1

    # --- Assemble full-length heading array. ---
    heading = np.empty(N)         # shape: (N,)
    heading[1:-1] = heading_mid   # central diff for interior frames
    heading[-1]   = heading_last  # backward diff for last frame
    heading[0]    = heading_first # forward diff for first frame

    return heading  # shape: (N,)


def compute_speed_xy(xy: np.ndarray, DT: float = 0.1):
    """Compute per-frame velocity components from a 2-D (x, y) trajectory.

    Uses central differences for interior frames and one-sided differences for
    boundary frames.  The time step DT defaults to 0.1 s (10 Hz).

    Formulas (central differences for interior frames):
        vx[t] = (x[t+1] - x[t-1]) / (2 * DT)
        vy[t] = (y[t+1] - y[t-1]) / (2 * DT)

    Boundary (forward / backward differences):
        vx[0]   = (x[1] - x[0]) / DT
        vx[N-1] = (x[N-1] - x[N-2]) / DT

    Speed magnitude:
        v[t] = sqrt(vx[t]^2 + vy[t]^2)

    Args:
        xy (np.ndarray): 2-D position trajectory.
            shape: (N, 2) — columns are [x, y].
        DT (float): Time step between consecutive frames in seconds.
            Default: 0.1 s.

    Returns:
        vx (np.ndarray): X-component of velocity at each frame.
            shape: (N,).
        vy (np.ndarray): Y-component of velocity at each frame.
            shape: (N,).
        v (np.ndarray): Speed magnitude at each frame.
            shape: (N,).
    """
    x, y = xy[:, 0], xy[:, 1]   # shape: (N,) each
    N = len(x)

    # --- Central differences for interior frames. ---
    vx_mid = (x[2:] - x[:-2]) / (2 * DT)   # shape: (N-2,)
    vy_mid = (y[2:] - y[:-2]) / (2 * DT)   # shape: (N-2,)

    # --- Forward differences for the first frame. ---
    vx_first = (x[1] - x[0]) / DT  # scalar
    vy_first = (y[1] - y[0]) / DT  # scalar

    # --- Backward differences for the last frame. ---
    vx_last  = (x[-1] - x[-2]) / DT  # scalar
    vy_last  = (y[-1] - y[-2]) / DT  # scalar

    # --- Assemble velocity arrays. ---
    vx = np.empty(N)           # shape: (N,)
    vy = np.empty(N)           # shape: (N,)

    vx[1:-1], vy[1:-1] = vx_mid, vy_mid      # central diff for interior
    vx[-1],   vy[-1]   = vx_last,  vy_last   # backward diff for last frame
    vx[0],    vy[0]    = vx_first, vy_first  # forward diff for first frame

    # --- Speed magnitude: sqrt(vx^2 + vy^2). ---
    v = np.hypot(vx, vy)  # shape: (N,)

    return vx, vy, v
    # All outputs shape: (N,)


# ---------------------------------------------------------------------------
# Input packaging helpers for predicted and ground-truth trajectories
# ---------------------------------------------------------------------------

def deal_pred_input(pred_traj_data: np.ndarray):
    """Package a raw predicted (x, y) trajectory into the 10-attribute MTR format.

    Extracts kinematic quantities (heading, velocity) from the raw 2-D path and
    augments them with assumed vehicle bounding-box dimensions and a validity
    flag.  The final observation (last row) is used as the center object state.

    Assumed vehicle dimensions:
        length = 4.5 m,  width = 2.0 m,  height = 1.8 m.

    Args:
        pred_traj_data (np.ndarray): Predicted trajectory positions.
            shape: (N, 2) — columns are [x, y].

    Returns:
        center_objects (np.ndarray): State of the last frame used as center.
            shape: (1, 10) — [x, y, z, l, w, h, heading, vx, vy, valid].
        obj_trajs_past (np.ndarray): Full history in 10-attribute format.
            shape: (1, N, 10).
        obj_types (np.ndarray[str]): Type label array.
            shape: (1,) — always ['TYPE_VEHICLE'].
        center_indices (np.ndarray[int]): Index of center object in obj_trajs_past.
            shape: (1,) — always [0].
        sdc_index (None): SDC index, currently unused.
        timestamps (np.ndarray[float32]): Absolute timestamps at 0.1 s intervals.
            shape: (N,).
        obj_trajs_future: None (future labels not used for FTD inference).
    """
    # Unpack 2-D position columns.
    x = pred_traj_data[:, 0]   # shape: (N,)
    y = pred_traj_data[:, 1]   # shape: (N,)

    # Elevation: assume flat-ground driving (z = 0 everywhere).
    z      = np.zeros_like(x)          # shape: (N,)

    # Bounding-box dimensions: constant assumed values for a typical vehicle.
    length = np.zeros_like(x) + 4.5   # shape: (N,) — 4.5 m long
    width  = np.zeros_like(x) + 2.0   # shape: (N,) — 2.0 m wide
    height = np.zeros_like(x) + 1.8   # shape: (N,) — 1.8 m tall

    # Heading angle derived from position differences via central differences.
    heading = compute_heading_xy(pred_traj_data)  # shape: (N,)

    # Velocity components derived from position differences.
    vx, vy, _ = compute_speed_xy(pred_traj_data)  # shape: (N,) each

    # Validity flag: all frames are valid for predicted trajectories.
    valid = np.ones_like(x)  # shape: (N,)

    # Stack into the 10-attribute per-frame format used by MTR.
    # Column order: [x, y, z, length, width, height, heading, vx, vy, valid]
    obj_trajs_past = np.stack(
        (x, y, z, length, width, height, heading, vx, vy, valid), axis=-1
    )
    # shape: (N, 10)

    # Use the last frame as the center object state.
    center_objects = obj_trajs_past[-1]  # shape: (10,)

    # Wrap in a batch dimension to match the expected (num_center_objects, 10) format.
    # Assume a single vehicle as the only (center) object.
    obj_types = np.array('TYPE_VEHICLE').reshape(1)  # shape: (1,) — always vehicle
    center_indices = np.array([0]).astype(np.int32)   # shape: (1,) — index 0 in obj_trajs_past
    sdc_index = None  # SDC index is unused in the FTD pipeline

    # Timestamps at 10 Hz (0.1 s intervals).
    timestamps = np.arange(len(x)).astype(np.float32) * 0.1  # shape: (N,)

    obj_trajs_future = None  # Future labels not required for feature extraction

    return (
        center_objects[None],  # shape: (1, 10)
        obj_trajs_past[None],  # shape: (1, N, 10)  — add object batch dim
        obj_types,             # shape: (1,)
        center_indices,        # shape: (1,)
        sdc_index,             # None
        timestamps,            # shape: (N,)
        obj_trajs_future,      # None
    )


def gt_2_ego(gt_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Convert a global-frame ground-truth trajectory into ego-centric local frame.

    The transformation involves two steps:
        1. Translate so the first point of the trajectory is the origin.
        2. Rotate by -yaw (the inverse of the ego vehicle's yaw) so that the
           ego vehicle faces the +X axis.

    The rotation matrix for a counter-clockwise angle theta is:
        R = [[ cos(theta),  sin(theta)],
             [-sin(theta),  cos(theta)]]

    After rotation, the columns are swapped (Z ↔ X) and the new X axis is
    negated to match the driving convention where Z points forward and X points
    rightward.

    Args:
        gt_xy (np.ndarray): Global trajectory positions.
            shape: (T, 2) — columns are [x, y] in world frame.
        yaw (float): Ego vehicle yaw angle in radians (world frame).

    Returns:
        gt (np.ndarray): Ego-centric trajectory positions.
            shape: (T, 2) — columns are [X_ego, Z_ego].
    """
    theta = torch.tensor(-yaw, dtype=torch.float32)  # scalar — negate yaw to un-rotate
    gt    = torch.from_numpy(gt_xy)                   # shape: (T, 2)

    # --- Step 1: Translate so the first point becomes the origin. ---
    origin = gt[0]          # shape: (2,)
    rel_gt = gt - origin    # shape: (T, 2)

    # --- Step 2: Build the 2-D rotation matrix R(-yaw). ---
    # This is a passive rotation that aligns the ego heading to +X.
    # R = [[ cos(-yaw),  sin(-yaw)],
    #      [-sin(-yaw),  cos(-yaw)]]
    R = torch.tensor([
        [ torch.cos(theta),  torch.sin(theta)],   # first row
        [-torch.sin(theta),  torch.cos(theta)]    # second row
    ])
    # shape: (2, 2)

    # Apply the rotation: each (x, y) row is multiplied by R on the right.
    # gt_local[t] = rel_gt[t] @ R   =>  gt_local[:, 0] = Z_forward, gt_local[:, 1] = X_right
    gt_local = torch.matmul(rel_gt, R)    # shape: (T, 2)

    # --- Step 3: Swap columns so that column 0 = X_right, column 1 = Z_forward. ---
    gt_local[:, [0, 1]] = gt_local[:, [1, 0]]  # column swap: X <-> Z

    # --- Step 4: Negate the new column 0 (X) to match right-hand convention. ---
    gt_local[:, 0] = -gt_local[:, 0]

    gt = gt_local.numpy()   # shape: (T, 2) — column 0 = X_ego, column 1 = Z_ego
    return gt


def deal_gt_input(gt_traj_data: np.ndarray):
    """Package a raw ground-truth (x, y) trajectory into the 10-attribute MTR format.

    Identical in structure to ``deal_pred_input`` but operates on ground-truth
    positions.  Assumed vehicle dimensions are the same (4.5 × 2.0 × 1.8 m).

    Args:
        gt_traj_data (np.ndarray): Ground-truth trajectory positions.
            shape: (N, 2) — columns are [x, y].

    Returns:
        center_objects (np.ndarray): State of the last frame used as center.
            shape: (1, 10).
        obj_trajs_past (np.ndarray): Full history in 10-attribute format.
            shape: (1, N, 10).
        obj_types (np.ndarray[str]): shape: (1,) — always ['TYPE_VEHICLE'].
        center_indices (np.ndarray[int]): shape: (1,) — always [0].
        sdc_index (None): Unused.
        timestamps (np.ndarray[float32]): shape: (N,).
        obj_trajs_future: None.
    """
    # Unpack 2-D position columns.
    x = gt_traj_data[:, 0]   # shape: (N,)
    y = gt_traj_data[:, 1]   # shape: (N,)

    # Assume flat-ground driving.
    z      = np.zeros_like(x)          # shape: (N,)

    # Standard vehicle bounding-box dimensions.
    length = np.zeros_like(x) + 4.5   # shape: (N,)
    width  = np.zeros_like(x) + 2.0   # shape: (N,)
    height = np.zeros_like(x) + 1.8   # shape: (N,)

    # Heading and velocity from finite differences.
    heading = compute_heading_xy(gt_traj_data)   # shape: (N,)
    vx, vy, _ = compute_speed_xy(gt_traj_data)  # shape: (N,) each

    # All ground-truth frames are marked as valid.
    valid = np.ones_like(x)  # shape: (N,)

    # Stack into the 10-attribute format: [x, y, z, l, w, h, heading, vx, vy, valid]
    obj_trajs_past = np.stack(
        (x, y, z, length, width, height, heading, vx, vy, valid), axis=-1
    )
    # shape: (N, 10)

    # Use the last frame as the center object.
    center_objects = obj_trajs_past[-1]  # shape: (10,)

    # Packaging metadata.
    obj_types = np.array('TYPE_VEHICLE').reshape(1)  # shape: (1,)
    center_indices = np.array([0]).astype(np.int32)   # shape: (1,)
    sdc_index = None
    timestamps = np.arange(len(x)).astype(np.float32) * 0.1  # shape: (N,)
    obj_trajs_future = None

    return (
        center_objects[None],  # shape: (1, 10)
        obj_trajs_past[None],  # shape: (1, N, 10)
        obj_types,             # shape: (1,)
        center_indices,        # shape: (1,)
        sdc_index,             # None
        timestamps,            # shape: (N,)
        obj_trajs_future,      # None
    )


# ---------------------------------------------------------------------------
# Feature extraction via sliding-window polyline encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(preds, gts, stride: int = 10):
    """Extract MTR polyline features from predicted and ground-truth trajectories.

    For each trajectory a sliding window of 11 consecutive frames is extracted
    with the given stride.  Each window is passed through the MTR context
    encoder's agent polyline MLP to obtain a fixed-size feature vector.
    Window features are averaged (mean pooling) to produce one feature vector
    per trajectory.

    Window parameters:
        window length = 11 frames
        stride = ``stride`` frames (default 10)

    Args:
        preds (list[np.ndarray]): List of predicted trajectory arrays.
            Each element shape: (T_i, 2) — columns are [x, y].
        gts (list[np.ndarray]): List of ground-truth trajectory arrays.
            Each element shape: (T_j, 2) — columns are [x, y].
        stride (int): Sliding-window step size in frames.  Default: 10.

    Returns:
        Fp (np.ndarray): Feature matrix for predicted trajectories.
            shape: (N_preds, D_feat) where D_feat is the polyline encoder output dim.
        Fg (np.ndarray): Feature matrix for ground-truth trajectories.
            shape: (N_gts, D_feat).
    """
    # Load MTR configuration and set global seeds for determinism.
    cfg = parse_config()

    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True   # ensure reproducible CUDA ops
    torch.backends.cudnn.benchmark = False       # disable auto-tuning for reproducibility

    # --- Load the pre-trained MTR model and isolate the polyline encoder MLP. ---
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    # Load checkpoint weights (weights_only=False allows loading full pickled state).
    model.load_state_dict(
        torch.load('ckpt/mtr-epoch=28-step=176552.ckpt', weights_only=False)['state_dict']
    )
    # Extract only the agent polyline encoder sub-module and move to GPU.
    mlp = model.context_encoder.agent_polyline_encoder.cuda()
    mlp.eval()  # disable dropout and batch-norm update

    Fp = []  # list of per-trajectory feature vectors for predicted trajectories
    Fg = []  # list of per-trajectory feature vectors for ground-truth trajectories

    # -----------------------------------------------------------------------
    # Process predicted trajectories.
    # -----------------------------------------------------------------------
    for pred_traj_data in preds:
        # pred_traj_data shape: (T, 2)

        pred_this_feature = []  # accumulate window features for this trajectory

        # Slide a window of 11 frames over the trajectory with the given stride.
        for idx in range(0, pred_traj_data.shape[0] - 11 + 1, stride):
            # Extract an 11-frame window starting at frame idx.
            this_pred_traj_data = pred_traj_data[idx : idx + 11]   # shape: (11, 2)

            # Package the window into MTR's 10-attribute format.
            this_pred_traj_data = deal_pred_input(this_pred_traj_data)

            # Build the ego-centric feature tensor for this window.
            obj_trajs, obj_trajs_mask = generate_centered_trajs_for_agents(*this_pred_traj_data)
            # obj_trajs      shape: (1, 1, 11, T+18) on CUDA
            # obj_trajs_mask shape: (1, 1, 11) on CUDA

            # Append the binary validity mask as a final feature channel.
            obj_trajs_in = torch.cat(
                (obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1
            )
            # obj_trajs_in shape: (1, 1, 11, T+19)

            # Run the polyline encoder MLP; index [0, 0] selects the single
            # center object's feature vector from the output.
            obj_polylines_feature = mlp(obj_trajs_in, obj_trajs_mask)[0, 0]
            # shape: (D_feat,)

            # Guard against NaN features (can occur with degenerate windows).
            has_nan = torch.isnan(obj_polylines_feature).any()
            if has_nan:
                import pdb
                pdb.set_trace()  # breakpoint for debugging NaN cases
                continue         # skip this window

            pred_this_feature.append(obj_polylines_feature)
            # each element shape: (D_feat,)

        # Skip this trajectory if no valid windows were found.
        if len(pred_this_feature) < 1:
            continue

        # Stack window features and average (mean pooling across time windows).
        pred_this_feature = torch.stack(pred_this_feature, dim=0)
        # shape: (num_windows, D_feat)

        pred_this_feature_mean = pred_this_feature.mean(0)
        # shape: (D_feat,)

        # Use the mean as the trajectory-level feature.
        pred_this_feature = pred_this_feature_mean
        # shape: (D_feat,)

        Fp.append(pred_this_feature.cpu().numpy())
        # each element shape: (D_feat,)

    # -----------------------------------------------------------------------
    # Process ground-truth trajectories (same logic as predicted).
    # -----------------------------------------------------------------------
    for gt_traj_data in gts:
        # gt_traj_data shape: (T, 2)

        gt_this_feature = []  # accumulate window features

        for idx in range(0, gt_traj_data.shape[0] - 11 + 1, stride):
            # Extract an 11-frame window.
            this_gt_traj_data = gt_traj_data[idx : idx + 11]   # shape: (11, 2)

            # Package into MTR format.
            this_gt_traj_data = deal_gt_input(this_gt_traj_data)

            # Build ego-centric feature tensor.
            obj_trajs, obj_trajs_mask = generate_centered_trajs_for_agents(*this_gt_traj_data)
            # obj_trajs      shape: (1, 1, 11, T+18)
            # obj_trajs_mask shape: (1, 1, 11)

            obj_trajs_in = torch.cat(
                (obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1
            )
            # obj_trajs_in shape: (1, 1, 11, T+19)

            obj_polylines_feature = mlp(obj_trajs_in, obj_trajs_mask)[0, 0]
            # shape: (D_feat,)

            gt_this_feature.append(obj_polylines_feature)

        # Stack and average window features.
        gt_this_feature = torch.stack(gt_this_feature, dim=0)
        # shape: (num_windows, D_feat)

        gt_this_feature_mean = gt_this_feature.mean(0)
        # shape: (D_feat,)

        gt_this_feature = gt_this_feature_mean
        # shape: (D_feat,)

        Fg.append(gt_this_feature.cpu().numpy())
        # each element shape: (D_feat,)

    # Stack all per-trajectory vectors into 2-D feature matrices.
    Fp = np.stack(Fp, axis=0).reshape(-1, Fp[0].shape[-1])
    # shape: (N_preds, D_feat)

    Fg = np.stack(Fg, axis=0).reshape(-1, Fg[0].shape[-1])
    # shape: (N_gts, D_feat)

    print(Fp.shape, Fg.shape)  # informational output for the caller

    return Fp, Fg
    # Fp shape: (N_preds, D_feat),  Fg shape: (N_gts, D_feat)


# ---------------------------------------------------------------------------
# Fréchet distance computation
# ---------------------------------------------------------------------------

import numpy as np
from scipy import linalg


def compute_fid_feats(
        X_real: np.ndarray,         # shape: (N, D)
        X_fake: np.ndarray,         # shape: (M, D)
        *,
        eps: float = 1e-6,          # diagonal jitter added to covariance matrices
        unbiased: bool = True,      # if True divide by N-1 (unbiased covariance estimate)
        clip_negative: bool = True  # clip numerically negative FID values to 0
) -> float:
    """Compute the Fréchet distance between two sets of feature vectors.

    The Fréchet distance (used in FID / FTD) between two multivariate Gaussian
    distributions N(mu_r, Sigma_r) and N(mu_g, Sigma_g) is:

        FD = ||mu_r - mu_g||^2
             + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r @ Sigma_g))

    The matrix square root sqrt(Sigma_r @ Sigma_g) is computed via
    scipy.linalg.sqrtm.  A small diagonal jitter ``eps * I`` is added to each
    covariance matrix before the product to improve numerical stability.

    Args:
        X_real (np.ndarray): Feature matrix for real (ground-truth) samples.
            shape: (N, D).
        X_fake (np.ndarray): Feature matrix for generated (predicted) samples.
            shape: (M, D).
        eps (float): Diagonal perturbation added to covariance matrices to
            prevent singularity.  Default: 1e-6.
        unbiased (bool): If True, covariance is divided by N-1 (unbiased
            estimator), matching the standard FID computation.  If False,
            divides by N (biased estimator).  Default: True.
        clip_negative (bool): If True, values slightly below zero caused by
            floating-point noise are clipped to 0.0.  Default: True.

    Returns:
        fid (float): Fréchet distance >= 0.

    Raises:
        ValueError: If scipy.linalg.sqrtm produces a significantly imaginary
            result, indicating a near-singular or ill-conditioned covariance.
    """
    # Cast to float64 for numerical precision during matrix operations.
    Xr = np.asarray(X_real, dtype=np.float64)   # shape: (N, D)
    Xg = np.asarray(X_fake, dtype=np.float64)   # shape: (M, D)

    # --- Compute means. ---
    mu_r = Xr.mean(axis=0)   # shape: (D,)
    mu_g = Xg.mean(axis=0)   # shape: (D,)
    diff = mu_r - mu_g        # shape: (D,)

    # --- Compute regularised covariance matrices. ---
    cov_opt = dict(rowvar=False, bias=not unbiased)   # rowvar=False: each row is an observation

    # Add eps * I for numerical stability (prevents singular covariance).
    sigma_r = np.cov(Xr, **cov_opt) + eps * np.eye(Xr.shape[1])  # shape: (D, D)
    sigma_g = np.cov(Xg, **cov_opt) + eps * np.eye(Xg.shape[1])  # shape: (D, D)

    # --- Compute the matrix product whose square root enters the trace term. ---
    cov_prod = sigma_r @ sigma_g   # shape: (D, D)

    # Matrix square root via Schur decomposition (scipy implementation).
    # disp=False returns (sqrtm, error_estimate) rather than raising on error.
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)
    # covmean shape: (D, D)

    # scipy.linalg.sqrtm may return complex values for near-singular inputs;
    # keep only the real part if the imaginary component is negligible.
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            raise ValueError(
                "sqrtm returned a significantly imaginary result; "
                "the covariance matrix may be singular."
            )
        covmean = covmean.real  # shape: (D, D) — discard negligible imaginary part

    # --- Assemble the Fréchet distance. ---
    # FD = (mu_r - mu_g)^T (mu_r - mu_g) + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r Sigma_g))
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)  # scalar

    # Clip tiny negative values that can arise from floating-point noise.
    if clip_negative and fid < 0:
        fid = 0.0

    return float(fid)


def get_ftd(
    pred_traj,              # list of np.ndarray, each shape: (T_i, 2)
    gt_traj,                # list of np.ndarray, each shape: (T_j, 2)
    stride: int = 1,        # sliding-window stride for infer()
    eps: float = 1e-6,      # diagonal jitter for compute_fid_feats()
) -> float:
    """Compute the Fréchet Trajectory Distance (FTD) between predicted and GT trajectories.

    FTD is computed in two stages:
        1. ``infer``:           Extract MTR polyline features for both sets of trajectories.
        2. ``compute_fid_feats``: Compute the Fréchet distance between the two
                                  feature distributions.

    Args:
        pred_traj (list[np.ndarray]): Predicted trajectory list.
            Each element shape: (T_i, 2).
        gt_traj (list[np.ndarray]): Ground-truth trajectory list.
            Each element shape: (T_j, 2).
        stride (int): Stride for the sliding window in ``infer``.  Default: 1.
        eps (float): Diagonal regularisation for covariance in
            ``compute_fid_feats``.  Default: 1e-6.

    Returns:
        ftd (float): Fréchet Trajectory Distance (lower is better).
    """
    # Stage 1: extract feature matrices from the MTR polyline encoder.
    Fp, Fg = infer(pred_traj, gt_traj, stride=stride)
    # Fp shape: (N_preds, D_feat),  Fg shape: (N_gts, D_feat)

    # Stage 2: compute Fréchet distance between the two feature distributions.
    return compute_fid_feats(Fp, Fg)  # scalar float
