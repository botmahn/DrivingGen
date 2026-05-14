"""
dataset.py
==========
Data structures and visualisation utilities for the visual odometry (VO)
pipeline used in the DrivingGen benchmark evaluation.

This module provides:

1. DatasetHandler
   A lightweight container that wraps pre-loaded RGB frames, metric depth maps,
   and a camera intrinsic matrix into a single object consumed by the VO
   pipeline (vo.py).  On construction it converts every RGB frame to grayscale
   via OpenCV so that feature detectors (SIFT) operate on single-channel images.

2. visualize_camera_movement
   Overlays optical-flow-style arrows between matched feature points on a pair
   of consecutive frames to give a qualitative sense of camera displacement.

3. estimate_yaw_from_xy
   Converts a sequence of (x, y) 2-D positions into per-frame heading angles
   (yaw) using forward finite differences and atan2.

4. draw_car
   Renders oriented car-shaped rectangles (rotated bounding boxes) onto a
   matplotlib Axes object at every position in a trajectory.

5. visualize_trajectory
   Produces a 2x2 matplotlib figure comparing the estimated ego trajectory,
   ground-truth ego trajectory, and trajectories of focal (surrounding) agents
   side-by-side in the bird's-eye (X-Z) plane.
"""

import os

import math
import numpy as np
import cv2 as cv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


class DatasetHandler:
    """Container for a driving video clip used by the visual odometry pipeline.

    Attributes
    ----------
    num_frames : int
        Total number of frames in the clip.
    images : list of np.ndarray
        Grayscale versions of every RGB frame.
        Each element has shape (H, W) with dtype uint8.
    images_rgb : list of np.ndarray
        Original RGB frames in uint8 format.
        Each element has shape (H, W, 3).
    depth_maps : list of np.ndarray
        Metric depth maps aligned with the RGB frames.
        Each element has shape (H, W) with values in metres (float64).
    k : np.ndarray
        Camera intrinsic matrix, shape (3, 3).
        Layout:
          [[fx,  0, cx],
           [ 0, fy, cy],
           [ 0,  0,  1]]
    """

    def __init__(self, rgb_np_list, depth_np_list, k):
        """Initialise the handler from pre-loaded numpy arrays.

        Parameters
        ----------
        rgb_np_list : list of np.ndarray
            Ordered list of RGB frames, each of shape (H, W, 3), dtype uint8.
        depth_np_list : list of np.ndarray
            Ordered list of depth maps, each of shape (H, W), dtype float64,
            values in metres.
        k : array-like, shape (3, 3)
            Camera intrinsic matrix (focal lengths and principal point).
        """
        # Store the total frame count so downstream code can iterate without
        # re-computing len() each time.
        self.num_frames = len(rgb_np_list)

        # Set up paths
        # root_dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.image_dir = os.path.join(root_dir_path, 'data/rgb')
        # self.depth_dir = os.path.join(root_dir_path, 'data/depth')

        # Initialise list that will hold grayscale images; filled in the loop below.
        self.images = []

        # Keep the original colour frames for visualisation (arrow overlays,
        # saving debug images).  No copy is made to avoid doubling memory.
        self.images_rgb = rgb_np_list       # list of (H, W, 3) uint8 arrays

        # Depth maps are used by the PnP solver in vo.py to lift 2-D pixel
        # correspondences into 3-D object points.
        self.depth_maps = depth_np_list     # list of (H, W) float64 arrays

        # self.k = np.array([[640, 0, 640],
        #                    [0, 480, 480],
        #                    [0,   0,   1]], dtype=np.float32)

        # Store camera intrinsics.  The matrix relates pixel coordinates to
        # normalised camera-plane coordinates via:
        #   [u, v, 1]^T = K * [Xc/Zc, Yc/Zc, 1]^T
        self.k = k                          # shape: (3, 3)

        # Read first frame
        # self.read_frame()

        # Convert every RGB frame to grayscale once at construction time.
        # SIFT (and most classical feature detectors) operate on single-channel
        # images, so pre-converting avoids repeated conversions in the VO loop.
        # cv.COLOR_RGB2GRAY applies the standard luminance formula:
        #   Y = 0.299*R + 0.587*G + 0.114*B
        for img in rgb_np_list:
            # img shape: (H, W, 3) uint8  ->  grayscale shape: (H, W) uint8
            self.images.append(cv.cvtColor(img, cv.COLOR_RGB2GRAY))

    def read_frame(self):
        """Legacy entry point: load depth then RGB from disk (not used in the
        current array-based workflow; kept for backward compatibility)."""
        self._read_depth()
        self._read_image()

    def _read_image(self):
        """Load grayscale and colour frames from the on-disk 'data/rgb' folder.

        Files are expected to be named 'frame_00001.png', 'frame_00002.png', …
        This method is part of the legacy file-system interface and is NOT
        called when the handler is constructed with pre-loaded numpy arrays.
        """
        for i in range(1, self.num_frames + 1):
            # Zero-pad the frame index to 5 digits to match the naming convention
            # used by the CARLA simulator exporter (e.g. 00001, 00042, 01000).
            zeroes = "0" * (5 - len(str(i)))
            im_name = "{0}/frame_{1}{2}.png".format(self.image_dir, zeroes, str(i))

            # Load as grayscale (flags=0 == cv.IMREAD_GRAYSCALE).
            # Grayscale images have shape (H, W).
            self.images.append(cv.imread(im_name, flags=0))

            # Load again as BGR, then reverse the channel order to RGB so that
            # matplotlib (which expects RGB) renders correct colours.
            # Shape: (H, W, 3) uint8
            self.images_rgb.append(cv.imread(im_name)[:, :, ::-1])

            # Print a combined loading progress bar.  Images are loaded in the
            # second half of the total work (depth is the first half), so the
            # offset (+ self.num_frames) places the progress bar in [50%, 100%].
            print("Data loading: {0}%".format(
                int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)),
                end="\r")

    def _read_depth(self):
        """Load depth maps from the on-disk 'data/depth' folder.

        Each depth file is a comma-separated text file produced by CARLA.
        Values are stored in kilometres and multiplied by 1000 to convert to
        metres (CARLA encodes depth as distance / far_clip, stored in the range
        [0, 1]; the * 1000 factor here scales the raw values to metres).

        This method is part of the legacy file-system interface.
        """
        for i in range(1, self.num_frames + 1):
            # Build zero-padded filename matching the CARLA depth exporter format.
            zeroes = "0" * (5 - len(str(i)))
            depth_name = "{0}/frame_{1}{2}.dat".format(self.depth_dir, zeroes, str(i))

            # np.loadtxt parses the CSV into a 2-D float64 array.
            # Multiplying by 1000 converts the CARLA-normalised depth values
            # (stored in [0, 1] range) to metric metres.
            # shape: (H, W) float64
            depth = np.loadtxt(
                depth_name,
                delimiter=',',
                dtype=np.float64) * 1000.0
            self.depth_maps.append(depth)

            # Print progress for the first half (depth loading).
            print("Data loading: {0}%".format(
                int(i / (self.num_frames * 2 - 1) * 100)),
                end="\r")


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    """Overlay feature correspondence arrows on a frame to show camera motion.

    For each matched point pair (p1 in frame t, p2 in frame t+1), the function
    draws:
      - a green circle at the source location p1
      - a green arrow from p1 to p2 indicating the optical-flow direction
      - a blue circle at the destination location p2

    This gives an intuitive visualisation of how the camera moved between two
    consecutive frames.

    Parameters
    ----------
    image1 : np.ndarray
        RGB image at time t, shape (H, W, 3).
    image1_points : list of (u, v)
        2-D pixel coordinates of matched features in image1.
    image2 : np.ndarray
        RGB image at time t+1, shape (H, W, 3).
    image2_points : list of (u, v)
        Corresponding 2-D pixel coordinates in image2.
    is_show_img_after_move : bool, optional
        If True, annotate and return image2 (the destination frame) instead of
        the source frame.  Useful for side-by-side comparison.

    Returns
    -------
    np.ndarray
        Annotated copy of image1 (or image2 when is_show_img_after_move=True).
        Shape: (H, W, 3).
    """
    # Work on copies to avoid mutating the caller's arrays.
    image1 = image1.copy()   # shape: (H, W, 3)
    image2 = image2.copy()   # shape: (H, W, 3)

    for i in range(0, len(image1_points)):
        # Convert float coordinates to integer pixel indices required by OpenCV.
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        # Draw a small green circle at the feature's position in frame t.
        # The circle marks the starting position of the optical-flow vector.
        cv.circle(image1, p1, 5, (0, 255, 0), 1)

        # Draw an arrow from the frame-t position to the frame-(t+1) position.
        # The arrow direction encodes where the scene point moved in image space,
        # which is the *inverse* of the camera motion projected onto the image.
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)

        # Draw a blue circle at the feature's destination in frame t+1.
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            # When showing the destination frame, annotate that frame as well.
            cv.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        # Return the annotated destination (t+1) frame.
        return image2
    else:
        # Return the annotated source (t) frame with arrows drawn on it.
        return image1


def estimate_yaw_from_xy(xs, ys):
    """Compute per-frame heading (yaw) angles from a sequence of (x, y) positions.

    Uses forward finite differences on the trajectory and atan2 to obtain the
    instantaneous heading at each time step.  The last frame replicates the
    second-to-last heading because a forward difference cannot be computed for
    the final position.

    Parameters
    ----------
    xs : array-like, shape (N,)
        X-coordinates of the trajectory in metres (or any consistent unit).
    ys : array-like, shape (N,)
        Y-coordinates of the trajectory.

    Returns
    -------
    yaws : np.ndarray, shape (N,)
        Heading angle in radians at each time step, measured from the positive
        X-axis in the standard mathematical (counter-clockwise positive) sense.
    """
    # Stack x and y into a single 2-D array for vectorised differencing.
    # shape: (N, 2)
    xys = np.stack([xs, ys], axis=-1)

    # Compute consecutive displacement vectors: delta[i] = pos[i+1] - pos[i].
    # shape: (N-1, 2)
    deltas = np.diff(xys, axis=0)

    # atan2(dy, dx) gives the angle of each displacement vector.
    # This is the geometric heading in the XY plane, measured in radians.
    # shape: (N-1,)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])

    # The final frame has no successor, so its heading is approximated by the
    # heading of the previous frame (constant-velocity extrapolation of direction).
    # shape: (N,)
    yaws = np.append(yaws, yaws[-1])  # Replicate last frame's heading direction
    return yaws


from matplotlib.patches import Polygon

def draw_car(ax, xs, ys, length=4.5, width=2.0, color='red', zorder=1):
    """Draw oriented car-shaped rectangles along a trajectory on a matplotlib axis.

    At every position (x, y) in the trajectory the function:
      1. Estimates the instantaneous heading via estimate_yaw_from_xy.
      2. Builds the four corner points of a rectangle of the given dimensions
         in the vehicle's local frame (centred at the origin, nose pointing +X).
      3. Rotates the corners by the heading angle using a 2-D rotation matrix.
      4. Translates the rotated corners to world coordinates and adds a filled
         semi-transparent Polygon patch to the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to draw the patches.
    xs : array-like, shape (N,)
        World X-coordinates of the vehicle centre at each time step.
    ys : array-like, shape (N,)
        World Y-coordinates (or Z in bird's-eye convention) at each time step.
    length : float, optional
        Vehicle length in the same units as xs/ys (default: 4.5 m).
    width : float, optional
        Vehicle width (default: 2.0 m).
    color : str or colour spec, optional
        Fill colour of the rectangle patches.
    zorder : int, optional
        Matplotlib drawing order (higher values appear on top).
    """
    # Obtain per-frame heading angles in radians.  shape: (N,)
    yaws = estimate_yaw_from_xy(xs, ys)

    for x, y, yaw in zip(xs, ys, yaws):
        # Define the four corner points in the vehicle's LOCAL coordinate frame.
        # The vehicle body is centred at the origin; the nose points in the +X
        # direction.  Corners are listed in counter-clockwise order.
        # shape: (4, 2)
        corners = np.array([
            [ length/2,  width/2],   # front-left
            [ length/2, -width/2],   # front-right
            [-length/2, -width/2],   # rear-right
            [-length/2,  width/2],   # rear-left
        ])

        # 2-D rotation matrix for heading angle yaw (radians).
        # Rotates a vector from the vehicle frame into the world (XY) frame:
        #   [Xw]   [cos(yaw)  -sin(yaw)] [Xlocal]
        #   [Yw] = [sin(yaw)   cos(yaw)] [Ylocal]
        # shape: (2, 2)
        rot = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        # Apply rotation and translation to bring corners into world frame.
        # corners @ rot.T rotates each row (corner) from local to world frame.
        # Adding [x, y] translates to the vehicle's world position.
        # shape: (4, 2)
        transformed = corners @ rot.T + np.array([x, y])

        # Create a filled semi-transparent polygon patch and add it to the axes.
        # alpha=0.3 ensures overlapping patches from nearby time steps remain
        # visually distinguishable.
        patch = Polygon(transformed, closed=True, edgecolor='none', facecolor=color, alpha=0.3, zorder=1)
        ax.add_patch(patch)

        # Optional: draw a line from the vehicle centre to the front mid-point
        # to indicate the nose/heading direction (currently disabled).
        # head = np.array([length/2, 0]) @ rot.T + np.array([x, y])
        # ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=1.5, zorder=zorder)


def visualize_trajectory(trajectory, outdir, gt=None, others=None, others_gt=None, map_scale=1, car_length=4.5, car_width=2.0, draw_polygon=0):
    """Generate a 2x2 figure comparing estimated and ground-truth trajectories.

    The figure layout is:
      [0,0] traj_main_plt  -- Overlay of all available trajectories (ego + GT + focal agents)
      [0,1] gt_plt         -- GT-only ego and focal-agent ground truth
      [1,0] ego_plt        -- Estimated ego trajectory alone
      [1,1] others_plt     -- Estimated focal-agent trajectories alone

    All sub-plots use the bird's-eye (X-Z) coordinate convention, where X is
    the lateral axis and Z is the forward (depth) axis of the vehicle.

    Parameters
    ----------
    trajectory : np.ndarray, shape (3, N)
        Estimated ego-vehicle trajectory.  Each column is a 3-D world position
        [X, Y, Z]^T.  Y is the vertical axis (height); X and Z form the
        ground plane.
    outdir : str
        Directory path where the output figure 'pose.jpg' will be saved.
    gt : np.ndarray or None, shape (N, 2)
        Ground-truth ego trajectory columns [X, Z].  If None, no GT curve
        is plotted.
    others : tuple or np.ndarray or None
        Focal-agent estimated trajectories.  May be a plain array of shape
        (N, 2) or a 3-tuple (others, others_w_ego, others_w_opt) for comparing
        different post-processing variants.  If None, omitted.
    others_gt : list of np.ndarray or None
        List of ground-truth trajectories for surrounding agents.  Each element
        has shape (T, 2) with columns [X, Z].  Only agents that move more than
        10 units along X are plotted (to suppress parked vehicles).
    map_scale : float, optional
        Unused scale factor kept for API compatibility.
    car_length : float, optional
        Length of the oriented rectangle drawn for each car (metres).
    car_width : float, optional
        Width of the oriented rectangle drawn for each car (metres).
    draw_polygon : int (bool), optional
        When non-zero, oriented car rectangles are rendered on top of the
        trajectory lines.
    """
    # Unpack X Y Z each trajectory point
    locX = []   # lateral (X) positions of ego vehicle
    locY = []   # vertical (Y) positions — unused in 2-D bird's-eye plots
    locZ = []   # forward (Z) positions of ego vehicle

    # Track the Y-axis range for potential future use in axis scaling.
    # These are initialised to ±inf so any real value will update them.
    maxY = -math.inf
    minY = math.inf

    # Iterate over trajectory columns to unpack per-frame positions.
    # trajectory shape: (3, N) — columns are world positions [X, Y, Z]^T
    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]   # shape: (3,)

        locX.append(current_pos.item(0))   # X coordinate (lateral)
        locY.append(current_pos.item(1))   # Y coordinate (height, unused in 2-D)
        locZ.append(current_pos.item(2))   # Z coordinate (forward depth)

        # Track Y bounds for potential vertical-axis range clipping.
        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    # auxY_line is the midpoint of the first and last Y positions; kept here
    # as a reference value for potential symmetric Y-axis centering.
    auxY_line = locY[0] + locY[-1]

    # Apply a clean white-background style and a sans-serif font globally.
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create the 2x2 subplot grid.  figsize=(10, 8) gives enough resolution
    # for four sub-plots to be legible when saved at 100 dpi.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100)

    # Assign each sub-plot a descriptive variable name for readability.
    traj_main_plt = axes[0, 0]   # combined overlay panel
    gt_plt        = axes[0, 1]   # ground-truth-only panel
    ego_plt       = axes[1, 0]   # estimated ego-only panel
    others_plt    = axes[1, 1]   # focal-agents panel

    gspec = gridspec.GridSpec(1, 1)  # retained for possible future grid layout

    # Retrieve the default colour cycle so each trajectory gets a consistent,
    # visually distinct colour across all four sub-plots.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Plot the estimated ego trajectory (X vs Z, bird's-eye view) ---
    toffset = 1.06   # title y-offset above the axes (unused; overwritten below)
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)

    if draw_polygon:
        # Draw oriented car rectangles along the ego path on the combined and
        # ego-only panels.
        draw_car(traj_main_plt, locX, locZ, car_length, car_width, color=colors[0], zorder=1)
        draw_car(ego_plt, locX, locZ, car_length, car_width, color=colors[0], zorder=1)

    # Plot the ego trajectory as a connected scatter (".-") so individual
    # frame positions are visible as dots and the path is shown as a line.
    # locX is lateral, locZ is forward; zorder=6 ensures ego is on top.
    traj_main_plt.plot(locX, locZ, ".-", label="Ego", zorder=6, linewidth=1, markersize=4, color=colors[0])
    ego_plt.plot(locX, locZ, ".-", label="Ego", zorder=6, linewidth=1, markersize=4, color=colors[0])

    # Initialise axis bounds from the ego trajectory; they will be expanded
    # as additional trajectories are added.
    max_x, min_x = np.max(locX), np.min(locX)
    max_y, min_y = np.max(locZ), np.min(locZ)

    # --- Plot the ground-truth ego trajectory if provided ---
    if gt is not None:
        # gt shape: (N, 2) — columns [X, Z]
        if draw_polygon:
            draw_car(traj_main_plt, gt[:, 0], gt[:, 1], car_length, car_width, color=colors[1], zorder=1)
            draw_car(gt_plt, gt[:, 0], gt[:, 1], car_length, car_width, color=colors[1], zorder=1)

        # Plot GT at zorder=1 so it appears behind the estimated trajectory.
        traj_main_plt.plot(gt[:, 0], gt[:, 1], ".-", label="EgoGT", zorder=1, linewidth=1, markersize=4, color=colors[1])
        gt_plt.plot(gt[:, 0], gt[:, 1], ".-", label="EgoGT", zorder=1, linewidth=1, markersize=4, color=colors[1])

        # Expand the bounding box to include the GT trajectory.
        max_x = max(max_x, np.max(gt[:, 0]))
        min_x = min(min_x, np.min(gt[:, 0]))
        max_y = max(max_y, np.max(gt[:, 1]))
        min_y = min(min_y, np.min(gt[:, 1]))

    # --- Plot focal-agent estimated trajectories if provided ---
    if others is not None:
        # The 'others' argument can encode up to three trajectory variants:
        #   others          -- base focal-agent trajectory
        #   others_w_ego    -- focal-agent with ego conditioning
        #   others_w_opt    -- focal-agent with optimisation post-processing
        # Unpack with a try/except so a plain array also works.
        try:
            others, others_w_ego, others_w_opt = others
        except:
            # If unpackable, treat as a single trajectory and disable variants.
            others_w_ego = None
            others_w_opt = None

        # others shape: (N, 2) — columns [X, Z]
        if draw_polygon:
            draw_car(traj_main_plt, others[:, 0], others[:, 1], car_length, car_width, color=colors[2], zorder=1)
            draw_car(others_plt, others[:, 0], others[:, 1], car_length, car_width, color=colors[2], zorder=1)

        traj_main_plt.plot(others[:, 0], others[:, 1], ".-", label="Focal", zorder=3, linewidth=1, markersize=4, color=colors[2])
        others_plt.plot(others[:, 0], others[:, 1], ".-", label="Focal", zorder=3, linewidth=1, markersize=4, color=colors[2])

        # Expand bounding box to include focal-agent trajectory.
        max_x = max(max_x, np.max(others[:, 0]))
        min_x = min(min_x, np.min(others[:, 0]))
        max_y = max(max_y, np.max(others[:, 1]))
        min_y = min(min_y, np.min(others[:, 1]))

        if others_w_ego is not None:
            # Expand bounds only; this variant is currently not plotted to
            # avoid cluttering the figure (line is commented out).
            # others_w_ego shape: (N, 2)
            max_x = max(max_x, np.max(others_w_ego[:, 0]))
            min_x = min(min_x, np.min(others_w_ego[:, 0]))
            max_y = max(max_y, np.max(others_w_ego[:, 1]))
            min_y = min(min_y, np.min(others_w_ego[:, 1]))

        if others_w_opt is not None:
            # others_w_opt shape: (N, 2) — optimised focal-agent trajectory
            if draw_polygon:
                draw_car(traj_main_plt, others_w_opt[:, 0], others_w_opt[:, 1], car_length, car_width, color=colors[3], zorder=1)
                draw_car(others_plt, others_w_opt[:, 0], others_w_opt[:, 1], car_length, car_width, color=colors[3], zorder=1)

            # Plot at zorder=6 to appear on top; it typically represents the
            # best-quality prediction and should be most visible.
            traj_main_plt.plot(others_w_opt[:, 0], others_w_opt[:, 1], ".-", label="Focal_w_opt", zorder=6, linewidth=1, markersize=4, color=colors[3])
            others_plt.plot(others_w_opt[:, 0], others_w_opt[:, 1], ".-", label="Focal_w_opt", zorder=6, linewidth=1, markersize=4, color=colors[3])

            max_x = max(max_x, np.max(others_w_opt[:, 0]))
            min_x = min(min_x, np.min(others_w_opt[:, 0]))
            max_y = max(max_y, np.max(others_w_opt[:, 1]))
            min_y = min(min_y, np.min(others_w_opt[:, 1]))

    # --- Plot surrounding-agent ground-truth trajectories if provided ---
    if others_gt is not None:
        count = 0
        for o_gt in others_gt:
            # o_gt shape: (T, 2) — per-agent GT, columns [Z, X] (note: swapped)
            # Filter out agents with less than 10 units of X-displacement to
            # suppress stationary or barely-moving vehicles that would clutter
            # the plot without adding useful comparison signal.
            if abs(o_gt[-1, 0] - o_gt[0, 0]) > 10:
                count += 1
                if count == 2:
                    # Only plot the second agent that passes the motion threshold
                    # (an arbitrary selection to keep the figure uncluttered).
                    print(f'len: {o_gt.shape[0]}')
                    if draw_polygon:
                        # Note: columns are [Z, X] in o_gt, so we pass
                        # o_gt[:,1] as X and o_gt[:,0] as Z.
                        draw_car(traj_main_plt, o_gt[:, 1], o_gt[:, 0], car_length, car_width, color=colors[4], zorder=1)
                        draw_car(gt_plt, o_gt[:, 1], o_gt[:, 0], car_length, car_width, color=colors[4], zorder=1)

                    traj_main_plt.plot(o_gt[:, 1], o_gt[:, 0], ".-", label="FocalGT", zorder=2, linewidth=1, markersize=4, color=colors[4])
                    gt_plt.plot(o_gt[:, 1], o_gt[:, 0], ".-", label="FocalGT", zorder=2, linewidth=1, markersize=4, color=colors[4])

                    max_x = max(max_x, np.max(o_gt[:, 1]))
                    min_x = min(min_x, np.min(o_gt[:, 1]))
                    max_y = max(max_y, np.max(o_gt[:, 0]))
                    min_y = min(min_y, np.min(o_gt[:, 0]))

    def set_plot(this_plot):
        """Apply consistent axis formatting to all four sub-plots.

        Sets axis labels, computes axis limits with a small padding margin so
        no trajectory point is clipped, and adds a legend.

        Parameters
        ----------
        this_plot : matplotlib.axes.Axes
            The sub-plot to configure.
        """
        this_plot.set_xlabel("X")   # lateral axis label
        this_plot.set_ylabel("Z")   # forward (depth) axis label

        # Add 3-unit padding on both sides of the data extent to ensure no
        # trajectory point sits right on the axis boundary.
        this_plot.set_xlim([min_x - 3, max_x + 3])
        this_plot.set_ylim([-3, max_y + 3])

        # Place the legend in the upper-right corner (loc=1) with a visible
        # frame so it does not blend into dense trajectory regions.
        this_plot.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Apply the same axis configuration to every sub-plot for visual consistency.
    set_plot(traj_main_plt)
    set_plot(gt_plt)
    set_plot(ego_plt)
    set_plot(others_plt)

    # plt.show() renders the figure interactively (blocks until window is closed
    # when running in a standard Python process).
    plt.show()

    # Save the figure to disk for logging/reporting.  JPEG at the default
    # quality setting is sufficient for trajectory visualisations.
    plt.savefig(os.path.join(outdir, 'pose.jpg'))
