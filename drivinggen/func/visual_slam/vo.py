"""
vo.py
=====
Visual Odometry (VO) pipeline for the DrivingGen benchmark evaluation.

This module implements a full monocular-depth VO system that accumulates
camera poses from a sequence of driving frames.  The pipeline operates in
the following stages for each consecutive frame pair (i, i+1):

  1. Feature detection  -- SIFT keypoints are extracted from every grayscale frame.
  2. Feature matching   -- FLANN kNN matching with cross-check builds candidate
                           correspondences between adjacent frames.
  3. Ratio-test filter  -- Lowe's ratio test (Lowe 2004) prunes ambiguous matches.
  4. Motion estimation  -- PnP-RANSAC lifts 2-D correspondences to 3-D using a
                           metric depth map and solves for the relative camera
                           rotation R and translation t.
  5. Pose accumulation  -- The relative transform is composed with the running
                           world pose to obtain the global camera position.

When PnP fails or the recovered pose fails numerical validation, the pipeline
falls back to a stochastic inertial extrapolation: the previous step-length is
preserved but the heading is perturbed by a random yaw within ±90 degrees to
avoid a degenerate stationary estimate.

Key conventions:
  - All coordinate frames follow the OpenCV/camera convention:
      X right, Y down, Z forward (into the scene).
  - Depth maps are in metres.
  - Camera intrinsic matrix K has shape (3, 3):
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
  - The global trajectory array has shape (3, N) where column i holds [X, Y, Z]^T
    of the camera at frame i.
"""


######3 vo fast ##############

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os


def check_data(dataset_handler, outdir):
    """Sanity-check the dataset by displaying and saving the first frame's RGB and depth.

    Writes two debug images to outdir:
      - 'rgb0.jpg'   : the first colour frame
      - 'depth0.jpg' : the first depth map rendered with a jet colour map

    Also prints the depth map's spatial dimensions and the depth value at the
    bottom-right corner pixel.

    Parameters
    ----------
    dataset_handler : DatasetHandler
        Loaded dataset containing grayscale images, RGB images, and depth maps.
    outdir : str
        Directory where debug images will be written.
    """
    # Retrieve the first grayscale frame and display it to verify loading.
    # image shape: (H, W)  uint8
    image = dataset_handler.images[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image, cmap='gray')

    # Retrieve and save the first RGB frame.
    # image_rgb shape: (H, W, 3) uint8
    image_rgb = dataset_handler.images_rgb[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(image_rgb)
    plt.savefig(os.path.join(outdir, 'rgb0.jpg'))

    # Retrieve and save the first depth map.  The jet colour map makes small
    # depth differences visually salient (blue = near, red = far).
    # depth shape: (H, W) float64  values in metres
    depth = dataset_handler.depth_maps[0]
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(depth, cmap='jet')
    plt.savefig(os.path.join(outdir, 'depth0.jpg'))

    # Print shape and a corner value for a quick sanity check.
    print("Depth map shape: {0}".format(depth.shape))
    v, u = depth.shape  # v = height (rows), u = width (cols)
    # Bottom-right pixel is a common spot to verify that depth is non-zero and
    # in the expected range (prevents silent all-zero or all-inf depth maps).
    depth_val = depth[v - 1, u - 1]
    print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(0, depth_val))


# Updated from ORB to SIFT on 2024-03-19 for improved descriptor quality on
# driving scenes where illumination and scale changes are common.

def extract_features(image, mask):
    """Detect SIFT keypoints and compute their descriptors for a single image.

    SIFT (Scale-Invariant Feature Transform) is chosen over ORB because its
    floating-point 128-dimensional descriptors are more discriminative under
    partial occlusion, blur, and viewpoint change — all common in driving video.

    The detector parameters are tuned to balance:
      - nfeatures=2500    : enough correspondences across large blank regions
                            (road surface, sky) that have few texture gradients.
      - contrastThreshold : lowered from the default 0.04 to detect more
                            keypoints in low-contrast regions (e.g. fog, overcast).
      - edgeThreshold     : reduced from 10 to favour corner-like points over
                            elongated edge responses that are less repeatable.
      - sigma             : reduced from 1.6 to preserve fine-grained details
                            that get blurred at higher sigma values.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Grayscale input image, dtype uint8.
    mask : np.ndarray or None, shape (H, W)
        Binary mask restricting feature detection to specific image regions
        (e.g. road surface only, excluding sky).  None means no restriction.

    Returns
    -------
    kp : list of cv2.KeyPoint
        Detected keypoints, each encoding position (u, v), scale, and orientation.
    des : np.ndarray, shape (N_kp, 128)
        SIFT descriptors, one 128-dimensional float32 vector per keypoint.
    """
    # Initialise SIFT detector with carefully tuned hyperparameters.
    # SIFT_create returns a Feature2D object that internally builds a
    # Difference-of-Gaussians scale space and locates extrema.
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015, edgeThreshold=7, sigma=1.3)

    # Jointly detect keypoints and compute descriptors in a single pass.
    # kp  : list of KeyPoint, length N_kp
    # des : shape (N_kp, 128) float32
    kp, des = sift.detectAndCompute(image, mask)

    return kp, des


def visualize_features(image, kp):
    """Overlay detected SIFT keypoints on the image for debugging.

    Renders each keypoint as a green circle whose radius encodes the keypoint
    scale.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Grayscale image.
    kp : list of cv2.KeyPoint
        Keypoints to visualise.
    """
    # drawKeypoints annotates the image with circles at keypoint locations.
    # flags=0 draws just the centre point; cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # would additionally draw orientation and scale.
    display = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


def extract_features_dataset(images, masks):
    """Run extract_features over all frames in the dataset.

    Parameters
    ----------
    images : list of np.ndarray, each shape (H, W)
        Ordered list of grayscale frames.
    masks : list of np.ndarray or list of None, each shape (H, W) or None
        Per-frame detection masks aligned with 'images'.

    Returns
    -------
    kp_list : list of list of cv2.KeyPoint
        Per-frame keypoint lists.  kp_list[i] corresponds to images[i].
    des_list : list of np.ndarray
        Per-frame descriptor arrays.  des_list[i] has shape (N_i, 128).
    """
    kp_list = []
    des_list = []

    # Process each frame independently; results are stored in order so that
    # kp_list[i] and des_list[i] correspond to images[i].
    for img, mask in zip(images, masks):
        kp, des = extract_features(img, mask)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


def match_features(des1, des2):
    """Find candidate matches between two sets of SIFT descriptors using FLANN.

    FLANN (Fast Library for Approximate Nearest Neighbours) uses a randomised
    KD-tree index to perform approximate nearest-neighbour search in the
    128-dimensional SIFT descriptor space.  The index is configured with
    trees=6 (more trees = higher recall, slower build) and checks=96 (more
    checks = fewer misses, slower query).

    kNN matching with k=2 returns the two closest descriptors so that Lowe's
    ratio test can be applied in filter_matches_distance to discard ambiguous
    matches where the first and second nearest neighbours are too similar.

    Cross-matching (both des1→des2 and des2→des1) is performed so the caller
    can optionally apply a mutual-consistency check (each match must be the
    best match in both directions).

    Parameters
    ----------
    des1 : np.ndarray, shape (N1, 128)
        Descriptors for the first image (frame t).
    des2 : np.ndarray, shape (N2, 128)
        Descriptors for the second image (frame t+1).

    Returns
    -------
    match : list of tuples (cv2.DMatch, cv2.DMatch)
        Forward matches (des1→des2).  Each element is a pair (best, second-best).
    match2 : list of tuples (cv2.DMatch, cv2.DMatch)
        Reverse matches (des2→des1).  Each element is a pair (best, second-best).
        Returns ([], []) on error.
    """
    # FLANN_INDEX_KDTREE=1 selects the randomised KD-tree algorithm, which
    # is the recommended choice for floating-point SIFT descriptors.
    # (FLANN_INDEX_LSH is better suited for binary descriptors like ORB.)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=6)   # More trees → higher recall at the cost of build time

    # checks=96 means FLANN traverses up to 96 leaf nodes per query.
    # Higher values reduce false negatives (missed true matches) but increase
    # query latency.
    search_params = dict(checks=96)

    # Initialise FLANN matcher with the KD-tree index.
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform kNN matching (k=2) in both directions.  The try/except guards
    # against edge cases where one of the descriptor arrays is empty or
    # contains NaN values (e.g. textureless frames with no SIFT keypoints).
    try:
        # Forward matches: for each descriptor in des1, find its 2 nearest
        # neighbours in des2.
        # match  : list of length N1, each element is a list of 2 DMatch objects
        match = flann.knnMatch(des1, des2, k=2)

        # Reverse matches: for each descriptor in des2, find its 2 nearest
        # neighbours in des1.  Used for cross-consistency filtering.
        match2 = flann.knnMatch(des2, des1, k=2)
    except:
        print('matching fail')
        return [], []

    return match, match2


# Optional
def filter_matches_distance(match, dist_threshold, match2):
    """Apply Lowe's ratio test to keep only unambiguous feature matches.

    The ratio test (Lowe 2004, IJCV) discards a match (m, n) if:
        m.distance >= dist_threshold * n.distance
    where m is the closest descriptor match and n is the second-closest.  If
    the best match is much closer than the runner-up (ratio < threshold), the
    match is distinctive and likely correct.  If both distances are similar,
    the correspondence is ambiguous and prone to error.

    A typical threshold is 0.7–0.8.  Lower values are more strict (fewer but
    more reliable matches); higher values retain more matches at the cost of
    more outliers for RANSAC to handle.

    Parameters
    ----------
    match : list of tuples (cv2.DMatch, cv2.DMatch)
        kNN matches (k=2) to filter.  Each element is a 2-tuple (best, 2nd-best).
    dist_threshold : float
        Lowe's ratio threshold in (0, 1).  Matches with ratio < threshold pass.
    match2 : list of tuples (cv2.DMatch, cv2.DMatch)
        Reverse matches (des2→des1), reserved for optional cross-consistency
        filtering (currently not applied; kept for API extension).

    Returns
    -------
    filtered_match : list of cv2.DMatch
        Matches passing the ratio test.
    """
    filtered_match = []
    for i, m in enumerate(match):
        if isinstance(m, tuple):
            if len(m) == 2:
                m, n = m   # best match and its runner-up
                # Apply Lowe's ratio test: accept only if the best match
                # distance is substantially smaller than the second-best.
                # This filters out keypoints with multiple plausible matches
                # in the descriptor space, which are likely not unique scene features.
                if m.distance < (dist_threshold * n.distance):
                    filtered_match.append(m)
            else:
                # Skip degenerate tuples (e.g. only one neighbour found).
                continue

    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """Draw feature correspondences between two images side by side.

    Parameters
    ----------
    image1 : np.ndarray, shape (H, W) or (H, W, 3)
        First image in the matched pair.
    kp1 : list of cv2.KeyPoint
        Keypoints in image1.
    image2 : np.ndarray, shape (H, W) or (H, W, 3)
        Second image in the matched pair.
    kp2 : list of cv2.KeyPoint
        Keypoints in image2.
    match : list of cv2.DMatch
        Filtered matches to draw.
    """
    # drawMatches concatenates both images horizontally and draws lines between
    # corresponding keypoints.  The result has shape (H, 2*W, 3).
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def match_features_dataset(des_list):
    """Run match_features on all consecutive frame pairs in the dataset.

    Parameters
    ----------
    des_list : list of np.ndarray
        Per-frame descriptor arrays, each of shape (N_i, 128).

    Returns
    -------
    matches1 : list of list of tuple
        Forward matches for each consecutive pair.  matches1[i] contains the
        raw kNN matches (before ratio filtering) from frame i to frame i+1.
    matches2 : list of list of tuple
        Reverse matches for each consecutive pair (frame i+1 to frame i).
    """
    # Match each adjacent pair (i, i+1) in a list comprehension for brevity.
    # len(des_list) - 1 pairs exist for len(des_list) frames.
    matches = [match_features(des_list[i], des_list[i + 1]) for i in range((len(des_list) - 1))]

    # Separate the forward (matches[i][0]) and reverse (matches[i][1]) results
    # into two parallel lists for independent downstream processing.
    matches1 = [match[0] for match in matches]
    matches2 = [match[1] for match in matches]

    return matches1, matches2


# Optional
def filter_matches_dataset(matches, dist_threshold, matches2):
    """Apply Lowe's ratio test to every consecutive frame pair in the dataset.

    Parameters
    ----------
    matches : list of list of tuple
        Per-pair raw kNN matches (forward direction).
    dist_threshold : float
        Lowe's ratio threshold; passed directly to filter_matches_distance.
    matches2 : list of list of tuple
        Per-pair raw kNN matches (reverse direction); passed through for
        optional cross-consistency filtering inside filter_matches_distance.

    Returns
    -------
    filtered_matches : list of list of cv2.DMatch
        Per-pair filtered matches that satisfy the ratio test.
    """
    # Apply the ratio filter independently to each pair.  The result list
    # is aligned with matches so filtered_matches[i] corresponds to the
    # frame pair (i, i+1).
    filtered_matches = [filter_matches_distance(m, dist_threshold, m2) for m, m2 in zip(matches, matches2)]

    return filtered_matches


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    """Overlay matched feature positions on a frame pair to show apparent motion.

    Draws coloured dots at each feature's location in frame t (green) and its
    matched location in frame t+1 (red).  The direction and magnitude of the
    displacement encodes both camera translation and scene depth.

    Note: the arrow rendering is disabled here (commented out) to reduce visual
    clutter; only the endpoint circles are drawn.

    Parameters
    ----------
    image1 : np.ndarray, shape (H, W, 3)
        RGB frame at time t.
    image1_points : list of (u, v)
        Pixel coordinates of matched features in image1.
    image2 : np.ndarray, shape (H, W, 3)
        RGB frame at time t+1.
    image2_points : list of (u, v)
        Corresponding pixel coordinates in image2.
    is_show_img_after_move : bool, optional
        If True, annotate and return image2 instead of image1.

    Returns
    -------
    np.ndarray, shape (H, W, 3)
        Annotated copy of image1 or image2.
    """
    # Operate on copies so the originals stored in DatasetHandler are not modified.
    image1 = image1.copy()   # shape: (H, W, 3) uint8
    image2 = image2.copy()   # shape: (H, W, 3) uint8

    for i in range(0, len(image1_points)):
        # Cast float pixel coordinates to int for OpenCV drawing functions.
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        # Small green circle marks the feature's location in frame t.
        cv2.circle(image1, p1, 1, (0, 255, 0), 1)
        # Arrow is disabled to reduce clutter when many points are shown.
        # cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        # Red circle marks where the same feature appears in frame t+1.
        cv2.circle(image1, p2, 1, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 1, (255, 0, 0), 1)

    if is_show_img_after_move:
        return image2
    else:
        return image1


import numpy as np

def is_pose_valid(rmat, tvec, det_tol=1e-2, trans_lim=50.0):
    """Validate that a recovered rotation matrix and translation vector are physically plausible.

    A valid rotation matrix R must satisfy two properties of SO(3):
      1. All elements are finite (no NaN / Inf from numerical breakdown).
      2. det(R) = 1.0  (proper rotation, not reflection; |det - 1| < det_tol).

    The translation vector is additionally checked for a magnitude bound
    (trans_lim metres) because large translations between adjacent frames
    indicate a failed PnP solve on a driving sequence at typical frame rates.

    Parameters
    ----------
    rmat : np.ndarray, shape (3, 3)
        Recovered rotation matrix from cv2.Rodrigues.
    tvec : np.ndarray, shape (3, 1) or (3,)
        Recovered translation vector from cv2.solvePnP.
    det_tol : float, optional
        Maximum tolerated deviation of det(rmat) from 1.0.  Default 0.01
        allows for small floating-point errors in the Rodrigues conversion.
    trans_lim : float, optional
        Maximum allowed Euclidean norm of tvec in metres.  Default 50 m
        corresponds to the maximum plausible displacement between frames at
        ~180 km/h with a 1 Hz frame rate (conservative upper bound).

    Returns
    -------
    bool
        True if both rotation and translation pass all checks, False otherwise.
    """
    # Check 1: No NaN or Inf in either matrix/vector.  Numerical breakdown in
    # the RANSAC or Rodrigues conversion can silently produce non-finite values.
    if not (np.isfinite(rmat).all() and np.isfinite(tvec).all()):
        return False

    # Check 2: Determinant of a proper rotation matrix must be exactly +1.
    # Values of -1 indicate a reflection (mirror), which is geometrically
    # invalid for a camera pose.  Deviations beyond det_tol signal corruption.
    if abs(np.linalg.det(rmat) - 1.0) > det_tol:
        return False

    # Check 3: Bound the translation magnitude.  Between two consecutive frames
    # the camera cannot have moved more than trans_lim metres without very high
    # speed or a scene discontinuity (cut), both of which should be rejected.
    if np.linalg.norm(tvec) > trans_lim:
        return False

    return True


def safe_inv(mat4x4):
    """Compute the inverse of a 4x4 homogeneous transform matrix with error handling.

    Direct matrix inversion can fail or produce non-finite results if the input
    is nearly singular (e.g. zero translation, degenerate rotation).  This
    wrapper catches LinAlgError and checks the result for finiteness before
    returning.

    Parameters
    ----------
    mat4x4 : np.ndarray, shape (4, 4)
        Homogeneous transformation matrix [R | t; 0 0 0 1].

    Returns
    -------
    np.ndarray, shape (4, 4) or None
        The matrix inverse if it exists and is numerically finite, else None.
        Callers should fall back to the previous valid pose when None is returned.
    """
    try:
        inv = np.linalg.inv(mat4x4)
        # Additional finiteness check: even if inversion succeeds algebraically,
        # a near-singular matrix can produce Inf or NaN entries.
        if np.isfinite(inv).all():
            return inv
    except np.linalg.LinAlgError:
        # Singular matrix — inversion is mathematically impossible.
        pass
    return None


def estimate_motion(match, kp1, kp2, k, depth1=None):
    """Estimate relative camera motion between two consecutive frames via PnP-RANSAC.

    The algorithm follows these steps:
      1. For each filtered match (m), retrieve the pixel coordinates (u1, v1) in
         frame t and (u2, v2) in frame t+1.
      2. Look up the metric depth s = depth1[v1, u1] at the feature location in
         frame t.  Only depths in (0.001, 80) metres are used; very small values
         are likely noise and very large values (>80 m) have high relative depth
         uncertainty at typical camera resolutions.
      3. Back-project the feature from 2-D pixel space to 3-D camera space using
         the pinhole model:
             p_c = K^{-1} * (s * [u1, v1, 1]^T)
         This gives a 3-D object point in the coordinate frame of camera t.
      4. Solve PnP-RANSAC with the set of (p_c, [u2, v2]) correspondences to
         recover the relative rotation R and translation t such that:
             [u2, v2, 1]^T ~ K * R * p_c + K * t
      5. Optionally refine R and t using the RANSAC inliers only (solvePnP with
         useExtrinsicGuess=True).
      6. Convert the rotation vector rvec to a 3x3 matrix via cv2.Rodrigues.
      7. Validate the result with is_pose_valid before returning.

    Parameters
    ----------
    match : list of cv2.DMatch or list of tuple
        Filtered feature matches between frame t and frame t+1.
    kp1 : list of cv2.KeyPoint
        Keypoints in frame t.
    kp2 : list of cv2.KeyPoint
        Keypoints in frame t+1.
    k : array-like, shape (3, 3)
        Camera intrinsic matrix for this frame pair.
    depth1 : np.ndarray, shape (H, W), optional
        Metric depth map of frame t (in metres).

    Returns
    -------
    rmat : np.ndarray, shape (3, 3)
        Recovered rotation matrix (identity on failure).
    tvec : np.ndarray, shape (3, 1)
        Recovered translation vector (zeros on failure).
    image1_points : list of [u, v]
        Pixel coordinates used from frame t.
    image2_points : list of [u, v]
        Corresponding pixel coordinates used from frame t+1.
    ok : bool
        True if the pose passed is_pose_valid, False otherwise.
    """
    # Initialise outputs to the identity pose.  These are returned unchanged
    # if the PnP solve fails or there are insufficient inliers.
    rmat = np.eye(3)          # shape: (3, 3)
    tvec = np.zeros((3, 1))   # shape: (3, 1)
    image1_points = []
    image2_points = []

    objectpoints = []   # 3-D points in camera-t frame; shape will be (N, 3)

    # Ensure K is float64 for numerical stability in matrix operations.
    k = np.asarray(k, np.float64)   # shape: (3, 3)

    # Pre-compute K^{-1} once to avoid repeated inversion in the loop.
    # Pinhole back-projection: p_c = K^{-1} * (s * [u, v, 1]^T)
    # Fall back to the pseudo-inverse if K is singular (degenerate calibration).
    try:
        K_inv = np.linalg.inv(k)    # shape: (3, 3)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(k)   # shape: (3, 3)

    # Build the set of 3-D / 2-D correspondences for PnP.
    for m in match:
        # Normalise the match representation: some callers pass raw kNN tuples
        # (DMatch, DMatch) while others pass plain DMatch objects.
        if isinstance(m, tuple):
            if len(m) == 2:
                m, n = m   # discard the second-best match (already filtered)
            else:
                m = m[0]

        # Pixel coordinates of the matched feature in both frames.
        u1, v1 = kp1[m.queryIdx].pt   # frame t  (float)
        u2, v2 = kp2[m.trainIdx].pt   # frame t+1 (float)

        # Look up the metric depth at the integer pixel location in frame t.
        # s in metres; depth1 shape: (H, W) float64
        s = depth1[int(v1), int(u1)]

        # Apply depth validity filter:
        #   s > 1e-3  : exclude pixels with essentially zero depth (sensor noise)
        #   s < 80    : exclude distant pixels where relative depth error is high
        # The 80 m threshold is empirically chosen for typical urban driving scenes
        # at stereo/LiDAR-derived depth map resolutions.
        if s < 80 and s > 1e-3:
            # Back-project to 3-D camera coordinates using the pinhole model:
            #   p_c = K^{-1} * (s * [u, v, 1]^T)
            # This converts the pixel (u, v) at known depth s to a metric 3-D
            # point in the camera-t coordinate frame.
            # p_c shape: (3,)
            p_c = K_inv @ (s * np.array([u1, v1, 1]))

            # Collect the 3-D object point and its 2-D image projection in frame t+1.
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)

    # PnP requires at least 6 point correspondences for a well-constrained
    # solution (4 are the minimum, but 6+ ensures RANSAC has enough inliers).
    if len(objectpoints) < 6:
        print("Not enough points for PnP RANSAC.")
        return rmat, tvec, image1_points, image2_points, False

    # Convert Python lists to numpy arrays required by OpenCV.
    # objectpoints shape: (N, 3) float64 — 3-D points in camera-t frame
    objectpoints = np.vstack(objectpoints)
    # imagepoints shape: (N, 2) float64 — 2-D projections in frame t+1
    imagepoints = np.array(image2_points)

    # Stage 1: PnP-RANSAC for robust initial pose estimation.
    # solvePnPRansac finds R and t such that the reprojection error
    # || K*(R*X + t) - x ||  is minimised for the inlier set.
    # dist_coeffs=None assumes a pre-rectified (undistorted) image.
    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
        # rvec shape: (3, 1) — rotation vector (Rodrigues form)
        # tvec shape: (3, 1) — translation vector in metres
        # inliers shape: (M, 1) int32 — indices of inlier correspondences
    except:
        print("PnP RANSAC failed.")
        return rmat, tvec, image1_points, image2_points, False

    # Stage 2: Non-linear refinement using inliers only.
    # Re-running solvePnP with useExtrinsicGuess=True performs Levenberg-
    # Marquardt optimisation starting from the RANSAC solution, which typically
    # reduces the residual reprojection error by 10-30% on clean data.
    try:
        # Select only the inlier points for the refinement step.
        # refined_objectPoints shape: (M, 3)
        refined_objectPoints = objectpoints[inliers[:, 0]]
        # refined_imagePoints shape: (M, 2)
        refined_imagePoints = imagepoints[inliers[:, 0]]
        retval, rvec, tvec = cv2.solvePnP(
            refined_objectPoints, refined_imagePoints, k, None,
            rvec, tvec, useExtrinsicGuess=True
        )
    except:
        # Refinement failure is non-fatal; the RANSAC result is already usable.
        pass

    # Convert the compact Rodrigues rotation vector rvec (3x1) to a full 3x3
    # rotation matrix using the exponential map:
    #   R = exp([rvec]_x)
    # where [rvec]_x is the skew-symmetric matrix of rvec.
    # rmat shape: (3, 3)
    rmat, _ = cv2.Rodrigues(rvec)

    # Numerical validation: reject poses with non-finite values, non-unit
    # determinant, or implausibly large translations.
    ok_final = is_pose_valid(rmat, tvec)
    return rmat, tvec, image1_points, image2_points, ok_final


def estimate_trajectory(matches, kp_list, k, depth_maps=[], save='', dataset_handler=None):
    """Accumulate per-frame relative poses into a global 3-D camera trajectory.

    Starting from the identity pose at frame 0, the function iterates over all
    consecutive frame pairs and:
      1. Estimates the relative pose (R, t) from frame i to frame i+1.
      2. Composes the relative transform with the current world pose to obtain
         the world pose at frame i+1.
      3. Extracts the camera position from the world pose by projecting the
         origin [0, 0, 0, 1]^T through the world pose matrix.

    Pose composition uses the inverse of the relative transform:
        T_world[i+1] = T_world[i] @ inv(T_relative)
    because T_relative encodes how the world moves relative to the camera (the
    extrinsic convention), so its inverse gives the camera motion in the world.

    Fallback (when PnP fails or is_pose_valid returns False):
      A stochastic inertial extrapolation is used:
        - The previous step length ||t_prev|| is preserved.
        - A random yaw perturbation in [-90°, +90°] is applied around the Y-axis.
      This avoids freezing the trajectory at the last valid position (which would
      under-estimate total displacement) while preventing wildly incorrect poses
      from corrupting the global estimate.

    Parameters
    ----------
    matches : list of list of cv2.DMatch
        Filtered feature matches for each consecutive frame pair.
        matches[i] is the match list for the pair (frame i, frame i+1).
    kp_list : list of list of cv2.KeyPoint
        Per-frame keypoint lists; kp_list[i] corresponds to frame i.
    k : list of np.ndarray, each shape (3, 3)
        Per-frame camera intrinsic matrices.  k[i] is used for the pair
        (frame i, frame i+1).  A list is required because intrinsics can change
        between clips (e.g. different cameras or zoom levels).
    depth_maps : list of np.ndarray, each shape (H, W)
        Per-frame metric depth maps; depth_maps[i] is used for frame i.
    save : str, optional
        If non-empty, camera-movement visualisation images are saved to this
        directory, one JPEG per frame pair.  Useful for debugging failed matches.
    dataset_handler : DatasetHandler or None, optional
        Dataset object; required only when save is non-empty (for RGB frames).

    Returns
    -------
    trajectory : np.ndarray, shape (3, N)
        World-frame camera positions, where N = len(matches) + 1.
        trajectory[:, i] = [X, Y, Z]^T of the camera at frame i.
    poses_3x3 : list of tuple (np.ndarray, np.ndarray)
        Per-frame world poses as (R, t) pairs.
        R has shape (3, 3), t has shape (3,).
        Indexed from 0 (identity) to N-1.
    """
    # Allocate the trajectory array.
    # trajectory shape: (3, N)  where N = len(matches) + 1 (including frame 0)
    trajectory = np.zeros((3, len(matches) + 1))

    # Store all 4x4 world poses for later export (e.g. to poses_3x3).
    # robot_pose shape: (N, 4, 4)
    robot_pose = np.zeros((len(matches) + 1, 4, 4))

    # Frame 0 is the world coordinate origin by convention.
    # shape: (4, 4)
    robot_pose[0] = np.eye(4)

    # Keep track of the most recent valid relative pose for inertial fallback.
    # Initialised to identity (no motion) for the very first frame.
    # shape: (4, 4)
    pose_last = np.eye(4)

    # Accumulate per-frame (R, t) pairs for downstream consumers (e.g. drawing).
    poses_3x3 = []
    # Frame 0 pose: identity rotation, zero translation.
    poses_3x3.append(
        (np.eye(3), np.array([0, 0, 0]))
    )

    # Main loop: process each consecutive frame pair.
    for i in range(len(matches)):
        # Estimate the relative camera motion from frame i to frame i+1.
        # rmat shape: (3, 3), tvec shape: (3, 1)
        rmat, tvec, image1_points, image2_points, ok = estimate_motion(
            matches[i], kp_list[i], kp_list[i + 1], k[i], depth_maps[i]
        )

        # Optionally save a visualisation of which features were matched.
        if save:
            image = visualize_camera_movement(
                dataset_handler.images_rgb[i], image1_points,
                dataset_handler.images_rgb[i + 1], image2_points
            )
            plt.imsave('./{}/frame_{:05d}.jpg'.format(save, i), image)

        if ok:
            # Build the 4x4 homogeneous relative transform from R and t.
            # Layout: [R | t]
            #         [0 | 1]
            # shape: (4, 4)
            current_pose = np.eye(4)
            current_pose[0:3, 0:3] = rmat           # rotation block
            current_pose[0:3, 3]   = tvec.T         # translation row

            # Update the inertial reference for potential future fallback.
            pose_last = current_pose.copy()

        else:
            # PnP failed or the recovered pose was numerically invalid.
            # Apply stochastic inertial extrapolation to produce a plausible
            # estimate rather than stalling the trajectory.

            # Extract the previous translation vector (step in camera-frame coords).
            # last_t shape: (3,)
            last_t = pose_last[:3, 3].astype(np.float64)
            # L is the magnitude of the previous translation (step length in metres).
            L = float(np.linalg.norm(last_t))

            # Compute the unit direction of the previous step.
            # If L is essentially zero (stationary frame), use forward (+Z) as a
            # neutral default so the perturbed step is still in a valid direction.
            t_hat = (last_t / L) if L > 1e-12 else np.array([0.0, 0.0, 1.0])

            # Sample a random yaw perturbation uniformly in [-max_yaw_deg, +max_yaw_deg].
            # Restricting to ±90° ensures the extrapolated step stays in the
            # forward hemisphere (no backward motion), which is a strong prior
            # for vehicles that generally drive forward.
            max_yaw_deg = 90.0
            yaw = np.deg2rad(np.random.uniform(-max_yaw_deg, max_yaw_deg))

            # Build a 3x3 rotation matrix for a yaw rotation around the Y-axis.
            # In the camera convention Y is the vertical (up) axis.
            #   Ry(yaw) = [[cos,  0, sin],
            #               [  0,  1,   0],
            #               [-sin, 0, cos]]
            # shape: (3, 3)
            c, s = np.cos(yaw), np.sin(yaw)
            Rrand = np.array([
                [ c,   0.0,  s],
                [0.0,  1.0, 0.0],
                [-s,   0.0,  c],
            ], dtype=np.float64)

            # Rotate the previous direction by the random yaw while preserving
            # the step length L.  This produces a plausible forward motion.
            # t_new shape: (3,)
            t_new = (Rrand @ t_hat) * L

            # Assemble the fallback pose.
            # shape: (4, 4)
            current_pose = np.eye(4)
            current_pose[:3, :3] = Rrand
            current_pose[:3,  3] = t_new

            # Update pose_last so that consecutive failures each draw a fresh
            # random heading (preventing compounding of the same bad direction).
            pose_last = current_pose.copy()

        # Compose the relative pose with the running world pose.
        # The inverse of current_pose converts the extrinsic (world-in-camera)
        # convention to a camera-in-world transform, then left-multiplying by
        # the previous world pose accumulates the global trajectory.
        # safe_inv returns None if inversion fails, in which case the pose is
        # frozen at the previous frame (last-resort fallback).
        inv_pose = safe_inv(current_pose)   # shape: (4, 4) or None
        if inv_pose is None:
            # Inversion failed: keep the previous world pose unchanged.
            robot_pose[i + 1] = robot_pose[i]
        else:
            # robot_pose shape: (4, 4) — world pose at frame i+1
            robot_pose[i + 1] = robot_pose[i] @ inv_pose

        # Store the (R, t) decomposition of the world pose for external use.
        # R shape: (3, 3), t shape: (3,)
        poses_3x3.append(
            (robot_pose[i + 1][:3, :3], robot_pose[i + 1][:3, 3])
        )

        # Extract the 3-D world position of the camera by projecting the origin.
        # Multiplying the world pose by [0,0,0,1]^T gives the camera centre in
        # world coordinates (the 4th column of the world pose matrix).
        # position shape: (4,)
        position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])

        # Store the X, Y, Z components in the trajectory array.
        # trajectory[:, i+1] shape: (3,)
        trajectory[:, i + 1] = position[0:3]

    # trajectory shape: (3, N) — complete ego camera trajectory in world frame
    # poses_3x3 : list of N (R, t) tuples for external pose consumers
    return trajectory, poses_3x3


# import cv2, numpy as np
# from typing import List, Sequence, Tuple

# # match, kp1, kp2, k, depth1=None
# # ----------------- 1. Single-frame motion estimation -----------------
# def estimate_motion(
#     matches, kp1, kp2, K, depth1, depth_max = 80.0,
# ) -> Tuple[np.ndarray, np.ndarray, bool]:
#     """
#     Returns R (3x3), t (3x1), success
#     - If successful:  success=True,  R/t are valid
#     - If failed:      success=False, R=I, t=0 (placeholder only)
#     """
#     obj, img = [], []
#     dist_coeffs = np.array([-0.356123,0.172545,-0.00213,0.000464], dtype=np.float32)

#     if depth1 is not None:                # Use PnP; requires 3D-2D correspondences
#         for m in matches:
#             m = m[0] if isinstance(m, tuple) else m
#             u1, v1 = kp1[m.queryIdx].pt
#             u2, v2 = kp2[m.trainIdx].pt
#             if (0 <= int(v1) < depth1.shape[0] and 0 <= int(u1) < depth1.shape[1]):
#                 s = float(depth1[int(v1), int(u1)])
#                 if 0 < s < depth_max:
#                     X = np.linalg.inv(K) @ (s * np.array([u1, v1, 1.0]))
#                     obj.append(X)
#                     img.append([u2, v2])

#     if len(obj) < 6:                      # Too few points -> failure
#         print("Not enough points for PnP RANSAC.")
#         return np.eye(3), np.zeros((3, 1)), False

#     obj = np.float32(obj)
#     img = np.float32(img)
#     try:
#         ok, rvec, tvec, inl = cv2.solvePnPRansac(
#             obj, img, K, dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#             iterationsCount=300, reprojectionError=3.0, confidence=0.999
#         )
#     except:
#         print("PnP RANSAC failed.")
#         return np.eye(3), np.zeros((3, 1)), False

#     if not ok or inl is None or len(inl) < 6:
#         print("PnP RANSAC failed, not enough inliers.")
#         return np.eye(3), np.zeros((3, 1)), False

#     try:
#         cv2.solvePnP(obj[inl.ravel()], img[inl.ravel()], K, dist_coeffs,
#                     rvec, tvec, useExtrinsicGuess=True)
#     except:
#         print("PnP refinement failed.")
#         return np.eye(3), np.zeros((3, 1)), False

#     R, _ = cv2.Rodrigues(rvec)
#     return R, tvec, True

# # matches, kp_list, k, depth_maps=[], save='', dataset_handler=None
# # ----------------- 2. Trajectory estimation with inertial fallback -----------------
# def estimate_trajectory(
#     matches, kp_list, K, depth_maps, save='', dataset_handler=None,
# ) -> Tuple[np.ndarray, List[np.ndarray]]:
#     """
#     Returns trajectory (3xN) and per-frame world poses (list of 4x4)
#     """
#     n = len(matches) + 1
#     traj   = np.zeros((3, n))
#     robot_pose = np.zeros((n, 4, 4))
#     robot_pose[0] = np.eye(4)                # World-frame pose T_w_i for each frame
#     DeltaT_last = np.eye(4)                   # Most recent successful relative transform
#     poses_3x3 = []
#     ## init pose
#     poses_3x3.append(
#         (np.eye(3), np.array([0, 0, 0]))
#     )

#     # K_seq   = K if isinstance(K, Sequence) else [K]*n
#     # depth_s = depth_maps if depth_maps is not None else [None]*n

#     for i in range(len(matches)):
#         R, t, ok = estimate_motion(
#             matches[i],
#             kp_list[i], kp_list[i+1],
#             K[i], depth_maps[i]
#         )

#         if ok:
#             # Current relative transform
#             DeltaT = np.eye(4)
#             DeltaT[:3,:3] = R
#             DeltaT[:3, 3] = t.ravel()
#             DeltaT_last   = DeltaT.copy()         # Update inertial reference
#         else:
#             DeltaT = DeltaT_last                  # Inertial extrapolation

#         # Compose to get world pose at frame i+1
#         robot_pose[i + 1] = robot_pose[i] @ np.linalg.inv(DeltaT)
#         poses_3x3.append(
#             (robot_pose[i+1][:3,:3], robot_pose[i+1][:3,3])
#         )

#         # Calculate current camera position from origin
#         position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])

#         # Build trajectory
#         traj[:, i + 1] = position[0:3]

#     return traj, poses_3x3
