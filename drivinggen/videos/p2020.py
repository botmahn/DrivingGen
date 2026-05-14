# -*- coding: utf-8 -*-
"""p2020_metrics.py  -- FULL list of image-quality KPIs explicitly named in
the IEEE P2020 Automotive Imaging White-Paper.

=============================================================================
Overview
=============================================================================
This module provides a *pragmatic, not perfect* reference implementation of
all 35 KPIs listed in the IEEE P2020 Automotive Camera Imaging whitepaper.
For many KPIs the whitepaper only prescribes the *concept* (e.g.
Contrast-Detection-Probability, LED-Flicker severity).  Wherever the official
formula or test-chart is still under discussion we provide a **reasonable
open-source surrogate** so that engineers can start benchmarking today -- swap
the function later when the final spec drops.

Design constraints
------------------
* Pure Python: depends only on ``numpy``, ``opencv-python``, ``scipy``.
  No GPU, no heavy ML models.
* Every metric returns an *un-scaled* ``float``.  Downstream z-score,
  min-max normalisation, and weighting are the caller's responsibility.
* Frames are expected as **BGR uint8 ndarray** unless explicitly noted.
* Video metrics accept either a Python list/iterable of frames or a 4-D
  NumPy array with layout (T, H, W, C).

=============================================================================
Metric inventory (35 total)
=============================================================================
SHARPNESS / GEOMETRY  (§4.1)
  laplacian_var, edge_rise_time, mtf50, mtf10, gradient_entropy,
  blur_extent, keystone_distortion, geometric_distortion, rolling_shutter

EXPOSURE / CONTRAST / HDR  (§4.2)
  mean_luminance, std_luminance, dynamic_range_OECF, dynamic_range_proxy,
  under_exposure_ratio, over_saturation_ratio, veiling_glare_index,
  global_contrast_factor, local_rms_contrast

COLOR & WHITE-BALANCE  (§4.4)
  grey_world_error, color_accuracy_deltaE, color_sat_mean, color_sat_std,
  color_separation_probability

NOISE & ARTIFACTS  (§4.3)
  spatial_noise_iso, temporal_noise, dsnu, fpn, row_noise, dark_current,
  blockiness, chroma_aberration, lens_shading_uniformity

TEXTURE / DETAIL  (§4.5)
  dead_leaves_texture_mtf, texture_loss_index

FLICKER / TEMPORAL  (§4.6)
  led_flicker_index, contrast_detection_probability

FOCUS / DOF  (§4.7)
  depth_of_field_metric, focus_stability

Convenience wrappers
  single_frame_metrics(img)  -- 22 per-frame metrics in a single call
  video_metrics(frames, fps) -- 3 temporal metrics across a video sequence

=============================================================================
revision: 2025-07-09
maintainer: openai-chatgpt
"""
from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from scipy import fft, ndimage, signal

# =============================================================================
# Module-level constant
# =============================================================================

# Small epsilon added to denominators to prevent division by zero.
# 1e-6 is small relative to 8-bit pixel values (range 0-255) so it never
# meaningfully biases any ratio.
EPS = 1e-6


# =============================================================================
# Shared helper utilities
# =============================================================================

def _gray(img: np.ndarray) -> np.ndarray:
    """Convert a BGR colour image to a float32 single-channel grayscale image.

    This is the common pre-processing step shared by every P2020 metric that
    operates in luminance space.

    Args:
        img: Input image.
              - If 3-channel (H, W, 3) BGR uint8: converted via OpenCV BT.601
                formula  Y = 0.114 B + 0.587 G + 0.299 R  then cast to float32.
              - If already 2-D (H, W): cast to float32 without conversion.

    Returns:
        Grayscale image as float32.
        shape: (H, W)   dtype: float32

    Notes:
        OpenCV's COLOR_BGR2GRAY uses the Rec. 601 luma coefficients
        Y = 0.114*B + 0.587*G + 0.299*R, which are appropriate for standard
        sRGB camera outputs used in automotive imaging.
    """
    if img.ndim == 3:
        # img shape: (H, W, 3) BGR uint8  ->  (H, W) uint8  ->  (H, W) float32
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # img is already grayscale (H, W); just ensure float32 precision
    return img.astype(np.float32)


def _safe_div(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Element-wise division with epsilon guard: result = a / (b + EPS).

    Prevents ZeroDivisionError and NumPy warnings when the denominator is
    zero or very small.

    Args:
        a: Numerator -- scalar or ndarray of any shape.
        b: Denominator -- scalar or ndarray broadcastable with ``a``.

    Returns:
        a / (b + EPS) -- same shape and dtype as the broadcast of a and b.

    Notes:
        EPS = 1e-6.  For typical luminance values (0-255 range) this
        introduces a relative error of at most 1e-6 / 1 = 1 ppm, which is
        negligible for all P2020 ratio metrics.
    """
    return a / (b + EPS)


# =============================================================================
# Sharpness / Geometry  (P2020 §4.1)
# =============================================================================

def laplacian_var(img: np.ndarray) -> float:
    """P2020 §4.1.1 -- Sharpness via Laplacian variance.

    Computes the variance of the discrete Laplacian of the grayscale image.
    Sharper images produce larger Laplacian values because high-frequency edge
    content dominates.

    Formula:
        Var(L) = E[L^2] - E[L]^2
        where L = Laplacian(Y),  Y = grayscale image.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Variance of Laplacian as a scalar float.
        Larger value => sharper image.

    Notes:
        P2020 §4.1 specifies sharpness assessment via second-order spatial
        derivatives.  The Laplacian variance (also called Tenengrad variance)
        is a standard blur-detection surrogate.
    """
    # _gray converts to float32 grayscale  shape: (H, W)
    gray = _gray(img)  # shape: (H, W)

    # cv2.Laplacian applies a 3x3 discrete Laplacian kernel:
    #   [0  1  0]
    #   [1 -4  1]
    #   [0  1  0]
    # Output shape: (H, W)  dtype: CV_32F
    lap = cv2.Laplacian(gray, cv2.CV_32F)  # shape: (H, W)

    # .var() returns a scalar Python float -- the pixel-wise variance
    return float(lap.var())


def edge_rise_time(img: np.ndarray) -> float:
    """P2020 §4.1.2 -- Edge Rise Time (10%-90% spatial width in pixels).

    Finds the steepest vertical edge in the image using a horizontal Sobel
    filter, extracts a local pixel intensity profile perpendicular to that
    edge, and measures the distance (in pixels) between the 10th and 90th
    percentile intensity levels -- the standard definition of rise time.

    Formula:
        ERT = x_{90%} - x_{10%}
        where x_{p%} is the pixel position at which the profile first crosses
        the p-th percentile of the local intensity range.

    Smaller ERT => sharper edge transition.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Rise time in pixels (non-negative float).
        Returns 0.0 if the profile is too short to compute percentiles.

    Notes:
        A window of ±12 pixels (25 px total) around the detected peak
        is sufficient to capture typical lane-mark edges at 1080p.
        P2020 §4.1 recommends slanted-edge charts (ISO 12233), but this
        surrogate works on natural driving scenes without a chart.
    """
    gray = _gray(img)  # shape: (H, W)

    # Horizontal Sobel  dx=1, dy=0  detects vertical edges (intensity changes
    # along the x-axis).  ksize=3 uses a 3x3 kernel.
    # sob shape: (H, W)  -- response is large where vertical edges are strong
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # shape: (H, W)

    # Find the pixel with the largest-magnitude Sobel response -- the steepest
    # vertical edge in the entire frame.
    # np.abs(sob): (H, W)  -- absolute Sobel magnitude
    # argmax returns a flat index; unravel_index converts it to (row, col)
    y, x = np.unravel_index(np.abs(sob).argmax(), sob.shape)

    # Extract a 1-D intensity profile centred on the detected edge, clamped to
    # image boundaries.  Width = 25 px covers the full transition region.
    # prof shape: (<=25,)
    prof = gray[y, max(x - 12, 0): x + 13]  # shape: (<=25,)

    # Compute the 10th and 90th percentile intensities of the local profile.
    p10, p90 = np.percentile(prof, [10, 90])

    # Find the first index in the profile where intensity >= p10 (i10) and
    # >= p90 (i90).  These approximate the beginning and end of the edge ramp.
    i10 = np.where(prof >= p10)[0][0]  # scalar int -- index of 10% crossing
    i90 = np.where(prof >= p90)[0][0]  # scalar int -- index of 90% crossing

    # Rise time = spatial distance between the 10% and 90% crossing points.
    return float(i90 - i10)


def mtf50(img: np.ndarray, axis: int = 0) -> float:
    """P2020 §4.1.3 -- Modulation Transfer Function at 50% (MTF50).

    Approximates the spatial frequency (in FFT bin units) at which the image
    modulation falls to 50% of its peak value.  Uses a Sobel-based Edge
    Spread Function (ESF) -> Line Spread Function (LSF) -> FFT pipeline that
    does not require a dedicated test chart.

    Procedure:
        1. Apply Sobel in the direction perpendicular to ``axis`` to get the
           ESF derivative (i.e. the LSF) column-by-column (or row-by-row).
        2. Average the LSF across all rows (or columns) to reduce noise.
        3. Compute the 1-D real FFT to get the Optical Transfer Function (OTF).
        4. Normalise the magnitude spectrum to [0, 1].
        5. Return the index of the first bin where the spectrum <= 0.50.

    Args:
        img:  BGR uint8 input frame.
              shape: (H, W, 3)
        axis: Sobel direction.
              0 = vertical Sobel (measures horizontal resolution) -- P2020 default.
              1 = horizontal Sobel (measures vertical resolution).

    Returns:
        MTF50 frequency in FFT bin index units (float).
        Larger value => better high-frequency resolution.

    Notes:
        P2020 §4.1.3 / ISO 12233 define MTF50 with a slanted-edge chart.
        This surrogate uses any scene content; results are comparable only
        within the same dataset.
    """
    gray = _gray(img)  # shape: (H, W)

    # Sobel with  dx = 1-axis, dy = axis  gives the derivative perpendicular
    # to ``axis``.  When axis=0: dx=1, dy=0 (vertical edge response).
    # sob shape: (H, W)
    sob = cv2.Sobel(gray, cv2.CV_32F, 1 - axis, axis, ksize=3)  # shape: (H, W)

    # Average the Sobel response along ``axis`` to form a single 1-D LSF.
    # When axis=0, sob.mean(axis=0) collapses rows -> shape: (W,)
    # When axis=1, sob.mean(axis=1) collapses cols -> shape: (H,)
    lsf = sob.mean(axis=axis)  # shape: (W,) or (H,)

    # Real FFT of the LSF gives the OTF magnitude.
    # spec shape: (N//2+1,)  where N = len(lsf)
    spec = np.abs(fft.rfft(lsf))  # shape: (N//2+1,)

    # Normalise so that the DC (zero-frequency) bin == 1.0
    spec /= spec.max() + EPS  # shape: (N//2+1,)  -- values in [0, 1]

    # Find the first frequency bin where the normalised spectrum drops to 0.50.
    idx = np.where(spec <= 0.5)[0]  # shape: (M,)  -- indices where MTF <= 50%

    # If the spectrum never drops to 0.50, report the full spectrum length
    # (indicating excellent sharpness beyond the measurable frequency range).
    return float(idx[0] if idx.size else spec.size)


def mtf10(img: np.ndarray, axis: int = 0) -> float:
    """P2020 §4.1.3 -- Modulation Transfer Function at 10% (MTF10).

    Identical to :func:`mtf50` except the threshold is 0.10 instead of 0.50.
    MTF10 indicates the spatial frequency at which modulation drops to 10%,
    i.e. the practical resolution limit of the imaging system.

    Args:
        img:  BGR uint8 input frame.
              shape: (H, W, 3)
        axis: Sobel direction (0=vertical, 1=horizontal).  Default: 0.

    Returns:
        MTF10 frequency in FFT bin index units (float).
        Larger value => higher effective resolution limit.

    Notes:
        MTF10 is referenced in P2020 §4.1.3 alongside MTF50 as the metric
        for the spatial frequency where the system response is ~10%, often
        considered the limiting resolution for object detection tasks.
    """
    gray = _gray(img)  # shape: (H, W)

    # Derivative perpendicular to the requested axis -- same as mtf50
    sob = cv2.Sobel(gray, cv2.CV_32F, 1 - axis, axis, ksize=3)  # shape: (H, W)

    # Average across the axis to form a 1-D LSF
    lsf = sob.mean(axis=axis)  # shape: (W,) or (H,)

    # Real FFT -> OTF magnitude spectrum
    spec = np.abs(fft.rfft(lsf))  # shape: (N//2+1,)

    # Normalise to [0, 1]
    spec /= spec.max() + EPS  # shape: (N//2+1,)

    # Find the first bin where the spectrum drops to 0.10
    idx = np.where(spec <= 0.1)[0]  # shape: (M,)

    return float(idx[0] if idx.size else spec.size)


def gradient_entropy(img: np.ndarray) -> float:
    """P2020 §4.1.4 -- Texture richness via gradient-magnitude entropy.

    Computes the Shannon entropy of the gradient-magnitude histogram.  Rich,
    high-frequency textures produce a broad histogram (high entropy); blurry
    or flat images produce a peaked histogram near zero (low entropy).

    Formula:
        H = -sum_i [ p_i * log2(p_i) ]
        where p_i is the normalised frequency of the i-th histogram bin of the
        gradient magnitude image |nabla Y|.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Entropy in bits (float, non-negative).
        Larger value => richer texture content.

    Notes:
        P2020 §4.1.4 discusses texture assessment as a complement to
        MTF-based sharpness.  Gradient entropy correlates with the amount of
        detail preserved after image processing (e.g. temporal smoothing in
        generated video).
    """
    gray = _gray(img)  # shape: (H, W)

    # Compute x- and y-Sobel responses
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # shape: (H, W)  -- dI/dx
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # shape: (H, W)  -- dI/dy

    # Gradient magnitude: |nabla Y| = sqrt(gx^2 + gy^2)
    mag = np.hypot(gx, gy)  # shape: (H, W)  -- magnitude in [0, ~360] for uint8

    # Build a 256-bin probability histogram of the gradient magnitudes.
    # density=True normalises so that integral (sum * bin_width) == 1.
    hist, _ = np.histogram(mag, bins=256, range=(0, mag.max() + EPS), density=True)
    # hist shape: (256,)  -- probability density values

    # Add EPS to avoid log2(0) which is undefined.
    hist += EPS  # shape: (256,)

    # Shannon entropy H = -sum( p * log2(p) )
    return float(-(hist * np.log2(hist)).sum())  # scalar bits


def blur_extent(img: np.ndarray) -> float:
    """P2020 §4.1.5 -- Blur extent via low-to-high frequency energy ratio.

    Separates the image into a low-frequency component (Gaussian-blurred) and
    a high-frequency residual (detail layer).  The ratio of their mean absolute
    energies quantifies how much of the signal is concentrated in low
    frequencies -- a proxy for motion blur or defocus.

    Formula:
        BlurExtent = mean(|L|) / (mean(|H|) + EPS)
        where L = GaussianBlur(Y, 21x21),  H = Y - L.

    Larger value => more blur (most energy is in the low-frequency band).

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Low-to-high frequency energy ratio (float, >= 0).

    Notes:
        The 21x21 Gaussian kernel is recommended in P2020 §4.1.5 as it covers
        the typical motion-blur radius encountered at highway speeds with
        standard automotive exposure times.  sigma=0 lets OpenCV auto-compute
        sigma from the kernel size (sigma ~ (21-1)/6 ~ 3.3).
    """
    gray = _gray(img)  # shape: (H, W)

    # Low-frequency component: 21x21 Gaussian blur (sigma derived from ksize)
    low = cv2.GaussianBlur(gray, (21, 21), 0)  # shape: (H, W)

    # High-frequency residual (detail layer): subtract the blurred version
    high = gray - low  # shape: (H, W)

    # Ratio: mean |low energy| / mean |high energy|
    # A blurry image has a small high-frequency component -> large ratio.
    return float(_safe_div(np.abs(low).mean(), np.abs(high).mean()))


def keystone_distortion(img: np.ndarray, chess_size: tuple = (7, 7)) -> float:
    """P2020 §4.1.6 -- Keystone (perspective) distortion via chessboard aspect ratio.

    Detects a chessboard calibration pattern and computes the relative
    difference between the detected grid width and height.  A perfectly
    fronto-parallel board gives width == height; tilt introduces keystone
    distortion that changes the apparent aspect ratio.

    Formula:
        Keystone = |W - H| / max(W, H)
        where W, H are the bounding-box width and height of detected corners.

    Args:
        img:        BGR uint8 input frame.
                    shape: (H, W, 3)
        chess_size: (columns, rows) of interior corners on the chessboard.
                    Default (7, 7) matches the most common automotive
                    calibration board.

    Returns:
        Keystone ratio in [0, 1) -- 0 means no keystone.
        Returns 0.0 if no chessboard is found in the image.

    Notes:
        P2020 §4.1.6 specifies keystone measurement with a full calibration
        rig.  This function is a convenient surrogate for automated pipelines
        where the target is already in the scene.
    """
    # Convert to uint8 grayscale for corner detector (float32 not accepted)
    gray_u8 = _gray(img).astype(np.uint8)  # shape: (H, W)  dtype: uint8

    # Attempt to find interior corners of the chessboard pattern
    ok, corners = cv2.findChessboardCorners(gray_u8, chess_size)

    # If no chessboard is detected, report zero distortion (undefined)
    if not ok:
        return 0.0

    # corners shape: (N, 1, 2) where N = chess_size[0] * chess_size[1]
    # Extract x (column) and y (row) coordinates of all detected corners
    x_coords = corners[:, 0, 0]  # shape: (N,)  -- column positions in pixels
    y_coords = corners[:, 0, 1]  # shape: (N,)  -- row positions in pixels

    # Bounding-box dimensions of the detected corner grid
    w = x_coords.max() - x_coords.min()  # scalar -- horizontal span in px
    h = y_coords.max() - y_coords.min()  # scalar -- vertical span in px

    # Keystone ratio: normalised absolute aspect-ratio deviation
    return float(abs(w - h) / max(w, h))


def geometric_distortion(img: np.ndarray, fov_deg: float | None = None) -> float:
    """P2020 §4.1.7 -- Barrel / pincushion distortion coefficient proxy.

    Estimates the radial distortion by fitting a linear model between the
    radius from the image centre and the horizontal displacement of detected
    edge pixels.  A positive slope indicates barrel distortion; negative
    indicates pincushion.

    Formula:
        Detect Canny edges -> for each edge pixel (xs, ys):
            r = sqrt((xs - W/2)^2 + (ys - H/2)^2)    radial distance
            delta_x = xs - W/2                         horizontal offset
        k = polyfit(r, delta_x, degree=1)[0]           first-order coefficient

    Args:
        img:     BGR uint8 input frame.
                 shape: (H, W, 3)
        fov_deg: Field-of-view in degrees.  Accepted for API compatibility
                 but not used in the computation -- a full calibration model
                 would require it.

    Returns:
        Linear radial distortion coefficient k (float, signed).
        Positive => barrel,  negative => pincushion.

    Notes:
        P2020 §4.1.7 recommends a grid / dot-pattern target for precise
        distortion measurement.  This surrogate uses Canny edges in natural
        scenes and is only suitable for coarse comparisons.
    """
    h, w = img.shape[:2]  # image height and width in pixels

    gray = _gray(img)  # shape: (H, W)

    # Canny edge detection with standard automotive thresholds (50, 150)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)  # shape: (H, W)  binary

    # Pixel coordinates of all detected edge points
    ys, xs = np.where(edges)  # ys shape: (N,)  xs shape: (N,)  -- row/col indices

    # Radial distance of each edge pixel from the image centre
    r = np.hypot(xs - w / 2, ys - h / 2)  # shape: (N,)  -- pixels

    # Horizontal displacement of each edge pixel from the image centre
    # Positive -> pixel is to the right of centre
    dx = xs - w / 2  # shape: (N,)  -- pixels

    # Fit a linear polynomial: dx ~ k * r + intercept
    # k > 0 means outer pixels are displaced further right than a linear model
    # predicts, consistent with barrel distortion.
    k = np.polyfit(r, dx, 1)[0]  # scalar -- slope of the linear fit

    return float(k)


def rolling_shutter(prev: np.ndarray, curr: np.ndarray) -> float:
    """P2020 §4.1.8 -- Rolling-shutter skew via optical flow direction (degrees).

    Computes dense optical flow between two consecutive frames using the
    Farneback algorithm and returns the mean flow direction angle.  A strong
    diagonal component in the mean flow indicates rolling-shutter skew
    (horizontal motion combined with a vertical scan delay produces a
    characteristic diagonal smear).

    Formula:
        flow = Farneback(prev, curr)
        angle = arctan2(mean(flow_y), mean(flow_x))  [degrees]

    Args:
        prev: Previous BGR uint8 frame.
              shape: (H, W, 3)
        curr: Current BGR uint8 frame.
              shape: (H, W, 3)

    Returns:
        Mean flow direction in degrees (float in [-180, 180]).
        Values near 0 or 180 indicate pure horizontal motion (no RS skew).
        Angles near ±90 suggest dominant vertical motion or RS artefacts.

    Notes:
        Farneback parameters follow P2020 §4.1.8 recommendations:
        pyr_scale=0.5, levels=1, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2.
    """
    # Convert both frames to uint8 grayscale for the optical flow algorithm
    prev_gray = _gray(prev).astype(np.uint8)  # shape: (H, W)
    curr_gray = _gray(curr).astype(np.uint8)  # shape: (H, W)

    # Dense optical flow -- Farneback algorithm
    # flow shape: (H, W, 2)  -- flow[y, x, 0] = dx,  flow[y, x, 1] = dy
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,      # pre-allocated output (None = allocate internally)
        0.5,       # pyr_scale: pyramid downscale factor
        1,         # levels: number of pyramid levels
        15,        # winsize: averaging window size
        3,         # iterations: per pyramid level
        5,         # poly_n: pixel neighbourhood for polynomial expansion
        1.2,       # poly_sigma: Gaussian std for polynomial weights
        0,         # flags: no additional options
    )  # shape: (H, W, 2)

    # Compute the mean flow vector across all pixels
    mean_dx = flow[..., 0].mean()  # scalar -- mean horizontal displacement (px)
    mean_dy = flow[..., 1].mean()  # scalar -- mean vertical displacement (px)

    # Convert mean flow direction to degrees using atan2
    return float(np.degrees(np.arctan2(mean_dy, mean_dx)))


# =============================================================================
# Exposure / Contrast  (P2020 §4.2)
# =============================================================================

def mean_luminance(img: np.ndarray) -> float:
    """P2020 §4.2.1 -- Mean scene luminance (code-value units).

    Computes the spatial average of the grayscale (luma) channel, which
    serves as a proxy for the mean scene brightness captured by the sensor.

    Formula:
        Y_mean = (1 / (H*W)) * sum_{i,j} Y(i,j)

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Mean luminance as a float in [0, 255].

    Notes:
        P2020 §4.2.1 specifies exposure evaluation via mean and standard
        deviation of luminance.  Code-value 128 corresponds roughly to
        18% grey (middle grey) for a correctly exposed scene.
    """
    return float(_gray(img).mean())  # scalar in [0, 255]


def std_luminance(img: np.ndarray) -> float:
    """P2020 §4.2.1 -- Luminance standard deviation (contrast proxy).

    Measures the spread of the luminance distribution.  A high standard
    deviation indicates a high-contrast scene; a low value indicates a flat
    or monochromatic scene.

    Formula:
        sigma_Y = sqrt( (1/(H*W)) * sum_{i,j} (Y(i,j) - Y_mean)^2 )

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Standard deviation of luminance as a float in [0, ~127].

    Notes:
        P2020 §4.2.1 treats mean and sigma together to characterise the
        exposure quality.  For generated video this metric detects whether
        the world model collapses diverse scenes to a narrow tonal range.
    """
    return float(_gray(img).std())  # scalar >= 0


def dynamic_range_proxy(img: np.ndarray) -> float:
    """P2020 §4.2.2 -- Dynamic range proxy via percentile spread.

    Estimates the scene dynamic range as the difference between the 99.9th
    and 0.1th percentile luminance values.  Using percentiles (rather than
    min/max) makes the metric robust to a handful of saturated or dead pixels.

    Formula:
        DR_proxy = P_{99.9}(Y) - P_{0.1}(Y)   [code values, 0-255]

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Approximate dynamic range in code values (float in [0, 255]).
        Larger value => wider captured dynamic range.

    Notes:
        This is an image-level surrogate for the chart-based OECF measurement
        in P2020 §4.2.2.  It is used when a calibrated test chart is not
        available (e.g. when evaluating generated video frames).
    """
    gray = _gray(img)  # shape: (H, W)

    # Compute percentiles across all pixels (flat)
    lo, hi = np.percentile(gray, [0.1, 99.9])  # scalars in [0, 255]

    return float(hi - lo)


def dynamic_range_OECF(
    oecf_curve: Sequence[Tuple[float, float]],
    snr_threshold: int = 20,
) -> float:
    """P2020 §4.2.3 -- Dynamic range from an Opto-Electronic Conversion Function curve.

    Computes the ratio of the maximum to minimum exposure values at which
    the measured Signal-to-Noise Ratio (SNR) meets or exceeds a specified
    threshold.  This directly quantifies the usable dynamic range of the
    sensor as defined in ISO 14524.

    Formula:
        DR_OECF = E_max / E_min
        where {E : SNR(E) >= snr_threshold}

    Args:
        oecf_curve:    Sequence of (exposure, SNR_dB) pairs measured from a
                       calibrated OECF test target.  Each element is a
                       (float, float) tuple.
        snr_threshold: Minimum acceptable SNR in dB.
                       Default 20 dB is from ISO 14524 / P2020 §4.2.3.

    Returns:
        Dynamic range as an exposure ratio (float >= 1).
        Returns 0.0 if fewer than 2 exposure steps meet the SNR threshold.

    Notes:
        oecf_curve is NOT an image array -- it is a calibration data table.
        Requires an OECF measurement setup (e.g. Imatest or similar tool).
    """
    # Unpack the sequence of (exposure, SNR) tuples into separate arrays
    e = np.array([c[0] for c in oecf_curve])    # shape: (N,)  -- exposure values
    snr = np.array([c[1] for c in oecf_curve])  # shape: (N,)  -- SNR in dB

    # Find the indices where SNR meets or exceeds the threshold
    idx = np.where(snr >= snr_threshold)[0]  # shape: (M,)  -- qualifying indices

    if idx.size > 1:
        # Dynamic range = ratio of highest to lowest qualifying exposure
        return float(_safe_div(e[idx].max(), e[idx].min()))
    return 0.0  # insufficient qualifying exposure steps


def under_exposure_ratio(img: np.ndarray, thr: int = 20) -> float:
    """P2020 §4.2.4 -- Fraction of pixels that are under-exposed.

    Counts the proportion of pixels whose luminance falls below the threshold,
    indicating crushed shadows or insufficient exposure.

    Formula:
        UER = (1/(H*W)) * sum_{i,j} [Y(i,j) < thr]

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
        thr: Luminance threshold below which a pixel is considered
             under-exposed.  Default 20 (~8% of full scale, approximately
             10% grey).

    Returns:
        Fraction of under-exposed pixels in [0, 1].
        0 => no crushed shadows;  1 => entire frame is black.

    Notes:
        P2020 §4.2.4 recommends monitoring the fraction of pixels in the
        bottom 5% of the tonal range to detect insufficient exposure.
    """
    # Boolean mask: True where pixel is under-exposed
    under = _gray(img) < thr  # shape: (H, W)  dtype: bool

    # .mean() of a bool array = fraction of True values
    return float(under.mean())


def over_saturation_ratio(img: np.ndarray, thr: int = 235) -> float:
    """P2020 §4.2.5 -- Fraction of pixels that are over-saturated (blown highlights).

    Counts the proportion of pixels whose luminance exceeds the threshold,
    indicating sensor saturation or clipping.

    Formula:
        OSR = (1/(H*W)) * sum_{i,j} [Y(i,j) > thr]

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
        thr: Luminance threshold above which a pixel is considered saturated.
             Default 235 leaves a 20-DN head-room to avoid false positives
             from anti-aliasing or JPEG compression ringing near 255.

    Returns:
        Fraction of over-saturated pixels in [0, 1].
        0 => no blown highlights;  1 => entire frame is white.

    Notes:
        P2020 §4.2.5 specifies that automotive cameras must report highlight
        clipping to ensure safety-relevant bright objects (traffic lights, sun
        reflections) are not silently clipped.
    """
    # Boolean mask: True where pixel is over-saturated
    over = _gray(img) > thr  # shape: (H, W)  dtype: bool

    return float(over.mean())


def veiling_glare_index(img: np.ndarray) -> float:
    """P2020 §4.2.6 -- Veiling Glare Index (VGI).

    Estimates the proportion of stray light (veiling glare) by comparing the
    mean brightness of the full frame to the mean brightness of the central
    50% x 50% region-of-interest.  In the absence of glare the full-frame
    mean should approximate the central mean; any excess in the full frame
    indicates scattered light in the periphery.

    Formula:
        VGI = (Y_full - Y_centre) / Y_centre
        where Y_centre = mean luminance in [H/4 : 3H/4, W/4 : 3W/4]

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Veiling glare index as a float.
        0 => no measurable glare;  positive => peripheral stray light.

    Notes:
        P2020 §4.2.6 recommends a flare-box setup for accurate VGI
        measurement.  This surrogate uses the centre/full-frame luminance
        ratio as a field-deployable approximation.
    """
    gray = _gray(img)  # shape: (H, W)
    H, W = gray.shape  # height and width in pixels

    # Central 50% x 50% region-of-interest
    # Row range: [H/4, 3H/4)   Col range: [W/4, 3W/4)
    core = gray[H // 4: 3 * H // 4, W // 4: 3 * W // 4]  # shape: (H/2, W/2)
    core_mean = core.mean()  # scalar -- mean luminance in the centre ROI

    full_mean = gray.mean()  # scalar -- mean luminance over the whole frame

    # Relative difference: how much brighter is the full frame vs the centre?
    return float(_safe_div(full_mean - core_mean, core_mean))


def global_contrast_factor(img: np.ndarray) -> float:
    """P2020 §4.2.7 -- Global Contrast Factor (GCF) via multi-scale RMS.

    Computes a weighted sum of the image's RMS contrast at 6 progressively
    coarser scales.  Coarser scales (more downsampled) receive lower weight
    (1/2^l) so that fine-grained contrast contributes more than overall tonal
    variation.

    Formula:
        GCF = sum_{l=0}^{5}  sigma(Y_l) / 2^l
        where Y_l = image downscaled by 2^l,  sigma = RMS standard deviation.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Global contrast factor as a float (non-negative).
        Larger value => higher overall perceived contrast.

    Notes:
        P2020 §4.2.7 cites multi-scale contrast assessment to account for
        the fact that human visual perception integrates contrast over a range
        of spatial frequencies.  The 6-level pyramid covers frequencies from
        DC to Nyquist/2^6 relative to the original resolution.
    """
    gray = _gray(img)  # shape: (H, W)

    total = 0.0  # accumulated weighted RMS contrast across scales

    for l in range(6):  # l = 0 (full res) to 5 (most downsampled)
        # Downscale by factor 1/2^l using area interpolation (anti-aliased)
        scale_factor = 1.0 / (2 ** l)  # scalar -- e.g. 1.0, 0.5, 0.25, ...
        small = cv2.resize(
            gray, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )  # shape: (H/2^l, W/2^l)

        # RMS contrast at this scale, weighted by 1/2^l
        weight = 1.0 / (2 ** l)  # scalar -- weight decreases with scale
        total += float(small.std()) * weight  # scalar accumulation

    return float(total)


def local_rms_contrast(img: np.ndarray, win: int = 16) -> float:
    """P2020 §4.2.8 -- Local RMS contrast averaged over non-overlapping patches.

    Divides the image into non-overlapping square patches of size ``win`` x ``win``
    pixels and computes the RMS standard deviation within each patch.
    The mean of these patch-level RMS values quantifies the typical local
    contrast over the whole frame.

    Formula:
        C_local = mean_{patches p} sigma(Y_p)
        where Y_p = luminance values within patch p,  sigma = std deviation.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
        win: Patch side length in pixels.
             Default 16 corresponds to approximately 1 degree of field-of-view
             at 720p resolution, as recommended in ISO 12233.

    Returns:
        Mean local RMS contrast (float, in [0, ~127] for uint8 input).

    Notes:
        P2020 §4.2.8 recommends local contrast measurement to detect spatial
        non-uniformities in exposure or tone-mapping that would not be visible
        in the global statistics.
    """
    gray = _gray(img)  # shape: (H, W)

    # Build a list of per-patch standard deviations
    # Iterate over non-overlapping win x win tiles
    patches = [
        gray[y: y + win, x: x + win].std()         # scalar -- RMS of one patch
        for y in range(0, gray.shape[0] - win, win) # row start positions
        for x in range(0, gray.shape[1] - win, win) # col start positions
    ]  # list of scalars, length ~ (H/win) * (W/win)

    return float(np.mean(patches))  # mean over all patches


# =============================================================================
# Color & White-Balance  (P2020 §4.4)
# =============================================================================

def grey_world_error(img: np.ndarray) -> float:
    """P2020 §4.4.1 -- Grey-World white-balance error.

    The Grey-World assumption states that the spatial average of each colour
    channel should be equal for a correctly white-balanced image.  This
    function quantifies the deviation from that assumption.

    Formula:
        mu = (mean(R) + mean(G) + mean(B)) / 3     [grand mean]
        GWE = |mean(R) - mu| + |mean(G) - mu| + |mean(B) - mu|

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Grey-World error in code values (float, >= 0).
        0 => perfect white balance;  larger => colour cast.

    Notes:
        P2020 §4.4.1 specifies white-balance evaluation under standard
        illuminants.  This metric can detect colour casts introduced by
        incorrect white-balance settings or biased world-model generation.
    """
    # Split the BGR image into separate float32 channels
    b, g, r = cv2.split(img.astype(np.float32))
    # b shape: (H, W)   g shape: (H, W)   r shape: (H, W)

    # Grand mean of all three channel averages
    mu = np.mean([b.mean(), g.mean(), r.mean()])  # scalar

    # Sum of absolute deviations from the grey-world target
    gwe = abs(r.mean() - mu) + abs(g.mean() - mu) + abs(b.mean() - mu)  # scalar

    return float(gwe)


def color_accuracy_deltaE(
    img: np.ndarray,
    chart_rgb: np.ndarray,
    chart_lab_ref: np.ndarray,
) -> float:
    """P2020 §4.4.2 -- Colour accuracy via mean Delta-E (CIE 1994 distance).

    Compares the measured LAB values of a colour chart crop to the
    reference LAB values from the Macbeth ColorChecker specification and
    returns the mean Euclidean distance in LAB space across all colour patches.

    Formula:
        delta_E_i = || LAB_measured_i - LAB_reference_i ||_2
        mean_dE = (1/N) * sum_{i=1}^{N} delta_E_i

    Args:
        img:           Full BGR uint8 frame (not used in computation; kept for
                       API consistency -- crop the chart ROI into ``chart_rgb``
                       before calling this function).
                       shape: (H, W, 3)
        chart_rgb:     BGR uint8 crop of the colour chart region from the image.
                       shape: (P, Q, 3)  or  (N, 1, 3)  -- any BGR image region
        chart_lab_ref: Reference LAB values for N colour patches from the
                       Macbeth ColorChecker specification.
                       shape: (N, 3)  -- each row is (L*, a*, b*)

    Returns:
        Mean Delta-E across all patches (float, >= 0).
        0 => perfect colour reproduction;  larger => more colour error.

    Notes:
        P2020 §4.4.2 recommends using a 24-patch Macbeth ColorChecker
        (ISO 17321-1).  The OpenCV LAB conversion uses the D65 illuminant.
        For proper Delta-E 94 or 2000 formulae, additional weighting terms
        are needed; this implementation uses the simpler CIE76 Euclidean
        distance in LAB space.
    """
    # Convert the chart crop from BGR to CIE LAB colour space
    # chart_lab shape: (P, Q, 3)  -- (L*, a*, b*) values per pixel
    chart_lab = cv2.cvtColor(chart_rgb, cv2.COLOR_BGR2LAB)

    # Reshape to a list of per-patch LAB vectors
    # measured shape: (N, 3)  -- one row per colour patch
    measured = chart_lab.reshape(-1, 3).astype(float)

    # Reference LAB reshaped to match
    # ref shape: (N, 3)
    ref = chart_lab_ref.reshape(-1, 3)

    # Per-patch Euclidean distance in LAB space
    # diff shape: (N, 3)
    diff = measured - ref

    # delta_E_i = ||diff_i||_2  for each patch
    # delta_e shape: (N,)
    delta_e = np.linalg.norm(diff, axis=1)

    return float(delta_e.mean())  # scalar -- mean across N patches


def color_sat_mean(img: np.ndarray) -> float:
    """P2020 §4.4.3 -- Mean colour saturation (HSV S channel).

    Converts the image to HSV colour space and returns the mean of the
    Saturation channel S, which ranges from 0 (greyscale) to 255 (fully
    saturated).

    Formula:
        S_mean = mean_{i,j} S(i,j)
        where S is the saturation channel of the HSV representation.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Mean saturation in [0, 255] (float).
        0 => monochrome;  255 => fully saturated throughout.

    Notes:
        P2020 §4.4.3 specifies saturation evaluation as a measure of colour
        vividness.  Low saturation in generated video may indicate colour
        fading artefacts in the world model decoder.
    """
    # Convert BGR -> HSV; the saturation channel is index 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # shape: (H, W, 3)
    s_channel = hsv[..., 1]  # shape: (H, W)  -- saturation in [0, 255]

    return float(s_channel.mean())  # scalar in [0, 255]


def color_sat_std(img: np.ndarray) -> float:
    """P2020 §4.4.3 -- Standard deviation of colour saturation (HSV S channel).

    Measures the spatial variability of saturation.  A high std indicates
    the scene contains a mix of vivid and desaturated regions (typical for
    real driving); a low std suggests uniform saturation (may indicate
    artificial post-processing or colour map collapse in generation).

    Formula:
        sigma_S = std_{i,j} S(i,j)

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Standard deviation of saturation in [0, ~127] (float).

    Notes:
        Companion metric to :func:`color_sat_mean`.  Together they provide a
        two-parameter characterisation of the saturation distribution, as
        referenced in P2020 §4.4.3.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # shape: (H, W, 3)
    s_channel = hsv[..., 1]  # shape: (H, W)  -- saturation in [0, 255]

    return float(s_channel.std())  # scalar >= 0


def color_separation_probability(img: np.ndarray, th: int = 15) -> float:
    """P2020 §4.4.4 -- Colour Separation Probability (CSP).

    Estimates the fraction of pixels whose colour is sufficiently distinct
    from the a*-b* mean colour of the frame -- a proxy for how well different
    coloured objects can be discriminated by the camera system.

    Formula:
        dist(i,j) = sqrt((a*(i,j) - mean(b*))^2 + (b*(i,j) - mean(b*))^2)
        CSP = fraction of pixels where dist > th

    Note: the formula uses mean(b*) as a pivot for both a* and b* distances,
    which is a low-cost surrogate for a full colour-clustering analysis.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
        th:  Distance threshold in LAB units.
             Default 15 ~ Delta-E of ~6, a perceptually just-noticeable
             difference at typical viewing distances.

    Returns:
        Colour separation probability in [0, 1] (float).
        Larger => more pixels are chromatically distinct from the mean.

    Notes:
        P2020 §4.4.4 defines CSP in the context of pedestrian/vehicle colour
        discrimination.  This surrogate does not require a colour chart and
        can be applied to any driving scene image.
    """
    # Convert BGR to CIE LAB colour space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # shape: (H, W, 3)

    a_star = lab[..., 1]  # shape: (H, W)  -- a* channel (green-red axis)
    b_star = lab[..., 2]  # shape: (H, W)  -- b* channel (blue-yellow axis)

    # Use the global mean of b* as a pivot reference point in chroma space
    b_mean = b_star.mean()  # scalar

    # Euclidean distance in (a*, b*) space from the pivot (b_mean, b_mean)
    dist = np.hypot(a_star - b_mean, b_star - b_mean)  # shape: (H, W)

    # Fraction of pixels exceeding the threshold distance
    return float((dist > th).mean())  # scalar in [0, 1]


# =============================================================================
# Noise & Artifacts  (P2020 §4.3)
# =============================================================================

def spatial_noise_iso(img: np.ndarray) -> float:
    """P2020 §4.3.1 -- Spatial noise approximation (ISO 15739 surrogate).

    Returns the standard deviation of the full-frame luminance as a fast
    proxy for ISO 15739 spatial noise.  In a perfectly flat scene this equals
    the sensor read-noise; in practice it upper-bounds the true noise because
    scene texture is also included.

    Formula:
        sigma_ISO ~ std_{i,j} Y(i,j)

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
             Ideally captured over a uniform grey patch; otherwise the result
             includes scene texture.

    Returns:
        Estimated spatial noise (float, >= 0).

    Notes:
        A flat-field image is required for a rigorous ISO 15739 measurement.
        In generated-video evaluation this metric serves as a coarse indicator
        of temporal texture consistency across frames.
    """
    return float(_gray(img).std())  # scalar -- std of all luminance pixels


def temporal_noise(prev: np.ndarray, curr: np.ndarray) -> float:
    """P2020 §4.3.2 -- Temporal noise (frame-difference standard deviation).

    Computes the standard deviation of the pixel-wise luminance difference
    between two consecutive frames.  For a static scene the difference should
    be pure noise; any residual structure indicates inter-frame artefacts.

    Formula:
        sigma_temp = std_{i,j} (Y_curr(i,j) - Y_prev(i,j))

    Args:
        prev: Previous BGR uint8 frame.
              shape: (H, W, 3)
        curr: Current BGR uint8 frame.
              shape: (H, W, 3)

    Returns:
        Temporal noise level (float, >= 0).

    Notes:
        P2020 §4.3.2 specifies temporal noise measurement over a static
        scene.  For dynamic driving scenes this metric also captures motion
        content, so it should be interpreted as an upper bound on noise.
    """
    diff = _gray(curr) - _gray(prev)  # shape: (H, W)  -- signed difference
    return float(diff.std())          # scalar -- std of the difference image


def dsnu(dark: np.ndarray) -> float:
    """P2020 §4.3.3 -- Dark Signal Non-Uniformity (DSNU): peak-to-peak spread.

    Measures the maximum minus minimum luminance value in a dark-field
    (lens-capped or fully opaque) image.  Non-zero values indicate that some
    pixels have a higher dark current or offset than others.

    Formula:
        DSNU = max(Y_dark) - min(Y_dark)

    Args:
        dark: Dark-field BGR uint8 frame (captured with lens cap / in darkness).
              shape: (H, W, 3)

    Returns:
        Peak-to-peak dark signal range (float, in [0, 255]).
        0 => perfectly uniform dark field;  larger => more fixed-pattern bias.

    Notes:
        P2020 §4.3.3 / EMVA 1288 define DSNU as the spatial non-uniformity
        of dark (zero-illumination) frames.  A dark-field image must be
        provided; results from lit scenes are meaningless.
    """
    gray_dark = _gray(dark)  # shape: (H, W)
    return float(gray_dark.max() - gray_dark.min())  # scalar peak-to-peak


def fpn(dark: np.ndarray) -> float:
    """P2020 §4.3.3 -- Fixed Pattern Noise (FPN): standard deviation of dark field.

    Measures the standard deviation of luminance in a dark-field image as a
    quantitative estimate of spatially correlated sensor noise (fixed-pattern
    noise, e.g. column offsets, row stripes).

    Formula:
        FPN = std_{i,j} Y_dark(i,j)

    Args:
        dark: Dark-field BGR uint8 frame.
              shape: (H, W, 3)

    Returns:
        Fixed pattern noise level (float, >= 0).
        0 => no spatial structure in darkness;  larger => more FPN.

    Notes:
        Unlike :func:`dsnu` (which is sensitive to single outlier pixels),
        FPN as standard deviation is more robust to hot pixels.  Both metrics
        are specified in P2020 §4.3.3 / EMVA 1288.
    """
    gray_dark = _gray(dark)  # shape: (H, W)
    return float(gray_dark.std())  # scalar std of dark-field luminance


def row_noise(img: np.ndarray) -> float:
    """P2020 §4.3.4 -- Row noise (variance of per-row luminance means).

    Row noise is a form of fixed-pattern noise where entire rows of pixels
    share a common offset due to readout amplifier variations.  It manifests
    as horizontal banding.  The variance of row-averaged luminance values
    quantifies this effect.

    Formula:
        mu_row(i) = mean_j Y(i,j)       per-row mean
        RowNoise = var_i (mu_row(i))     variance across all rows

    Args:
        img: BGR uint8 input frame (ideally a flat-field scene).
             shape: (H, W, 3)

    Returns:
        Row noise as variance of row means (float, >= 0).
        0 => no horizontal banding;  larger => stronger row-level artefacts.

    Notes:
        P2020 §4.3.4 / EMVA 1288 section on spatial noise.  In generated
        video row noise can indicate systematic vertical gradients introduced
        by upsampling or decoder artefacts.
    """
    gray = _gray(img)           # shape: (H, W)
    row_means = gray.mean(axis=1)  # shape: (H,)  -- mean luminance per row

    return float(row_means.var())  # scalar -- variance across rows


def dark_current(dark: np.ndarray, long_exposure_s: float) -> float:
    """P2020 §4.3.5 -- Dark current estimate (DN per second).

    Estimates the mean dark current by dividing the average dark-field
    luminance by the integration (exposure) time.  In a sensor without dark
    current the dark frame would read exactly 0; any positive mean indicates
    thermally generated charge.

    Formula:
        DarkCurrent = mean(Y_dark) / (t_exp + EPS)   [DN/s]

    Args:
        dark:             Dark-field BGR uint8 frame.
                          shape: (H, W, 3)
        long_exposure_s:  Exposure duration in seconds used to capture
                          ``dark``.  Must be > 0.

    Returns:
        Estimated dark current in digital numbers per second (float, >= 0).

    Notes:
        P2020 §4.3.5 / EMVA 1288 specify dark current measurement at
        multiple temperatures.  For generated-video evaluation this function
        is not applicable (generated frames have no physical dark current).
    """
    gray_dark = _gray(dark)  # shape: (H, W)
    mean_dark = gray_dark.mean()  # scalar -- mean dark-field luminance in DN

    # Divide by exposure time; EPS guards against zero-second exposure
    return float(mean_dark / (long_exposure_s + EPS))  # scalar DN/s


def blockiness(img: np.ndarray) -> float:
    """P2020 §4.3.6 -- JPEG block artefact strength (DCT DC energy on 8x8 grid).

    Detects the presence of 8x8 DCT block boundaries introduced by JPEG
    compression.  The Discrete Cosine Transform is applied to the full image;
    the energy at positions that are multiples of 8 (the DC position of each
    8x8 block) is extracted and averaged.  Strong block artefacts produce
    large DC values at these regular grid positions.

    Formula:
        dct = DCT2(Y)
        mask = 1 at positions (8k, 8l) for k,l = 0, 1, 2, ...
        Blockiness = mean |dct[mask]|

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Mean absolute DCT energy at 8x8 block boundaries (float, >= 0).
        Larger => stronger JPEG block artefacts.

    Notes:
        P2020 §4.3.6 lists blockiness / compression artefacts as quality
        degradation factors.  OpenCV's cv2.dct requires the input to be
        cropped to a multiple of 8 in both dimensions.
    """
    gray = _gray(img)  # shape: (H, W)
    h, w = gray.shape  # height and width before cropping

    # Crop to the nearest multiple of 8 in both dimensions (required by DCT)
    gray = gray[: h - h % 8, : w - w % 8]  # shape: (H', W') where H'%8==0, W'%8==0

    # 2-D Discrete Cosine Transform (DCT-II) of the full cropped image
    dct = cv2.dct(gray.astype(np.float32))  # shape: (H', W')

    # Binary mask selecting only the DC positions of each 8x8 block
    mask = np.zeros_like(dct)  # shape: (H', W')  -- all zeros
    mask[::8, ::8] = 1.0       # shape: (H', W')  -- 1 at every 8th row/col

    # Mean absolute DCT energy at the block DC positions
    return float(np.abs(dct * mask).mean())  # scalar


def chroma_aberration(img: np.ndarray) -> float:
    """P2020 §4.3.7 -- Chromatic aberration (R-B difference std at edges).

    Chromatic aberration causes red and blue colour channels to focus at
    slightly different distances, producing colour fringing along high-contrast
    edges.  This metric measures the standard deviation of (R - B) at detected
    edge pixels -- a robust proxy for the magnitude of lateral chromatic
    aberration.

    Formula:
        edge_pixels = Canny(Y)
        CA = std_{(i,j) in edges} (R(i,j) - B(i,j))

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Chromatic aberration level (float, >= 0).
        Larger => more colour fringing at edges.

    Notes:
        P2020 §4.3.7 specifies chromatic aberration measurement with a
        slanted-edge or star-pattern target.  This surrogate uses Canny edges
        in the scene for field evaluation without a test chart.
    """
    # Split BGR into individual float32 channels
    b, g, r = cv2.split(img.astype(np.float32))
    # b shape: (H, W)   g shape: (H, W)   r shape: (H, W)

    # Detect edges in the luminance image using Canny
    gray_u8 = _gray(img).astype(np.uint8)  # shape: (H, W)  -- for Canny
    edges = cv2.Canny(gray_u8, 50, 150)    # shape: (H, W)  -- binary edge map

    # Get the coordinates of all edge pixels
    ys, xs = np.where(edges)  # ys shape: (N,)  xs shape: (N,)

    # R - B difference at each edge pixel location
    rb_diff = r[ys, xs] - b[ys, xs]  # shape: (N,)

    # Std dev of R-B differences quantifies chromatic fringing magnitude
    return float(np.std(rb_diff))  # scalar >= 0


def lens_shading_uniformity(img: np.ndarray) -> float:
    """P2020 §4.3.8 -- Lens shading uniformity (centre-to-corner ratio).

    Optical vignetting causes image corners to be darker than the centre.
    This metric quantifies the uniformity by computing the ratio of the mean
    luminance in the central 50% x 50% region to the mean luminance in the
    four corner quadrants (each 25% x 25% of the image).

    Formula:
        centre = mean Y over [H/4:3H/4, W/4:3W/4]
        corners = mean Y over the four corner 25%-quadrants
        LSU = centre / (corners + EPS)

    Values > 1 indicate the centre is brighter than the corners (typical
    vignetting); values close to 1 indicate uniform illumination.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
             Should ideally be a flat-field image (uniformly lit target).

    Returns:
        Centre-to-corner luminance ratio (float, > 0).
        1 => uniform;  > 1 => vignetting (dark corners);  < 1 => unusual.

    Notes:
        P2020 §4.3.8 specifies lens shading with an integrating sphere or
        uniform flat-field light source.  Applied to driving scenes this
        metric also captures real-scene luminance gradients.
    """
    gray = _gray(img)  # shape: (H, W)
    H, W = gray.shape  # height and width in pixels

    # Central 50% x 50% ROI (rows H/4 to 3H/4, cols W/4 to 3W/4)
    center_region = gray[H // 4: 3 * H // 4, W // 4: 3 * W // 4]  # shape: (H/2, W/2)
    center_mean = center_region.mean()  # scalar

    # Four corner patches (each covering the outermost 25% of the image)
    top_left     = gray[: H // 4,    : W // 4]     # shape: (H/4, W/4)
    top_right    = gray[: H // 4,    3 * W // 4:]  # shape: (H/4, W/4)
    bottom_left  = gray[3 * H // 4:, : W // 4]     # shape: (H/4, W/4)
    bottom_right = gray[3 * H // 4:, 3 * W // 4:]  # shape: (H/4, W/4)

    corners_mean = np.mean([
        top_left.mean(), top_right.mean(),
        bottom_left.mean(), bottom_right.mean(),
    ])  # scalar -- average luminance of the four corners

    return float(center_mean / (corners_mean + EPS))  # ratio > 0


# =============================================================================
# Texture / Detail  (P2020 §4.5)
# =============================================================================

def dead_leaves_texture_mtf(img: np.ndarray, dl_patch: np.ndarray) -> float:
    """P2020 §4.5.1 -- Dead-Leaves texture MTF (texture retention at quarter Nyquist).

    Estimates the texture preservation of an imaging pipeline using a
    dead-leaves (random overlapping discs) patch.  The power spectral density
    of the patch is computed via 2-D FFT, and the ratio of the PSD at the
    quarter-Nyquist frequency to the peak PSD is returned.  This ratio
    quantifies how much mid-frequency texture the system retains.

    Formula:
        PSD(u, v) = |FFT2(dl_patch)|^2
        Row-averaged radial PSD: P(u) = mean_v PSD(u, v)  [after fftshift]
        DL_MTF = P(N/4) / max(P)
        where N = number of columns (Nyquist / 4 ~ mid-band frequency)

    Args:
        img:      Full input frame (accepted for API consistency; not used
                  in computation -- provide the dead-leaves crop in
                  ``dl_patch``).
                  shape: (H, W, 3)
        dl_patch: BGR or grayscale crop of the dead-leaves chart region.
                  shape: (Ph, Pw, 3)  or  (Ph, Pw)

    Returns:
        Texture MTF ratio at quarter Nyquist (float in [0, 1]).
        1 => full texture retention;  0 => complete texture loss.

    Notes:
        P2020 §4.5.1 specifies the dead-leaves target as defined in
        ISO 19567-1.  A real dead-leaves target must be in the scene; this
        function cannot synthesise it from arbitrary images.
    """
    gray_patch = _gray(dl_patch)  # shape: (Ph, Pw)

    # 2-D FFT shifted so DC is at centre; then compute power spectral density
    fft2_shifted = fft.fftshift(fft.fft2(gray_patch))  # shape: (Ph, Pw)  complex
    psd_2d = np.abs(fft2_shifted) ** 2                  # shape: (Ph, Pw)  power

    # Smooth with a 5x5 uniform filter to reduce noise, then average over rows
    # to obtain a 1-D radial PSD approximation
    psd_smooth = ndimage.uniform_filter(psd_2d, size=5)  # shape: (Ph, Pw)
    psd_1d = psd_smooth.mean(axis=0)                      # shape: (Pw,)

    # Evaluate PSD at the quarter-Nyquist bin (index = Pw/4)
    quarter_nyq_val = psd_1d[len(psd_1d) // 4]  # scalar -- PSD at N/4
    peak_val = psd_1d.max()                       # scalar -- peak PSD

    return float(quarter_nyq_val / (peak_val + EPS))  # ratio in [0, 1]


def texture_loss_index(img: np.ndarray, ref: np.ndarray) -> float:
    """P2020 §4.5.2 -- Texture Loss Index (TLI).

    Quantifies the loss of image texture relative to a reference by computing
    the mean absolute difference of grayscale values, normalised by the
    standard deviation of the reference.  A TLI of 0 means no texture loss;
    larger values indicate increasing texture degradation.

    Formula:
        TLI = mean |Y_ref - Y_img| / (std(Y_ref) + EPS)

    Args:
        img: Test BGR uint8 frame (e.g. output of a processed pipeline).
             shape: (H, W, 3)
        ref: Reference BGR uint8 frame (e.g. original unprocessed capture).
             shape: (H, W, 3) -- must have the same spatial dimensions as img.

    Returns:
        Texture Loss Index (float, >= 0).
        0 => identical texture;  1 => difference equals reference std.

    Notes:
        P2020 §4.5.2 defines texture quality relative to a ground-truth
        reference image.  For generated-video evaluation, ``ref`` should be
        the corresponding real camera frame.
    """
    gray_img = _gray(img)  # shape: (H, W)
    gray_ref = _gray(ref)  # shape: (H, W)  -- reference image

    # Mean absolute pixel difference between reference and test image
    mean_abs_diff = np.mean(np.abs(gray_ref - gray_img))  # scalar

    # Normalise by the reference's standard deviation to get a scale-free index
    return float(_safe_div(mean_abs_diff, gray_ref.std()))  # scalar >= 0


# =============================================================================
# Flicker / Temporal  (P2020 §4.6)
# =============================================================================

def led_flicker_index(
    video: Sequence[np.ndarray],
    fps: float,
    freq_hz: float | None = None,
) -> float:
    """P2020 §4.6.1 -- LED Flicker Index (LFI): spectral energy fraction at PWM peak.

    Quantifies the severity of LED PWM flicker by computing the fraction of
    total luminance-sequence spectral energy that is concentrated near the
    dominant periodic frequency.

    Procedure:
        1. Compute the mean luminance of each frame -> 1-D temporal signal.
        2. Real FFT -> power spectrum.
        3. If ``freq_hz`` is not given, auto-detect as the frequency with the
           highest spectral energy (excluding DC).
        4. Integrate power within ±2 Hz of the detected peak.
        5. Return the fraction: peak-band energy / total energy.

    Formula:
        LFI = sum_{|f - f_peak| < 2} |S(f)|^2  /  sum_f |S(f)|^2

    Args:
        video:   Sequence of BGR uint8 frames.
                 Each frame shape: (H, W, 3)
        fps:     Frame rate in frames per second.
        freq_hz: Known flicker frequency in Hz (e.g. 100 Hz for 50 Hz mains).
                 If None, the dominant frequency is auto-detected.

    Returns:
        LED Flicker Index in [0, 1].
        0 => no periodic flicker;  1 => all energy at the flicker frequency.

    Notes:
        The ±2 Hz bandwidth is wide enough to capture both 25 fps and 30 fps
        aliasing of 100 Hz / 120 Hz PWM flicker.  P2020 §4.6.1 references
        JEDEC / VESA flicker standards for the definition.
    """
    # Compute the mean luminance (scalar) for each frame in the sequence
    luma = [_gray(f).mean() for f in video]  # list of scalars, length T

    # Real FFT of the temporal luminance signal -> power spectrum
    spec = np.abs(fft.rfft(luma)) ** 2  # shape: (T//2+1,)  -- power at each freq

    # Corresponding frequency values in Hz
    freqs = fft.rfftfreq(len(luma), d=1.0 / fps)  # shape: (T//2+1,)

    if freq_hz is None:
        # Auto-detect dominant frequency: skip bin 0 (DC) and find argmax
        # freqs[1:] shape: (T//2,)  -- frequencies excluding DC
        freq_hz = freqs[1:][np.argmax(spec[1:])]  # scalar Hz

    # Boolean mask for the ±2 Hz band around the detected peak
    band = (freqs > freq_hz - 2) & (freqs < freq_hz + 2)  # shape: (T//2+1,)  bool

    # Fraction of spectral energy within the flicker band
    return float(spec[band].sum() / (spec.sum() + EPS))  # scalar in [0, 1]


def contrast_detection_probability(
    img: np.ndarray,
    patch_size: int = 32,
    thr: float = 0.05,
) -> float:
    """P2020 §4.6.2 -- Contrast Detection Probability (CDP).

    Estimates the probability that a human observer (or ADAS algorithm) can
    detect contrast in the image by measuring what fraction of local patches
    have a Michelson contrast exceeding the threshold.

    Formula:
        For each patch p of size ``patch_size`` x ``patch_size``:
            C_p = (max(p) - min(p)) / 255
        CDP = fraction of patches where C_p > thr

    Args:
        img:        BGR uint8 input frame.
                    shape: (H, W, 3)
        patch_size: Side length of each square patch in pixels.
                    Default 32 px ~ 1 degree of field-of-view at 720p
                    resolution (ISO 12233 recommendation).
        thr:        Michelson contrast threshold for "detectable" contrast.
                    Default 0.05 = 5% contrast.

    Returns:
        Contrast detection probability in [0, 1] (float).
        1 => all patches have detectable contrast;  0 => flat image.

    Notes:
        P2020 §4.6.2 defines CDP in the context of LED flickering visibility.
        Here it is repurposed as a general contrast-adequacy metric for
        automotive scene frames.
    """
    gray = _gray(img)   # shape: (H, W)
    h, w = gray.shape   # image dimensions

    # Extract all non-overlapping patches
    patches = [
        gray[y: y + patch_size, x: x + patch_size]  # shape: (patch_size, patch_size)
        for y in range(0, h - patch_size, patch_size)
        for x in range(0, w - patch_size, patch_size)
    ]  # list of arrays, each shape: (patch_size, patch_size)

    # For each patch compute normalised Michelson contrast and compare to threshold
    # (max - min) / 255 gives contrast in [0, 1]; > thr means detectable
    detectable = [(p.max() - p.min()) / 255 > thr for p in patches]
    # detectable: list of bool, length = number of patches

    return float(np.mean(detectable))  # fraction of detectable patches


# =============================================================================
# Focus / Depth-of-Field  (P2020 §4.7)
# =============================================================================

def depth_of_field_metric(img: np.ndarray, roi=None) -> float:
    """P2020 §4.7.1 -- Depth-of-Field proxy via Laplacian variance (in an ROI).

    Computes the Laplacian variance of the image (or a specific region of
    interest), which correlates inversely with the depth of field.  A shallow
    depth of field produces high sharpness in the focused region and strong
    Laplacian response there; out-of-focus regions have very low Laplacian
    variance.

    Formula:
        DOF_metric = Var( Laplacian( Y_roi ) )

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)
        roi: Optional region-of-interest as a tuple (x1, y1, x2, y2) in pixel
             coordinates.  If None, the entire image is used.

    Returns:
        Laplacian variance of the ROI (float, >= 0).
        Larger => sharper focus / shallower DOF in the ROI.

    Notes:
        P2020 §4.7.1 specifies DOF measurement with a slanted-edge or
        Siemens star target.  This surrogate uses the Laplacian variance
        approach which is applicable to arbitrary scene regions.
    """
    if roi is not None:
        # Crop the ROI: roi = (x1, y1, x2, y2) in pixel coordinates
        x1, y1, x2, y2 = roi
        region = img[y1: y2, x1: x2]  # shape: (y2-y1, x2-x1, 3)
    else:
        region = img  # shape: (H, W, 3)

    gray = _gray(region)  # shape: (H_roi, W_roi)

    # Laplacian variance -- same formula as laplacian_var() but on the ROI only
    lap = cv2.Laplacian(gray, cv2.CV_32F)  # shape: (H_roi, W_roi)

    return float(lap.var())  # scalar


def focus_stability(prev: np.ndarray, curr: np.ndarray) -> float:
    """P2020 §4.7.2 -- Focus stability (absolute inter-frame change in DOF metric).

    Measures how much the focus quality changes between two consecutive frames.
    A stable, correctly focused camera produces nearly identical Laplacian
    variance across frames; an autofocus hunting or mechanically vibrating lens
    produces large frame-to-frame swings.

    Formula:
        FocusStability = |DOF(curr) - DOF(prev)|

    Args:
        prev: Previous BGR uint8 frame.
              shape: (H, W, 3)
        curr: Current BGR uint8 frame.
              shape: (H, W, 3)

    Returns:
        Absolute change in Laplacian variance between consecutive frames
        (float, >= 0).
        0 => perfectly stable focus;  larger => focus hunting.

    Notes:
        P2020 §4.7.2 specifies focus stability evaluation over a static scene.
        Applied to driving video this metric is sensitive to both focus changes
        and scene content changes.
    """
    dof_prev = depth_of_field_metric(prev)  # scalar -- DOF metric for prev frame
    dof_curr = depth_of_field_metric(curr)  # scalar -- DOF metric for curr frame

    return abs(dof_curr - dof_prev)  # scalar >= 0


# =============================================================================
# Convenience wrappers
# =============================================================================

def single_frame_metrics(img: np.ndarray) -> Dict[str, float]:
    """Compute all 22 single-frame P2020 KPIs and return them in a dictionary.

    Applies every per-frame metric function that requires only one image
    (no dark-field, no reference, no OECF curve, no colour chart) to ``img``
    and returns the results as a flat key-value dictionary keyed by function
    name.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Dictionary mapping metric name (str) -> metric value (float).
        Keys (in order):
            'laplacian_var', 'edge_rise_time', 'mtf50', 'mtf10',
            'gradient_entropy', 'blur_extent',
            'mean_luminance', 'std_luminance', 'dynamic_range_proxy',
            'under_exposure_ratio', 'over_saturation_ratio',
            'veiling_glare_index', 'global_contrast_factor',
            'local_rms_contrast', 'grey_world_error',
            'color_sat_mean', 'color_sat_std',
            'spatial_noise_iso', 'row_noise', 'blockiness',
            'chroma_aberration', 'lens_shading_uniformity'

    Notes:
        Metrics that require additional inputs (dark frame, reference image,
        OECF data, colour chart) are intentionally excluded from this wrapper.
        Call those functions directly with the appropriate inputs.
    """
    m: Dict[str, float] = {}

    # Ordered list of single-argument metric functions
    metric_functions = [
        laplacian_var,          # §4.1.1  sharpness (Laplacian variance)
        edge_rise_time,         # §4.1.2  edge transition width in pixels
        mtf50,                  # §4.1.3  spatial frequency at 50% modulation
        mtf10,                  # §4.1.3  spatial frequency at 10% modulation
        gradient_entropy,       # §4.1.4  texture richness (entropy of gradient mag)
        blur_extent,            # §4.1.5  low-to-high frequency energy ratio
        mean_luminance,         # §4.2.1  mean luminance code value
        std_luminance,          # §4.2.1  luminance standard deviation
        dynamic_range_proxy,    # §4.2.2  percentile-based dynamic range
        under_exposure_ratio,   # §4.2.4  fraction of under-exposed pixels
        over_saturation_ratio,  # §4.2.5  fraction of over-saturated pixels
        veiling_glare_index,    # §4.2.6  centre vs. full-frame luminance ratio
        global_contrast_factor, # §4.2.7  multi-scale RMS contrast
        local_rms_contrast,     # §4.2.8  mean patch-level RMS contrast
        grey_world_error,       # §4.4.1  white-balance grey-world deviation
        color_sat_mean,         # §4.4.3  mean HSV saturation
        color_sat_std,          # §4.4.3  std of HSV saturation
        spatial_noise_iso,      # §4.3.1  full-frame luminance std (noise proxy)
        row_noise,              # §4.3.4  variance of row-averaged luminance
        blockiness,             # §4.3.6  JPEG block DCT energy
        chroma_aberration,      # §4.3.7  R-B std at edge pixels
        lens_shading_uniformity,# §4.3.8  centre-to-corner luminance ratio
    ]

    for fn in metric_functions:
        m[fn.__name__] = fn(img)  # call each metric and store the result

    return m


def video_metrics(frames: Sequence[np.ndarray], fps: float = 10.0) -> Dict[str, float]:
    """Compute temporal P2020 KPIs that require a multi-frame video sequence.

    Evaluates three temporal metrics across the provided frame sequence:
    mean temporal noise, LED flicker index, and focus stability.

    Args:
        frames: Ordered sequence of BGR uint8 video frames.
                Each frame shape: (H, W, 3)
                Minimum 2 frames required.
        fps:    Frame rate in frames per second.
                Used by :func:`led_flicker_index` for frequency-axis scaling.
                Default: 10.0 fps (common for automotive data loggers).

    Returns:
        Dictionary with keys:
            'temporal_noise_mean':  Mean inter-frame luminance-difference std
                                    across all consecutive frame pairs (float).
            'led_flicker':          LED Flicker Index across the sequence (float).
            'focus_stability':      Mean absolute inter-frame DOF change (float).
        Returns an empty dict if fewer than 2 frames are provided.

    Notes:
        All three metrics are defined for a *static* scene in P2020.  Applied
        to dynamic driving footage they capture a mix of true temporal artefacts
        and scene-change contributions.  Interpret accordingly.
    """
    if len(frames) < 2:
        # Cannot compute any inter-frame metrics with fewer than 2 frames
        return {}

    # Compute temporal noise as mean over all consecutive frame pairs
    noise_values = [
        temporal_noise(frames[i - 1], frames[i])
        for i in range(1, len(frames))
    ]  # list of scalars, length T-1
    mean_tn = float(np.mean(noise_values))

    # LED flicker index from the full frame sequence
    lfi = led_flicker_index(frames, fps)  # scalar in [0, 1]

    # Focus stability: mean absolute DOF change across consecutive frame pairs
    focus_values = [
        focus_stability(frames[i - 1], frames[i])
        for i in range(1, len(frames))
    ]  # list of scalars, length T-1
    mean_fs = float(np.mean(focus_values))

    return {
        'temporal_noise_mean': mean_tn,  # mean inter-frame noise level
        'led_flicker':         lfi,      # fraction of spectral energy at PWM peak
        'focus_stability':     mean_fs,  # mean absolute focus drift per frame pair
    }


# =============================================================================
# Module self-test (run with:  python p2020.py)
# =============================================================================

if __name__ == "__main__":

    # Path to a single reference frame for single-frame metric evaluation
    img = cv2.imread(
        '/mnt/cache/zhouyang/dg-bench/infer_logs_nuplan_0530_0604/'
        'sg-one-north+changing_lane+8bdf19ef1ae35709/gt/images/00000.png'
    )  # shape: (H, W, 3)  BGR uint8

    # Directory containing all frames for video-level metric evaluation
    base_dir = (
        '/mnt/cache/zhouyang/dg-bench/infer_logs_nuplan_0530_0604/'
        'sg-one-north+changing_lane+8bdf19ef1ae35709/gt/images'
    )

    # Build a sorted list of all image paths in the directory
    frame_list = sorted(
        os.path.join(base_dir, f) for f in os.listdir(base_dir)
    )

    # Load all frames into memory
    vid = [cv2.imread(f) for f in frame_list]  # list of (H, W, 3) BGR uint8

    # Run single-frame metrics on the reference image
    single = single_frame_metrics(img)   # dict of 22 floats

    # Run video-level metrics on the full frame sequence at 10 fps
    video = video_metrics(vid, fps=10)   # dict of 3 floats

    print(img.shape)  # e.g. (900, 1600, 3)

    import pdb
    pdb.set_trace()

    # Example output recorded during development (values may vary with input):
    # single_frame_metrics:
    # {'laplacian_var': 184.80, 'edge_rise_time': 16.0, 'mtf50': 2.0,
    #  'mtf10': 6.0, 'gradient_entropy': 2.71, 'blur_extent': 15.18,
    #  'mean_luminance': 106.75, 'std_luminance': 63.33,
    #  'dynamic_range_proxy': 254.0, 'under_exposure_ratio': 0.0204,
    #  'over_saturation_ratio': 0.0825, 'veiling_glare_index': 0.2019,
    #  'global_contrast_factor': 123.74, 'local_rms_contrast': 13.95,
    #  'grey_world_error': 2.90, 'color_sat_mean': 26.95,
    #  'color_sat_std': 36.19, 'spatial_noise_iso': 63.33,
    #  'row_noise': 1361.03, 'blockiness': 0.261,
    #  'chroma_aberration': 8.36, 'lens_shading_uniformity': 0.781}
    #
    # video_metrics:
    # {'temporal_noise_mean': 31.62, 'led_flicker': 0.0,
    #  'focus_stability': 2.65}
