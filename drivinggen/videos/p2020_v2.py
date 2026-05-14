# -*- coding: utf-8 -*-
"""p2020_high_priority_metrics_v2.py  -- ADAS-critical KPI subset (priority >= 8 pts)
========================================================================================

Overview
--------
This module implements the 14 highest-priority metrics from the IEEE P2020
Automotive Camera Imaging whitepaper, selected specifically for their
relevance to ADAS (Advanced Driver Assistance Systems) and generated-video
quality evaluation.

Unlike ``p2020.py`` (which covers all 35 metrics), this module focuses on the
subset most likely to reveal failures in world-model video generation:

  - Dynamic range collapse (HDR scene statistics, §5)
  - Sharpness regression (MTF / contrast transfer, §6)
  - Geometric distortion (radial lens distortion proxy, §3)
  - Flare / stray light (centre vs. periphery brightness, §2)
  - LED flicker / temporal modulation (PWM alias energy, §7)
  - Texture richness and blur (gradient entropy, blur extent)
  - Chromatic aberration (R-B fringing at edges)

Design decisions
----------------
* All functions operate on **BGR uint8 ndarray** input (standard OpenCV format).
* Three lightweight dependencies only: ``numpy``, ``opencv-python``, ``scipy``.
* Every metric returns an unscaled ``float``; normalisation is the caller's job.
* Video metrics accept a Python list / iterable of frames OR a 4-D ndarray
  with layout (T, H, W, C).
* ``single_frame_metrics`` currently runs only ``mmp_alias`` (the most
  diagnostic metric for generated video); uncomment others as needed.
* ``video_metrics`` runs ``mmp_alias`` on the full sequence.

Metric inventory (14)
---------------------
DYNAMIC-RANGE / HDR  (§5)
    frame_dynamic_range_proxy, sequence_dynamic_range_proxy,
    temporal_exposure_jitter

SHARPNESS / RESOLUTION  (§6)
    mtf50, mtf10, _mtf_sobel (internal helper),
    contrast_transfer_accuracy, edge_rise_time

GEOMETRY  (§3)
    total_distortion

FLARE / STRAY-LIGHT  (§2)
    flare_attenuation

FLICKER / TEMPORAL  (§7)
    flicker_modulation_power, fmp_alias,
    modulation_mitigation_probability, mmp_alias

TEXTURE / DETAIL
    gradient_entropy, blur_extent, chroma_aberration

Convenience wrappers
    single_frame_metrics(img)        -- currently runs mmp_alias only
    video_metrics(frames, fps=30)    -- runs mmp_alias on the sequence
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from scipy import fft, stats


# =============================================================================
# Module-level constant
# =============================================================================

# Tiny epsilon for safe division; 1e-9 is well below any meaningful signal
# level in 8-bit (0-255) or float (0.0-1.0) imagery.
EPS = 1e-9


# =============================================================================
# Shared helper utilities
# =============================================================================

def _gray(img: np.ndarray) -> np.ndarray:
    """Convert a BGR colour image (or existing grayscale) to float32 grayscale.

    Applies OpenCV's Rec. 601 BT luma formula:
        Y = 0.114 * B + 0.587 * G + 0.299 * R
    when the input is 3-channel, otherwise simply casts to float32.

    Args:
        img: Input image.
             - 3-channel BGR uint8: shape (H, W, 3)
             - 2-channel grayscale (any numeric dtype): shape (H, W)

    Returns:
        Single-channel float32 grayscale image.
        shape: (H, W)   dtype: float32

    Notes:
        All P2020 v2 metrics call this helper as their first step so that
        each metric function can accept either BGR colour or pre-converted
        grayscale input without branching logic.
    """
    if img.ndim == 3:
        # Convert BGR colour -> grayscale using Rec. 601 coefficients
        # img shape: (H, W, 3) BGR uint8  ->  (H, W) uint8  ->  (H, W) float32
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)  # shape: (H, W)  dtype: float32


def _safe_div(a, b):
    """Safely divide a by b by adding EPS to the denominator.

    Args:
        a: Numerator (scalar or ndarray).
        b: Denominator (scalar or ndarray broadcastable with a).

    Returns:
        a / (b + EPS) -- same type/shape as the broadcast result.
    """
    return a / (b + EPS)


# =============================================================================
# Dynamic-Range / HDR  (P2020 §5)
# =============================================================================

def frame_dynamic_range_proxy(
    img: np.ndarray,
    p_lo: float = 0.1,
    p_hi: float = 99.9,
    assume_gamma: float | None = None,
) -> float:
    """P2020 §5 -- Per-frame dynamic range proxy via luminance percentile spread.

    Estimates the captured dynamic range of a single frame without a
    calibrated test chart, using the spread between extreme luminance
    percentiles.  Optionally applies gamma de-correction to convert code
    values into approximate linear light units, then reports the result in
    exposure value (EV) stops.

    Two output modes controlled by ``assume_gamma``:

        Mode A -- code-value difference (assume_gamma is None):
            DR = P_{p_hi}(Y) - P_{p_lo}(Y)    [0-255 code values]

        Mode B -- EV stops (assume_gamma is a float):
            Y_lin = (Y / 255) ^ assume_gamma   [approx. linear light]
            DR_EV = log2(P_{p_hi}(Y_lin) + EPS) - log2(P_{p_lo}(Y_lin) + EPS)
                  = log2( (P_hi + EPS) / (P_lo + EPS) )

    Args:
        img:          BGR uint8 input frame.
                      shape: (H, W, 3)
        p_lo:         Lower percentile for dark-end anchor.
                      Default 0.1 excludes ~0.1% of pixels as dead/hot.
        p_hi:         Upper percentile for bright-end anchor.
                      Default 99.9 excludes ~0.1% of pixels as saturated.
        assume_gamma: If None (default): return raw 8-bit code-value range.
                      If a float (e.g. 2.2 for sRGB): apply inverse gamma to
                      convert to approximate linear light, then return EV range.

    Returns:
        Dynamic range estimate as a float:
            - Mode A (no gamma): value in [0, 255] (code-value difference).
            - Mode B (gamma given): value in EV stops (log2 scale, typically
              0-8 for typical automotive scenes).

    Notes:
        P2020 §5 specifies DR measurement with a calibrated OECF chart.
        This surrogate is intended for generated-video evaluation where no
        chart is present.  Mode B provides a fairer cross-codec comparison
        when mixing SDR / HDR / tone-mapped generated clips.
    """
    # Convert to float32 grayscale (8-bit code values in [0, 255])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # gray shape: (H, W)  dtype: float32  values: [0, 255]

    if assume_gamma:
        # Apply inverse gamma: convert code value -> approximate linear light
        norm = gray / 255.0         # shape: (H, W)  values: [0, 1] normalised
        gray = np.power(norm, assume_gamma)  # shape: (H, W)  approx. linear [0, 1]

    # Compute percentile anchors across all pixels (flattened)
    lo, hi = np.percentile(gray, [p_lo, p_hi])  # scalars

    if assume_gamma:
        # EV stops: DR = log2(hi / lo)  -- log ratio of linear light levels
        return float(np.log2((hi + EPS) / (lo + EPS)))  # scalar in EV
    else:
        # Raw code-value range: DR = hi - lo
        return float(hi - lo)  # scalar in [0, 255]


def sequence_dynamic_range_proxy(
    video: Sequence[np.ndarray],
    p_lo: float = 0.1,
    p_hi: float = 99.9,
    assume_gamma: float | None = None,
) -> float:
    """P2020 §5 -- Sequence-level dynamic range proxy via pooled luminance histogram.

    Extends :func:`frame_dynamic_range_proxy` to an entire video sequence by
    pooling all pixel luminance values from all frames into a single histogram
    before computing the percentile spread.  This gives the dynamic range of
    the *scene* (across time) rather than any single frame.

    Two output modes -- same as :func:`frame_dynamic_range_proxy`:
        Mode A (assume_gamma=None):  returns 8-bit code-value difference.
        Mode B (assume_gamma=float): returns EV stops on linear light scale.

    Args:
        video:        Sequence of BGR uint8 frames.
                      Each frame shape: (H, W, 3)
        p_lo:         Lower percentile.  Default 0.1.
        p_hi:         Upper percentile.  Default 99.9.
        assume_gamma: None -> code-value range; float (e.g. 2.2) -> EV range.

    Returns:
        Sequence dynamic range (float) in code values or EV stops.

    Notes:
        Pooling all pixels across all frames before computing percentiles
        ensures that transient bright or dark events (e.g. a flash or a
        tunnel entry) are included in the range estimate rather than being
        averaged away by per-frame processing.
    """
    # Pool all grayscale pixels from all frames into a single flat array.
    # Each frame contributes H*W pixels; the concatenation has T*H*W elements.
    all_pix = np.concatenate(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).ravel() for f in video]
    ).astype(np.float32)
    # all_pix shape: (T*H*W,)  dtype: float32  values: [0, 255]

    if assume_gamma:
        # Inverse gamma: convert code values to approximate linear light
        norm = all_pix / 255.0             # shape: (T*H*W,)  values: [0, 1]
        all_pix = np.power(norm, assume_gamma)  # shape: (T*H*W,)  linear [0, 1]

    # Compute percentile anchors over the entire pooled distribution
    lo, hi = np.percentile(all_pix, [p_lo, p_hi])  # scalars

    if assume_gamma:
        # Log2 ratio gives EV stop range on the linear light scale
        return float(np.log2((hi + EPS) / (lo + EPS)))  # scalar in EV
    else:
        # Code-value range (0-255 scale)
        return float(hi - lo)  # scalar in [0, 255]


def temporal_exposure_jitter(video: Sequence[np.ndarray]) -> float:
    """P2020 §5 -- Temporal exposure jitter (std of frame-to-frame brightness change).

    Quantifies how erratically the mean frame brightness fluctuates over time.
    This is distinct from periodic flicker (which has a regular frequency):
    jitter is random, frame-level brightness variation caused by auto-exposure
    hunting or unstable tone mapping in the video generator.

    Formula:
        lum(t) = mean_{i,j} Y(t, i, j)        per-frame mean luminance
        jitter = std( diff(lum) )
               = std( lum(1)-lum(0), lum(2)-lum(1), ... )

    Args:
        video: Sequence of BGR uint8 frames.
               Each frame shape: (H, W, 3)

    Returns:
        Standard deviation of frame-to-frame luminance differences (float, >= 0).
        0 => perfectly stable brightness;  larger => more erratic auto-exposure.

    Notes:
        np.diff computes lum[t] - lum[t-1] for each consecutive pair, giving
        a length-(T-1) array of first-order differences.  The std of this
        measures temporal variability independently of any linear drift trend.
    """
    # Compute the mean luminance (scalar) for every frame in the sequence
    lum = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in video]
    # lum: list of scalars, length T

    # First-order differences: delta_lum[t] = lum[t+1] - lum[t]
    delta_lum = np.diff(lum)  # shape: (T-1,)

    # Standard deviation of brightness jumps quantifies exposure instability
    return float(np.std(delta_lum))  # scalar >= 0


# =============================================================================
# Sharpness / Resolution  (P2020 §6)
# =============================================================================

def _mtf_sobel(img: np.ndarray, axis: int, thr: float) -> float:
    """Internal helper -- compute a single-axis MTF at a given modulation threshold.

    Implements the Sobel Edge Spread Function (ESF) -> Line Spread Function
    (LSF) -> real FFT pipeline to approximate the Optical Transfer Function
    (OTF) magnitude, then finds the first frequency bin where the normalised
    OTF drops to ``thr``.

    Procedure:
        1. Apply Sobel filter in the direction perpendicular to ``axis``.
        2. Average the Sobel response along ``axis`` to form a 1-D LSF.
        3. Real FFT of the LSF -> OTF magnitude spectrum.
        4. Normalise the spectrum to [0, 1] by dividing by its maximum.
        5. Return the index of the first bin where spectrum <= thr.

    Args:
        img:  BGR uint8 or grayscale float32 input image.
              shape: (H, W, 3)  or  (H, W)
        axis: Direction selector:
              0 -> vertical Sobel (measures horizontal resolution / vertical edges)
              1 -> horizontal Sobel (measures vertical resolution / horizontal edges)
        thr:  Modulation threshold (e.g. 0.50 for MTF50, 0.10 for MTF10).

    Returns:
        Frequency bin index where the normalised OTF first drops to ``thr``
        (float).  Larger value => better resolution.  Returns ``spec.size``
        (i.e. Nyquist) if the modulation never drops to ``thr``.

    Notes:
        This function is deliberately private (name prefix ``_``) because its
        return value is in FFT bin units, not cycles-per-pixel.  The public
        wrappers ``mtf50`` and ``mtf10`` call it and may average results from
        both axes.
    """
    gray = _gray(img)  # shape: (H, W)  float32

    # Sobel: dx = 1 - axis,  dy = axis
    # When axis=0: dx=1, dy=0 -> responds to vertical edges (horizontal frequency).
    # When axis=1: dx=0, dy=1 -> responds to horizontal edges (vertical frequency).
    sob = cv2.Sobel(gray, cv2.CV_32F, 1 - axis, axis, ksize=3)  # shape: (H, W)

    # Average Sobel response along the ``axis`` dimension to form a 1-D LSF.
    # axis=0: mean over rows -> shape: (W,)
    # axis=1: mean over cols -> shape: (H,)
    lsf = sob.mean(axis=axis)  # shape: (W,) or (H,)

    # Real FFT of the LSF gives the complex OTF; take absolute value.
    # spec shape: (N//2+1,) where N = len(lsf)
    spec = np.abs(fft.rfft(lsf))  # shape: (N//2+1,)  -- OTF magnitude

    # Normalise so the peak == 1.0 (typically the DC bin is the largest)
    spec /= spec.max() + EPS  # shape: (N//2+1,)  values in [0, 1]

    # Find the first bin index where the normalised spectrum <= thr
    idx = np.flatnonzero(spec <= thr)  # shape: (M,)  -- indices below threshold

    # If the spectrum never drops to thr, the resolution exceeds the FFT limit
    return float(idx[0] if idx.size else spec.size)


def mtf50(img: np.ndarray, axis: int | str = "both") -> float:
    """P2020 §6.4.3 -- MTF50 surrogate via Sobel ESF pipeline.

    Estimates the spatial frequency (in FFT bin units) at which the image
    Modulation Transfer Function drops to 50% of its peak.  This is the
    primary sharpness metric in P2020 §6.

    Two operating modes:

        axis in {0, 1}: Evaluate only the specified axis.
            axis=0 -- vertical Sobel (horizontal resolution, consistent with
                       P2020 §6 default and the original p2020.py).
            axis=1 -- horizontal Sobel (vertical resolution).

        axis = "both" or None: Evaluate both axes independently and return
            their mean.  This is more robust for natural driving scenes and
            generated frames that may have anisotropic content.

    Args:
        img:  BGR uint8 input frame.
              shape: (H, W, 3)
        axis: Axis selector.  Default "both" (bidirectional average).

    Returns:
        MTF50 in FFT bin index units (float).
        Larger value => sharper image, higher spatial resolution.

    Notes:
        P2020 §6.4.3 / ISO 12233 define MTF50 with a slanted-edge chart.
        This surrogate uses scene content and is not directly comparable to
        chart-based measurements, but is consistent within the same dataset.
    """
    if axis in (0, 1):
        # Single-axis evaluation
        return _mtf_sobel(img, axis, 0.50)  # scalar

    # Bidirectional average -- more noise-robust on arbitrary scene content
    val0 = _mtf_sobel(img, 0, 0.50)  # scalar -- horizontal resolution
    val1 = _mtf_sobel(img, 1, 0.50)  # scalar -- vertical resolution

    return float((val0 + val1) * 0.5)  # scalar -- mean of both axes


def mtf10(img: np.ndarray, axis: int | str = "both") -> float:
    """P2020 §6.4.3 -- MTF10 surrogate via Sobel ESF pipeline.

    Identical to :func:`mtf50` except the modulation threshold is 0.10 rather
    than 0.50.  MTF10 indicates the frequency at which the system can just
    barely resolve detail -- the practical resolution limit.

    Args:
        img:  BGR uint8 input frame.
              shape: (H, W, 3)
        axis: Axis selector ("both", 0, or 1).  Default "both".

    Returns:
        MTF10 in FFT bin index units (float).
        Larger value => better limiting resolution.

    Notes:
        MTF10 is referenced in P2020 §6.4.3 alongside MTF50.  It is more
        sensitive to aliasing and noise than MTF50, making it a useful
        diagnostic for generated video that may introduce high-frequency
        noise artefacts.
    """
    if axis in (0, 1):
        return _mtf_sobel(img, axis, 0.10)  # scalar

    val0 = _mtf_sobel(img, 0, 0.10)  # scalar -- horizontal MTF10
    val1 = _mtf_sobel(img, 1, 0.10)  # scalar -- vertical MTF10

    return float((val0 + val1) * 0.5)  # scalar -- bidirectional mean


def contrast_transfer_accuracy(
    img: np.ndarray,
    tgt_contrasts: Sequence[float] = (0.1, 0.5, 0.9),
    patch_size: int = 16,
) -> float:
    """P2020 §6.5 KPI-3 -- Contrast Transfer Accuracy (CTA) at multiple contrast levels.

    Evaluates how faithfully the imaging / generation pipeline reproduces
    scene contrast at low, medium, and high contrast levels.  The metric
    measures the ratio of the *observed* mean Michelson contrast (across all
    image patches) to each *target* contrast level.  A ratio of 1 means
    perfect contrast reproduction; < 1 means contrast compression; > 1 means
    contrast enhancement.

    Michelson contrast of a patch:
        C = (I_max - I_min) / (I_max + I_min)

    Procedure:
        1. Downsample the image by 4x (makes patch statistics more robust
           to noise by averaging 4x4 pixel blocks into each sample).
        2. Divide the downsampled image into non-overlapping 16x16 patches.
        3. Compute Michelson contrast C for each patch.
        4. For each target contrast level t in ``tgt_contrasts``:
               slope_t = mean(C_patches) / t
        5. Return the mean of all slopes.

    Args:
        img:           BGR uint8 input frame.
                       shape: (H, W, 3)
        tgt_contrasts: Sequence of target contrast levels to evaluate.
                       Default (0.1, 0.5, 0.9) tests low / mid / high contrast,
                       as recommended in P2020 §6.5.
        patch_size:    Side length of each evaluation patch after 4x downsampling.
                       Default 16 px.

    Returns:
        Mean contrast transfer slope across all target levels (float).
        1.0 => perfect reproduction;  < 1 => contrast compressed;
        > 1 => contrast enhanced.

    Notes:
        P2020 §6.5 requires contrast measurement with a multi-contrast chart
        (e.g. SFRplus or eSFR).  This surrogate computes the metric from
        arbitrary scene content, which mixes the system's contrast transfer
        with the scene's own contrast distribution.
    """
    gray = _gray(img)  # shape: (H, W)  float32

    # 4x downsampling with INTER_AREA (averages 4x4 blocks, anti-aliased)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    # small shape: (H/4, W/4)  -- downsampled image

    h, w = small.shape  # dimensions after downsampling

    # Compute Michelson contrast for every non-overlapping patch
    vals = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            p = small[y: y + patch_size, x: x + patch_size]  # shape: (patch_size, patch_size)
            # Michelson contrast: C = (max - min) / (max + min)
            c = _safe_div(p.max() - p.min(), p.max() + p.min())  # scalar in [0, 1]
            vals.append(c)
    # vals: list of scalars, length ~ (H/4/patch_size) * (W/4/patch_size)

    if not vals:
        # No valid patches (image too small after downsampling)
        return 0.0

    mean_c = np.mean(vals)  # scalar -- mean observed Michelson contrast

    # For each target level, compute the slope = observed / target
    slopes = [mean_c / (t + EPS) for t in tgt_contrasts]
    # slopes: list of scalars, length = len(tgt_contrasts)

    return float(np.mean(slopes))  # scalar -- mean slope across all target levels


def edge_rise_time(img: np.ndarray, window: int = 12) -> float:
    """P2020 §4.1.2 / §6 -- Edge Rise Time (10%-90% width in pixels), bidirectional.

    Measures the sharpness of the steepest edge in the image by finding the
    number of pixels required for the intensity profile to transition from
    10% to 90% of its local range.  Both vertical and horizontal edges are
    evaluated and the result is their mean, giving a more complete picture
    of directional sharpness.

    Definition of edge rise distance (ERD) for a 1-D profile:
        p10 = 10th percentile of profile
        p90 = 90th percentile of profile
        i10 = first index where profile >= p10
        i90 = first index where profile >= p90
        ERD = max(i90 - i10, 0)   [pixels]

    Smaller ERD => sharper edge transition.

    Args:
        img:    BGR uint8 input frame.
                shape: (H, W, 3)
        window: Half-width (in pixels) of the local profile around the
                detected edge peak.  Default 12 -> 25-pixel profile.
                Sufficient for typical automotive lane-mark edges at 1080p.

    Returns:
        Mean edge rise distance in pixels (float, >= 0).
        0 => step-function edge;  large => blurry edge.

    Notes:
        This function always returns a value (never None); if the profile is
        too short or flat, the returned width approaches 0.  Consistent with
        ISO 12233 rise-time measurement concept as referenced in P2020 §6.
    """
    gray = _gray(img)  # shape: (H, W)
    assert gray.ndim == 2, "edge_rise_time expects a 2-D (grayscale) image."
    h, w = gray.shape  # image dimensions

    # Horizontal Sobel: dx=1, dy=0 -- large response at vertical edges
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # shape: (H, W)

    # Vertical Sobel: dx=0, dy=1 -- large response at horizontal edges
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # shape: (H, W)

    # Locate the peak absolute response for each Sobel direction
    y_v, x_v = np.unravel_index(np.argmax(np.abs(sob_x)), sob_x.shape)  # row, col
    y_h, x_h = np.unravel_index(np.argmax(np.abs(sob_y)), sob_y.shape)  # row, col

    # Extract local intensity profiles perpendicular to each detected edge
    # Vertical edge -> profile is horizontal (along columns)
    prof_vert = gray[y_v, max(0, x_v - window): min(w, x_v + window + 1)]
    # shape: (<=2*window+1,)  -- row slice centred on the vertical edge peak

    # Horizontal edge -> profile is vertical (along rows)
    prof_horiz = gray[max(0, y_h - window): min(h, y_h + window + 1), x_h].ravel()
    # shape: (<=2*window+1,)  -- column slice centred on the horizontal edge peak

    def _erd_from_profile(prof: np.ndarray) -> float:
        """Compute the 10%-90% edge rise distance from a 1-D intensity profile.

        Args:
            prof: 1-D array of luminance values across an edge.
                  shape: (N,)

        Returns:
            Rise distance in pixels (float >= 0).
            Returns 0.0 for degenerate (flat or length-1) profiles.
        """
        prof = np.asarray(prof, dtype=float)  # shape: (N,)
        n = prof.size

        if n <= 1:
            return 0.0  # degenerate profile -- no measurable rise time

        # 10th and 90th percentile of the profile's intensity range
        p10, p90 = np.percentile(prof, [10, 90])  # scalars

        # Find the first index where the profile crosses each threshold
        i10_arr = np.flatnonzero(prof >= p10)  # shape: (M10,)
        i90_arr = np.flatnonzero(prof >= p90)  # shape: (M90,)

        # Use the first crossing; fall back to extremes if threshold not found
        i10 = int(i10_arr[0]) if i10_arr.size else 0          # scalar int
        i90 = int(i90_arr[0]) if i90_arr.size else (n - 1)    # scalar int

        # Ensure non-negative (i90 should >= i10 for a rising edge, but clip)
        return float(max(i90 - i10, 0))

    erd_v = _erd_from_profile(prof_vert)   # scalar -- ERD for vertical edge
    erd_h = _erd_from_profile(prof_horiz)  # scalar -- ERD for horizontal edge

    # Return the mean of both directions for a direction-agnostic sharpness score
    return float((erd_v + erd_h) * 0.5)


# =============================================================================
# Geometry  (P2020 §3)
# =============================================================================

def total_distortion(img: np.ndarray, outer_frac: float = 0.8) -> float:
    """P2020 §3.5.1 -- Total radial distortion (signed proxy, uncalibrated).

    Estimates the overall radial lens distortion without a calibration target
    by fitting a linear model between a "ideal" uniformly-spaced radius
    sequence and the actual sorted radii of detected Canny edge pixels.
    The residual of the fit in the outer-image region is the distortion proxy.

    Interpretation of the sign:
        Positive -> outer edge pixels are further from centre than the linear
                    model predicts => barrel distortion.
        Negative -> outer edge pixels are closer to centre than predicted
                    => pincushion distortion.
    (Note: the implementation always returns abs(td) -- callers should call
    the function and interpret the raw value; sign logic is shown in the body.)

    Formula:
        1. Detect Canny edges.  Compute observed radii: r = ||p - centre||_2
        2. Sort r ascending -> r_sorted.  Build ideal linear sequence
           s = linspace(0, Rmax, M).
        3. Linear regression: r_sorted ~ a + b * s  (removes scale/offset).
        4. Per-point relative deviation: rel = (r_sorted - r_ref) / (s + eps)
        5. Take the median of rel in the outer ``outer_frac`` fraction of s.
        6. Return |median| as the unsigned total distortion proxy.

    Args:
        img:        BGR uint8 input frame.
                    shape: (H, W, 3)
        outer_frac: Fraction of the sorted-radius range to consider as
                    "outer image".  Default 0.8 (outer 20% of the range).
                    Distortion is more pronounced near image edges and more
                    stable to measure there.

    Returns:
        Total distortion proxy (float, >= 0, dimensionless relative units).
        E.g. 0.02 ~ 2% radial distortion.
        Returns float('nan') if fewer than 50 edge pixels are found.

    Notes:
        A full calibration-based measurement (P2020 §3.5.1) requires a
        regular dot-grid or checkerboard target.  This surrogate works on
        natural scenes but is only suitable for coarse comparison and may be
        affected by non-uniform scene content.
    """
    # Prepare uint8 grayscale for Canny (requires uint8 input)
    gray_u8 = _gray(img).astype(np.uint8)  # shape: (H, W)  dtype: uint8
    h, w = gray_u8.shape[:2]               # image height and width in pixels

    # Detect edges with standard automotive Canny thresholds
    edges = cv2.Canny(gray_u8, 50, 150)  # shape: (H, W)  dtype: uint8  binary
    ys, xs = np.where(edges)              # ys shape: (N,)  xs shape: (N,)

    if xs.size < 50:
        # Insufficient edge pixels to fit a reliable model
        return float("nan")

    # Image centre in floating-point coordinates
    cx, cy = w * 0.5, h * 0.5  # scalars

    # Observed radial distance of each edge pixel from the image centre
    r = np.hypot(xs - cx, ys - cy)  # shape: (N,)  -- pixels

    # Maximum possible radius = half-diagonal of the image
    Rmax = float(np.hypot(cx, cy))  # scalar -- pixels

    # Sort observed radii in ascending order
    order = np.argsort(r)  # shape: (N,)  -- sort indices
    r_sorted = r[order]    # shape: (N,)  -- sorted observed radii

    M = r_sorted.size  # number of edge pixels

    # "Ideal" uniformly-spaced radius sequence: what we'd expect if radii were
    # perfectly linearly distributed from 0 to Rmax (no distortion).
    s = np.linspace(0.0, Rmax, M, dtype=np.float64)  # shape: (M,)

    # Linear regression: r_sorted ~ a + b * s
    # This removes the overall scale factor and DC offset, isolating the
    # non-linear (distortion) component.
    a, b = np.polyfit(s, r_sorted, 1)  # scalars -- intercept and slope
    r_ref = a + b * s                  # shape: (M,)  -- linear prediction

    # Per-point relative deviation: (r_obs - r_ref) / s
    # Dividing by s (rather than r_ref) avoids instability at s=0.
    rel = (r_sorted - r_ref) / (s + 1e-6)  # shape: (M,)  -- relative deviation

    # Focus on the outer fraction of the image where distortion is most visible
    k0 = int(np.clip(outer_frac * M, 0, M - 1))  # scalar -- start index of outer region
    rel_outer = rel[k0:] if k0 < M else rel        # shape: (M - k0,)

    # Median of relative deviations in the outer region (robust to outliers)
    td = float(np.nanmedian(rel_outer))  # scalar -- signed distortion proxy

    # Return absolute value (unsigned magnitude); caller can check sign if needed
    return td if td > 0 else -td


# =============================================================================
# Flare / Stray-Light  (P2020 §2)
# =============================================================================

def flare_attenuation(
    img: np.ndarray,
    peak_thr: int = 240,
    bg_thr: int = 60,
    inner_ratio: float = 0.10,
    outer_band: tuple = (0.40, 0.45),
) -> float:
    """P2020 §2.5 KPI-2 -- Flare Attenuation (centre/periphery brightness ratio).

    Estimates the camera's ability to suppress flare (veiling glare and stray
    light) by comparing the mean luminance of the central image region to the
    mean luminance of an annular band in the periphery.

    Interpretation:
        Smaller value => better flare suppression (centre is not being
        artificially brightened by scattered light from a bright source).
        Larger value => significant veiling glare / stray light.

    Formula:
        r(i,j) = sqrt((j - W/2)^2 + (i - H/2)^2) / diag   [normalised radius]
        inner_mask = r < inner_ratio
        outer_mask = outer_band[0] < r < outer_band[1]
        FlareAttenuation = mean(Y[inner_mask]) / mean(Y[outer_mask])

    where diag = sqrt(H^2 + W^2) is the full image diagonal.

    Args:
        img:         BGR uint8 input frame.
                     shape: (H, W, 3)
        peak_thr:    Accepted as API parameter for diagnostics; not used in
                     the current computation (gating logic was removed to
                     ensure the function always returns a value).
        bg_thr:      Accepted as API parameter for diagnostics; not used.
        inner_ratio: The inner circular region is defined as r < inner_ratio
                     times the image diagonal.  Default 0.10.
        outer_band:  The outer annular region is defined as
                     outer_band[0] * diag < r < outer_band[1] * diag.
                     Default (0.40, 0.45).

    Returns:
        Flare attenuation ratio (float, > 0).
        Values < 1 indicate the periphery is brighter than the centre
        (unusual); values >> 1 indicate severe veiling glare.
        Never returns None; falls back to whole-image statistics for tiny
        images where the masks would be empty.

    Notes:
        P2020 §2.5 defines flare attenuation with a specific high-luminance
        point source target.  This surrogate uses the natural scene's centre
        vs. periphery luminance ratio and is most meaningful when a bright
        light source (sun, headlight) is present in the image.
    """
    gray = _gray(img)  # shape: (H, W)  float32
    assert gray.ndim == 2, "flare_attenuation expects a grayscale image."

    h, w = gray.shape  # image height and width in pixels

    # Full diagonal length (used as the normalisation radius)
    diag = float(np.hypot(h, w))  # scalar in pixels

    # Per-pixel distance from image centre, using open grid (no full meshgrid)
    # np.ogrid[:h, :w] generates index arrays without broadcasting overhead
    yy, xx = np.ogrid[:h, :w]  # yy shape: (H, 1)  xx shape: (1, W)
    r = np.hypot(xx - w * 0.5, yy - h * 0.5)  # shape: (H, W)  -- pixels from centre

    # Boolean masks defining the two measurement regions
    inner_mask = (r < inner_ratio * diag)                         # shape: (H, W)  bool
    outer_mask = (r > outer_band[0] * diag) & (r < outer_band[1] * diag)  # shape: (H, W)  bool

    # Fallback for very small images or extreme parameter settings
    if not inner_mask.any():
        # Inner mask is empty -- use the entire image as the inner region
        inner_mask = np.ones_like(gray, dtype=bool)  # shape: (H, W)  all True
    if not outer_mask.any():
        # Outer mask is empty -- use everything outside the inner mask
        outer_mask = ~inner_mask  # shape: (H, W)  -- complement of inner

    inner_mean = float(gray[inner_mask].mean())  # scalar -- mean in centre region
    outer_mean = float(gray[outer_mask].mean())  # scalar -- mean in outer annulus

    # Safe division: if outer is essentially black, use 1e-6 as denominator
    denom = outer_mean if outer_mean > 1e-6 else 1e-6

    return inner_mean / denom  # ratio > 0


# =============================================================================
# Flicker / Temporal  (P2020 §7)
# =============================================================================

def flicker_modulation_power(video: Sequence[np.ndarray], fps: float = 10) -> float:
    """P2020 §7.5 -- Flicker Modulation Power (FMP): spectral energy ratio at PWM peak.

    Quantifies the strength of periodic luminance modulation (flicker) by
    computing the fraction of the temporal luminance spectrum's total power
    that is concentrated near the dominant oscillation frequency.

    Procedure:
        1. Compute mean luminance per frame -> temporal signal luma(t).
        2. Real FFT -> power spectrum: S(f) = |FFT(luma)|^2.
        3. Identify the dominant frequency peak (excluding DC at f=0).
        4. Integrate power in the ±2 Hz band around the peak.
        5. Return the ratio: band_power / total_power.

    Formula:
        FMP = sum_{|f - f_peak| < 2} S(f)  /  sum_f S(f)

    Args:
        video: Sequence of BGR uint8 frames.
               Each frame shape: (H, W, 3)
        fps:   Frame rate in frames per second.  Default 10.
               Required for correct frequency-axis scaling.

    Returns:
        Flicker Modulation Power in [0, 1] (float).
        0 => no periodic flicker;  1 => all energy at flicker frequency.

    Notes:
        P2020 §7.5 references the VESA Flicker Index and JEDEC JESD51-51 for
        the formal definition of flicker modulation power.  This implementation
        is a frequency-domain proxy suitable for any frame rate >= 4 fps.
    """
    # Compute the mean luminance (scalar) for each frame
    luma = [_gray(f).mean() for f in video]  # list of scalars, length T

    # Power spectral density of the temporal luminance signal
    spec = np.abs(fft.rfft(luma)) ** 2   # shape: (T//2+1,)  -- power
    freqs = fft.rfftfreq(len(luma), d=1.0 / fps)  # shape: (T//2+1,)  -- Hz

    if len(freqs) < 3:
        # Too few frames to identify a meaningful spectral peak
        return 0.0

    # Dominant frequency: find the bin with maximum power, excluding DC (bin 0)
    peak = freqs[1:][np.argmax(spec[1:])]  # scalar -- peak frequency in Hz

    # ±2 Hz band around the dominant peak
    band = (freqs > peak - 2) & (freqs < peak + 2)  # shape: (T//2+1,)  bool

    # Fraction of total power in the flicker band
    return float(spec[band].sum() / (spec.sum() + EPS))  # scalar in [0, 1]


def fmp_alias(
    video: Sequence[np.ndarray],
    fps: float = 10,
    min_peak: float = 0.2,
) -> float:
    """P2020 §7.5 -- Proxy FMP for low frame-rate footage (alias-aware).

    An improved variant of :func:`flicker_modulation_power` designed for
    video captured at low frame rates (e.g. 10 fps from automotive data
    loggers) where the Nyquist frequency is low and scene brightness drift
    can masquerade as flicker.

    Improvements over the basic FMP:
        - Ignores spectral peaks below ``min_peak`` Hz (treats them as slow
          scene luminance drift rather than true flicker).
        - Uses an adaptive bandwidth: max(0.5 Hz, 20% of peak frequency),
          so very low-frequency peaks get a narrow band and high-frequency
          peaks get a wider band.

    Formula:
        bw = max(0.5, 0.2 * peak_f)
        FMP_alias = sum_{|f - f_peak| < bw} S(f)  /  sum_f S(f)

    Args:
        video:    Sequence of BGR uint8 frames.
                  Each frame shape: (H, W, 3)
        fps:      Frame rate in frames per second.  Default 10.
        min_peak: Minimum frequency (Hz) for a peak to be classified as
                  flicker rather than scene brightness drift.  Default 0.2 Hz.

    Returns:
        Proxy FMP in [0, 1] (float).
        Returns 0.0 if no peak above ``min_peak`` is found (no detectable
        flicker at this frame rate).

    Notes:
        At 10 fps the Nyquist limit is 5 Hz, so 50 Hz / 60 Hz LED PWM
        flicker cannot be directly detected -- only its aliases.  This
        function is designed to detect those aliases reliably.
    """
    lum = np.array([_gray(f).mean() for f in video])  # shape: (T,)  -- mean luma

    # Power spectrum of the temporal luminance signal
    spec = np.abs(fft.rfft(lum)) ** 2                 # shape: (T//2+1,)  -- power
    freqs = fft.rfftfreq(len(lum), d=1.0 / fps)       # shape: (T//2+1,)  -- Hz

    if len(freqs) < 3:
        return 0.0  # too few frames for spectral analysis

    # Find dominant frequency excluding DC (bin 0)
    peak_idx = np.argmax(spec[1:]) + 1  # scalar int -- index of dominant peak
    peak_f = freqs[peak_idx]             # scalar Hz -- dominant frequency

    if peak_f < min_peak:
        # The dominant frequency is too low to be genuine flicker
        return 0.0  # treat as no measurable flicker

    # Adaptive bandwidth: at least 0.5 Hz, or 20% of the peak frequency
    bw = max(0.5, 0.2 * peak_f)  # scalar Hz

    # Band mask: frequencies within ±bw of the peak
    band = (freqs > peak_f - bw) & (freqs < peak_f + bw)  # shape: (T//2+1,)  bool

    # Fraction of total spectral energy in the flicker band
    return float(spec[band].sum() / (spec.sum() + EPS))  # scalar in [0, 1]


def modulation_mitigation_probability(video: Sequence[np.ndarray]) -> float:
    """P2020 §7.6 KPI-A -- Modulation Mitigation Probability (MMP).

    Estimates the fraction of video frames whose mean luminance deviates by
    less than 5% from the sequence grand-mean.  A well-stabilised camera
    (or one with effective flicker mitigation) produces a flat luminance
    time series where almost all frames are within 5% of the mean.

    Formula:
        lum(t) = mean_{i,j} Y(t, i, j)
        grand_mean = mean_t lum(t)
        dev(t) = |lum(t) - grand_mean| / (grand_mean + EPS)
        MMP = fraction of t where dev(t) < 0.05

    Args:
        video: Sequence of BGR uint8 frames.
               Each frame shape: (H, W, 3)

    Returns:
        Modulation Mitigation Probability in [0, 1] (float).
        1.0 => all frames within 5% of mean (excellent mitigation).
        0.0 => no frames within 5% (severe unmitigated flicker).

    Notes:
        P2020 §7.6 defines MMP as the probability that a randomly chosen frame
        passes the 5% modulation criterion.  The threshold 5% corresponds to
        the VESA Flicker Index boundary for "low flicker risk" at typical
        automotive viewing distances.
    """
    # Per-frame mean luminance values
    means = np.array([_gray(f).mean() for f in video])  # shape: (T,)

    grand_mean = means.mean()  # scalar -- temporal mean over the whole sequence

    # Relative deviation of each frame from the grand mean
    dev = np.abs(means - grand_mean) / (grand_mean + EPS)  # shape: (T,)

    # Fraction of frames where the deviation is below the 5% threshold
    return float((dev < 0.05).mean())  # scalar in [0, 1]


def mmp_alias(
    video: Sequence[np.ndarray],
    fps: float = 10.0,
    band_hz: float = 0.5,
    thr: float = 0.05,
    win_sec: float = 3.0,
    hop_ratio: float = 0.5,
) -> float:
    """P2020 §7.6 -- Proxy MMP via sliding-window spectral analysis.

    An improved variant of :func:`modulation_mitigation_probability` that
    uses a sliding-window FFT approach.  This makes the metric applicable
    to long video sequences with non-stationary flicker characteristics,
    and avoids the limitation of comparing each frame against a global
    mean (which is itself affected by slow scene changes).

    Algorithm:
        1. Compute per-frame mean luminance -> temporal signal lum(t).
        2. Global FFT to identify the dominant alias frequency f_peak
           (excluding DC and any peak < 0.2 Hz which is scene drift).
        3. For each overlapping window of ``win_sec`` seconds (hop = 50%):
               a. FFT the windowed luminance segment.
               b. Compute the alias band energy fraction A around f_peak.
               c. Count the window as a "success" if A < thr.
        4. Return: (number of successful windows) / (total windows).

    Formula:
        For each window x[s : s+win]:
            A = sum_{|f - f_peak| < band_hz} |FFT(x)|^2  /  sum_f |FFT(x)|^2
            success = (A < thr)
        MMP_alias = hits / total

    Args:
        video:     Sequence of BGR uint8 frames.
                   Each frame shape: (H, W, 3)
        fps:       Frame rate in frames per second.  Default 10.0.
        band_hz:   Half-bandwidth (Hz) around the peak when computing the
                   alias band energy in each window.  Default 0.5 Hz.
        thr:       Energy fraction below which a window is classified as
                   "flicker successfully mitigated".  Default 0.05 (5%).
        win_sec:   Window duration in seconds.  Default 3.0 s.
                   At 10 fps: win = 30 frames.
        hop_ratio: Fractional overlap between consecutive windows.
                   Default 0.5 (50% overlap -> hop = win / 2 frames).

    Returns:
        Proxy MMP in [0, 1] (float).
        1.0 => flicker alias energy is below threshold in every window
               (good mitigation or no flicker).
        0.0 => alias energy exceeds threshold in every window (severe
               unmitigated flicker alias).
        Returns 0.0 for sequences shorter than 4 frames.
        Returns 1.0 if the dominant frequency is < 0.2 Hz (scene drift,
        not genuine flicker).

    Notes:
        P2020 §7.6 specifies MMP as the probability of modulation mitigation.
        The sliding-window approach here handles non-stationary flicker
        more robustly than a single global FFT.  This is the recommended
        metric for evaluating generated video in the DrivingGen benchmark.
    """
    T = len(video)  # total number of frames in the sequence

    if T < 4:
        # Too few frames to perform any meaningful spectral analysis
        return 0.0

    # Compute the mean luminance (scalar) for every frame
    lum = np.array([_gray(f).mean() for f in video], dtype=np.float32)
    # lum shape: (T,)  -- temporal luminance signal

    # ── Step 1: Find the dominant alias frequency using the full sequence ──

    # Power spectral density of the full luminance signal
    spec = np.abs(fft.rfft(lum)) ** 2           # shape: (T//2+1,)  -- power
    freqs = fft.rfftfreq(T, d=1.0 / fps)         # shape: (T//2+1,)  -- Hz

    # Dominant frequency: exclude DC (bin 0) then find argmax
    peak_idx = np.argmax(spec[1:]) + 1           # scalar int -- index of peak
    peak_f = freqs[peak_idx]                      # scalar Hz -- dominant frequency

    if peak_f < 0.2:
        # Below 0.2 Hz is considered slow scene luminance drift, not flicker.
        # All windows are treated as "successfully mitigated".
        return 1.0

    # ── Step 2: Configure the sliding window ──

    # Window size in frames: at least 8 frames, target win_sec * fps frames
    win = max(8, int(round(win_sec * fps)))  # scalar int -- frames per window

    # Hop size: 50% overlap by default; at least 1 frame
    hop = max(1, int(round(win * hop_ratio)))  # scalar int -- frames per hop

    # If the window is larger than the sequence, use the whole sequence
    if win > T:
        win = T
        hop = max(1, T // 2)  # fallback: 50% of the full sequence

    # Frequency band boundaries for the alias region
    band_lo = peak_f - band_hz  # scalar Hz -- lower band edge
    band_hi = peak_f + band_hz  # scalar Hz -- upper band edge

    # ── Step 3: Evaluate each window ──

    hits = 0    # number of windows with alias energy < thr
    total = 0   # total number of windows evaluated

    for s in range(0, T - win + 1, hop):
        # Extract the windowed luminance segment
        x = lum[s: s + win]  # shape: (win,)  -- window of mean luminance values

        # FFT of this window
        X = np.abs(fft.rfft(x)) ** 2         # shape: (win//2+1,)  -- power
        F = fft.rfftfreq(len(x), d=1.0 / fps)  # shape: (win//2+1,)  -- Hz

        # Alias band mask for this window's frequency axis
        band = (F > band_lo) & (F < band_hi)  # shape: (win//2+1,)  bool

        # Alias energy fraction: how much of the total power is in the band?
        A = X[band].sum() / (X.sum() + EPS)  # scalar in [0, 1]

        # Count as a "hit" if the alias energy is below the mitigation threshold
        hits += float(A < thr)   # adds 1.0 if below threshold, 0.0 otherwise
        total += 1                # increment window counter

    # Return the fraction of successfully mitigated windows
    return float(hits / total) if total > 0 else 0.0  # scalar in [0, 1]


# =============================================================================
# Texture / Detail
# =============================================================================

def gradient_entropy(img: np.ndarray) -> float:
    """P2020 §4.1.4 -- Texture richness via gradient-magnitude Shannon entropy.

    Computes the entropy of the gradient-magnitude histogram as a measure of
    texture complexity.  Scenes with diverse edge orientations and strengths
    (rich textures) produce a broad, flat histogram (high entropy); blurry
    or uniform scenes produce a narrow peak near zero (low entropy).

    Formula:
        gx = Sobel(Y, dx=1),  gy = Sobel(Y, dy=1)
        mag = sqrt(gx^2 + gy^2)
        hist = 256-bin normalised frequency distribution of mag
        H = -sum_i  p_i * log2(p_i)   [Shannon entropy in bits]

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Gradient magnitude entropy in bits (float, >= 0).
        Larger value => richer, more complex texture.

    Notes:
        Gradient entropy is referenced in P2020 §4.1.4 as a texture-richness
        measure.  In generated-video evaluation, low entropy compared to real
        frames indicates over-smoothing in the decoder.
    """
    gray = _gray(img)  # shape: (H, W)  float32

    # Compute horizontal and vertical Sobel responses
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # shape: (H, W)  -- dI/dx
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # shape: (H, W)  -- dI/dy

    # Gradient magnitude: ||nabla Y|| = sqrt(gx^2 + gy^2)
    mag = np.hypot(gx, gy)  # shape: (H, W)  -- magnitude [0, ~360] for uint8

    # Normalised 256-bin histogram of gradient magnitudes
    # density=True: normalise so that integral (sum * bin_width) == 1
    hist, _ = np.histogram(mag, bins=256, range=(0, mag.max() + EPS), density=True)
    # hist shape: (256,)  -- probability density values

    # Prevent log2(0) by adding epsilon before taking the logarithm
    hist += EPS  # shape: (256,)  -- all values now > 0

    # Shannon entropy H = -sum_i p_i * log2(p_i)
    return float(-(hist * np.log2(hist)).sum())  # scalar in bits


def blur_extent(img: np.ndarray) -> float:
    """P2020 §4.1.5 -- Blur extent via low-to-high frequency energy ratio.

    Separates the image into a low-frequency component (via Gaussian blur)
    and a high-frequency residual (detail layer).  The ratio of their mean
    absolute energies quantifies the degree of blur: more blur means more
    energy in the low-frequency band.

    Formula:
        L = GaussianBlur(Y, kernel=21x21)   [low-frequency component]
        H_detail = Y - L                    [high-frequency residual]
        BlurExtent = mean(|L|) / (mean(|H_detail|) + EPS)

    Larger ratio => more energy in low frequencies => more blur.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Blur extent ratio (float, >= 0).
        1 => equal low and high frequency energy.
        >> 1 => heavily blurred.

    Notes:
        The 21x21 Gaussian kernel (sigma auto-derived by OpenCV as ~3.5)
        is recommended in P2020 §4.1.5 as it covers the typical motion-blur
        radius for highway driving scenes.
    """
    gray = _gray(img)  # shape: (H, W)  float32

    # Low-frequency component: 21x21 Gaussian blur (sigma=0 -> auto from ksize)
    low = cv2.GaussianBlur(gray, (21, 21), 0)  # shape: (H, W)

    # High-frequency residual (detail layer): subtract the blurred version
    high = gray - low  # shape: (H, W)  -- values near 0 for blurry images

    # Energy ratio: mean |low| / mean |high|
    # A blurry image has small |high| -> large ratio
    return float(_safe_div(np.abs(low).mean(), np.abs(high).mean()))  # scalar >= 0


def chroma_aberration(img: np.ndarray, on_empty: str = "nan") -> float:
    """P2020 §4.3.7 -- Chromatic aberration (standard deviation of R-B at edges).

    Chromatic aberration (CA) causes the red and blue channels to focus at
    slightly different image planes, producing coloured fringing along
    high-contrast edges.  This metric measures the spatial variability of
    (R - B) at detected edge pixels; high std indicates strong lateral CA.

    Formula:
        edge_pixels = Canny(Y,  low_thr=50, high_thr=150)
        CA = std_{(i,j) in edge_pixels} (R(i,j) - B(i,j))

    Args:
        img:      BGR uint8 input frame.
                  shape: (H, W, 3)
        on_empty: Behaviour when no edge pixels are found:
                  "nan"  (default) -> return float('nan').
                         Recommended for statistical pipelines where NaN
                         values can be excluded from aggregation.
                  "zero" -> return 0.0.
                         Use when downstream code cannot handle NaN.

    Returns:
        Chromatic aberration level in DN (float, >= 0 or nan).
        Larger => more colour fringing at edges.

    Notes:
        P2020 §4.3.7 specifies CA measurement with a slanted-edge or
        dead-leaves target.  This surrogate uses scene edges and is
        appropriate for field evaluation of generated video where dedicated
        targets are absent.
    """
    # Split BGR into individual float32 channels
    b, g, r = cv2.split(img.astype(np.float32))
    # b shape: (H, W)   g shape: (H, W)   r shape: (H, W)

    # Detect edges in the grayscale image using the Canny algorithm
    gray_u8 = _gray(img).astype(np.uint8)  # shape: (H, W)  uint8 for Canny
    edges = cv2.Canny(gray_u8, 50, 150)    # shape: (H, W)  binary edge map

    # Pixel coordinates of all detected edge points
    ys, xs = np.where(edges > 0)  # ys shape: (N,)  xs shape: (N,)

    if ys.size == 0:
        # No edges detected -- chromatic aberration is unmeasurable
        return float('nan') if on_empty == "nan" else 0.0

    # R - B channel difference at each edge pixel
    diff = r[ys, xs] - b[ys, xs]  # shape: (N,)  -- signed difference

    # Remove any non-finite values (e.g. from unexpected image artefacts)
    diff = diff[np.isfinite(diff)]  # shape: (M,)  M <= N

    if diff.size == 0:
        return float('nan') if on_empty == "nan" else 0.0

    # Standard deviation of R-B differences at edge pixels
    return float(np.std(diff))  # scalar >= 0


# =============================================================================
# Convenience wrappers
# =============================================================================

def single_frame_metrics(img: np.ndarray) -> Dict[str, float]:
    """Compute the per-frame P2020 v2 high-priority KPIs for a single image.

    Currently only runs :func:`mmp_alias` (the most diagnostic metric for
    generated-video evaluation).  Additional metrics listed in the function
    body can be re-enabled by uncommenting them.

    Args:
        img: BGR uint8 input frame.
             shape: (H, W, 3)

    Returns:
        Dictionary mapping metric name (str) -> metric value (float).
        Currently returns: {'mmp_alias': <value>}

    Notes:
        Most single-frame metrics in this module (MTF50, distortion, etc.)
        are also valid per-frame metrics; they are currently commented out in
        the dispatcher to keep evaluation cost low during batch generation.
        Uncomment as needed for more detailed analysis.
    """
    out: Dict[str, float] = {}

    for fn in [
        # frame_dynamic_range_proxy,   # §5   per-frame DR estimate
        # mtf50,                       # §6   spatial frequency at 50% MTF
        # mtf10,                       # §6   spatial frequency at 10% MTF
        # contrast_transfer_accuracy,  # §6.5 contrast transfer slope
        # edge_rise_time,              # §6   10%-90% edge rise in pixels
        # total_distortion,            # §3   radial distortion proxy
        # flare_attenuation,           # §2   centre/periphery brightness ratio
        # gradient_entropy,            # §4.1 texture entropy (bits)
        # blur_extent,                 # §4.1 low/high frequency energy ratio
        # chroma_aberration,           # §4.3 R-B std at edges
    ]:
        out[fn.__name__] = fn(img)

    return out


def video_metrics(frames: Sequence[np.ndarray], fps: float = 30.0) -> Dict[str, float]:
    """Compute video-level P2020 v2 high-priority KPIs for a frame sequence.

    Currently runs only :func:`mmp_alias` (sliding-window Modulation
    Mitigation Probability), which is the primary metric for evaluating
    temporal flicker artefacts in generated video.

    Args:
        frames: Ordered sequence of BGR uint8 video frames.
                Each frame shape: (H, W, 3)
                Minimum 2 frames required (recommended >= 30 for stable FFT).
        fps:    Frame rate in frames per second.  Default 30.0 fps.
                Passed through to :func:`mmp_alias` for frequency scaling.

    Returns:
        Dictionary mapping metric name (str) -> metric value (float).
        Currently returns: {'mmp_alias': <value>}
        Returns an empty dict if fewer than 2 frames are provided.

    Notes:
        Additional temporal metrics (sequence DR, temporal exposure jitter,
        FMP alias, full MMP) are listed and commented out in the dispatcher.
        Enable them selectively to balance evaluation thoroughness vs. speed.
    """
    if len(frames) < 2:
        # Cannot compute inter-frame metrics with fewer than 2 frames
        return {}

    out: Dict[str, float] = {}

    for fn in [
        # sequence_dynamic_range_proxy,       # §5   pooled DR across all frames
        # fmp_alias,                          # §7.5 adaptive-bandwidth FMP proxy
        mmp_alias,                            # §7.6 sliding-window MMP (primary)
        # temporal_exposure_jitter,           # §5   std of frame-to-frame brightness
        # flicker_modulation_power,           # §7.5 basic FMP (fixed ±2 Hz band)
        # modulation_mitigation_probability,  # §7.6 simple 5%-deviation MMP
    ]:
        out[fn.__name__] = fn(frames)

    return out


# End of file
