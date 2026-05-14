# =============================================================================
# video_distribution.py
# =============================================================================
# Computes the Fréchet Video Distance (FVD) between a set of generated videos
# and a set of real (ground-truth) videos stored on disk.
#
# Design overview:
#   - FVD is the video analogue of FID (Fréchet Inception Distance).  It embeds
#     videos with a 3D CNN (typically I3D pre-trained on Kinetics) and measures
#     the Fréchet distance between the multivariate Gaussians fitted to the real
#     and generated embedding distributions.  A lower FVD indicates that the
#     generated videos are statistically closer to the real data distribution.
#   - The metric variant ``fvd2048_100f`` uses 2048 reference videos and 100
#     frames per video.  100 frames is the target clip length for DrivingGen's
#     evaluation protocol; using the full clip rather than a sub-sampled version
#     captures long-range temporal dynamics (e.g. consistent lane-keeping,
#     smooth camera motion) that shorter clips would miss.
#   - The implementation delegates to ``calc_metrics_`` from stylegan-v
#     (Skorokhodov et al., 2022) rather than reimplementing FVD from scratch.
#     stylegan-v's implementation handles dataset loading, feature extraction,
#     Gaussian fitting, and Fréchet distance computation in one call.
#
# Commented-out alternatives:
#   - ``common_metrics_on_video_quality`` (commented path) was an earlier
#     dependency that has been replaced by stylegan-v for consistency with the
#     rest of the evaluation suite.
#   - ``fvd2048_16f`` / ``fvd2048_128f`` variants are kept as comments for
#     reference; only the 100-frame variant is active.
#
# Usage:
#   from drivinggen.videos.video_distribution import get_fvd
#   fvd_score = get_fvd(fake_path="/data/generated", gt_path="/data/real")
# =============================================================================

import sys
import os

# ---------------------------------------------------------------------------
# Third-party stylegan-v dependency
# ---------------------------------------------------------------------------
# stylegan-v provides a robust, GPU-accelerated FVD implementation that is
# consistent with the evaluation protocol used in the original stylegan-v
# paper.  We add its source root to sys.path at import time so that its
# internal relative imports resolve correctly without requiring an editable
# install.
sys.path.append(os.path.abspath('third_parties/stylegan-v'))
from src.scripts.calc_metrics_for_dataset import calc_metrics_

# ---------------------------------------------------------------------------
# NOTE: The following alternative FVD implementation (common_metrics_on_video_quality)
# was used in an earlier version of this pipeline.  It has been superseded by the
# stylegan-v implementation above but is retained here for reference.
#
# sys.path.append('/mnt/cache/zhouyang/dg-bench/common_metrics_on_video_quality')
# from calculate_fvd import calculate_fvd
#
# def get_fvd(video_tensor_1, video_tensor_2):
#     fvd = calculate_fvd(video_tensor_1, video_tensor_2, 'cuda',
#                         method='styleganv', only_final=True)
#     return fvd['value'][0]
# ---------------------------------------------------------------------------


def get_fvd(fake_path: str, gt_path: str) -> float:
    """Compute the Fréchet Video Distance (FVD) between generated and real videos.

    FVD measures the statistical distance between the distributions of real and
    generated videos in I3D feature space.  Lower FVD indicates that the
    generated videos are closer to the real data distribution in terms of both
    per-frame appearance and temporal dynamics.

    This function uses the ``fvd2048_100f`` metric variant, which:
        - Embeds 2048 videos per split (real and generated) with a pre-trained
          I3D network.
        - Operates on 100-frame clips, capturing long-range temporal coherence
          that is essential for driving scenario evaluation.

    Why 100 frames?
        Driving sequences require evaluating sustained behaviours over time
        (e.g. consistent steering, stable depth cues, smooth ego-motion).  Short
        16-frame windows (``fvd2048_16f``) sample only ~0.5 s of 30 fps footage,
        which is insufficient to penalise temporal drift or flickering that
        accumulates over longer horizons.  100 frames (~3.3 s) gives a more
        realistic assessment of temporal quality.

    Why stylegan-v's ``calc_metrics_``?
        The stylegan-v implementation is GPU-accelerated, supports multi-GPU
        setups, includes data caching (``use_cache``), and produces results that
        are directly comparable to scores reported in the stylegan-v paper and
        related driving generation work.

    Args:
        fake_path (str): Path to the directory containing generated video frames
            or video files to be evaluated.  The directory layout must match the
            format expected by stylegan-v's dataset loader.
        gt_path (str): Path to the directory containing real (ground-truth) video
            frames or video files.  Must use the same layout as ``fake_path``.

    Returns:
        float: The FVD score for the ``fvd2048_100f`` variant.  Values are
            non-negative; lower is better (0 would mean the two distributions
            are identical, which never occurs in practice).

    Notes:
        - ``mirror=False``: Horizontal flipping augmentation is disabled because
          driving scenes have a strong left/right asymmetry (road markings, lane
          positions) and mirroring would corrupt the feature statistics.
        - ``resolution=256``: Videos are centre-cropped/resized to 256×256
          before I3D feature extraction, matching the I3D pre-training resolution.
        - ``gpus=1``: Feature extraction runs on a single GPU.  Increase for
          large datasets if multiple GPUs are available.
        - ``use_cache=False``: Feature caches are not reused between runs to
          ensure results always reflect the current content of ``fake_path`` and
          ``gt_path``.
        - ``num_runs=1``: A single evaluation run is performed.  Multiple runs
          can be used to estimate variance, but one run is standard for benchmarks.
        - ``calc_metrics_`` returns a list of result dicts (one per metric
          variant requested); since only ``fvd2048_100f`` is requested, the
          result is at index 0.

    Example:
        >>> score = get_fvd(
        ...     fake_path="/data/drivinggen/generated_videos",
        ...     gt_path="/data/drivinggen/real_videos",
        ... )
        >>> print(f"FVD: {score:.2f}")
        FVD: 142.57
    """
    print('Calculating FVD...')

    # Call stylegan-v's metric computation function.
    # Only the 100-frame FVD variant is requested; additional variants
    # (16f, 128f, 128f_subsample8f) are commented out below for reference.
    fvd = calc_metrics_(
        # Alternative metric variants (disabled):
        # metrics=['fvd2048_16f', 'fvd2048_128f', 'fvd2048_128f_subsample8f'],
        metrics=['fvd2048_100f'],    # 100-frame FVD, 2048 reference videos
        real_data_path=gt_path,      # Ground-truth video directory
        fake_data_path=fake_path,    # Generated video directory
        mirror=False,                # No horizontal flip: driving has directional asymmetry
        resolution=256,              # Resize to 256×256 to match I3D pre-training resolution
        gpus=1,                      # Single-GPU feature extraction
        verbose=True,                # Print progress to stdout
        use_cache=False,             # Always recompute; do not load stale caches
        num_runs=1                   # Single evaluation pass (standard for benchmarks)
    )
    # fvd is a list of dicts, one per requested metric.
    # fvd[0] corresponds to 'fvd2048_100f'.
    # fvd[0]['results'] maps metric name -> scalar score.

    # Return the scalar FVD score for the 100-frame variant.
    # Alternative return statements for multi-variant evaluation (disabled):
    # return [
    #     fvd[0]['results']['fvd2048_16f'],
    #     fvd[1]['results']['fvd2048_128f'],
    #     fvd[2]['results']['fvd2048_128f_subsample8f'],
    # ]
    return fvd[0]['results']['fvd2048_100f']
