"""
video_sub_q.py
==============
Subjective Quality assessment for the DrivingGen benchmark.

Uses **CLIP-IQA+** (no-reference image quality assessment trained on human
opinion scores) to predict the perceived quality of each video frame without
needing a reference image.

Why CLIP-IQA+?
--------------
CLIP-IQA+ is an opinion-aware, no-reference IQA metric that leverages the
rich visual-semantic features of CLIP.  It correlates well with human Mean
Opinion Scores (MOS) across diverse distortion types — including compression
artefacts, blur, and hallucination patterns typical of generative video
models.  It does not require a clean reference frame, which is essential for
evaluating generated content where no "ground truth" frame exists.

Pipeline for each video:
  1. Open each frame image (via ``open_image``).
  2. Resize to 512×512 (CLIP-IQA+ input size) and convert to a normalised
     float32 tensor.
  3. Stack all frames into a single batch tensor (shape: T×3×512×512).
  4. Run CLIP-IQA+ inference on the batch.
  5. Average the per-frame scores to get a single video-level quality value.

Dataset score:
    The mean of per-video scores across all videos in the dataset.

Optional z-score normalisation (``subjective_quality_zscore_rescale_infer``)
is provided for calibrating generated scores against a GT distribution, but
is not used in the default evaluation path.
"""

# Third-party imports
import pyiqa
import pyiqa.models
import pyiqa.models.inference_model
import torch
import numpy as np
from PIL import Image
from pyiqa.models.inference_model import InferenceModel
from torchvision import transforms
from torchvision.transforms import ToTensor
from typing import Tuple, List

# Local imports
from .metrics.base_metrics import open_image   # unified image loader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

# CLIP-IQA+ inference model; None until init_subjective_quality_model() is called
subjective_quality_model: InferenceModel = None


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_subjective_quality_model():
    """
    Lazy-load the CLIP-IQA+ no-reference image quality model.

    ``pyiqa.create_metric('clipiqa+')`` downloads and caches the model
    weights on first call.  Subsequent calls (within the same Python process)
    are no-ops because the singleton is already set.

    Why lazy loading?
        Other modules in DrivingGen may not need IQA at all.  Deferring
        model loading avoids unnecessary GPU memory allocation.

    Side effects:
        Sets the global ``subjective_quality_model`` variable.
    """
    global subjective_quality_model
    if subjective_quality_model is not None:
        return  # already loaded; skip

    # Create the CLIP-IQA+ metric; pyiqa handles weight download + placement
    subjective_quality_model = pyiqa.create_metric('clipiqa+')


# ---------------------------------------------------------------------------
# Z-score normalisation helper (reserved for calibrated evaluation)
# ---------------------------------------------------------------------------

def subjective_quality_zscore_rescale_infer(
    arr,
    target_low: float = 0.0,
    target_high: float = 1.0,
    gt=None,
) -> np.ndarray:
    """
    Z-score normalise raw CLIP-IQA+ scores and linearly rescale to a target
    range using statistics derived from a reference (GT) dataset.

    This function calibrates generated-video scores relative to a known
    distribution, so that a score of 0.5 represents "average GT quality"
    regardless of the absolute CLIP-IQA+ output range.

    Why z-score normalisation?
    --------------------------
    Raw CLIP-IQA+ scores vary with content type and camera characteristics.
    Z-scoring against the GT distribution removes this systematic offset,
    making cross-model comparisons fair.

    Score type is currently hard-coded to ``True`` (higher = better), which
    applies a direct linear rescaling.  Other modes (``False``, ``'mid'``,
    ``'row'``) are retained for future use with metrics where lower or
    mid-range values are desirable.

    Args:
        arr (array-like, shape (N,)): Raw CLIP-IQA+ scores for N videos.
        target_low (float): Lower bound of the output range.  Default: 0.0.
        target_high (float): Upper bound of the output range.  Default: 1.0.
        gt (tuple[float, float, float, float]): Reference distribution
            statistics ``(gt_mean, gt_std, gt_z_min, gt_z_max)`` obtained
            from the GT dataset at the 5th and 95th percentiles.

    Returns:
        np.ndarray, shape (N,): Rescaled quality scores in
            ``[target_low, target_high]``.
    """
    arr = np.asarray(arr, dtype=np.float32)   # shape: (N,)

    gt_mean, gt_std, gt_z_min, gt_z_max = gt

    # Step 1: z-score normalisation
    z = (arr - gt_mean) / (gt_std + 1e-9)   # shape: (N,); eps avoids div/0

    # Step 2: clip to GT percentile range and rescale to [0, 1]
    scaled = np.clip(
        (z - gt_z_min) / (gt_z_max - gt_z_min + 1e-9),
        0, 1
    )   # shape: (N,)

    # Step 3: choose score direction
    score_type = True  # True = higher raw score = higher quality
    if score_type is True:
        score = scaled                      # higher = better
    elif score_type is False:
        score = 1 - scaled                  # lower raw = better
    elif score_type == 'mid':
        # Penalise both extremes; mid-range scores are best
        score = 1 - 2 * (scaled - 0.5) ** 2
    elif score_type == 'row':
        # Special mode: values below 0.2 are treated as "perfect" (e.g. no
        # noise is always good); above 0.2 quality decreases linearly
        score = np.where(
            scaled < 0.2,
            1.0,
            1.0 - np.clip((scaled - 0.2) / 0.8, 0.0, 1.0)
        )

    # Step 4: map from [0, 1] to [target_low, target_high]
    return score * (target_high - target_low) + target_low  # shape: (N,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_subjective_quality(video_list: list) -> float:
    """
    Compute dataset-level subjective quality using CLIP-IQA+.

    For each video, all frames are:
      1. Opened as PIL images via ``open_image``.
      2. Resized to 512×512 (the spatial resolution expected by CLIP-IQA+).
      3. Converted to a float32 tensor in [0, 1].
      4. Stacked into a batch and sent to CUDA.

    CLIP-IQA+ then returns one quality score per frame (shape: (T,));
    these are averaged to produce a single video-level score.

    The dataset score is the mean of all per-video scores.

    Args:
        video_list (list[list[str | PIL.Image.Image]]): Outer list over
            videos; each inner list contains either file paths (str) or
            PIL Image objects for the video frames.

    Returns:
        float: Mean CLIP-IQA+ score across all videos and frames.
            Range: approximately [0, 1]; higher = better perceived quality.
    """
    scores = []  # one scalar per video

    # Ensure the model is loaded
    global subjective_quality_model
    init_subjective_quality_model()

    # ------------------------------------------------------------------
    # Inner helper: preprocess a list of frames into a batched CUDA tensor
    # ------------------------------------------------------------------
    def _process_image(rendered_images) -> torch.Tensor:
        """
        Preprocess a list of frames into a single batched tensor for CLIP-IQA+.

        Args:
            rendered_images (list[str | PIL.Image.Image]): Frame images as
                file paths or PIL Images.

        Returns:
            torch.Tensor: Batched frame tensor on CUDA.
                shape: (T, 3, 512, 512), dtype float32, values in [0, 1].
        """
        # Transform: resize to 512×512 then convert to float tensor [0, 1]
        preprocessing = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),    # converts PIL [0,255] → float [0,1]
        ])

        rendered_images_tensors: List[torch.Tensor] = []
        for image in rendered_images:
            if isinstance(image, str):
                # Load from disk first, then apply transforms
                image = preprocessing(open_image(image))   # shape: (3, 512, 512)
            else:
                # Already a PIL Image; apply transforms directly
                image = preprocessing(image)               # shape: (3, 512, 512)
            rendered_images_tensors.append(image)

        # Stack along batch dimension
        img: torch.Tensor = torch.stack(rendered_images_tensors).to('cuda')
        # shape: (T, 3, 512, 512)
        return img

    print('=========================Start Subjective Quality=========================')

    for rendered_images in tqdm(video_list):
        # Preprocess all frames for this video into a single batch
        imgs = _process_image(rendered_images)   # shape: (T, 3, 512, 512)

        with torch.no_grad():
            # CLIP-IQA+ returns one score per image in the batch
            score = subjective_quality_model(imgs)  # shape: (T,) or scalar

        # Average over all frames to get a single video quality value
        score = score.mean()                        # scalar tensor
        scores.append(score.item())                 # convert to Python float

    raw_mean = float(np.array(scores).mean())
    print(f'raw: {raw_mean}')   # expected GT reference: ~0.554

    # Return dataset mean quality score
    return raw_mean
