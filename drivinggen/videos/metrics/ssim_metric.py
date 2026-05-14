# =============================================================================
# ssim_metric.py
# =============================================================================
# Wrapper around TorchMetrics' StructuralSimilarityIndexMeasure (SSIM) for
# use within the DrivingGen evaluation pipeline.
#
# Design notes:
#   - SSIM is a full-reference metric: it always requires a clean ground-truth
#     image alongside the generated image being scored.
#   - Unlike pixel-level MSE/PSNR, SSIM models three perceptual components
#     separately — luminance, contrast, and structure — making it more
#     correlated with human quality judgements on natural images.
#   - Input images are converted from NumPy HWC uint8 to CHW float tensors via
#     ``_process_np_to_tensor`` (inherited from BaseMetric).  No spatial resize
#     is applied; callers must ensure matched resolutions.
# =============================================================================

import numpy as np

from .base_metrics import BaseMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure


class StructuralSimilarityIndexMeasureMetric(BaseMetric):
    """SSIM full-reference image quality metric.

    The Structural Similarity Index Measure (SSIM) evaluates perceptual image
    quality by jointly assessing three components of structural information:

        1. **Luminance** – mean intensity comparison.
        2. **Contrast** – standard deviation comparison.
        3. **Structure** – normalised cross-correlation of local patches.

    This decomposition makes SSIM more aligned with the human visual system than
    simple pixel-level error metrics such as MSE or PSNR, which weight all
    spatial positions and error types equally.

    Scoring convention:
        - Score range: [-1.0, 1.0] (in practice [0.0, 1.0] for natural images).
        - **Higher is better**: a score of 1.0 means the two images are
          structurally identical; lower scores indicate increasing distortion.

    Full-reference design:
        SSIM requires a distortion-free reference image to serve as the quality
        anchor.  For DrivingGen this reference is the ground-truth video frame
        from the evaluation dataset; the score measures how faithfully the
        generated frame reproduces its structural content.

    Reference:
        Wang, Z. et al. (2004). Image Quality Assessment: From Error Visibility
        to Structural Similarity. IEEE Transactions on Image Processing, 13(4).
        https://doi.org/10.1109/TIP.2003.819861

    TorchMetrics documentation:
        https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html

    Attributes:
        _metric (StructuralSimilarityIndexMeasure): The TorchMetrics SSIM
            module, placed on ``self._device``.
        _device (torch.device): Inherited from BaseMetric; CUDA if available,
            otherwise CPU.
    """

    def __init__(self) -> None:
        """Initialise the SSIM metric module and move it to the target device.

        Calls ``BaseMetric.__init__()`` to configure ``self._device``, then
        instantiates ``StructuralSimilarityIndexMeasure`` with default
        parameters (11×11 Gaussian kernel, sigma=1.5, data range inferred from
        the input tensor range) and places it on the device.
        """
        super().__init__()
        # Default TorchMetrics SSIM settings match the original Wang et al.
        # paper configuration (window size 11, sigma 1.5).
        # Placing the module on self._device ensures it is co-located with
        # the input tensors produced by _process_np_to_tensor.
        self._metric = StructuralSimilarityIndexMeasure().to(self._device)

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        """Compute the SSIM score for a single rendered / reference image pair.

        Args:
            rendered_image (np.ndarray): The generated / predicted image frame.
                Expected shape: (H, W, 3), dtype uint8, pixel values in [0, 255].
            reference_image (np.ndarray): The undistorted ground-truth reference.
                Expected shape: (H, W, 3), dtype uint8, pixel values in [0, 255].
                Must have the same spatial resolution as ``rendered_image``.

        Returns:
            float: The SSIM index in [-1.0, 1.0] (typically [0.0, 1.0] for
                natural image comparisons).  A value of 1.0 indicates perfect
                structural similarity.

        Notes:
            ``_process_np_to_tensor`` converts both NumPy arrays to float32
            tensors with a leading batch dimension and moves them to
            ``self._device``:
                img1: # shape: (1, 3, H, W), float32, values in [0, 1]
                img2: # shape: (1, 3, H, W), float32, values in [0, 1]

            ``.detach()`` prevents gradient tracking (evaluation only), and
            ``.item()`` extracts the Python float from the resulting 0-d tensor.
        """
        # Convert HWC uint8 NumPy arrays to (1, 3, H, W) float tensors on device.
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)
        # img1: # shape: (1, 3, H, W) — rendered frame, float32 in [0, 1]
        # img2: # shape: (1, 3, H, W) — reference frame, float32 in [0, 1]

        # Forward pass through SSIM; result is a scalar tensor.
        # .detach() drops the computational graph (eval mode, no backprop needed).
        # .item() converts the 0-d tensor to a plain Python float.
        score: float = self._metric(img1, img2).detach().item()
        return score
