# =============================================================================
# lpips_metric.py
# =============================================================================
# Wrapper around TorchMetrics' LearnedPerceptualImagePatchSimilarity (LPIPS)
# for use within the DrivingGen evaluation pipeline.
#
# Design notes:
#   - AlexNet backbone is chosen over VGG-16 because it is significantly faster
#     while achieving comparable perceptual correlation with human judgements on
#     the BAPPS benchmark (Zhang et al., 2018).  For large-scale video evaluation
#     this speed advantage is critical.
#   - LPIPS operates in the range [0, 1] (approximately); lower is better,
#     meaning the generated frame is perceptually closer to the reference.
#   - Input images are converted from NumPy HWC uint8 to CHW float tensors via
#     ``_process_np_to_tensor`` inherited from BaseMetric.  No spatial resize is
#     applied in this path; callers should ensure matched resolutions.
# =============================================================================

import numpy as np

from .base_metrics import BaseMetric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LearnedPerceptualImagePatchSimilarityMetric(BaseMetric):
    """LPIPS perceptual similarity metric backed by an AlexNet feature extractor.

    LPIPS (Learned Perceptual Image Patch Similarity) measures how similar two
    images look to a human observer by comparing deep feature activations rather
    than raw pixel values.  It correlates strongly with human perceptual
    judgements and is widely used in generative model evaluation.

    Scoring convention:
        - Score range: approximately [0.0, 1.0].
        - **Lower is better**: a score of 0.0 indicates perfect perceptual
          similarity; a score close to 1.0 indicates high perceptual dissimilarity.

    Backbone choice (AlexNet):
        AlexNet is used instead of VGG-16 or SqueezeNet because it provides the
        best balance between speed and correlation with human judgements (as
        reported in Zhang et al., "The Unreasonable Effectiveness of Deep Features
        as a Perceptual Metric", CVPR 2018).  For DrivingGen, which may evaluate
        hundreds of video frames, runtime efficiency is a practical priority.

    Reference:
        Zhang, R. et al. (2018). The Unreasonable Effectiveness of Deep Features
        as a Perceptual Metric. CVPR 2018.
        https://arxiv.org/abs/1801.03924

    TorchMetrics documentation:
        https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html

    Attributes:
        _metric (LearnedPerceptualImagePatchSimilarity): The TorchMetrics LPIPS
            model with AlexNet backbone, placed on ``self._device``.
        _device (torch.device): Inherited from BaseMetric; set to CUDA if
            available, otherwise CPU.
    """

    def __init__(self) -> None:
        """Initialise the LPIPS metric with an AlexNet backbone.

        Calls ``BaseMetric.__init__()`` to set ``self._device``, then
        instantiates a ``LearnedPerceptualImagePatchSimilarity`` model using
        the ``"alex"`` (AlexNet) network and moves it to the target device.
        """
        super().__init__()
        # AlexNet backbone: fast inference, good human perceptual correlation.
        # Placing the model on self._device ensures it is co-located with the
        # input tensors produced by _process_np_to_tensor.
        self._metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self._device
        )

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        """Compute the LPIPS perceptual similarity score for a single image pair.

        Args:
            rendered_image (np.ndarray): The generated / predicted image frame.
                Expected shape: (H, W, 3), dtype uint8, pixel values in [0, 255].
            reference_image (np.ndarray): The ground-truth reference frame.
                Expected shape: (H, W, 3), dtype uint8, pixel values in [0, 255].
                Must have the same spatial resolution as ``rendered_image``.

        Returns:
            float: The LPIPS score in approximately [0.0, 1.0].
                Lower values indicate greater perceptual similarity.

        Notes:
            ``_process_np_to_tensor`` converts both NumPy arrays to float32
            tensors with a leading batch dimension and moves them to
            ``self._device``:
                img1: # shape: (1, 3, H, W), float32, values in [0, 1]
                img2: # shape: (1, 3, H, W), float32, values in [0, 1]

            ``.detach()`` prevents gradient tracking through the metric
            computation (not needed for evaluation), and ``.item()`` extracts
            the Python float from the scalar tensor.
        """
        # Convert HWC uint8 NumPy arrays to (1, 3, H, W) float tensors on device.
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)
        # img1: # shape: (1, 3, H, W) — rendered frame, float32 in [0, 1]
        # img2: # shape: (1, 3, H, W) — reference frame, float32 in [0, 1]

        # Forward pass through LPIPS; result is a scalar tensor.
        # .detach() drops the computational graph (eval mode, no gradients needed).
        # .item() converts the 0-d tensor to a plain Python float.
        score: float = self._metric(img1, img2).detach().item()
        return score
