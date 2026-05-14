# =============================================================================
# base_metrics.py
# =============================================================================
# Abstract base infrastructure for all image quality metrics used in DrivingGen.
#
# Design overview:
#   - Device utilities (is_cuda_available, get_torch_device_name, get_torch_device)
#     centralise GPU/CPU selection so every metric class uses the same device.
#   - Image I/O helpers (is_url, open_image) let callers pass either a local file
#     path or an HTTP(S) URL; both are normalised to an RGB PIL Image before
#     preprocessing.
#   - BaseMetric (ABC) owns the canonical preprocessing pipeline:
#       resize to 512×512  ->  ToTensor  ->  stack into batch  ->  move to device
#     512×512 is chosen because it is the native resolution used by IQA-PyTorch
#     models and large enough to preserve fine texture detail that perceptual
#     metrics depend on.
#   - IQAPytorchMetric extends BaseMetric for metrics backed by pyiqa, loading
#     a named model and placing it on the correct device automatically.
# =============================================================================

from abc import ABC, abstractmethod
import urllib
from typing import List, Tuple, Union

import numpy as np
import pyiqa
import pyiqa.models
import pyiqa.models.inference_model
import requests
import torch
from PIL import Image
from pyiqa.models.inference_model import InferenceModel
from torchvision import transforms
from torchvision.transforms import ToTensor


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Return True when at least one CUDA-capable GPU is visible to PyTorch.

    Returns:
        bool: True if ``torch.cuda.is_available()`` reports a GPU, else False.

    Notes:
        This is a thin wrapper so the rest of the codebase never directly
        imports ``torch.cuda`` for availability checks, making it easier to
        mock in unit tests.
    """
    return torch.cuda.is_available()


def get_torch_device_name() -> str:
    """Return the canonical device string for the current hardware.

    Returns:
        str: ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.

    Notes:
        Downstream code uses this string to construct ``torch.device`` objects
        or to pass as a ``device`` argument to model constructors.
    """
    return "cuda" if is_cuda_available() else "cpu"


def get_torch_device() -> torch.device:
    """Return a ``torch.device`` representing the best available hardware.

    Returns:
        torch.device: A CUDA device when a GPU is present, else the CPU device.

    Notes:
        All metric classes call this once in ``__init__`` and cache the result
        in ``self._device`` so that tensors and models are co-located.
    """
    return torch.device(get_torch_device_name())


# ---------------------------------------------------------------------------
# Image I/O utilities
# ---------------------------------------------------------------------------

def is_url(location: str) -> bool:
    """Determine whether a string is an HTTP(S) URL.

    Args:
        location (str): A filesystem path or a URL string.

    Returns:
        bool: True if ``location`` begins with ``http`` or ``https``, else False.

    Notes:
        Uses ``urllib.parse.urlparse`` for robust scheme detection rather than
        a naive ``startswith`` check, which would miss edge cases such as
        ``//example.com/image.png`` (scheme-relative URLs).
    """
    return urllib.parse.urlparse(location).scheme in ["http", "https"]


def open_image(image_location: str) -> Image.Image:
    """Load an image from a local path or a remote URL as an RGB PIL Image.

    Args:
        image_location (str): Either a local filesystem path (e.g.
            ``"/data/frame_001.png"``) or an HTTP(S) URL pointing to an image.

    Returns:
        Image.Image: The loaded image converted to RGB mode.  Converting to RGB
        ensures consistent 3-channel input regardless of whether the source
        image is RGBA, grayscale, or palette-indexed.

    Notes:
        Remote images are streamed via ``requests.get(..., stream=True)`` to
        avoid loading the entire response into memory before PIL can begin
        decoding.
    """
    image: Image.Image
    if is_url(image_location):
        # Stream the HTTP response so large images are not fully buffered
        # before PIL starts decoding.
        image = Image.open(requests.get(image_location, stream=True).raw)
    else:
        image = Image.open(image_location)

    # Force 3-channel RGB so downstream ToTensor always produces (3, H, W).
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """Abstract base class shared by every image quality metric in DrivingGen.

    Responsibilities:
        - Stores a reference to the underlying metric model (``self._metric``)
          and the target compute device (``self._device``).
        - Provides ``_process_image`` and ``_process_images`` for PIL / path
          inputs, and ``_process_np_to_tensor`` for raw NumPy array inputs.
        - Declares ``_compute_scores`` as an abstract method that every
          concrete subclass must implement.

    Attributes:
        _metric: The underlying metric model or callable.  Set to ``None``
            here; concrete subclasses assign a specific model in their
            ``__init__``.
        _device (torch.device): The device on which tensors and models live.
    """

    def __init__(self) -> None:
        # _metric is intentionally None at this level; subclasses assign
        # a concrete model (e.g. LearnedPerceptualImagePatchSimilarity).
        self._metric = None
        # Determine GPU vs CPU once and reuse throughout the object lifetime.
        self._device: torch.device = get_torch_device()

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _process_image(
        self,
        rendered_images: List[Union[str, Image.Image]],
    ) -> torch.Tensor:
        """Preprocess a list of images into a batched tensor (no reference).

        Used by no-reference (NR) metrics that score a single image without
        a ground-truth counterpart.

        Args:
            rendered_images (List[Union[str, Image.Image]]): Each element is
                either a filesystem path / URL string or an already-open PIL
                Image.  All formats are normalised before stacking.

        Returns:
            torch.Tensor: Batched image tensor on ``self._device``.
                # shape: (B, 3, 512, 512)
                where B = len(rendered_images).

        Notes:
            Images are resized to 512×512 before converting to a tensor.
            512×512 is the standard resolution expected by IQA-PyTorch models
            and large enough to preserve perceptually relevant texture detail.
            ``ToTensor`` scales pixel values from [0, 255] to [0.0, 1.0].
        """
        # Build a deterministic pipeline: resize then tensorise.
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),  # Normalise spatial resolution
                transforms.ToTensor(),           # (H, W, C) uint8 -> (C, H, W) float32 in [0,1]
            ]
        )

        rendered_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Accept either a path/URL string or a pre-loaded PIL Image.
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            # Each element: # shape: (3, 512, 512)
            rendered_images_.append(image)

        # Stack individual tensors along a new batch dimension.
        img: torch.Tensor = torch.stack(rendered_images_).to(self._device)
        # shape: (B, 3, 512, 512)
        return img

    def _process_images(
        self,
        rendered_images: List[Union[str, Image.Image]],
        reference_image: Union[str, Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess a rendered batch and a single reference image for
        full-reference metric evaluation.

        The same reference image is replicated once per rendered image so that
        the two returned tensors have identical batch sizes and can be passed
        directly to a pairwise metric.

        Args:
            rendered_images (List[Union[str, Image.Image]]): Batch of generated
                or rendered images to evaluate, given as paths/URLs or PIL Images.
            reference_image (Union[str, Image.Image]): The single ground-truth
                image used as the quality reference.  It is broadcast across
                the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A pair ``(img1, img2)`` where:
                img1 – rendered images:  # shape: (B, 3, 512, 512)
                img2 – reference copies: # shape: (B, 3, 512, 512)
                Both tensors reside on ``self._device``.

        Notes:
            Broadcasting the reference avoids allocating a second copy of the
            pixel data until ``torch.stack`` is called, keeping peak memory
            roughly proportional to batch size rather than 2×batch size.
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        # Preprocess the reference image once; it is reused for every rendered
        # image in the batch.
        if isinstance(reference_image, str):
            reference_image = preprocessing(open_image(reference_image))
        # reference_image: # shape: (3, 512, 512)

        rendered_images_: List[torch.Tensor] = []
        reference_images_: List[torch.Tensor] = []

        for image in rendered_images:
            # Preprocess each rendered frame individually.
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            # image: # shape: (3, 512, 512)
            rendered_images_.append(image)

            # Append the same reference tensor for every rendered image so
            # that img1[i] and img2[i] are always a matched pair.
            reference_images_.append(reference_image)

        img1: torch.Tensor = torch.stack(rendered_images_).to(self._device)
        # shape: (B, 3, 512, 512)

        img2: torch.Tensor = torch.stack(reference_images_).to(self._device)
        # shape: (B, 3, 512, 512)

        return img1, img2

    def _process_np_to_tensor(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a pair of NumPy HWC images to batched CHW float tensors.

        This is the fast path for callers that already hold images in memory
        as NumPy arrays (e.g. extracted video frames), avoiding the extra
        PIL encode/decode round-trip used by ``_process_images``.

        Args:
            rendered_image (np.ndarray): The generated / predicted frame.
                Expected shape: (H, W, 3), dtype uint8 with values in [0, 255].
            reference_image (np.ndarray): The ground-truth frame.
                Expected shape: (H, W, 3), dtype uint8 with values in [0, 255].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(img1, img2)`` where each
                tensor has:
                # shape: (1, 3, H, W)
                dtype float32, values in [0.0, 1.0], on ``self._device``.

        Notes:
            ``ToTensor()`` handles the HWC -> CHW transposition and uint8 ->
            float32 / 255 normalisation in a single pass.  ``unsqueeze(0)``
            inserts the batch dimension expected by TorchMetrics operators.
            No spatial resize is performed here; it is the caller's
            responsibility to pass matching resolutions.
        """
        # ToTensor: (H, W, 3) uint8 -> (3, H, W) float32 in [0,1]
        # unsqueeze(0): (3, H, W) -> (1, 3, H, W)
        img1: torch.Tensor = ToTensor()(rendered_image).unsqueeze(0).to(self._device)
        # shape: (1, 3, H, W)

        img2: torch.Tensor = ToTensor()(reference_image).unsqueeze(0).to(self._device)
        # shape: (1, 3, H, W)

        return img1, img2

    @abstractmethod
    def _compute_scores(self, *args):
        """Compute and return the metric score(s) for the given inputs.

        This method is intentionally left abstract so that each subclass can
        define the exact call signature (e.g. a single image, a pair of
        images, a list of frames) that matches its underlying model's API.

        Subclasses must:
            1. Call one of the ``_process_*`` helpers to obtain device tensors.
            2. Forward the tensors through ``self._metric``.
            3. Detach and return a Python scalar (``float``) or a list/dict
               of scalars as appropriate.
        """
        pass


# ---------------------------------------------------------------------------
# IQA-PyTorch convenience subclass
# ---------------------------------------------------------------------------

class IQAPytorchMetric(BaseMetric):
    """BaseMetric subclass that wraps any metric available through pyiqa.

    ``pyiqa`` (IQA-PyTorch) provides a unified interface to dozens of
    no-reference and full-reference image quality assessment models.  This
    class handles model creation and device placement, leaving only
    ``_compute_scores`` to be implemented by concrete subclasses.

    Args:
        metric_name (str): The metric identifier recognised by
            ``pyiqa.create_metric``, e.g. ``"niqe"``, ``"brisque"``,
            ``"lpips"``.

    Attributes:
        _metric (InferenceModel): The loaded pyiqa model, already moved to
            ``self._device``.

    Notes:
        Calling ``.to(self._device)`` on the pyiqa model ensures that both
        the model weights and intermediate feature maps are computed on the
        same device as the input tensors produced by the preprocessing helpers,
        preventing device-mismatch errors at inference time.
    """

    def __init__(self, metric_name: str) -> None:
        # Initialise device and set _metric to None (BaseMetric.__init__).
        super().__init__()
        # Overwrite _metric with the actual pyiqa model, placed on the target device.
        self._metric: InferenceModel = self._create_metric(metric_name).to(self._device)

    def _create_metric(self, metric: str) -> InferenceModel:
        """Instantiate a pyiqa metric by name.

        Args:
            metric (str): The metric name passed to ``pyiqa.create_metric``,
                e.g. ``"niqe"`` or ``"musiq"``.

        Returns:
            InferenceModel: The pyiqa model object, on CPU at this point;
                the caller is responsible for moving it to the correct device.

        Notes:
            ``pyiqa.create_metric`` handles weight downloading and caching
            automatically on the first call, so no manual checkpoint management
            is needed.
        """
        metric_model: InferenceModel = pyiqa.create_metric(metric)
        return metric_model
