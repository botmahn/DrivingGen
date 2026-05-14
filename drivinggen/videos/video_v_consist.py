"""
video_v_consist.py
==================
Visual (scene-level) Consistency metric for the DrivingGen benchmark.

Combines two complementary signals to assess the temporal coherence of a
generated driving video:

  1. **SEA-RAFT optical flow** — quantifies how much the scene is moving
     between consecutive frames.  The per-frame-pair median flow magnitude
     forms a "motion series" that drives *keyframe selection*.

  2. **DINOv3-ViT-H16+ embeddings** — dense, semantically rich 1024-d
     appearance descriptors extracted at the selected keyframes.  Adjacent-
     keyframe cosine similarity measures how smoothly the scene appearance
     evolves over time.

Why arc-length keyframe selection?
-----------------------------------
Uniform temporal sampling is suboptimal: during slow-motion segments many
consecutive frames are visually identical (wasted budget), while during
fast-motion segments a single step can span a large appearance change.

Instead, we treat the cumulative sum of per-frame flow magnitudes as the
"arc length" of the trajectory in appearance space.  Selecting keyframes
that are equidistant in arc length guarantees that each pair captures a
similar amount of visual change — making the cosine-similarity statistics
more interpretable and comparable across scenes with different dynamics.

The number of keyframes K is linearly interpolated between ``min_k`` and
``max_k`` based on the mean motion magnitude, so static scenes use fewer
keyframes (lower memory / compute) while dynamic scenes use more.

Embeddings and flow series are cached to disk (pickle) to avoid redundant
computation across evaluation runs.
"""

# Standard-library imports
import os
import sys
import math
import pickle
import types
from typing import Tuple, List

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Compatibility shim for older PyTorch versions that lack torch.compiler.
# ---------------------------------------------------------------------------
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    # Always return False; this matches the default behaviour of newer PyTorch.
    torch.compiler.is_compiling = lambda: False

from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image as load_image_hf

# ---------------------------------------------------------------------------
# SEA-RAFT path setup.  The third-party code lives in third_parties/SEA-RAFT.
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath('third_parties/SEA-RAFT'))
sys.path.append(os.path.abspath('third_parties/SEA-RAFT/core'))

import argparse
from core.raft import RAFT
from core.utils.utils import load_ckpt
from config.parser import parse_args


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------

# DINOv3 tuple: (model, processor) — None until init_dinov3_model() is called
dinov3 = None

# SEA-RAFT model and its parsed argument namespace
flow_model = None
flow_args  = None


# ---------------------------------------------------------------------------
# DINOv3 initialisation
# ---------------------------------------------------------------------------

def init_dinov3_model():
    """
    Lazy-load the DINOv3-ViT-H16+ model and its image processor.

    Why DINOv3 ``pooler_output``?
    ------------------------------
    ``pooler_output`` is the global [CLS] token representation (shape: 1024-d
    for ViT-H), which aggregates spatial information from all patch tokens via
    self-attention.  For scene-level consistency this is preferable to
    patch-level features because:
      - It is invariant to minor spatial shifts / translations.
      - It captures the overall scene semantics (road layout, lighting, sky)
        rather than local texture details.
      - Cosine similarity of [CLS] embeddings is well-calibrated due to
        DINOv3's contrastive pre-training objective.

    The model is loaded once and stored in the module-level ``dinov3`` tuple
    ``(model, processor)``; repeated calls are no-ops.

    Side effects:
        Sets the global ``dinov3`` variable.
    """
    global dinov3
    if dinov3 is not None:
        return  # already initialised; skip

    model_dir = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    # AutoImageProcessor handles resizing and normalisation per the ViT-H spec
    processor = AutoImageProcessor.from_pretrained(model_dir)

    # Load weights with automatic GPU/CPU device placement
    model = AutoModel.from_pretrained(model_dir, device_map='auto')

    dinov3 = model, processor   # cache for subsequent calls


# ---------------------------------------------------------------------------
# SEA-RAFT optical flow initialisation and helpers
# ---------------------------------------------------------------------------

def init_flow_model():
    """
    Lazy-load the SEA-RAFT optical flow model.

    SEA-RAFT is a lightweight recurrent flow estimator derived from RAFT.
    We use the KITTI-tuned checkpoint because driving sequences share similar
    motion statistics (forward translation + mild rotation).

    The model is placed on CUDA and set to eval mode (no gradient tracking).
    Config and checkpoint paths are hard-coded to the bundled third-party
    directory.

    Side effects:
        Sets the global ``flow_model`` and ``flow_args`` variables.
    """
    global flow_model, flow_args
    if flow_model is not None:
        return  # already initialised; skip

    args = {
        "cfg":  "third_parties/SEA-RAFT/config/eval/kitti-M.json",
        "path": "ckpt/Tartan-C-T-TSKH-kitti432x960-M.pth",
    }
    args = argparse.Namespace(**args)
    args = parse_args(args)   # merges JSON config into the namespace

    # Instantiate and load the RAFT model
    model = RAFT(args)
    load_ckpt(model, args.path)  # loads pre-trained weights from disk
    model.to('cuda')
    model.eval()

    flow_model = model
    flow_args  = args


def load_image(imfile: str) -> torch.Tensor:
    """
    Load a single frame from disk and prepare it for SEA-RAFT inference.

    Args:
        imfile (str): Absolute path to the image file.

    Returns:
        torch.Tensor: Normalised image tensor on CUDA.
            shape: (1, 3, H, W), dtype float32, values in [0, 255].
    """
    image = cv2.imread(imfile)                          # BGR uint8 (H, W, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # RGB uint8 (H, W, 3)
    # Convert to (3, H, W) float tensor then add batch dimension
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    # shape: (3, H, W)
    image = image[None].to('cuda')                      # shape: (1, 3, H, W)
    return image


def forward_flow(image1: torch.Tensor,
                 image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single SEA-RAFT forward pass to compute optical flow.

    Uses ``torch.amp.autocast`` for mixed-precision inference, which reduces
    GPU memory usage without measurable accuracy loss for flow estimation.

    Args:
        image1 (torch.Tensor): First frame.  shape: (1, 3, H, W).
        image2 (torch.Tensor): Second frame. shape: (1, 3, H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - **flow_final**: Optical flow field.  shape: (1, 2, H, W).
            - **info_final**: Auxiliary info tensor.  shape: (1, ?, H, W).
    """
    with torch.amp.autocast(device_type="cuda"):
        output = flow_model(image1, image2,
                            iters=flow_args.iters, test_mode=True)
    flow_final = output['flow'][-1]   # shape: (1, 2, H, W)  — last recurrent iter
    info_final = output['info'][-1]   # shape: (1, ?, H, W)
    return flow_final, info_final


def _compute_flow(image1: torch.Tensor,
                  image2: torch.Tensor) -> np.ndarray:
    """
    Compute optical flow between two frames, handling model-resolution scaling.

    SEA-RAFT may require images at a resolution different from the input.
    ``flow_args.scale`` encodes the power-of-two scaling factor.  We:
      1. Upsample both frames by ``2^scale`` before inference.
      2. Downsample the resulting flow by the same factor (and correct
         magnitudes accordingly) so the output is in the original resolution.

    Args:
        image1 (torch.Tensor): First frame.  shape: (1, 3, H, W).
        image2 (torch.Tensor): Second frame. shape: (1, 3, H, W).

    Returns:
        np.ndarray: Dense optical flow array.
            shape: (H, W, 2), where axis-2 contains (u, v) displacement
            in pixels (at the original resolution).
    """
    scale_factor = 2 ** flow_args.scale  # e.g. scale=1 → factor=2

    # Upsample to model's preferred resolution
    img1 = F.interpolate(image1, scale_factor=scale_factor,
                         mode='bilinear', align_corners=False)
    # shape: (1, 3, H*scale, W*scale)
    img2 = F.interpolate(image2, scale_factor=scale_factor,
                         mode='bilinear', align_corners=False)
    # shape: (1, 3, H*scale, W*scale)

    # Run the flow estimator at the upsampled resolution
    flow, info = forward_flow(img1, img2)
    # flow shape: (1, 2, H*scale, W*scale)

    # Downsample flow back to original resolution; multiply by 1/scale to
    # convert flow magnitudes from upsampled coordinates to original pixels.
    inv_scale = 0.5 ** flow_args.scale
    flow_down = F.interpolate(flow, scale_factor=inv_scale,
                              mode='bilinear', align_corners=False) * inv_scale
    # flow_down shape: (1, 2, H, W)

    # Move to CPU and convert to HWC numpy for downstream use
    flow_np = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
    # shape: (H, W, 2)
    return flow_np


# ---------------------------------------------------------------------------
# Motion series computation
# ---------------------------------------------------------------------------

def compute_motion_series(images: list) -> np.ndarray:
    """
    Compute per-adjacent-frame-pair median optical flow magnitudes.

    For each consecutive pair ``(images[t], images[t+1])`` we:
      1. Compute the dense optical flow field (shape: H × W × 2).
      2. Compute per-pixel magnitude: ``sqrt(u^2 + v^2)``.
      3. Take the *median* magnitude (robust to outliers from dynamic objects
         or occluded regions that produce erroneous large flow vectors).

    The resulting series ``mags[t]`` (length T-1) summarises how much the
    scene changed between frames.  It is used both to select keyframes and
    to adaptively set the number of keyframes K.

    Args:
        images (list[str]): Ordered list of frame file paths (length T ≥ 2).

    Returns:
        np.ndarray: Per-frame-pair median flow magnitudes.
            shape: (T-1,), dtype float32.
    """
    assert len(images) >= 2, "At least 2 frames required to compute motion."

    mags = []   # accumulate scalar magnitudes for each consecutive pair

    prev = images[0]   # path to the first frame
    for nxt in images[1:]:
        # Load both frames as (1, 3, H, W) CUDA tensors
        image1_f = load_image(prev)   # shape: (1, 3, H, W)
        image2_f = load_image(nxt)    # shape: (1, 3, H, W)

        # Compute dense flow; output shape: (H, W, 2)
        flow = _compute_flow(image1_f, image2_f)

        # Per-pixel Euclidean magnitude; shape: (H, W)
        flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Median is more robust than mean for scenes with moving foreground
        median_flow = float(torch.from_numpy(flow_magnitude).median().item())
        mags.append(median_flow)
        prev = nxt   # advance the sliding window

    mags = np.asarray(mags, dtype=np.float32)   # shape: (T-1,)
    return mags


# ---------------------------------------------------------------------------
# Arc-length-based keyframe selection
# ---------------------------------------------------------------------------

def select_indices_by_arc_length_abs(
    mags: np.ndarray,
    v_low: float = 0.4,
    v_high: float = 4.0,
    min_k: int = 4,
    max_k: int = 16,
    force_odd_gap: bool = False,
) -> List[int]:
    """
    Select K keyframe indices equidistant in optical-flow arc-length space.

    Motivation:
        A uniform temporal grid over-samples static segments and under-samples
        dynamic ones.  By treating cumulative flow as "arc length" we equate
        each inter-keyframe interval to the same amount of visual change,
        yielding a fairer and more interpretable consistency estimate.

    K selection:
        The number of keyframes K is linearly interpolated between ``min_k``
        (for near-static scenes, ``mean_flow ≤ v_low``) and ``max_k`` (for
        highly dynamic scenes, ``mean_flow ≥ v_high``).  This balances
        compute cost against temporal coverage.

    Arc-length sampling:
        The arc-length array S (length N, where N = T = number of frames) is
        ``S[i] = sum(mags[:i])``.  We pick K target values linearly spaced
        from 0 to S[-1] and find the frame whose cumulative flow is closest
        to each target.

    Args:
        mags (np.ndarray): Per-frame-pair flow magnitudes.
            shape: (T-1,), dtype float32.
        v_low (float): Motion threshold (px/frame) below which K = min_k.
            Default: 0.4.
        v_high (float): Motion threshold (px/frame) above which K = max_k.
            Default: 4.0.
        min_k (int): Minimum number of keyframes.  Default: 4.
        max_k (int): Maximum number of keyframes.  Default: 16.
        force_odd_gap (bool): If True, adjust adjacent keyframe pairs so
            their gap is odd (guarantees an integer midpoint exists).
            Default: False.

    Returns:
        List[int]: Sorted list of K keyframe indices (0-based, length K).
            Always includes index 0 and index N-1.
    """
    N = len(mags) + 1   # number of frames = number of flow pairs + 1

    if N <= 2:
        # Degenerate case: only 1 or 2 frames
        return list(range(N))

    # Mean per-frame motion magnitude (single scalar)
    m_bar = float(mags.mean()) if len(mags) else 0.0

    # Safeguard against coincident thresholds
    if v_high <= v_low:
        v_high = v_low + 1e-6

    # Interpolation ratio r ∈ [0, 1]: 0 = static, 1 = very dynamic
    r = (m_bar - v_low) / (v_high - v_low)
    r = float(np.clip(r, 0.0, 1.0))

    # Number of keyframes: linear interpolation between min_k and max_k
    K = int(round(min_k + r * (max_k - min_k)))
    K = int(np.clip(K, min_k, max_k))

    # --- Arc-length array ---
    # S[i] is the cumulative flow from frame 0 to frame i; shape: (N,)
    S = np.concatenate([[0.0], np.cumsum(mags)])  # shape: (N,)
    S_total = S[-1]                                # total arc length (scalar)

    if S_total <= 1e-6:
        # Nearly zero motion: fall back to uniform temporal sampling
        import pdb
        pdb.set_trace()
        return list(np.linspace(0, N - 1, num=min(K, N), dtype=int))

    # Target arc-length values, evenly spaced from 0 to S_total
    targets = np.linspace(0.0, S_total, num=min(K, N))  # shape: (K,)

    # --- Find frame indices closest to each target arc length ---
    idxs = [0]   # always include the first frame
    ptr  = 1     # pointer into S (advances monotonically)

    for t in targets[1:-1]:    # skip first (0.0) and last (S_total)
        # Advance ptr until S[ptr] just exceeds the target
        while ptr < N and S[ptr] < t:
            ptr += 1

        i = ptr
        # Snap to the nearer neighbour (ptr-1 or ptr)
        if i < N and abs(S[i] - t) > abs(S[i - 1] - t):
            i = i - 1
        idxs.append(int(i))

    idxs.append(N - 1)   # always include the last frame

    # --- Optional: force odd inter-keyframe gaps for integer midpoints ---
    if force_odd_gap and N >= 3 and len(idxs) >= 2:
        adj = [idxs[0]]
        for a, b in zip(idxs[:-1], idxs[1:]):
            # If gap is even, try shifting b right then a left
            if (b - a) % 2 == 0 and (b - a) >= 2:
                if b + 1 < N:
                    b = b + 1
                elif a - 1 >= 0:
                    a = a - 1
            if a <= adj[-1]:
                a = adj[-1]
            if b <= a:
                b = a + 1
            adj[-1] = a
            adj.append(b)
        # Remove duplicates while preserving order
        clean = [adj[0]]
        for x in adj[1:]:
            if x > clean[-1]:
                clean.append(x)
        idxs = clean

    return idxs


# ---------------------------------------------------------------------------
# Pair construction for consistency checking
# ---------------------------------------------------------------------------

def build_pairs_with_mid(idxs: List[int]) -> List[Tuple[int, int, int]]:
    """
    Build ``(i, j, center)`` triples from a list of keyframe indices.

    For each consecutive pair ``(idxs[k], idxs[k+1])`` with a gap of at
    least 2, computes the integer midpoint ``center = (i + j) // 2``.
    The midpoint can be used (e.g.) to check whether the scene is smooth
    at the halfway point between two keyframes.

    Args:
        idxs (List[int]): Sorted list of keyframe indices (from
            ``select_indices_by_arc_length_abs``).

    Returns:
        List[Tuple[int, int, int]]: List of ``(i, j, center)`` triples
            where ``j - i >= 2``.  Pairs with gap < 2 are skipped.
    """
    triples = []
    for i, j in zip(idxs[:-1], idxs[1:]):
        if j - i < 2:
            continue   # gap too small to have a meaningful midpoint
        c = (i + j) // 2   # integer midpoint (floor for even gaps)
        triples.append((i, j, c))
    return triples


# ---------------------------------------------------------------------------
# Main scene consistency computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_scene_consistency_v3(video_list: list, names: list = None) -> float:
    """
    Compute scene-level visual consistency for a list of generated videos.

    For each video:
      1. Compute (or load from cache) the optical flow motion series.
         shape: (T-1,)
      2. Compute (or load from cache) DINOv3 [CLS] embeddings for every
         frame.  shape: (T, D_feat=1024)
      3. Select K keyframe indices via arc-length equidistant sampling.
      4. Extract keyframe embeddings.  shape: (K, 1024)
      5. Compute **adjacent cosine similarity** between consecutive
         keyframe embeddings.  shape: (K-1,)
      6. Average to get a single scene score.

    Only the adjacent similarity ``s2`` is returned (the first-to-all
    similarity ``s1`` is computed but not included in the final score as
    scene content may legitimately change over long driving sequences).

    Args:
        video_list (list[list[str]]): Outer list over videos; each inner
            list contains frame file paths in chronological order.
        names (list[str] | None): Per-video directories for caching flow
            and DINOv3 embeddings.  Must have the same length as
            ``video_list``.

    Returns:
        float: Mean adjacent cosine similarity score across all videos.
            Range: approximately [0, 1]; higher = more consistent.
    """
    # Ensure both models are loaded
    global flow_model, flow_args
    init_flow_model()

    global dinov3
    init_dinov3_model()
    model, processor = dinov3

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)  # operates on feature dim

    print('================= Start Scene Consistency (motion-downsampled) =================')

    scores = []   # accumulate per-video consistency scores

    for idx, frames in tqdm(enumerate(video_list)):
        # ------------------------------------------------------------------
        # Step 1: Compute or load optical flow motion series
        # ------------------------------------------------------------------
        flow_cache = os.path.join(names[idx], 'flow.pkl')
        if not os.path.exists(flow_cache):
            os.makedirs(names[idx], exist_ok=True)
            # Run SEA-RAFT on all consecutive frame pairs
            mags = compute_motion_series(frames)  # shape: (T-1,)
            print(f'Storing flow to: {flow_cache}')
            with open(flow_cache, 'wb') as f:
                pickle.dump(mags, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(flow_cache, 'rb') as f:
                mags = pickle.load(f)              # shape: (T-1,)

        # ------------------------------------------------------------------
        # Step 2: Compute or load DINOv3 frame embeddings
        # ------------------------------------------------------------------
        dino_cache = os.path.join(names[idx], 'dino.pkl')
        if not os.path.exists(dino_cache):
            feats = []   # will become a list of 1024-d tensors
            for k in range(len(frames)):
                try:
                    img = load_image_hf(frames[k])   # HF-normalised PIL image
                except Exception:
                    import pdb
                    pdb.set_trace()

                inputs = processor(images=img, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model(**inputs)

                # pooler_output: [CLS] token; shape: (1, 1024) → squeeze to (1024,)
                feats.append(out.pooler_output.squeeze(0))  # shape: (1024,)

            print(f'Storing DINOv3 embeddings to: {dino_cache}')
            with open(dino_cache, 'wb') as f:
                pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dino_cache, 'rb') as f:
                feats = pickle.load(f)
            # feats: list of T tensors, each shape: (1024,)

        # ------------------------------------------------------------------
        # Step 3: Select keyframe indices via arc-length equidistant sampling
        # ------------------------------------------------------------------
        idxs = select_indices_by_arc_length_abs(
            mags, v_low=1.0, v_high=10.0, min_k=3, max_k=20
        )
        # idxs: list of K ints in [0, T-1]

        if len(idxs) < 2:
            # Not enough keyframes to compute similarity; degenerate score
            scores.append(0.0)
            continue

        print(idxs)  # log selected keyframe positions for debugging

        # ------------------------------------------------------------------
        # Step 4: Gather keyframe embeddings
        # ------------------------------------------------------------------
        feats_sel = [feats[k] for k in idxs]       # list of K tensors (1024,)
        F_mat = torch.stack(feats_sel)              # shape: (K, 1024)

        # ------------------------------------------------------------------
        # Step 5: Compute cosine similarities
        # ------------------------------------------------------------------

        # First-to-all: compare every keyframe to the first one
        F1 = F_mat[0].unsqueeze(0).expand(len(F_mat) - 1, -1)
        # shape: (K-1, 1024) — first keyframe broadcast
        F2 = F_mat[1:]
        # shape: (K-1, 1024) — frames 1 … K-1
        s1 = cos(F1, F2).clamp_min(0).mean()   # scalar

        # Adjacent: compare each keyframe to the next
        Fa = F_mat[:-1]   # shape: (K-1, 1024) — frames 0 … K-2
        Fb = F_mat[1:]    # shape: (K-1, 1024) — frames 1 … K-1
        s2 = cos(Fa, Fb).clamp_min(0).mean()   # scalar

        # Use adjacent similarity as the scene score (s1 excluded because
        # global scene appearance can legitimately change over long sequences)
        scores.append(float(s2))

    score = float(np.mean(scores)) if scores else 0.0
    print(f'[motion-downsampled] DINOv3 consistency raw: {score}')
    return score
